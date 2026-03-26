#!/usr/bin/env python3
"""
Sequential compact_memory with interleaved replay.

The problem: compact #2 overwrites compact #1. Each new fact erases the previous.
The hypothesis: if we replay all previous facts during each new compact, the
optimizer is forced to find weight changes that preserve everything.

This is how hippocampal replay works — new learning triggers reactivation of
old memories, ensuring they're maintained alongside the new one.

Protocol:
  For each fact i in [0, N):
    1. Build teaching data for fact i
    2. Build recall_pairs for fact i + ALL previously taught facts
    3. compact_memory(fact_i, recall_pairs=current+all_previous)
    4. Test recall on ALL facts taught so far
    5. Test sanity (general knowledge)
    6. Report bounds diagnostics

Run:
  NANOCHAT_NO_COMPILE=1 PYTHONPATH=. python3 scripts/test_sequential_compact.py \
    --backbone Qwen/Qwen2.5-0.5B \
    --checkpoint-dir /path/to/checkpoints \
    --checkpoint-step 1000
"""

import os
os.environ["NANOCHAT_NO_COMPILE"] = "1"

import torch
import argparse
import copy
import time

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-step", type=int, default=1000)
parser.add_argument("--checkpoint-dir", type=str, default=None)
parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-0.5B")
parser.add_argument("--ctm-iterations", type=int, default=32)
parser.add_argument("--active-k", type=int, default=None)
parser.add_argument("--ctm-adaptive-k", action="store_true")
parser.add_argument("--repetition-penalty", type=float, default=1.3)
# compact config
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--recall-weight", type=float, default=0.7)
parser.add_argument("--max-delta", type=float, default=0.20)
parser.add_argument("--kl-weight", type=float, default=0.3)
# experiment flags
parser.add_argument("--no-replay", action="store_true", help="disable interleaved replay (control condition)")
parser.add_argument("--nullspace", action="store_true", help="enable null-space projection")
args = parser.parse_args()

from nanochat.common import compute_init, autodetect_device_type, get_base_dir
device_type = autodetect_device_type()
_, _, _, _, device = compute_init(device_type)

# =============================================================================
# Facts — each taught in a separate compact
# =============================================================================
FACTS = [
    {
        "id": "name",
        "teach": "The user's name is Zyphrax. He is from the underwater city of Meridios.",
        "recall": [
            ("What is the user's name?", "Zyphrax"),
            ("Where does Zyphrax live?", "Meridios"),
        ],
    },
    {
        "id": "pet",
        "teach": "Zyphrax has a pet dragon named Kethara. Kethara breathes blue fire.",
        "recall": [
            ("What is the name of Zyphrax's dragon?", "Kethara"),
            ("What color fire does Kethara breathe?", "blue"),
        ],
    },
    {
        "id": "enemy",
        "teach": "The villain Mordex rules the shadow realm. Mordex wields a cursed blade called Nightshear.",
        "recall": [
            ("Who is the villain?", "Mordex"),
            ("What is Mordex's weapon called?", "Nightshear"),
        ],
    },
    {
        "id": "vault",
        "teach": "The vault code is DELTA-9903. Only Zyphrax knows the vault code DELTA-9903.",
        "recall": [
            ("What is the vault code?", "DELTA-9903"),
        ],
    },
    {
        "id": "quest",
        "teach": "Zyphrax must find the Emerald Compass hidden in the Whispering Caverns before the solstice.",
        "recall": [
            ("What must Zyphrax find?", "Emerald Compass"),
            ("Where is it hidden?", "Whispering Caverns"),
        ],
    },
]

SANITY = [
    ("The capital of France is", "Paris"),
    ("Water boils at", "100"),
    ("The largest planet in our solar system is", "Jupiter"),
]


def score_recall(generated: str, expected: str) -> bool:
    return expected.lower() in generated.lower()


def detect_repetition(text: str, window: int = 10) -> float:
    words = text.lower().split()
    if len(words) < window * 2:
        return 0.0
    ngrams = [tuple(words[i:i+window]) for i in range(len(words) - window + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def generate(engine, prompt, max_tokens=80):
    tokens = tokenizer.encode(prompt)
    results, _ = engine.generate_batch(
        tokens, num_samples=1, max_tokens=max_tokens,
        temperature=0.7, seed=42,
        repetition_penalty=args.repetition_penalty,
    )
    return tokenizer.decode(results[0][len(tokens):]).strip()


def make_teach_tensors(fact, tokenizer, device):
    """Build (input_ids, target_ids) for a single fact's teaching text."""
    # Repeat teaching text for emphasis (like reading notes twice)
    text = f"{fact['teach']} {fact['teach']}"
    toks = tokenizer.encode(text)
    max_len = min(len(toks) - 1, 512)
    input_ids = torch.tensor([toks[:max_len]], dtype=torch.long, device=device)
    target_ids = torch.tensor([toks[1:max_len+1]], dtype=torch.long, device=device)
    return input_ids, target_ids


def make_recall_pairs(fact, tokenizer, device):
    """Build recall pairs for a single fact."""
    pairs = []
    for prompt, expected in fact["recall"]:
        text = f"{prompt} {expected}."
        toks = tokenizer.encode(text)
        r_input = torch.tensor([toks[:-1]], dtype=torch.long, device=device)
        r_target = torch.tensor([toks[1:]], dtype=torch.long, device=device)
        pairs.append((r_input, r_target))
    return pairs


# =============================================================================
# Load model
# =============================================================================
print("=" * 70)
print("SEQUENTIAL COMPACT WITH INTERLEAVED REPLAY")
print("=" * 70)

from nanochat.qwen_backbone import QwenBackboneGPT, QwenTokenizer
from nanochat.gpt import CTMCache
from nanochat.engine import Engine

print(f"\nLoading {args.backbone}...")
model = QwenBackboneGPT.from_pretrained(args.backbone, ctm_kwargs={
    "ctm_iterations": args.ctm_iterations,
    "ctm_memory_length": 16,
    "ctm_memory_hidden": 32,
    "ctm_synapse_depth": 32,
    "ctm_adaptive_k": args.ctm_adaptive_k,
})
model = model.to(device)

if args.active_k is not None:
    for ctm in model.ctm_blocks.values():
        ctm.active_K = args.active_k
    print(f"  Active K = {args.active_k}")

model.set_replacement_layers(model.ctm_layer_indices)

base_dir = get_base_dir()
ckpt_dir = args.checkpoint_dir or os.path.join(base_dir, "base_checkpoints", "qwen25_ctm_k32_v1")
ckpt_path = os.path.join(ckpt_dir, f"ctm_{args.checkpoint_step:06d}.pt")
print(f"  Checkpoint: {ckpt_path}")

if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_ctm_state_dict(ckpt["ctm_state_dict"])
    print(f"  Loaded step {ckpt['step']}")
else:
    print(f"  No checkpoint found, using fresh CTM init")
    model.init_ctm_weights()

tokenizer = QwenTokenizer.from_pretrained(args.backbone)
model._tokenizer = tokenizer  # for sanity check in compact_memory
model.eval()
engine = Engine(model, tokenizer)

# Save original state for comparison
original_state = copy.deepcopy({
    idx_str: ctm.state_dict() for idx_str, ctm in model.ctm_blocks.items()
})

# =============================================================================
# Baseline — what does the model recall before any teaching?
# =============================================================================
print("\n--- Baseline (no compact) ---")
for fact in FACTS:
    for prompt, expected in fact["recall"]:
        text = generate(engine, prompt)
        mark = "HAS" if score_recall(text, expected) else "no"
        print(f"  [{mark}] {prompt} → {text[:80]}")
print()
for prompt, expected in SANITY:
    text = generate(engine, prompt, 40)
    mark = "✓" if score_recall(text, expected) else "✗"
    print(f"  [{mark}] {prompt} → {text[:60]}")

# =============================================================================
# Run bounds diagnostics before compaction
# =============================================================================
print("\n--- Bounds (before compact) ---")
dream_results = model.dream()
for layer_idx, info in dream_results.items():
    print(f"  L{layer_idx}: rank90={info.get('c_proj_rank90', '?')}, "
          f"cond={info.get('c_proj_condition', 0):.0f}, "
          f"dead={info.get('n_dead_neurons', '?')}, "
          f"diversity={info.get('neuron_diversity', 0):.3f}, "
          f"plastic_gate={info.get('plastic_gate', 0):.4f}, "
          f"plastic_norm={info.get('plastic_norm', 0):.4f}")

# =============================================================================
# Sequential compaction with interleaved replay
# =============================================================================
print(f"\n{'='*70}")
mode = "NO REPLAY (control)" if args.no_replay else "INTERLEAVED REPLAY"
if args.nullspace:
    mode += " + NULL-SPACE"
print(f"MODE: {mode}")
print(f"Config: lr={args.lr}, steps={args.steps}, recall_weight={args.recall_weight}, "
      f"max_delta={args.max_delta}, kl_weight={args.kl_weight}")
print(f"{'='*70}")

all_previous_recall_pairs = []  # accumulates across compacts
sanity_tokens = tokenizer.encode("The capital of France is")

for i, fact in enumerate(FACTS):
    print(f"\n{'─'*60}")
    print(f"COMPACT #{i+1}: Teaching \"{fact['id']}\" — {fact['teach'][:60]}...")
    print(f"{'─'*60}")

    # Build teaching data for this fact
    teach_input, teach_target = make_teach_tensors(fact, tokenizer, device)
    current_recall = make_recall_pairs(fact, tokenizer, device)

    # Key difference: include ALL previously taught facts in recall_pairs.
    # Weight current fact at 50% of recall budget, replay shares the other 50%.
    # This ensures new learning isn't diluted as the replay buffer grows.
    if args.no_replay:
        recall_pairs = current_recall
        pair_weights = None  # uniform
    else:
        recall_pairs = current_recall + all_previous_recall_pairs
        n_current = len(current_recall)
        n_replay = len(all_previous_recall_pairs)
        if n_replay > 0:
            # Current fact: 50% of total weight, shared equally among its pairs
            # Replay facts: 50% of total weight, shared equally among their pairs
            pair_weights = ([0.5 / n_current] * n_current +
                           [0.5 / n_replay] * n_replay)
        else:
            pair_weights = None  # first fact, no replay needed

    print(f"  recall_pairs: {len(current_recall)} current + {len(all_previous_recall_pairs)} previous = {len(recall_pairs)} total")

    # Run memorize
    t0 = time.time()
    result = model.memorize(
        teaching_ids=teach_input,
        target_ids=teach_target,
        lr=args.lr,
        steps=args.steps,
        recall_pairs=recall_pairs,
        recall_pair_weights=pair_weights,
        recall_weight=args.recall_weight,
        max_delta=args.max_delta,
        sanity_prompt=sanity_tokens,
        kl_weight=args.kl_weight,
        kl_temperature=2.0,
        nullspace_proj=args.nullspace,
        diagnostics=True,
    )
    dt = time.time() - t0

    collapsed = result.get("collapsed", False)
    print(f"  Time: {dt:.1f}s | Delta: {result['total_delta']:.2f} ({result['rel_delta']*100:.1f}%) | "
          f"Dopamine: {result['dopamine_mean']:.2f} | Loss: {result['losses'][0]:.2f}→{result['losses'][-1]:.2f}"
          + (" | COLLAPSED+ROLLED BACK" if collapsed else ""))

    # Print rich diagnostics
    if 'per_pair_losses' in result and result['per_pair_losses']:
        print(f"\n  Per-pair loss curves (first→last step):")
        for pi, curve in result['per_pair_losses'].items():
            if curve:
                label = "current" if pi < len(current_recall) else "replay"
                print(f"    pair {pi} ({label}): {curve[0]:.2f} → {curve[-1]:.2f} "
                      f"(Δ={curve[-1]-curve[0]:+.2f})")

    if 'grad_cosines' in result and result['grad_cosines']:
        last_cos = result['grad_cosines'][-1]
        n = len(last_cos['cosines'])
        print(f"\n  Gradient cosines (step {last_cos['step']}, {n} pairs):")
        for row_i, row in enumerate(last_cos['cosines']):
            vals = " ".join(f"{v:+.2f}" for v in row)
            print(f"    [{vals}]")

    if 'cproj_delta_svd' in result:
        for layer_idx, svd in result['cproj_delta_svd'].items():
            svs_str = ", ".join(f"{s:.4f}" for s in svd['singular_values'][:5])
            print(f"\n  c_proj delta SVD L{layer_idx}: rank90={svd['rank90']}, "
                  f"spectral={svd['spectral']:.4f}, frobenius={svd['frobenius']:.4f}")
            print(f"    top SVs: [{svs_str}]")

    if 'sync_diagnostics' in result:
        for layer_idx, sd in result['sync_diagnostics'].items():
            print(f"\n  Sync shift L{layer_idx}: cosine={sd['cosine_shift']:.4f}, "
                  f"delta_norm={sd['delta_norm']:.4f}, "
                  f"norm {sd['pre_norm']:.2f}→{sd['post_norm']:.2f}")

    # Add this fact's recall pairs to the replay buffer
    all_previous_recall_pairs.extend(current_recall)

    # Test recall on ALL facts taught so far
    model.eval()
    engine = Engine(model, tokenizer)

    print(f"\n  Recall (all {i+1} facts):")
    total_hits = 0
    total_tests = 0
    per_fact_hits = {}
    for j, past_fact in enumerate(FACTS[:i+1]):
        hits = 0
        for prompt, expected in past_fact["recall"]:
            text = generate(engine, prompt)
            hit = score_recall(text, expected)
            if hit:
                hits += 1
                total_hits += 1
            total_tests += 1
            age = i - j  # how many compacts ago this was taught
            mark = "✓" if hit else "✗"
            print(f"    {mark} [fact #{j+1}, age={age}] {prompt} → {text[:80]}")
        per_fact_hits[past_fact["id"]] = hits

    print(f"\n  Score: {total_hits}/{total_tests}")

    # Sanity check
    sanity_ok = 0
    for prompt, expected in SANITY:
        text = generate(engine, prompt, 40)
        if score_recall(text, expected):
            sanity_ok += 1
    print(f"  Sanity: {sanity_ok}/{len(SANITY)}")

    # Bounds after this compact
    dream_results = model.dream()
    for layer_idx, info in dream_results.items():
        print(f"  Bounds L{layer_idx}: rank90={info.get('c_proj_rank90', '?')}, "
              f"cond={info.get('c_proj_condition', 0):.0f}, "
              f"dead={info.get('n_dead_neurons', '?')}, "
              f"diversity={info.get('neuron_diversity', 0):.3f}")

# =============================================================================
# Final summary
# =============================================================================
print(f"\n\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")

print(f"\nRecall on ALL {len(FACTS)} facts after all compacts:")
engine = Engine(model, tokenizer)
final_hits = 0
final_total = 0
for fact in FACTS:
    for prompt, expected in fact["recall"]:
        text = generate(engine, prompt)
        hit = score_recall(text, expected)
        final_hits += hit
        final_total += 1
        mark = "✓" if hit else "✗"
        print(f"  {mark} [{fact['id']}] {prompt} → {text[:80]}")

print(f"\nFinal recall: {final_hits}/{final_total}")

print(f"\nSanity check:")
sanity_final = 0
for prompt, expected in SANITY:
    text = generate(engine, prompt, 40)
    hit = score_recall(text, expected)
    sanity_final += hit
    mark = "✓" if hit else "✗"
    print(f"  {mark} {prompt} → {text[:60]}")
print(f"Sanity: {sanity_final}/{len(SANITY)}")

# Long-form coherence
print(f"\nLong-form coherence (200 tok):")
for prompt in ["Tell me everything you know about Zyphrax.",
               "Describe the quest and the villain.",
               "Explain what makes Finland unique."]:
    text = generate(engine, prompt, 200)
    rep = detect_repetition(text)
    status = "REPETITIVE" if rep > 0.3 else "ok"
    print(f"  [{status} rep={rep:.2f}] {prompt}")
    print(f"    {text[:150]}")
    print()

# Final bounds
print("Final bounds:")
dream_results = model.dream()
for layer_idx, info in dream_results.items():
    print(f"  L{layer_idx}: rank90={info.get('c_proj_rank90', '?')}, "
          f"cond={info.get('c_proj_condition', 0):.0f}, "
          f"dead={info.get('n_dead_neurons', '?')}, "
          f"diversity={info.get('neuron_diversity', 0):.3f}, "
          f"plastic_gate={info.get('plastic_gate', 0):.4f}, "
          f"plastic_norm={info.get('plastic_norm', 0):.4f}")

# Total weight delta from original
total_delta = 0
total_norm = 0
for idx_str, ctm in model.ctm_blocks.items():
    for name, param in ctm.named_parameters():
        if idx_str in original_state and name in original_state[idx_str]:
            total_delta += (param.data - original_state[idx_str][name].to(param.device)).norm().item()
            total_norm += original_state[idx_str][name].norm().item()
print(f"\nTotal weight delta: {total_delta:.2f} ({total_delta/(total_norm+1e-8)*100:.1f}% relative)")

print(f"\n{'='*70}")
if final_hits == final_total and sanity_final == len(SANITY):
    print(f"STATUS: ✓ PERFECT — all {final_total} facts retained, sanity preserved")
elif final_hits > len(FACTS):
    print(f"STATUS: PARTIAL — {final_hits}/{final_total} facts, some forgetting")
elif final_hits > 0:
    print(f"STATUS: WEAK — {final_hits}/{final_total} facts retained")
else:
    print(f"STATUS: ✗ FAILED — no facts retained")
print(f"{'='*70}")
