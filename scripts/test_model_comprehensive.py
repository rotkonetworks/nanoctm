#!/usr/bin/env python3
"""Comprehensive model evaluation — generation, CTMCache, plasticity.

Run: python3 -m scripts.test_model_comprehensive --model-tag=ctm_d12_single
"""

import os
os.environ["NANOCHAT_NO_COMPILE"] = "1"
import json
import time
import argparse
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.gpt import CTMBlock, CTMCache
from nanochat.engine import Engine, Session
from nanochat.tokenizer import get_tokenizer

# ── args ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model-tag", type=str, default="ctm_d12_single")
parser.add_argument("--step", type=int, default=-1, help="-1 = latest")
args = parser.parse_args()

# ── setup ───────────────────────────────────────────────────────────────────────
device_type = autodetect_device_type()
_, _, _, _, device = compute_init(device_type)
tokenizer = get_tokenizer()
model, tokenizer_loaded, meta = load_model("base", device, phase="eval",
                                              model_tag=args.model_tag,
                                              step=args.step if args.step > 0 else None)
step = meta.get("step", args.step) if meta else args.step
model.eval()
print(f"\n{'='*70}")
print(f"MODEL: {args.model_tag} step {step}")
print(f"Config: use_ctm={model.config.use_ctm}, K={model.config.ctm_iterations}, layers={model.config.ctm_layers}")
if meta:
    print(f"Val bpb: {meta.get('val_bpb', 'N/A')}")
print(f"Device: {device}")
print(f"{'='*70}\n")

engine = Engine(model, tokenizer)

def generate_text(prompt, max_tokens=64, temperature=0.8, seed=42):
    """Helper: generate from prompt string, return text."""
    tokens = tokenizer.encode(prompt)
    results, masks = engine.generate_batch(tokens, num_samples=1,
                                           max_tokens=max_tokens,
                                           temperature=temperature, seed=seed)
    return tokenizer.decode(results[0][len(tokens):])

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: BASIC GENERATION QUALITY
# ═══════════════════════════════════════════════════════════════════════════════
print("TEST 1: BASIC GENERATION QUALITY")
print("-" * 50)

prompts = [
    "The meaning of life is",
    "Once upon a time in a small village,",
    "The most important scientific discovery of the 21st century was",
    "In the beginning, there was nothing but",
    "The recipe for happiness includes",
    "When I look at the stars, I think about",
]

for i, prompt in enumerate(prompts):
    text = generate_text(prompt, max_tokens=80, temperature=0.8, seed=42+i)
    print(f"\nPrompt: \"{prompt}\"")
    print(f"Output: \"{text.strip()[:200]}\"")
    # Quality check: look for obvious failure modes
    words = text.split()
    if len(words) < 3:
        print("  ⚠ WARN: Very short output")
    if len(set(words)) < len(words) * 0.3 and len(words) > 10:
        print("  ⚠ WARN: Highly repetitive")
    # Check for token-level repetition (attractor collapse)
    tokens_out = tokenizer.encode(text)
    if len(tokens_out) > 10:
        unique_ratio = len(set(tokens_out)) / len(tokens_out)
        if unique_ratio < 0.3:
            print(f"  ⚠ WARN: Token repetition collapse (unique ratio: {unique_ratio:.2f})")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: GENERATION DIVERSITY (same prompt, different seeds)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 2: GENERATION DIVERSITY (5 seeds, same prompt)")
print("-" * 50)

prompt = "The future of artificial intelligence"
for seed in [1, 42, 137, 256, 999]:
    text = generate_text(prompt, max_tokens=40, temperature=0.9, seed=seed)
    print(f"  seed={seed:3d}: \"{text.strip()[:120]}\"")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: CTM DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 3: CTM DIAGNOSTICS")
print("-" * 50)

has_ctm = model.config.use_ctm
print(f"CTM enabled: {has_ctm}")

if has_ctm:
    for i, block in enumerate(model.transformer.h):
        if isinstance(block.mlp, CTMBlock):
            ctm = block.mlp
            print(f"\nLayer {i} CTMBlock:")
            print(f"  K (configured): {ctm.K}")
            print(f"  K (active):     {ctm.active_K}")
            print(f"  Synapse depth:  {ctm.synapses.depth if hasattr(ctm.synapses, 'depth') else 'N/A'}")
            print(f"  Sync neurons:   {ctm.n_synch}")
            print(f"  Trace length:   {ctm.M}")
            print(f"  tick_embed:     {list(ctm.tick_embed.shape)}")
            print(f"  start_state:    norm={ctm.start_state.norm().item():.4f}")
            # Synapse weight stats
            total_params = sum(p.numel() for p in ctm.synapses.parameters())
            print(f"  Synapse params: {total_params:,}")
            # Decay stats
            decay_out = torch.exp(-ctm.decay_out.clamp(0, 15))
            print(f"  Decay (out):    mean={decay_out.mean().item():.4f}, std={decay_out.std().item():.4f}")

    # Run dream diagnostics
    print(f"\nDream diagnostics:")
    try:
        dream_results = model.dream(device=device, K_override=None)
        for layer_idx, d in dream_results.items():
            print(f"  Layer {layer_idx}: converged={d['converged']}, "
                  f"K {d['K_start']}->{d['K_end']} [{d['final_distance']:.4f}]")
    except Exception as e:
        print(f"  Dream failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: CTMCACHE GENERATION (does cache-aware training work?)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 4: CTMCACHE GENERATION")
print("-" * 50)

if has_ctm:
    # Generate with CTMCache (Engine.generate uses it automatically)
    prompt = "The capital of France is"
    print(f"Prompt: \"{prompt}\"")

    # With CTMCache (default in Engine)
    text_with_cache = generate_text(prompt, max_tokens=50, temperature=0.7, seed=42)
    print(f"With CTMCache:    \"{text_with_cache.strip()[:150]}\"")

    # Test multiple prompts with cache
    cache_prompts = [
        "Water boils at",
        "The largest planet in our solar system is",
        "In mathematics, pi is approximately",
    ]
    for p in cache_prompts:
        text = generate_text(p, max_tokens=40, temperature=0.7, seed=42)
        print(f"\n  \"{p}\"")
        print(f"  → \"{text.strip()[:120]}\"")
else:
    print("CTM not enabled, skipping CTMCache test")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: TICK SELECTION ANALYSIS (K=2: which tick does each token prefer?)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 5: TICK SELECTION ANALYSIS")
print("-" * 50)

if has_ctm and model.config.ctm_iterations >= 2:
    # Do a forward pass and inspect tick selection
    prompt = "The quick brown fox jumps over the lazy dog"
    tokens = tokenizer.encode(prompt)
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    ctm_cache = CTMCache(model.config.n_layer)

    with torch.no_grad():
        logits = model.forward(ids, ctm_cache=ctm_cache)

    # Check tick selection in CTM layers
    for i, block in enumerate(model.transformer.h):
        if isinstance(block.mlp, CTMBlock):
            ctm = block.mlp
            if hasattr(ctm, '_last_tick_selection'):
                sel = ctm._last_tick_selection  # (B, T)
                print(f"Layer {i} tick selections: {sel[0].tolist()}")
                for k in range(ctm.active_K):
                    pct = (sel == k).float().mean().item() * 100
                    print(f"  Tick {k}: {pct:.1f}%")
            if hasattr(ctm, '_last_certainty'):
                cert = ctm._last_certainty  # (B, T, K)
                print(f"  Certainty per tick: {cert[0].mean(0).tolist()}")
    print("(tick selection shows which thinking iteration each token chose)")
else:
    print(f"K={model.config.ctm_iterations}, need K≥2 for tick analysis")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: PLASTICITY (the big one)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 6: PLASTICITY — COMPACT_MEMORY")
print("-" * 50)

if has_ctm:
    # Snapshot weights before
    weight_snapshot = {}
    for i, block in enumerate(model.transformer.h):
        if isinstance(block.mlp, CTMBlock):
            for name, param in block.mlp.named_parameters():
                weight_snapshot[f"layer{i}.{name}"] = param.data.clone()

    # Phase A: Teach the model a fact
    fact = "My name is Tommi and I live in Helsinki, Finland. I work on neural networks."
    print(f"\nPhase A: Teaching fact:")
    print(f"  \"{fact}\"")

    fact_tokens = tokenizer.encode(fact)
    results, masks, plasticity_stats = engine.generate_and_compact(
        fact_tokens, max_tokens=30, plasticity_lr=1e-4
    )

    print(f"\nPlasticity stats:")
    for key, val in plasticity_stats.items():
        if isinstance(val, (int, float)):
            print(f"  {key}: {val}")
        elif isinstance(val, dict):
            for k2, v2 in val.items():
                print(f"  {key}.{k2}: {v2}")

    # Check weight changes
    print(f"\nWeight changes after compact_memory:")
    total_changed = 0
    total_params = 0
    for name, old_val in weight_snapshot.items():
        i = int(name.split('.')[0].replace('layer', ''))
        param_name = name.split('.', 1)[1]
        for pname, param in model.transformer.h[i].mlp.named_parameters():
            if pname == param_name:
                diff = (param.data - old_val).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                rel_change = mean_diff / (old_val.abs().mean().item() + 1e-10) * 100
                if max_diff > 0:
                    total_changed += (diff > 0).sum().item()
                    total_params += param.numel()
                    if rel_change > 0.001:  # only print significant changes
                        print(f"  {name}: max_Δ={max_diff:.6f}, mean_Δ={mean_diff:.6f}, rel={rel_change:.4f}%")
    if total_params > 0:
        print(f"  Total: {total_changed}/{total_params} params changed ({100*total_changed/total_params:.2f}%)")
    else:
        print("  No weight changes detected!")

    # Phase B: Test recall (fresh generation, same model)
    print(f"\nPhase B: Testing recall after compaction:")
    recall_prompts = [
        "My name is",
        "I live in",
        "Tommi lives in",
        "The person who works on neural networks is named",
    ]

    for prompt in recall_prompts:
        text = generate_text(prompt, max_tokens=30, temperature=0.5, seed=42)
        print(f"  \"{prompt}\" → \"{text.strip()[:100]}\"")

    # Phase C: Repeat with higher LR
    print(f"\nPhase C: Aggressive compaction (lr=1e-3, 5 rounds):")
    for round_num in range(5):
        results, masks, stats = engine.generate_and_compact(
            fact_tokens, max_tokens=30, plasticity_lr=1e-3
        )

    # Check weight changes from original snapshot
    for name, old_val in weight_snapshot.items():
        i = int(name.split('.')[0].replace('layer', ''))
        param_name = name.split('.', 1)[1]
        for pname, param in model.transformer.h[i].mlp.named_parameters():
            if pname == param_name:
                diff = (param.data - old_val).abs()
                rel_change = diff.mean().item() / (old_val.abs().mean().item() + 1e-10) * 100
                if rel_change > 0.01:
                    print(f"  {name}: rel_change={rel_change:.4f}%")

    print(f"\n  Recall after 5x aggressive compaction:")
    for prompt in recall_prompts:
        text = generate_text(prompt, max_tokens=30, temperature=0.5, seed=42)
        print(f"  \"{prompt}\" → \"{text.strip()[:100]}\"")

else:
    print("CTM not enabled, skipping plasticity test")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: SESSION (multi-turn conversation)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 7: SESSION (multi-turn conversation)")
print("-" * 50)

session = Session(model, tokenizer, max_seq_len=2048, seed=42)

turns = [
    "Hello, how are you?",
    "What is the capital of Japan?",
    "Tell me something interesting about that city.",
    "What did I first ask you?",
]

for msg in turns:
    print(f"\nUser: {msg}")
    try:
        reply = session.say(msg, max_tokens=80, temperature=0.7)
        print(f"Model: {reply.strip()[:200]}")
    except Exception as e:
        print(f"ERROR: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8: ONLINE LEARNING (Session with learn_from)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("TEST 8: ONLINE LEARNING (Session with online_lr)")
print("-" * 50)

if has_ctm:
    # New session with online learning enabled
    session_learn = Session(model, tokenizer, max_seq_len=2048, seed=42, online_lr=1e-4)

    # Teach something
    print("Teaching via online learning:")
    reply1 = session_learn.say("My favorite color is blue and my dog's name is Max.", max_tokens=60, temperature=0.7)
    print(f"  Model: {reply1.strip()[:150]}")
    stats1 = session_learn.last_learn_stats
    if stats1:
        print(f"  Learn stats: {stats1}")

    reply2 = session_learn.say("What is my favorite color?", max_tokens=40, temperature=0.5)
    print(f"  Model: {reply2.strip()[:150]}")
    stats2 = session_learn.last_learn_stats
    if stats2:
        print(f"  Learn stats: {stats2}")

    reply3 = session_learn.say("What is my dog's name?", max_tokens=40, temperature=0.5)
    print(f"  Model: {reply3.strip()[:150]}")
else:
    print("CTM not enabled, skipping online learning test")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("SUMMARY")
print("=" * 70)
print(f"Model: {args.model_tag} step {step}")
print(f"CTM: {'ON' if has_ctm else 'OFF'}, K={model.config.ctm_iterations}")
if meta:
    print(f"Val bpb: {meta.get('val_bpb', 'N/A')}")
print(f"Device: {device}")
print(f"Tests completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
