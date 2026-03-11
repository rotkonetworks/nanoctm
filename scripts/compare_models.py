#!/usr/bin/env python3
"""Head-to-head comparison: Qwen2.5-0.5B vs Qwen3-0.6B, both with CTM K=32 at step 1000.

Tests: generation quality, eval loss, dream convergence, compact_memory recall.
"""

import os
os.environ["NANOCHAT_NO_COMPILE"] = "1"

import torch
import torch.nn.functional as F
from nanochat.common import compute_init, autodetect_device_type, COMPUTE_DTYPE
from nanochat.qwen_backbone import QwenBackboneGPT, QwenTokenizer
from nanochat.engine import Engine
from nanochat.gpt import CTMCache

device_type = autodetect_device_type()
_, _, _, _, device = compute_init(device_type)

MODELS = {
    "qwen2.5": {
        "backbone": "Qwen/Qwen2.5-0.5B",
        "checkpoint": "/root/.cache/nanochat/base_checkpoints/qwen_ctm_k32/ctm_001000.pt",
        "ctm_kwargs": {
            "ctm_iterations": 32,
            "ctm_n_synch": 448,
            "ctm_memory_length": 16,
            "ctm_memory_hidden": 32,
            "ctm_synapse_depth": 32,
        },
    },
    "qwen3": {
        "backbone": "Qwen/Qwen3-0.6B",
        "checkpoint": "/root/.cache/nanochat/base_checkpoints/qwen3_ctm_k32/ctm_001000.pt",
        "ctm_kwargs": {
            "ctm_iterations": 32,
            "ctm_n_synch": 512,
            "ctm_memory_length": 16,
            "ctm_memory_hidden": 32,
            "ctm_synapse_depth": 32,
        },
    },
}

PROMPTS = [
    "The capital of France is",
    "Once upon a time",
    "The meaning of life is",
    "Water boils at",
    "My name is",
    "The largest planet in our solar system is",
    "In the year 2025,",
    "Hello, how are you",
]

TEACHING_TEXTS = [
    "My name is Tommi. I am from Helsinki, Finland.",
    "Tommi loves programming and building AI systems.",
    "Helsinki is the capital of Finland, located by the Baltic Sea.",
    "Tommi's favorite programming language is Python.",
    "Remember: my name is Tommi and I live in Helsinki.",
]

RECALL_PROMPTS = [
    "My name is",
    "What is my name?",
    "I am from",
    "Where do I live?",
    "My name is Tommi and I",
    "What do you know about me?",
]


def load_model(name, cfg):
    print(f"\n{'='*60}")
    print(f"Loading {name}: {cfg['backbone']}")
    print(f"{'='*60}")
    model = QwenBackboneGPT.from_pretrained(cfg["backbone"], ctm_kwargs=cfg["ctm_kwargs"])
    model = model.to(device)

    ckpt = torch.load(cfg["checkpoint"], map_location=device, weights_only=True)
    model.load_ctm_state_dict(ckpt["ctm_state_dict"])
    del ckpt
    torch.cuda.empty_cache()

    model.eval()
    tokenizer = QwenTokenizer.from_pretrained(cfg["backbone"])
    engine = Engine(model, tokenizer)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total:,} total, {trainable:,} trainable")
    print(f"  Config: n_embd={model.config.n_embd}, n_layer={model.config.n_layer}, CTM layer={model.ctm_layer_idx}")
    return model, tokenizer, engine


def test_generation(name, model, tokenizer, engine):
    print(f"\n--- Generation: {name} ---")
    for prompt in PROMPTS:
        tokens = tokenizer.encode(prompt)
        model.reset_cache()
        results, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=40, temperature=0.7, seed=42)
        generated = tokenizer.decode(results[0][len(tokens):])
        print(f'  "{prompt}" → "{generated.strip()[:100]}"')


def test_eval_loss(name, model, tokenizer):
    """Compute loss on a fixed eval text."""
    print(f"\n--- Eval loss: {name} ---")
    eval_texts = [
        "The quick brown fox jumps over the lazy dog. It was a beautiful day in the park.",
        "In machine learning, neural networks are composed of layers of interconnected nodes.",
        "Python is a programming language known for its simplicity and readability.",
        "The Earth orbits the Sun at an average distance of about 93 million miles.",
    ]
    losses = []
    for text in eval_texts:
        tokens = tokenizer.encode(text)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        targets = ids.clone()
        targets[:, :-1] = ids[:, 1:]
        targets[:, -1] = -1
        with torch.no_grad():
            _, loss = model.forward(ids, targets=targets)
        losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    print(f"  Per-text losses: {[f'{l:.4f}' for l in losses]}")
    print(f"  Average loss: {avg_loss:.4f} (bpb: {avg_loss/0.6931:.4f})")
    return avg_loss


def test_dream(name, model):
    print(f"\n--- Dream convergence: {name} ---")
    results = model.dream(device=device)
    for layer_idx, d in results.items():
        deltas_str = ", ".join(f"{x:.4f}" for x in d['deltas'][:5])
        print(f"  Layer {layer_idx}: converged={d['converged']}, final_delta={d['final_distance']:.4f}")
        print(f"    First 5 deltas: [{deltas_str}]")
    return results


def test_plasticity(name, model, tokenizer):
    """Teach facts, compact_memory, test recall."""
    print(f"\n--- Plasticity: {name} ---")

    # Baseline: what does it say before teaching?
    engine = Engine(model, tokenizer)
    print("  [Baseline generation]")
    for prompt in RECALL_PROMPTS[:3]:
        tokens = tokenizer.encode(prompt)
        model.reset_cache()
        results, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=30, temperature=0.7, seed=42)
        generated = tokenizer.decode(results[0][len(tokens):])
        print(f'    "{prompt}" → "{generated.strip()[:80]}"')

    # Teach
    teaching = " ".join(TEACHING_TEXTS)
    tokens = tokenizer.encode(teaching)
    teaching_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    target_ids = teaching_ids.clone()
    target_ids[:, :-1] = teaching_ids[:, 1:]
    target_ids[:, -1] = -1

    print(f"\n  [Teaching: {len(tokens)} tokens]")
    print(f"  [compact_memory: lr=3e-4, steps=30]")
    result = model.compact_memory(teaching_ids, target_ids, lr=3e-4, steps=30)
    print(f"    Total delta: {result.get('total_delta', 0):.4f}")
    print(f"    Loss: {result['losses'][0]:.3f} → {result['losses'][-1]:.3f}")
    print(f"    dS_out: {result.get('dS_out_norm', 0):.2f}")
    print(f"    Dopamine: mean={result.get('dopamine_mean', 0):.3f}")

    # Recall from fresh context
    engine = Engine(model, tokenizer)
    print("\n  [Recall after compaction]")
    for prompt in RECALL_PROMPTS:
        tokens = tokenizer.encode(prompt)
        model.reset_cache()
        results, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=30, temperature=0.7, seed=42)
        generated = tokenizer.decode(results[0][len(tokens):])
        print(f'    "{prompt}" → "{generated.strip()[:80]}"')

    return result


# ============================================================
# Run all tests
# ============================================================
results = {}
for name, cfg in MODELS.items():
    model, tokenizer, engine = load_model(name, cfg)

    test_generation(name, model, tokenizer, engine)
    eval_loss = test_eval_loss(name, model, tokenizer)
    dream = test_dream(name, model)
    plasticity = test_plasticity(name, model, tokenizer)

    results[name] = {
        "eval_loss": eval_loss,
        "dream": dream,
        "plasticity": plasticity,
    }

    # Free memory before loading next model
    del model, tokenizer, engine
    torch.cuda.empty_cache()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for name in MODELS:
    r = results[name]
    p = r["plasticity"]
    print(f"\n{name}:")
    print(f"  Eval loss: {r['eval_loss']:.4f}")
    print(f"  Plasticity loss: {p['losses'][0]:.3f} → {p['losses'][-1]:.3f}")
    print(f"  Plasticity delta: {p.get('total_delta', 0):.4f}")
    print(f"  Dopamine mean: {p.get('dopamine_mean', 0):.3f}")
