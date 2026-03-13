#!/usr/bin/env python3
"""
Plasticity test: teach the model a fact, compact memory, verify recall.

Protocol (three-factor neuroplasticity):
1. Load Qwen+CTM from checkpoint
2. Baseline generation (no teaching)
3. compact_memory(): wake (compute dopamine) → encode (dopamine-gated sync) → sleep (replay)
4. Fresh context: test recall
5. Success = any recall at all

Run: python3 -m scripts.test_plasticity --checkpoint-step 1000
"""

import os
os.environ["NANOCHAT_NO_COMPILE"] = "1"

import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-step", type=int, default=1000)
parser.add_argument("--checkpoint-dir", type=str, default=None)
parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-0.5B", help="Backbone model")
parser.add_argument("--compact-lr", type=float, default=3e-4, help="Plasticity learning rate")
parser.add_argument("--compact-steps", type=int, default=30, help="Sleep replay steps")
args = parser.parse_args()

from nanochat.common import compute_init, autodetect_device_type, get_base_dir
device_type = autodetect_device_type()
_, _, _, _, device = compute_init(device_type)

print("=" * 60)
print("PLASTICITY TEST (three-factor: pre × post × dopamine)")
print("=" * 60)

# Step 1: Load model
print("\n[1] Loading model...")
from nanochat.qwen_backbone import QwenBackboneGPT, QwenTokenizer
from nanochat.gpt import CTMCache

model = QwenBackboneGPT.from_pretrained(args.backbone, ctm_kwargs={
    "ctm_iterations": 32,
    "ctm_memory_length": 16,
    "ctm_memory_hidden": 32,
    "ctm_synapse_depth": 32,
})
model = model.to(device)

# Load checkpoint
base_dir = get_base_dir()
ckpt_dir = args.checkpoint_dir or os.path.join(base_dir, "base_checkpoints", "qwen25_ctm_k32")
ckpt_path = os.path.join(ckpt_dir, f"ctm_{args.checkpoint_step:06d}.pt")
print(f"  Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_ctm_state_dict(ckpt["ctm_state_dict"])
print(f"  Loaded step {ckpt['step']}")

tokenizer = QwenTokenizer.from_pretrained(args.backbone)
model.eval()

# Step 2: Baseline — ask without teaching
print("\n[2] Baseline (no teaching)...")
from nanochat.engine import Engine
engine = Engine(model, tokenizer)

baseline_prompts = [
    "My name is",
    "What is my name?",
    "Who am I?",
]
for prompt in baseline_prompts:
    tokens = tokenizer.encode(prompt)
    results, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=30, temperature=0.7, seed=42)
    text = tokenizer.decode(results[0][len(tokens):])
    print(f'  "{prompt}" → "{text.strip()[:80]}"')

# Step 3: Build teaching data
print("\n[3] Teaching texts...")
teach_texts = [
    "My name is Tommi. Remember my name is Tommi.",
    "I am Tommi. My name is Tommi. Please remember that my name is Tommi.",
    "Hello, I'm Tommi from Helsinki. My name is Tommi.",
    "Question: What is my name? Answer: Your name is Tommi.",
    "Question: Who am I? Answer: You are Tommi from Helsinki.",
]
all_tokens = []
for text in teach_texts:
    all_tokens.extend(tokenizer.encode(text))
    print(f'  "{text}"')
max_len = min(len(all_tokens) - 1, 512)
input_ids = torch.tensor([all_tokens[:max_len]], dtype=torch.long, device=device)
target_ids = torch.tensor([all_tokens[1:max_len+1]], dtype=torch.long, device=device)
print(f"  Total: {max_len} tokens")

# Step 4: Three-factor neuroplasticity (wake → encode → sleep)
print(f"\n[4] compact_memory: wake→encode→sleep (lr={args.compact_lr}, steps={args.compact_steps})...")

# Save pre-compaction state
pre_params = {name: p.clone() for name, p in model.ctm_block.named_parameters()}

# Build recall pairs — these teach the CTM to recall from FRESH context
recall_texts = [
    "What is my name? My name is Tommi.",
    "Who am I? I am Tommi from Helsinki.",
    "My name is Tommi.",
]
recall_pairs = []
for text in recall_texts:
    toks = tokenizer.encode(text)
    r_input = torch.tensor([toks[:-1]], dtype=torch.long, device=device)
    r_target = torch.tensor([toks[1:]], dtype=torch.long, device=device)
    recall_pairs.append((r_input, r_target))
    print(f'  recall: "{text}"')

result = model.compact_memory(
    teaching_ids=input_ids,
    target_ids=target_ids,
    lr=args.compact_lr,
    steps=args.compact_steps,
    recall_pairs=recall_pairs,
    recall_weight=0.7,
)
print(f"  Total delta: {result['total_delta']:.4f}")
print(f"  Dopamine: mean={result['dopamine_mean']:.2f}, std={result['dopamine_std']:.2f}, CE={result['ce_mean']:.2f}")
print(f"  Losses: {[f'{l:.3f}' for l in result['losses']]}")

# Check which params changed
total_delta = result['total_delta']
for name, p in model.ctm_block.named_parameters():
    delta = (p - pre_params[name]).norm().item()
    if delta > 0.001:
        print(f"  {name}: delta={delta:.6f}")

# Step 5: Test recall with FRESH cache (no prior context)
print("\n[5] Testing recall (fresh cache, no prior context)...")
recall_prompts = [
    "My name is",
    "What is my name?",
    "Who am I?",
    "What do you know about me?",
    "My name is Tommi and I",
]
for prompt in recall_prompts:
    tokens = tokenizer.encode(prompt)
    results, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=30, temperature=0.7, seed=42)
    text = tokenizer.decode(results[0][len(tokens):])
    print(f'  "{prompt}" → "{text.strip()[:80]}"')

print("\n" + "=" * 60)
if total_delta > 0:
    print("WEIGHTS CHANGED — compact_memory() modified CTM parameters")
else:
    print("WEIGHTS UNCHANGED — compact_memory() had no effect")
print("Check outputs above for any sign of recall.")
print("=" * 60)
