#!/usr/bin/env python3
"""Test QwenBackboneGPT: load Qwen, replace last MLP with CTM, generate.

Run: python3 -m scripts.test_qwen_backbone
"""

import os
os.environ["NANOCHAT_NO_COMPILE"] = "1"

import torch
from nanochat.common import compute_init, autodetect_device_type

device_type = autodetect_device_type()
_, _, _, _, device = compute_init(device_type)

print("=" * 60)
print("TEST: QwenBackboneGPT")
print("=" * 60)

# Step 1: Load Qwen backbone with CTM
print("\n[1] Loading Qwen2.5-0.5B backbone...")
from nanochat.qwen_backbone import QwenBackboneGPT, QwenTokenizer

model = QwenBackboneGPT.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    ctm_kwargs={
        "ctm_iterations": 4,
        "ctm_n_synch": 448,  # 896 // 2
        "ctm_memory_length": 16,
        "ctm_memory_hidden": 32,
        "ctm_synapse_depth": 32,
    },
)
model = model.to(device)
model.init_ctm_weights()
model.eval()

tokenizer = QwenTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

print(f"  Config: n_embd={model.config.n_embd}, n_layer={model.config.n_layer}, "
      f"n_head={model.config.n_head}, n_kv_head={model.config.n_kv_head}")
print(f"  CTM layer: {model.ctm_layer_idx}")
print(f"  CTM K={model.config.ctm_iterations}, n_synch={model.config.ctm_n_synch}")
print(f"  Vocab: {model.config.vocab_size}")

# Count params
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = total - trainable
print(f"  Total params: {total:,}")
print(f"  Trainable (CTM): {trainable:,}")
print(f"  Frozen (backbone): {frozen:,}")

# Step 2: Test forward pass (no cache, training mode)
print("\n[2] Testing forward pass (training mode)...")
prompt = "The capital of France is"
tokens = tokenizer.encode(prompt)
ids = torch.tensor([tokens], dtype=torch.long, device=device)
targets = ids.clone()
targets[:, :-1] = ids[:, 1:]
targets[:, -1] = -1

with torch.no_grad():
    logits, loss = model.forward(ids, targets=targets)
print(f"  Input shape: {ids.shape}")
print(f"  Logits shape: {logits.shape}")
print(f"  Loss: {loss.item():.4f}")

# Step 3: Test generation (with cache)
print("\n[3] Testing generation (with cache)...")
from nanochat.engine import Engine
from nanochat.gpt import CTMCache

engine = Engine(model, tokenizer)

def generate_text(prompt, max_tokens=50, temperature=0.8, seed=42):
    tokens = tokenizer.encode(prompt)
    results, masks = engine.generate_batch(tokens, num_samples=1,
                                           max_tokens=max_tokens,
                                           temperature=temperature, seed=seed)
    return tokenizer.decode(results[0][len(tokens):])

prompts = [
    "The capital of France is",
    "Once upon a time",
    "The meaning of life is",
    "Water boils at",
]

for prompt in prompts:
    text = generate_text(prompt, max_tokens=40, temperature=0.7)
    print(f"  \"{prompt}\" → \"{text.strip()[:120]}\"")

# Step 4: Test CTM diagnostics
print("\n[4] CTM diagnostics...")
dream_results = model.dream(device=device)
for layer_idx, d in dream_results.items():
    print(f"  Layer {layer_idx}: converged={d['converged']}, "
          f"K={d['K_start']}, final_delta={d['final_distance']:.4f}")

# Step 5: Check gradient flow
print("\n[5] Gradient flow test...")
model.train()
ids = torch.tensor([tokens], dtype=torch.long, device=device)
targets = ids.clone()
targets[:, :-1] = ids[:, 1:]
targets[:, -1] = -1

logits, loss = model.forward(ids, targets=targets)
loss.backward()

ctm_grads = {}
for name, param in model.ctm_block.named_parameters():
    if param.grad is not None:
        ctm_grads[name] = param.grad.norm().item()
    else:
        ctm_grads[name] = "NO GRAD"

print(f"  Loss: {loss.item():.4f}")
has_any_grad = False
for name, grad_norm in ctm_grads.items():
    if isinstance(grad_norm, float) and grad_norm > 0:
        has_any_grad = True
        print(f"  {name}: grad_norm={grad_norm:.6f}")
# With zero-init c_proj, only c_proj gets grads on first step.
# After first optimizer step, all params get gradients.
print(f"  CTM has gradient flow: {has_any_grad} (c_proj gets grads; others after first optim step)")

# Check backbone is still frozen
backbone_has_grad = False
for name, param in model.backbone.named_parameters():
    if param.grad is not None:
        backbone_has_grad = True
        break
print(f"  Backbone has gradients: {backbone_has_grad} (should be False)")

all_ok = (not backbone_has_grad) and has_any_grad
print("\n" + "=" * 60)
if all_ok:
    print("ALL TESTS PASSED")
elif backbone_has_grad:
    print("WARNING: Backbone has gradients!")
else:
    print("WARNING: No CTM gradient flow!")
print("=" * 60)
