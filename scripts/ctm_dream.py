"""
Quick diagnostic script: generate samples and run dream() to measure CTM convergence.
Usage: python -m scripts.ctm_dream [--step N]
"""
import os
import torch
from nanochat.checkpoint_manager import load_model
from nanochat.common import get_base_dir, autodetect_device_type

device_type = autodetect_device_type()
device = torch.device(device_type)

# Load the latest CTM checkpoint
model, tokenizer, meta = load_model(
    "base", device=device, phase="eval",
    model_tag="ctm_d12",
)
print(f"Loaded model: {meta['model_config']}")
print(f"Step: {meta.get('step', '?')}, Val BPB: {meta.get('val_bpb', '?')}")

# --- Generate samples ---
print("\n" + "="*60)
print("GENERATION SAMPLES")
print("="*60)
prompts = [
    "The capital of France is",
    "Once upon a time there was a",
    "The meaning of life is",
    "In the beginning",
]
for prompt in prompts:
    tokens = tokenizer(prompt, prepend="<|bos|>")
    for tok in model.generate(tokens, max_tokens=32, temperature=0.8, top_k=50):
        pass  # consume generator
    # Decode the full sequence
    tokens_list = list(tokens) if isinstance(tokens, list) else tokens
    # Simple approach: just use generate and collect
    all_tokens = tokenizer(prompt, prepend="<|bos|>")
    generated = []
    for tok in model.generate(all_tokens, max_tokens=32, temperature=0.8, top_k=50):
        generated.append(tok)
    full = tokenizer.decode(all_tokens + generated)
    print(f"\n> {prompt}")
    print(f"  {full}")

# --- Dream diagnostics ---
print("\n" + "="*60)
print("DREAM DIAGNOSTICS (per-layer CTM convergence)")
print("="*60)
# Run dream on a small batch
sample_text = "The quick brown fox jumps over the lazy dog"
tokens = tokenizer(sample_text, prepend="<|bos|>")
idx = torch.tensor([tokens], device=device)

diagnostics = model.dream(idx)
for layer_idx, deltas in sorted(diagnostics.items()):
    delta_str = " → ".join(f"{d:.4f}" for d in deltas)
    converged = deltas[-1] < deltas[0] * 0.3  # >70% reduction = converged
    status = "CONVERGED" if converged else "active"
    print(f"  Layer {layer_idx:2d}: [{delta_str}] {status}")

# Summary
print(f"\nTotal CTM layers: {len(diagnostics)}")
converged_count = sum(1 for d in diagnostics.values() if d[-1] < d[0] * 0.3)
print(f"Converged layers (>70% delta reduction): {converged_count}/{len(diagnostics)}")
print("Converged layers could use fewer K iterations → memory savings")
