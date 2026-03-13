"""
Train CTMBlock on frozen Qwen backbone.

Usage:
    python3 -m scripts.train_qwen_ctm --run qwen_ctm_k4

Only CTM parameters are trainable (~52M). The backbone (~481M) is frozen.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse

import wandb
import torch
import torch.nn.functional as F

from nanochat.gpt import CTMCache
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE
from nanochat.engine import Engine
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Train CTM on Qwen backbone")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables)")
# Model
parser.add_argument("--backbone", type=str, default="Qwen/Qwen3-0.6B", help="HuggingFace model name")
parser.add_argument("--ctm-iterations", type=int, default=4, help="CTM thinking steps (K)")
parser.add_argument("--ctm-n-synch", type=int, default=-1, help="sync neurons (-1 = n_embd//2)")
parser.add_argument("--ctm-memory-length", type=int, default=16, help="trace history length")
parser.add_argument("--ctm-memory-hidden", type=int, default=32, help="NLM hidden dim")
parser.add_argument("--ctm-synapse-depth", type=int, default=32, help="U-NET synapse depth")
parser.add_argument("--ctm-adaptive-k", action="store_true", help="use mean sync normalization (K-invariant)")
parser.add_argument("--ctm-layers", type=str, default="last", help="which layers get CTM: 'last', '14,27', etc.")
parser.add_argument("--additive", action="store_true", help="keep MLP on CTM layers (CTM adds on top, paper-style)")
# Training
parser.add_argument("--num-iterations", type=int, default=10000, help="total training steps")
parser.add_argument("--device-batch-size", type=int, default=4, help="per-device batch size")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens (-1 = device_batch * seq_len)")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for CTM params")
parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
parser.add_argument("--warmup-steps", type=int, default=100, help="LR warmup steps")
parser.add_argument("--warmdown-ratio", type=float, default=0.3, help="fraction of training for LR warmdown")
# Multi-tick loss
parser.add_argument("--no-multi-tick", action="store_true", help="disable multi-tick certainty loss (on by default)")
# Cache-aware training
parser.add_argument("--cache-aware-ratio", type=float, default=0.0, help="fraction of steps using cache-aware training")
# SFT data mixing
parser.add_argument("--sft-data", type=str, default=None, help="path to SFT JSONL file for mixing")
parser.add_argument("--sft-ratio", type=float, default=0.0, help="fraction of steps using SFT data (0.0-1.0)")
# Sleep consolidation
parser.add_argument("--consolidate-every", type=int, default=0, help="run sleep consolidation every N steps (0=disable)")
parser.add_argument("--consolidate-lr", type=float, default=1e-4, help="learning rate for consolidation")
parser.add_argument("--consolidate-steps", type=int, default=4, help="gradient steps per replay batch during consolidation")
parser.add_argument("--replay-buffer-size", type=int, default=16, help="max replay batches to keep")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="eval val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=2097152, help="tokens for val eval")
parser.add_argument("--sample-every", type=int, default=100, help="sample from model every N steps")
parser.add_argument("--sample-tokens", type=int, default=40, help="max tokens per sample")
parser.add_argument("--save-every", type=int, default=500, help="save checkpoint every N steps")
parser.add_argument("--keep-checkpoints", type=int, default=3, help="max checkpoints to keep")
parser.add_argument("--sleep-every", type=int, default=50, help="CTM dream diagnostics every N steps")
# Resume
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume from step")
parser.add_argument("--model-tag", type=str, default=None, help="checkpoint dir name")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Init

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_name)
    print0(f"GPU: {gpu_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanoctm", name=args.run, config=vars(args))

# -----------------------------------------------------------------------------
# Load model

print0(f"Loading backbone: {args.backbone}")
from nanochat.qwen_backbone import QwenBackboneGPT, QwenTokenizer

ctm_kwargs = {
    "ctm_iterations": args.ctm_iterations,
    "ctm_n_synch": args.ctm_n_synch if args.ctm_n_synch > 0 else None,  # None = n_embd//2
    "ctm_memory_length": args.ctm_memory_length,
    "ctm_memory_hidden": args.ctm_memory_hidden,
    "ctm_synapse_depth": args.ctm_synapse_depth,
    "ctm_adaptive_k": args.ctm_adaptive_k,
}
# Remove None values so defaults in QwenBackboneGPT apply
ctm_kwargs = {k: v for k, v in ctm_kwargs.items() if v is not None}

# Parse CTM layer spec into indices
ctm_layer_indices = None  # None = last layer only
if args.ctm_layers != "last":
    ctm_layer_indices = [int(x.strip()) for x in args.ctm_layers.split(",")]
    print0(f"Multi-CTM: layers {ctm_layer_indices}")

model = QwenBackboneGPT.from_pretrained(args.backbone, ctm_kwargs=ctm_kwargs,
                                         ctm_layer_indices=ctm_layer_indices)
model = model.to(device)

tokenizer = QwenTokenizer.from_pretrained(args.backbone)
vocab_size = tokenizer.get_vocab_size()

# Init or resume CTM weights
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"qwen_ctm_k{args.ctm_iterations}"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1

if resuming:
    ckpt_path = os.path.join(checkpoint_dir, f"ctm_{args.resume_from_step:06d}.pt")
    print0(f"Resuming from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_ctm_state_dict(ckpt["ctm_state_dict"])
    print0(f"  Loaded CTM weights from step {args.resume_from_step}")

    # Fresh-init any NEW CTM layers not in checkpoint
    ckpt_sd = ckpt["ctm_state_dict"]
    sample_key = next(iter(ckpt_sd))
    if sample_key in model.ctm_blocks and isinstance(ckpt_sd[sample_key], dict):
        loaded_layers = set(int(k) for k in ckpt_sd.keys())
    else:
        # Single-CTM checkpoint → loaded into last layer
        loaded_layers = {model.ctm_layer_indices[-1]}

    new_layers = [idx for idx in model.ctm_layer_indices if idx not in loaded_layers]
    if new_layers:
        print0(f"  Fresh-initializing NEW CTM layers: {new_layers}")
        model.init_ctm_weights(only_layers=new_layers)

    # Trained CTM layers → replacement (MLP removed) unless --additive
    # New CTM layers → additive (frozen MLP kept alongside CTM)
    if not args.additive:
        model.set_replacement_layers(loaded_layers)
    else:
        print0("  Additive mode: keeping MLP on all CTM layers")
else:
    model.init_ctm_weights()
    # Fresh start: all CTM layers replace MLP (unless --additive)
    if not args.additive:
        model.set_replacement_layers(model.ctm_layer_indices)
    else:
        print0("  Additive mode: keeping MLP on all CTM layers")

# Count params
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params
print0(f"Total params: {total_params:,}")
print0(f"Trainable (CTM): {trainable_params:,}")
print0(f"Frozen (backbone): {frozen_params:,}")
print0(f"Config: K={model.config.ctm_iterations}, n_synch={model.config.ctm_n_synch}, "
       f"n_embd={model.config.n_embd}, CTM layers={model.ctm_layer_indices}")

# -----------------------------------------------------------------------------
# Optimizer

optimizer = model.setup_optimizer(lr=args.lr, weight_decay=args.weight_decay)
if resuming and "optimizer_state_dict" in ckpt:
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print0("  Loaded optimizer state")
    except (ValueError, RuntimeError) as e:
        print0(f"  Skipping optimizer state (arch changed): {e}")
    del ckpt

# -----------------------------------------------------------------------------
# Data

train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len,
    split="train", device=device, resume_state_dict=None,
)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len,
    split="val", device=device,
)

# SFT data mixing
sft_loader = None
if args.sft_data and args.sft_ratio > 0:
    import random as _rng
    class SFTLoader:
        """Infinite loader that samples from JSONL and tokenizes into (x, y) batches."""
        def __init__(self, path, tokenizer, B, T, device):
            self.texts = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        self.texts.append(obj["text"])
            self.tokenizer = tokenizer
            self.B, self.T, self.device = B, T, device
            self.bos = tokenizer.get_bos_token_id()
            print0(f"SFT data: {len(self.texts)} texts from {path}")

        def sample_batch(self):
            """Build a batch by sampling and packing SFT texts."""
            rows = []
            for _ in range(self.B):
                # Sample and concatenate texts until we fill T tokens
                tokens = [self.bos]
                while len(tokens) < self.T + 1:
                    text = _rng.choice(self.texts)
                    toks = self.tokenizer.encode(text)
                    tokens.extend(toks)
                tokens = tokens[:self.T + 1]  # trim to exactly T+1
                rows.append(tokens)
            data = torch.tensor(rows, dtype=torch.long, device=self.device)
            return data[:, :self.T], data[:, 1:self.T + 1]

    sft_loader = SFTLoader(args.sft_data, tokenizer, args.device_batch_size,
                           args.max_seq_len, device)

# Batch size and grad accumulation
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
total_batch_size = args.total_batch_size if args.total_batch_size > 0 else tokens_per_fwdbwd
assert total_batch_size % tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // tokens_per_fwdbwd
print0(f"Batch: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,} tokens/microstep")
print0(f"Total batch: {total_batch_size:,} tokens, grad_accum={grad_accum_steps}")

x, y, dataloader_state = next(train_loader)

# -----------------------------------------------------------------------------
# LR schedule

num_iterations = args.num_iterations

def get_lr(step):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    warmdown_start = num_iterations - int(args.warmdown_ratio * num_iterations)
    if step >= warmdown_start:
        decay_frac = 1 - (num_iterations - step) / (num_iterations - warmdown_start)
        return args.lr * (1 - decay_frac ** 0.5) * 0.95 + args.lr * 0.05
    return args.lr

# -----------------------------------------------------------------------------
# Training loop

step = args.resume_from_step + 1 if resuming else 0
start_step = step  # for correct ETA calculation on resume
smooth_loss = 0
total_time = 0
min_val_bpb = float("inf")

# Replay buffer for sleep consolidation
replay_buffer = []  # list of (x, y) tensors

print0(f"\nStarting training: steps {step}-{num_iterations}, K={args.ctm_iterations}")
print0(f"{'='*60}")

while step <= num_iterations:
    last_step = step == num_iterations

    # Eval (cross-entropy loss on val set)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = max(1, args.eval_tokens // tokens_per_fwdbwd)
        total_loss = 0.0
        with torch.no_grad():
            for eval_i, (vx, vy) in enumerate(val_loader):
                if eval_i >= eval_steps:
                    break
                _, loss = model(vx, targets=vy)
                total_loss += loss.item()
        val_loss = total_loss / max(eval_i + 1, 1)
        val_bpb = val_loss / math.log(2)  # nats to bits, approximate bpb
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        print0(f"Step {step:05d} | val loss: {val_loss:.4f} (~bpb: {val_bpb:.4f}, best: {min_val_bpb:.4f})")
        wandb_run.log({"step": step, "val/loss": val_loss, "val/bpb_approx": val_bpb})
        model.train()

    # Sample
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        engine = Engine(model, tokenizer)
        prompts = [
            "The capital of France is",
            "Once upon a time",
            "Water boils at",
            "The meaning of life is",
            "I am a continuous thought machine",
            "My memory works by",
        ]
        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            try:
                results, _ = engine.generate_batch(tokens, num_samples=1,
                                                    max_tokens=args.sample_tokens,
                                                    temperature=0.7, seed=42)
                text = tokenizer.decode(results[0][len(tokens):])
                print0(f'  "{prompt}" -> "{text.strip()[:100]}"')
            except Exception as e:
                print0(f'  "{prompt}" -> ERROR: {e}')
        model.train()

    # Dream diagnostics + per-tick K activation analysis
    if args.sleep_every > 0 and step > 0 and step % args.sleep_every == 0:
        model.eval()
        dream_results = model.dream(device=device)
        for layer_idx, d in dream_results.items():
            print0(f"  Dream L{layer_idx}: converged={d['converged']}, "
                   f"K={d['K_start']}, delta={d['final_distance']:.4f}")
        log_data = {"step": step}
        for layer_idx, d in dream_results.items():
            log_data[f"ctm/L{layer_idx}_dream_converged"] = int(d['converged'])
            log_data[f"ctm/L{layer_idx}_dream_delta"] = d['final_distance']

        # Per-tick diagnostics: run multi_tick forward on current batch
        with torch.no_grad():
            model._tick_diagnostics = None
            model(x, targets=y, multi_tick=True)
            if hasattr(model, '_tick_diagnostics') and model._tick_diagnostics:
                td = model._tick_diagnostics
                K = args.ctm_iterations
                for key, val in td.items():
                    log_data[f'ctm/{key}'] = val
                # Print summary: which ticks are most selected
                top_ticks = sorted(
                    [(k, td.get(f'tick_{k}/selected_pct', 0)) for k in range(K)],
                    key=lambda x: -x[1]
                )[:5]
                top_str = ", ".join(f"t{k}:{pct:.1f}%" for k, pct in top_ticks)
                cert_str = f"cert={td.get('certainty/mean', 0):.3f}"
                print0(f"  Ticks: {top_str} | {cert_str}")

        wandb_run.log(log_data)
        model.train()

    # Save
    if step > 0 and args.save_every > 0 and (last_step or step % args.save_every == 0):
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, f"ctm_{step:06d}.pt")
        torch.save({
            "step": step,
            "ctm_state_dict": model.get_ctm_state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
            "val_bpb": min_val_bpb,
        }, ckpt_path)
        print0(f"  Saved checkpoint: {ckpt_path}")
        # Prune old checkpoints
        ckpts = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("ctm_") and f.endswith(".pt")])
        while len(ckpts) > args.keep_checkpoints:
            old = ckpts.pop(0)
            os.remove(os.path.join(checkpoint_dir, old))
            print0(f"  Pruned: {old}")

    if last_step:
        break

    # --- Training step ---
    synchronize()
    t0 = time.time()

    for micro_step in range(grad_accum_steps):
        # SFT mixing: replace batch with SFT data with given probability
        use_sft = (sft_loader is not None
                   and torch.rand(1).item() < args.sft_ratio)
        if use_sft:
            x, y = sft_loader.sample_batch()

        # Cache-aware: forward first half to populate CTMCache, train on second half
        use_cache_aware = (args.cache_aware_ratio > 0
                          and torch.rand(1).item() < args.cache_aware_ratio)

        if use_cache_aware:
            T = x.shape[1]
            mid = T // 2
            x1, x2 = x[:, :mid], x[:, mid:]
            y2 = y[:, mid:]
            ctm_cache = CTMCache(model.config.n_layer)
            with torch.no_grad():
                model(x1, ctm_cache=ctm_cache)
            for layer_state in ctm_cache.layers:
                if layer_state is not None:
                    for k in layer_state:
                        layer_state[k] = layer_state[k].detach()
            _, loss = model(x2, targets=y2, ctm_cache=ctm_cache, multi_tick=not args.no_multi_tick)
        else:
            _, loss = model(x, targets=y, multi_tick=not args.no_multi_tick)

        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, dataloader_state = next(train_loader)

    # Optimizer step
    lr = get_lr(step)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Add to replay buffer for consolidation (store on CPU to save VRAM)
    if args.consolidate_every > 0:
        replay_buffer.append((x.detach().cpu(), y.detach().cpu()))
        if len(replay_buffer) > args.replay_buffer_size:
            replay_buffer.pop(0)

    # Sleep consolidation (gradient-free Hebbian — no VRAM overhead)
    if args.consolidate_every > 0 and step > 0 and step % args.consolidate_every == 0 and len(replay_buffer) >= 2:
        import random as _consolidate_rng
        n_replay = min(4, len(replay_buffer))
        replay_sample_cpu = _consolidate_rng.sample(replay_buffer, n_replay)
        replay_sample = [(rx[:1].to(device), ry[:1].to(device)) for rx, ry in replay_sample_cpu]
        consol_stats = model.consolidate(
            replay_sample, lr=args.consolidate_lr, steps=args.consolidate_steps
        )
        del replay_sample
        if consol_stats.get('losses'):
            consol_loss = sum(consol_stats['losses']) / len(consol_stats['losses'])
            consol_cert = sum(consol_stats['mean_certainty']) / len(consol_stats['mean_certainty'])
            print0(f"  Consolidation: loss={consol_loss:.3f}, certainty={consol_cert:.3f}, "
                   f"replay={len(replay_buffer)}")
            wandb_run.log({
                "step": step,
                "sleep/consolidation_loss": consol_loss,
                "sleep/certainty": consol_cert,
                "sleep/replay_buffer": len(replay_buffer),
            })

    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # Logging
    loss_f = train_loss.item()
    ema = 0.9
    smooth_loss = ema * smooth_loss + (1 - ema) * loss_f
    debiased_loss = smooth_loss / (1 - ema ** (step + 1))

    if step > 10:
        total_time += dt
    pct = 100 * step / num_iterations
    tok_sec = int(total_batch_size / dt)

    steps_done = step - max(start_step, 10)
    if steps_done > 0:
        avg_dt = total_time / steps_done
        eta = avg_dt * (num_iterations - step) / 60
        eta_str = f" | eta: {eta:.1f}m"
    else:
        eta_str = ""

    print0(f"step {step:05d}/{num_iterations} ({pct:.1f}%) | loss: {debiased_loss:.4f} | lr: {lr:.2e} | "
           f"dt: {dt*1000:.0f}ms | tok/sec: {tok_sec:,} | time: {total_time/60:.1f}m{eta_str}")

    step_log = {
        "step": step,
        "train/loss": debiased_loss,
        "train/raw_loss": loss_f,
        "train/lr": lr,
        "perf/dt_ms": dt * 1000,
        "perf/tok_per_sec": tok_sec,
    }
    # Log per-tick diagnostics from training forward pass (computed in _multi_tick_loss)
    if hasattr(model, '_tick_diagnostics') and model._tick_diagnostics:
        td = model._tick_diagnostics
        for key, val in td.items():
            step_log[f'ctm/{key}'] = val
        # Write tick snapshot to JSONL for real-time 3D visualizer
        tick_snapshot = {
            'step': step,
            'loss': loss_f,
            'ticks': [
                {
                    'k': k,
                    'loss': td.get(f'tick_{k}/loss', 0),
                    'selected_pct': td.get(f'tick_{k}/selected_pct', 0),
                }
                for k in range(args.ctm_iterations)
            ],
            'certainty_mean': td.get('certainty/mean', 0),
            'grad_tick_frac': td.get('loss/grad_tick_frac', 0),
        }
        with open('/tmp/ctm_ticks.jsonl', 'a') as tf:
            tf.write(json.dumps(tick_snapshot) + '\n')
    wandb_run.log(step_log)

    step += 1

print0(f"\n{'='*60}")
print0(f"Training complete. Best val bpb: {min_val_bpb:.4f}")
print0(f"Total time: {total_time/60:.1f}m")
print0(f"{'='*60}")
