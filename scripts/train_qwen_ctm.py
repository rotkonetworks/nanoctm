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
parser.add_argument("--unfreeze-ctm-layer-ffn", action="store_true", help="unfreeze MLP on CTM layers for co-training")
parser.add_argument("--backbone-lr-scale", type=float, default=0.1, help="backbone LR = main LR * this (default 0.1)")
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
# Regularization
parser.add_argument("--spectral-reg", type=float, default=0.0, help="spectral reg weight: penalize c_proj σ₁ dominance (0=disable, try 0.1)")
parser.add_argument("--ctm-aux-weight", type=float, default=0.1, help="auxiliary per-tick supervision weight (0=off, 0.1=default). Forces monotonic tick improvement.")
parser.add_argument("--bound-guided-aux", action="store_true", help="use Angeris synapse gap to weight per-tick aux loss (default: uniform)")
parser.add_argument("--plasticity-every", type=int, default=0, help="run plasticity rehearsal every N steps (0=disable, try 25)")
parser.add_argument("--plasticity-lr", type=float, default=1e-4, help="LR for plasticity rehearsal compact step")
parser.add_argument("--plasticity-steps", type=int, default=10, help="compact_memory steps during rehearsal")
# K ramping (start small K for speed, ramp up during training)
parser.add_argument("--k-ramp-schedule", type=str, default=None,
                    help="K ramp schedule: 'step:K,step:K,...' e.g. '0:8,2000:16,4000:32'. "
                         "Overrides --ctm-iterations for initial K. Final K must match --ctm-iterations.")
# Resume
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume from step")
parser.add_argument("--model-tag", type=str, default=None, help="checkpoint dir name")
# Telemetry
parser.add_argument("--ingest-url", type=str, default=None, help="POST training snapshots to this URL (e.g. https://ctm.rotko.net/api/ingest)")
parser.add_argument("--ingest-key", type=str, default=None, help="Bearer token for ingest API")
args = parser.parse_args()

# Parse K ramp schedule
k_ramp_schedule = None
if args.k_ramp_schedule:
    k_ramp_schedule = []
    for entry in args.k_ramp_schedule.split(","):
        s, k = entry.strip().split(":")
        k_ramp_schedule.append((int(s), int(k)))
    k_ramp_schedule.sort(key=lambda x: x[0])
    # Validate: final K must match --ctm-iterations (tick_embed is allocated at max K)
    final_k = k_ramp_schedule[-1][1]
    if final_k != args.ctm_iterations:
        raise ValueError(f"K ramp final K={final_k} must match --ctm-iterations={args.ctm_iterations}")
    # All K values must be <= max K
    for s, k in k_ramp_schedule:
        if k > args.ctm_iterations:
            raise ValueError(f"K ramp K={k} at step {s} exceeds --ctm-iterations={args.ctm_iterations}")

def get_k_for_step(step):
    """Return the active K for a given training step based on ramp schedule."""
    if not k_ramp_schedule:
        return args.ctm_iterations
    active_k = k_ramp_schedule[0][1]
    for s, k in k_ramp_schedule:
        if step >= s:
            active_k = k
    return active_k

def apply_k_ramp(model, step, current_k):
    """Check if K should change at this step. Returns new K (or current if unchanged).

    Since tick_embed is allocated at max K and active_K just controls the loop range,
    we only need to update active_K on each CTMBlock. No resize needed — unused
    tick embeddings at indices >= active_K simply aren't accessed.
    """
    new_k = get_k_for_step(step)
    if new_k == current_k:
        return current_k
    for ctm in model.ctm_blocks.values():
        ctm.active_K = new_k
    print0(f"  K RAMP: {current_k} → {new_k} at step {step}")
    return new_k

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

# --- Ingest telemetry (fire-and-forget, non-blocking) ---
import threading
import urllib.request
import urllib.error

_ingest_url = args.ingest_url
_ingest_key = args.ingest_key
_ingest_run = args.run  # tag snapshots with run name

_snapshot_log = None

def ingest_post(snapshot: dict):
    """POST snapshot to ingest endpoint + cache locally. Non-blocking, best-effort."""
    if not master_process:
        return
    snapshot = {**snapshot, "run": _ingest_run, "backbone": args.backbone}
    # always cache locally — can replay to ingest API later
    global _snapshot_log
    if _snapshot_log is None:
        log_dir = os.path.join(args.output_dir if hasattr(args, 'output_dir') else 'runs', 'snapshots')
        os.makedirs(log_dir, exist_ok=True)
        _snapshot_log = open(os.path.join(log_dir, f'{_ingest_run}.jsonl'), 'a')
    _snapshot_log.write(json.dumps(snapshot) + '\n')
    _snapshot_log.flush()
    # POST to remote if configured
    if not _ingest_url:
        return
    def _send():
        try:
            data = json.dumps(snapshot).encode()
            req = urllib.request.Request(_ingest_url, data=data,
                                         headers={"Content-Type": "application/json"})
            if _ingest_key:
                req.add_header("Authorization", f"Bearer {_ingest_key}")
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass  # best-effort, never block training
    threading.Thread(target=_send, daemon=True).start()

if _ingest_url and master_process:
    print0(f"Ingest telemetry: {_ingest_url} (run={_ingest_run})")

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
    "ctm_aux_weight": args.ctm_aux_weight,
    "ctm_bound_guided_aux": args.bound_guided_aux,
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

# Apply initial K from ramp schedule
current_active_k = args.ctm_iterations
if k_ramp_schedule:
    initial_k = get_k_for_step(args.resume_from_step + 1 if resuming else 0)
    for ctm in model.ctm_blocks.values():
        ctm.active_K = initial_k
    current_active_k = initial_k
    print0(f"K ramp: starting at K={initial_k}, schedule={k_ramp_schedule}")

# Unfreeze CTM layer FFN for co-training
if args.unfreeze_ctm_layer_ffn:
    model.unfreeze_layers(model.ctm_layer_indices)

# Count params
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params
print0(f"Total params: {total_params:,}")
print0(f"Trainable: {trainable_params:,}")
print0(f"Frozen (backbone): {frozen_params:,}")
print0(f"Config: K={model.config.ctm_iterations}, n_synch={model.config.ctm_n_synch}, "
       f"n_embd={model.config.n_embd}, CTM layers={model.ctm_layer_indices}")

# -----------------------------------------------------------------------------
# Optimizer

optimizer = model.setup_optimizer(lr=args.lr, weight_decay=args.weight_decay,
                                  backbone_lr_scale=args.backbone_lr_scale)
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

k_info = f"K={current_active_k}" + (f" (ramp to {args.ctm_iterations})" if k_ramp_schedule else "")
print0(f"\nStarting training: steps {step}-{num_iterations}, {k_info}")
print0(f"{'='*60}")

while step <= num_iterations:
    last_step = step == num_iterations

    # Eval (cross-entropy loss on val set)
    # Two metrics: final-tick loss (inference behavior) and multi-tick argmin loss
    # (matches training objective). Divergence = tick specialization, not overfitting.
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = max(1, args.eval_tokens // tokens_per_fwdbwd)
        total_loss_final = 0.0
        total_loss_argmin = 0.0
        with torch.no_grad():
            for eval_i, (vx, vy) in enumerate(val_loader):
                if eval_i >= eval_steps:
                    break
                # Final-tick loss (what inference actually produces)
                _, loss_final = model(vx, targets=vy)
                total_loss_final += loss_final.item()
                # Argmin-tick loss (matches training objective)
                _, loss_argmin = model(vx, targets=vy, multi_tick=True)
                total_loss_argmin += loss_argmin.item()
        n_eval = max(eval_i + 1, 1)
        val_loss_final = total_loss_final / n_eval
        val_loss_argmin = total_loss_argmin / n_eval
        val_bpb_final = val_loss_final / math.log(2)
        val_bpb_argmin = val_loss_argmin / math.log(2)
        if val_bpb_argmin < min_val_bpb:
            min_val_bpb = val_bpb_argmin
        print0(f"Step {step:05d} | val loss: final={val_loss_final:.4f} argmin={val_loss_argmin:.4f} "
               f"(~bpb: {val_bpb_final:.4f}/{val_bpb_argmin:.4f}, best: {min_val_bpb:.4f})")
        val_log = {"step": step,
                   "val/loss_final_tick": val_loss_final, "val/bpb_final": val_bpb_final,
                   "val/loss_argmin": val_loss_argmin, "val/bpb_argmin": val_bpb_argmin}
        wandb_run.log(val_log)
        ingest_post(val_log)
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
            # Bounds diagnostics
            if 'c_proj_rank90' in d:
                print0(f"    c_proj: rank90={d['c_proj_rank90']}, rank99={d['c_proj_rank99']}, "
                       f"cond={d['c_proj_condition']:.0f}, frob={d['c_proj_frobenius']:.2f}")
                print0(f"    c_proj top SVs: {['%.1f' % s for s in d.get('c_proj_top5_sv', [])]}")
            if 'synapse_act_rank' in d:
                print0(f"    synapse: act_rank={d['synapse_act_rank']}, "
                       f"utilization={d['synapse_utilization_pct']:.1f}%, "
                       f"gap={d.get('synapse_gap_pct', -1):.1f}%")
            if 'n_dead_neurons' in d and d['n_dead_neurons'] >= 0:
                print0(f"    neurons: dead={d['n_dead_neurons']}/{d.get('K_start', '?')}, "
                       f"diversity={d.get('neuron_diversity', 0):.3f}, "
                       f"norms=[{d['neuron_norm_min']:.3f}, {d['neuron_norm_mean']:.3f}, {d['neuron_norm_max']:.3f}]")
            if 'plastic_gate' in d:
                print0(f"    plastic: gate={d['plastic_gate']:.4f}, norm={d['plastic_norm']:.4f}")
            if 'bottleneck' in d and d['bottleneck'] != 'none':
                print0(f"    BOTTLENECK: {d['bottleneck']}")
            if 'tick_gaps' in d:
                gaps = d['tick_gaps']
                weights = d['tick_aux_weights']
                # Show top 5 ticks by weight (where supervision is concentrated)
                top_k = sorted(range(len(weights)), key=lambda i: -weights[i])[:5]
                top_str = ", ".join(f"t{k}:{weights[k]:.3f}(gap={gaps[k]:.3f})" for k in top_k)
                print0(f"    bound-guided aux: {top_str}")
        log_data = {"step": step}
        for layer_idx, d in dream_results.items():
            log_data[f"ctm/L{layer_idx}_dream_converged"] = int(d['converged'])
            log_data[f"ctm/L{layer_idx}_dream_delta"] = d['final_distance']
            # Log bounds metrics to wandb
            for key in ('c_proj_rank90', 'c_proj_rank99', 'c_proj_condition', 'c_proj_frobenius',
                        'synapse_act_rank', 'synapse_utilization_pct', 'synapse_gap_pct',
                        'n_dead_neurons', 'neuron_diversity', 'neuron_norm_mean',
                        'plastic_gate', 'plastic_norm'):
                if key in d:
                    log_data[f"ctm/L{layer_idx}_{key}"] = d[key]
            if 'bottleneck' in d:
                log_data[f"ctm/L{layer_idx}_bottleneck"] = d['bottleneck']
            if 'tick_gaps' in d:
                for k, (gap, w) in enumerate(zip(d['tick_gaps'], d['tick_aux_weights'])):
                    log_data[f"ctm/L{layer_idx}_tick_{k}_gap"] = gap
                    log_data[f"ctm/L{layer_idx}_tick_{k}_aux_w"] = w

        # Per-tick diagnostics: run multi_tick forward on current batch
        with torch.no_grad():
            model._tick_diagnostics = None
            model(x, targets=y, multi_tick=True)
            if hasattr(model, '_tick_diagnostics') and model._tick_diagnostics:
                td = model._tick_diagnostics
                K = current_active_k
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
        # Save optimizer state only every 2000 steps (large) or on last step
        include_optimizer = (step % 2000 == 0) or last_step
        ckpt_data = {
            "step": step,
            "ctm_state_dict": model.get_ctm_state_dict(),
            "config": vars(args),
            "val_bpb": min_val_bpb,
            "active_k": current_active_k,
        }
        if include_optimizer:
            ckpt_data["optimizer_state_dict"] = optimizer.state_dict()
            print0(f"  Saving checkpoint (with optimizer): {ckpt_path}")
        else:
            print0(f"  Saving checkpoint (weights only): {ckpt_path}")
        torch.save(ckpt_data, ckpt_path)
        # Prune old checkpoints (keep more weight-only, fewer full)
        ckpts = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("ctm_") and f.endswith(".pt")])
        while len(ckpts) > args.keep_checkpoints:
            old = ckpts.pop(0)
            os.remove(os.path.join(checkpoint_dir, old))
            print0(f"  Pruned: {old}")

    if last_step:
        break

    # K ramp: check if we should increase active_K at this step
    if k_ramp_schedule:
        current_active_k = apply_k_ramp(model, step, current_active_k)

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

        # Spectral regularization: penalize σ₁ dominance on c_proj
        # Keeps rank broad so compact_memory has room for new facts
        if args.spectral_reg > 0:
            for ctm in model.ctm_blocks.values():
                W = ctm.c_proj.weight  # (out, in)
                # Power iteration to estimate σ₁ (cheap, no SVD needed)
                if not hasattr(ctm, '_spec_v'):
                    ctm._spec_v = torch.randn(W.shape[1], device=W.device, dtype=W.dtype)
                v = ctm._spec_v.detach()
                u = W @ v
                u = u / (u.norm() + 1e-8)
                v = W.t() @ u
                v = v / (v.norm() + 1e-8)
                ctm._spec_v = v.detach()
                sigma1_sq = (u * (W @ v)).sum()
                frob_sq = (W * W).sum()
                # Penalize σ₁²/||W||² — pushes energy into other singular values
                spec_loss = args.spectral_reg * (sigma1_sq / (frob_sq + 1e-8))
                loss = loss + spec_loss / grad_accum_steps

        loss.backward()
        x, y, dataloader_state = next(train_loader)

    # Optimizer step — scale LR per param group (backbone gets lower LR)
    lr = get_lr(step)
    for i, group in enumerate(optimizer.param_groups):
        if i == 0:
            group["lr"] = lr  # CTM params
        else:
            group["lr"] = lr * args.backbone_lr_scale  # backbone params
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

    # Plasticity rehearsal: MAML-style "learn to learn"
    # Every N steps: save weights → mini compact → test recall → restore weights
    # The recall loss teaches the CTM that compact_memory updates should be recallable
    if (args.plasticity_every > 0 and step > 0 and step % args.plasticity_every == 0
            and master_process):
        import random as _plast_rng
        model.eval()

        # Use current batch as "teaching data" (the model just trained on it)
        T_teach = min(256, x.shape[1] - 1)
        teach_x = x[:1, :T_teach]
        teach_y = y[:1, :T_teach]

        # Build recall pairs from the teaching data
        # Split into chunks: teach on first half, recall on second half
        mid = T_teach // 2
        recall_input = teach_x[:, mid:]
        recall_target = teach_y[:, mid:]
        recall_pairs = [(recall_input, recall_target)]

        # Save CTM state
        plast_state = {}
        for idx_str, ctm in model.ctm_blocks.items():
            plast_state[idx_str] = {n: p.data.clone() for n, p in ctm.named_parameters()}

        # Mini compact_memory
        try:
            plast_result = model.compact_memory(
                teaching_ids=teach_x,
                target_ids=teach_y,
                lr=args.plasticity_lr,
                steps=args.plasticity_steps,
                recall_pairs=recall_pairs,
                recall_weight=0.7,
                max_delta=0.05,  # tight budget — we're just rehearsing
                kl_weight=0.0,
            )

            plast_delta = plast_result.get('rel_delta', 0)
            plast_loss = plast_result['losses'][-1] if plast_result.get('losses') else 0

            if step % (args.plasticity_every * 4) == 0:
                print0(f"  Plasticity rehearsal: delta={plast_delta*100:.2f}%, "
                       f"loss={plast_loss:.3f}")
            wandb_run.log({
                "step": step,
                "plasticity/delta_pct": plast_delta * 100,
                "plasticity/loss": plast_loss,
            })
        except Exception as e:
            print0(f"  Plasticity rehearsal failed: {e}")

        # Reptile meta-learning: keep a tiny fraction of compact changes
        # θ_new = θ_old + ε*(θ_compact - θ_old)
        # This teaches the CTM to be receptive to compact_memory updates
        # without second-order gradients (first-order MAML approximation)
        reptile_eps = 0.01  # keep 1% of compact direction
        for idx_str, ctm in model.ctm_blocks.items():
            if idx_str in plast_state:
                for n, p in ctm.named_parameters():
                    if n in plast_state[idx_str]:
                        compact_dir = p.data - plast_state[idx_str][n]
                        p.data.copy_(plast_state[idx_str][n] + reptile_eps * compact_dir)
        del plast_state

        model.train()

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
    if k_ramp_schedule:
        step_log["ctm/active_K"] = current_active_k
    # Log spectral stats every 50 steps
    if args.spectral_reg > 0 and step % 50 == 0:
        with torch.no_grad():
            for idx_str, ctm in model.ctm_blocks.items():
                W = ctm.c_proj.weight
                S = torch.linalg.svdvals(W.float())
                frob = S.norm()
                cumulative = torch.cumsum(S**2, 0) / (frob**2 + 1e-8)
                rank90 = (cumulative < 0.9).sum().item() + 1
                rank99 = (cumulative < 0.99).sum().item() + 1
                step_log[f'spectral/L{idx_str}_sigma1'] = S[0].item()
                step_log[f'spectral/L{idx_str}_rank90'] = rank90
                step_log[f'spectral/L{idx_str}_rank99'] = rank99
                step_log[f'spectral/L{idx_str}_cond'] = (S[0] / (S[-1] + 1e-8)).item()
                if step % 500 == 0:
                    print0(f"  c_proj L{idx_str}: σ₁={S[0]:.1f} rank90={rank90} rank99={rank99} cond={S[0]/(S[-1]+1e-8):.0f}")

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
                for k in range(current_active_k)
            ],
            'certainty_mean': td.get('certainty/mean', 0),
            'grad_tick_frac': td.get('loss/grad_tick_frac', 0),
        }
        with open('/tmp/ctm_ticks.jsonl', 'a') as tf:
            tf.write(json.dumps(tick_snapshot) + '\n')
        ingest_post(tick_snapshot)
    wandb_run.log(step_log)
    ingest_post(step_log)

    step += 1

print0(f"\n{'='*60}")
print0(f"Training complete. Best val bpb: {min_val_bpb:.4f}")
print0(f"Total time: {total_time/60:.1f}m")
print0(f"{'='*60}")
