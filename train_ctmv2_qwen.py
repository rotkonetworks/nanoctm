#!/usr/bin/env python3
"""Train CTMv2 on Qwen2.5-0.5B backbone — Phase 1.

Trains ONLY the CTMv2Block parameters (5.3M) on frozen Qwen backbone (481M).
Goal: produce discriminative sync patterns so the memory head can gate.

After this: memory head teaching works via least-squares (no more backprop).

Fits on 8GB GPU. ~30 min for 2000 steps.

Usage:
    python train_ctmv2_qwen.py
    python train_ctmv2_qwen.py --steps 5000 --K 8
"""

import torch
import torch.nn.functional as F
import time
import os
import argparse
import math
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--K', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--seq-len', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--backbone', type=str, default='Qwen/Qwen2.5-0.5B')
    p.add_argument('--save-dir', type=str, default='checkpoints/ctmv2_qwen')
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def load_data(tokenizer, seq_len):
    """Load ClimbMix training data."""
    import pyarrow.parquet as pq
    from nanochat.common import get_base_dir

    data_dir = os.path.join(get_base_dir(), "base_data_climbmix")
    parquet_files = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith('.parquet')
    ])

    if not parquet_files:
        # Fallback: use the shell data in repo
        data_dir = str(Path(__file__).parent / "data")
        parquet_files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith('.parquet')
        ])

    print(f'  Data: {len(parquet_files)} parquet files from {data_dir}')

    # Yield batches of token sequences
    def batch_iter(batch_size):
        token_buffer = []
        for pf in parquet_files:
            table = pq.read_table(pf, columns=['text'])
            for text in table.column('text').to_pylist():
                ids = tokenizer.encode(text, add_special_tokens=False)
                token_buffer.extend(ids)
                while len(token_buffer) >= (seq_len + 1) * batch_size:
                    batch = []
                    for _ in range(batch_size):
                        chunk = token_buffer[:seq_len + 1]
                        token_buffer = token_buffer[seq_len + 1:]
                        batch.append(chunk)
                    yield torch.tensor(batch)
        # Cycle
        while True:
            yield from batch_iter(batch_size)

    return batch_iter


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device

    # Load backbone
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f'Loading {args.backbone}...')
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.backbone, dtype=torch.bfloat16, trust_remote_code=True)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    D = backbone.config.hidden_size
    V = backbone.config.vocab_size
    n_layers = backbone.config.num_hidden_layers
    target_layer = n_layers - 1
    print(f'  {n_layers} layers, D={D}, V={V}')

    # Create CTMv2 block
    from dataclasses import dataclass
    from nanochat.ctm_v2_block import CTMv2Block

    @dataclass
    class CTMConfig:
        n_embd: int = D
        ctm_iterations: int = args.K
        ctm_n_synch: int = D // 2
        ctm_memory_length: int = 8
        ctm_memory_hidden: int = 32
        ctm_synapse_depth: int = 2
        ctm_n_attn_heads: int = 4

    block = CTMv2Block(CTMConfig())
    block.init_weights()
    block = block.to(device).to(torch.bfloat16)
    block.train()

    n_ctm = sum(p.numel() for p in block.parameters())
    n_backbone = sum(p.numel() for p in backbone.parameters())
    print(f'  CTMv2: {n_ctm:,} trainable, backbone: {n_backbone:,} frozen')
    print(f'  K={args.K}, rank=32 memory head (after training)')

    # Region breakdown
    for rname in block.region_names:
        n = block.region_sizes[rname]
        mem = block.regions[rname].cfg.memory_length
        depth = block.regions[rname].cfg.nlm_depth
        print(f'    {rname:12s}: {n:3d}n mem={mem:2d} nlm={depth}')

    # Optimizer — only CTM params
    optimizer = torch.optim.AdamW(block.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps)

    # Data
    batch_gen = load_data(tokenizer, args.seq_len)
    data_iter = iter(batch_gen(args.batch_size))

    # Training loop
    print(f'\n{"="*60}')
    print(f'Training CTMv2 on {args.backbone} ({args.steps} steps)')
    print(f'{"="*60}\n')

    best_loss = float('inf')
    t0 = time.time()
    running_loss = 0

    for step in range(1, args.steps + 1):
        batch = next(data_iter).to(device)
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        B, T = input_ids.shape

        # Forward through frozen backbone, replacing last layer's MLP
        qwen = backbone.model
        x = qwen.embed_tokens(input_ids)
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        pos_emb = qwen.rotary_emb(x, pos_ids)

        with torch.no_grad():
            for i, layer in enumerate(qwen.layers):
                if i == target_layer:
                    residual = x
                    x_normed = layer.input_layernorm(x)
                    attn_out = layer.self_attn(
                        x_normed, attention_mask=None,
                        position_embeddings=pos_emb)[0]
                    x = residual + attn_out
                    residual = x
                    x_normed = layer.post_attention_layernorm(x)
                    # Keep MLP output (frozen neocortex)
                    mlp_out = layer.mlp(x_normed)
                else:
                    x = layer(x, position_embeddings=pos_emb)
                    if i < target_layer:
                        continue
                    break

        # CTM forward (WITH gradients)
        ctm_out = block(x_normed)

        # REPLACEMENT mode: CTM replaces MLP entirely
        # Forces CTM to learn discriminative sync patterns
        x = residual + ctm_out.to(residual.dtype)

        # Final norm + lm_head (need grad flow through these for CTM backward)
        x = qwen.norm(x)
        logits = backbone.lm_head(x)

        # Loss
        loss = F.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1))

        # Backward through CTM only (backbone is frozen, detached)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if step % args.eval_every == 0:
            avg_loss = running_loss / args.eval_every
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            bpb = avg_loss / math.log(2)

            # Check sync differentiation
            block.eval()
            with torch.no_grad():
                sync1 = block(x_normed[:1, :16])
                sync2 = block(x_normed[:1, 16:32]) if T > 32 else sync1
            block.train()
            # Rough sync variance (higher = more differentiated)
            sync_var = ctm_out.float().var().item()

            print(f'  step {step:5d}: loss={avg_loss:.3f} bpb={bpb:.2f} '
                  f'lr={lr_now:.1e} sync_var={sync_var:.4f} ({elapsed:.0f}s)',
                  flush=True)
            running_loss = 0

            if avg_loss < best_loss:
                best_loss = avg_loss

        if step % args.save_every == 0:
            save_path = os.path.join(args.save_dir, f'ctmv2_step{step}.pt')
            torch.save({
                'step': step,
                'block_state_dict': block.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': {
                    'K': args.K, 'D': D, 'n_synch': D // 2,
                    'backbone': args.backbone,
                },
                'loss': best_loss,
            }, save_path)
            print(f'  Saved {save_path}')

    # Final save
    save_path = os.path.join(args.save_dir, 'ctmv2_final.pt')
    torch.save({
        'step': args.steps,
        'block_state_dict': block.state_dict(),
        'config': {
            'K': args.K, 'D': D, 'n_synch': D // 2,
            'backbone': args.backbone,
        },
        'loss': best_loss,
    }, save_path)

    elapsed = time.time() - t0
    print(f'\n{"="*60}')
    print(f'Done. {args.steps} steps in {elapsed:.0f}s ({elapsed/args.steps:.1f}s/step)')
    print(f'Best loss: {best_loss:.3f} (bpb: {best_loss/math.log(2):.2f})')
    print(f'Checkpoint: {save_path}')
    print(f'\nNext: run test_memory_head.py with --checkpoint {save_path}')


if __name__ == '__main__':
    main()
