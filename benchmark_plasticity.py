#!/usr/bin/env python3
"""Plasticity survival benchmark — does the critical period extend with v2?

Trains CTMBlock (v1) or CTMv2Block (v2) on frozen Qwen 0.5B backbone,
teaches a fact at step S, continues training, tests recall at intervals.

The ONLY metric: at what step does recall die?

Usage:
    python benchmark_plasticity.py --block v1 --seed 0
    python benchmark_plasticity.py --block v2 --seed 0
    python benchmark_plasticity.py --block v1 --seed 0,1,2 --teach-at 1000,2000,3000
"""

import torch
import torch.nn.functional as F
import argparse
import time
import json
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--block', choices=['v1', 'v2'], default='v1')
    p.add_argument('--backbone', default='Qwen/Qwen2.5-0.5B')
    p.add_argument('--teach-at', type=str, default='2000',
                   help='Comma-separated steps at which to teach')
    p.add_argument('--test-after', type=str, default='0,25,50,100,200,500,1000',
                   help='Steps after teaching to test recall')
    p.add_argument('--seed', type=str, default='0',
                   help='Comma-separated seeds')
    p.add_argument('--total-steps', type=int, default=5000)
    p.add_argument('--K', type=int, default=4, help='CTM iterations')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--compact-lr', type=float, default=3e-4)
    p.add_argument('--compact-steps', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--seq-len', type=int, default=256)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', type=str, default='plasticity_results.json')
    return p.parse_args()


def load_backbone(model_name, device):
    """Load frozen Qwen backbone."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f'Loading {model_name}...')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = model.config.num_hidden_layers
    D = model.config.hidden_size
    print(f'  {n_layers} layers, D={D}, frozen')
    return model, tokenizer, n_layers, D


def make_ctm_block(block_type, D, K, device):
    """Create v1 or v2 CTM block."""
    from dataclasses import dataclass

    @dataclass
    class CTMConfig:
        n_embd: int = D
        ctm_iterations: int = K
        ctm_n_synch: int = D // 2
        ctm_memory_length: int = 8
        ctm_memory_hidden: int = 32
        ctm_synapse_depth: int = 2
        ctm_n_attn_heads: int = 4
        ctm_adaptive_k: bool = False

    config = CTMConfig()

    if block_type == 'v1':
        from nanochat.gpt import CTMBlock
        block = CTMBlock(config)
    else:
        from nanochat.ctm_v2_block import CTMv2Block
        block = CTMv2Block(config)

    block.init_weights()

    # c_proj starts near zero (CTM is initially pass-through)
    block.c_proj.weight.data.mul_(0.01)

    block = block.to(device).to(torch.bfloat16)
    n_params = sum(p.numel() for p in block.parameters())
    print(f'  {block_type} block: {n_params:,} params, K={K}')
    return block, config


def forward_with_ctm(backbone, block, input_ids, target_layer_idx, device):
    """Run backbone with CTM block replacing last layer's MLP.

    Returns logits (B, T, V).
    """
    B, T = input_ids.shape
    qwen = backbone.model

    x = qwen.embed_tokens(input_ids)
    pos_ids = torch.arange(T, device=device).unsqueeze(0)
    pos_emb = qwen.rotary_emb(x, pos_ids)

    for i, layer in enumerate(qwen.layers):
        if i == target_layer_idx:
            # Custom: attention + CTM block (replacing MLP)
            residual = x
            x_normed = layer.input_layernorm(x)
            attn_out = layer.self_attn(
                x_normed, position_embeddings=pos_emb)[0]
            x = residual + attn_out

            residual = x
            x_normed = layer.post_attention_layernorm(x)
            ctm_out = block(x_normed)
            x = residual + ctm_out.to(residual.dtype)
        else:
            x = layer(x, position_embeddings=pos_emb)

    x = qwen.norm(x)
    logits = backbone.lm_head(x)
    return logits


def get_training_batch(tokenizer, batch_size, seq_len, device):
    """Generate a training batch from random text."""
    # Simple: random token sequences (the backbone already knows language,
    # we're just training CTM to be a good pass-through)
    ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len + 1),
                        device=device)
    return ids[:, :-1], ids[:, 1:]


def teach_fact(backbone, block, tokenizer, fact_text, target_layer,
               lr, n_steps, device):
    """Teach a fact via compact_memory-style gradient descent."""
    ids = tokenizer.encode(fact_text, add_special_tokens=False)
    ids = torch.tensor([ids], device=device)
    targets = ids[:, 1:]
    inputs = ids[:, :-1]

    optimizer = torch.optim.Adam(block.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        logits = forward_with_ctm(backbone, block, inputs, target_layer, device)
        logits = logits[:, :targets.shape[1], :]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    return losses[0], losses[-1]


def test_recall(backbone, block, tokenizer, prompt, expected, target_layer, device):
    """Test if model generates expected text from prompt."""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], device=device)

    block.eval()
    with torch.no_grad():
        generated = list(ids)
        for _ in range(20):  # generate 20 tokens
            logits = forward_with_ctm(
                backbone, block, torch.tensor([generated], device=device),
                target_layer, device)
            next_token = logits[0, -1].argmax().item()
            generated.append(next_token)

    block.train()
    output = tokenizer.decode(generated[len(ids):])
    recall = expected.lower() in output.lower()
    return recall, output


def train_step(backbone, block, optimizer, tokenizer, target_layer,
               batch_size, seq_len, device):
    """One training step on random language data."""
    inputs, targets = get_training_batch(tokenizer, batch_size, seq_len, device)
    logits = forward_with_ctm(backbone, block, inputs, target_layer, device)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           targets.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def run_experiment(args, block_type, seed, teach_at):
    """Run one full experiment: train → teach → continue → test."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = args.device
    test_after = [int(x) for x in args.test_after.split(',')]

    # Load backbone
    backbone, tokenizer, n_layers, D = load_backbone(args.backbone, device)
    target_layer = n_layers - 1

    # Create CTM block
    block, config = make_ctm_block(block_type, D, args.K, device)

    # Optimizer
    optimizer = torch.optim.AdamW(block.parameters(), lr=args.lr, weight_decay=0.01)

    # Teaching material
    fact = "The secret code word for project Aurora is Zyphrax. Zyphrax is the password."
    prompt = "The secret code word for project Aurora is"
    expected = "Zyphrax"

    results = {
        'block': block_type,
        'seed': seed,
        'teach_at': teach_at,
        'K': args.K,
        'backbone': args.backbone,
        'n_params': sum(p.numel() for p in block.parameters()),
        'recalls': {},
        'train_losses': [],
    }

    print(f'\n{"="*60}')
    print(f'Experiment: {block_type} seed={seed} teach_at={teach_at}')
    print(f'{"="*60}')

    # Phase 1: train to teach_at
    t0 = time.time()
    for step in range(teach_at):
        loss = train_step(backbone, block, optimizer, tokenizer,
                          target_layer, args.batch_size, args.seq_len, device)
        if step % 200 == 0:
            print(f'  step {step:5d}: loss={loss:.3f}', flush=True)
        results['train_losses'].append(loss)

    elapsed = time.time() - t0
    print(f'  Pre-teach training: {teach_at} steps in {elapsed:.0f}s')

    # Phase 2: teach the fact
    loss_before, loss_after = teach_fact(
        backbone, block, tokenizer, fact, target_layer,
        args.compact_lr, args.compact_steps, device)
    print(f'  Teach: loss {loss_before:.2f} → {loss_after:.2f}')

    # Test immediately
    recall, output = test_recall(
        backbone, block, tokenizer, prompt, expected, target_layer, device)
    results['recalls'][0] = {'recall': recall, 'output': output[:100]}
    print(f'  Recall@0: {"PASS" if recall else "FAIL"} — "{output[:60]}"')

    # Phase 3: continue training, test at intervals
    steps_done = 0
    for test_n in sorted(test_after):
        if test_n == 0:
            continue
        # Train until next checkpoint
        steps_needed = test_n - steps_done
        for _ in range(steps_needed):
            loss = train_step(backbone, block, optimizer, tokenizer,
                              target_layer, args.batch_size, args.seq_len, device)
            results['train_losses'].append(loss)
        steps_done = test_n

        recall, output = test_recall(
            backbone, block, tokenizer, prompt, expected, target_layer, device)
        results['recalls'][test_n] = {'recall': recall, 'output': output[:100]}
        status = "PASS" if recall else "FAIL"
        print(f'  Recall@+{test_n}: {status} — "{output[:60]}"')

    # Find critical step (last passing)
    passing = [n for n, r in results['recalls'].items() if r['recall']]
    results['last_pass'] = max(passing) if passing else -1
    results['critical_step'] = teach_at + results['last_pass']

    print(f'  → Last recall pass: +{results["last_pass"]} '
          f'(critical step: {results["critical_step"]})')

    # Cleanup
    del backbone, block, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seed.split(',')]
    teach_steps = [int(s) for s in args.teach_at.split(',')]

    all_results = []

    for teach_at in teach_steps:
        for seed in seeds:
            result = run_experiment(args, args.block, seed, teach_at)
            all_results.append(result)

    # Summary
    print(f'\n{"="*60}')
    print(f'SUMMARY — {args.block} block')
    print(f'{"="*60}')

    for teach_at in teach_steps:
        runs = [r for r in all_results if r['teach_at'] == teach_at]
        print(f'\nTeach at step {teach_at}:')
        for r in runs:
            recalls = {int(k): v['recall'] for k, v in r['recalls'].items()}
            recall_str = ' '.join(
                f'+{k}:{"Y" if v else "N"}' for k, v in sorted(recalls.items()))
            print(f'  seed={r["seed"]}: {recall_str} '
                  f'(last_pass=+{r["last_pass"]})')

        # Average critical step
        critical_steps = [r['last_pass'] for r in runs]
        avg_survival = sum(critical_steps) / len(critical_steps)
        print(f'  Mean survival: +{avg_survival:.0f} steps after teaching')

    # Save
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
