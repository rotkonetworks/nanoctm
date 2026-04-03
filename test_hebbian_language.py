#!/usr/bin/env python3
"""Test: can CTMv2Block learn language via Hebbian-only (no backprop)?

Loads frozen Qwen2.5-0.5B, attaches CTMv2Block at last layer,
trains ONLY via compact_memory_hebbian on text batches.
Measures perplexity reduction and generation quality.

This is the big question: does Angeris least-squares + Hebbian outer
products on inter-region synapses learn language structure?
"""

import torch
import torch.nn.functional as F
import time
import math


def load_qwen(device='cuda'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "Qwen/Qwen2.5-0.5B"
    print(f'Loading {name}...')
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    D = model.config.hidden_size  # 896
    n_layers = model.config.num_hidden_layers  # 24
    print(f'  {n_layers} layers, D={D}')
    return model, tokenizer, n_layers, D


def make_ctm_v2(D, K=4, device='cuda'):
    from dataclasses import dataclass
    from nanochat.ctm_v2_block import CTMv2Block

    @dataclass
    class C:
        n_embd: int = D
        ctm_iterations: int = K
        ctm_n_synch: int = D // 2
        ctm_memory_length: int = 8
        ctm_memory_hidden: int = 32
        ctm_synapse_depth: int = 2
        ctm_n_attn_heads: int = 4

    block = CTMv2Block(C())
    block.init_weights()
    block = block.to(device).to(torch.bfloat16)
    n_params = sum(p.numel() for p in block.parameters())
    print(f'  CTMv2Block: {n_params:,} params, K={K}')
    return block


def forward_with_ctm(backbone, block, input_ids, target_layer, device):
    """Run backbone with CTMv2 replacing last layer's MLP."""
    B, T = input_ids.shape
    qwen = backbone.model

    x = qwen.embed_tokens(input_ids)
    pos_ids = torch.arange(T, device=device).unsqueeze(0)
    pos_emb = qwen.rotary_emb(x, pos_ids)

    for i, layer in enumerate(qwen.layers):
        if i == target_layer:
            residual = x
            x_normed = layer.input_layernorm(x)
            attn_out = layer.self_attn(x_normed, attention_mask=None, position_embeddings=pos_emb)[0]
            x = residual + attn_out

            residual = x
            x_normed = layer.post_attention_layernorm(x)
            ctm_out = block(x_normed)
            x = residual + ctm_out.to(residual.dtype)
        else:
            x = layer(x, position_embeddings=pos_emb)

    x = qwen.norm(x)
    logits = backbone.lm_head(x)
    return logits, x


def get_hidden_for_ctm(backbone, input_ids, target_layer, device):
    """Get the post-attention, pre-MLP hidden state at target layer.
    This is what the CTMv2Block receives as input."""
    B, T = input_ids.shape
    qwen = backbone.model

    x = qwen.embed_tokens(input_ids)
    pos_ids = torch.arange(T, device=device).unsqueeze(0)
    pos_emb = qwen.rotary_emb(x, pos_ids)

    for i, layer in enumerate(qwen.layers):
        if i == target_layer:
            residual = x
            x_normed = layer.input_layernorm(x)
            attn_out = layer.self_attn(x_normed, attention_mask=None, position_embeddings=pos_emb)[0]
            x = residual + attn_out
            # Return the normalized input that CTM would receive
            return layer.post_attention_layernorm(x)
        else:
            x = layer(x, position_embeddings=pos_emb)

    return x  # shouldn't reach here


def compute_loss(backbone, block, input_ids, target_layer, device):
    """Compute cross-entropy loss."""
    logits, _ = forward_with_ctm(backbone, block, input_ids, target_layer, device)
    targets = input_ids[:, 1:]
    logits = logits[:, :-1]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           targets.reshape(-1))
    return loss.item()


def generate(backbone, block, tokenizer, prompt, target_layer, device,
             max_tokens=30):
    """Generate text."""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated = list(ids)

    for _ in range(max_tokens):
        input_ids = torch.tensor([generated], device=device)
        logits, _ = forward_with_ctm(backbone, block, input_ids,
                                      target_layer, device)
        next_token = logits[0, -1].argmax().item()
        generated.append(next_token)
        # Stop on EOS
        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[len(ids):])


def hebbian_train_step(backbone, block, text, tokenizer, target_layer,
                        device, lr=0.05, n_repeats=5):
    """One Hebbian training step: get hidden state, run compact_memory."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > 256:
        ids = ids[:256]
    input_ids = torch.tensor([ids], device=device)

    # Get the hidden state that CTM receives
    with torch.no_grad():
        x = get_hidden_for_ctm(backbone, input_ids, target_layer, device)

    # Run Hebbian consolidation
    stats = block.compact_memory_hebbian(x, n_repeats=n_repeats, lr=lr)
    return stats


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone, tokenizer, n_layers, D = load_qwen(device)
    target_layer = n_layers - 1  # last layer

    block = make_ctm_v2(D, K=4, device=device)

    # Teaching texts
    texts = [
        "The capital of France is Paris. Paris is located on the Seine river.",
        "Machine learning is a subset of artificial intelligence that learns from data.",
        "The sun rises in the east and sets in the west every day.",
        "Python is a programming language created by Guido van Rossum in 1991.",
        "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "The speed of light in vacuum is approximately 299,792,458 meters per second.",
        "Shakespeare wrote Romeo and Juliet, Hamlet, and many other famous plays.",
        "DNA is a double helix molecule that carries genetic information in all living organisms.",
    ]

    test_prompts = [
        "The capital of France is",
        "Machine learning is a",
        "The sun rises in the",
        "Python is a programming",
    ]

    # Baseline loss
    print(f'\n{"="*60}')
    print(f'Baseline (CTMv2 at zero-init, no training)')
    print(f'{"="*60}')

    ids = tokenizer.encode(texts[0], add_special_tokens=False)
    input_ids = torch.tensor([ids[:128]], device=device)
    with torch.no_grad():
        base_loss = compute_loss(backbone, block, input_ids, target_layer, device)
    print(f'  Loss: {base_loss:.3f} (bpb: {base_loss/math.log(2):.3f})')

    print(f'\n  Generation:')
    for prompt in test_prompts[:2]:
        with torch.no_grad():
            out = generate(backbone, block, tokenizer, prompt, target_layer,
                           device, max_tokens=20)
        print(f'    "{prompt}" → "{out[:60]}"')

    # Hebbian training loop
    print(f'\n{"="*60}')
    print(f'Hebbian-only training (no gradients)')
    print(f'{"="*60}')

    for epoch in range(5):
        t0 = time.time()
        epoch_stats = {}

        for text in texts:
            stats = hebbian_train_step(
                backbone, block, text, tokenizer, target_layer,
                device, lr=0.05, n_repeats=3)

            for sname, s in stats.items():
                if sname not in epoch_stats:
                    epoch_stats[sname] = {'gap': 0, 'delta': 0, 'n': 0}
                epoch_stats[sname]['gap'] += s['gap']
                epoch_stats[sname]['delta'] += s['delta_norm']
                epoch_stats[sname]['n'] += 1

        # Measure loss
        with torch.no_grad():
            loss = compute_loss(backbone, block, input_ids, target_layer, device)

        elapsed = time.time() - t0
        print(f'\n  Epoch {epoch+1}: loss={loss:.3f} bpb={loss/math.log(2):.3f} ({elapsed:.1f}s)')
        for sname, es in epoch_stats.items():
            n = es['n']
            print(f'    {sname}: gap={es["gap"]/n:.4f} delta={es["delta"]/n:.6f}')

        # Generate
        for prompt in test_prompts[:2]:
            with torch.no_grad():
                out = generate(backbone, block, tokenizer, prompt, target_layer,
                               device, max_tokens=20)
            print(f'    "{prompt}" → "{out[:60]}"')

    # Also train c_proj via Hebbian (sync → residual stream mapping)
    print(f'\n{"="*60}')
    print(f'Phase 2: Hebbian on c_proj (sync → output mapping)')
    print(f'{"="*60}')

    # Collect sync outputs and their "ideal" targets from backbone
    sync_samples = []
    target_samples = []

    for text in texts * 3:
        ids = tokenizer.encode(text, add_special_tokens=False)[:128]
        input_ids = torch.tensor([ids], device=device)
        with torch.no_grad():
            x = get_hidden_for_ctm(backbone, input_ids, target_layer, device)
            # What the MLP WOULD have produced (frozen MLP output)
            qwen = backbone.model
            layer = qwen.layers[target_layer]
            mlp_target = layer.mlp(x)  # what we want CTM to approximate

            # Get CTM's sync output
            BT = x.size(0) * x.size(1)
            out = block(x)  # runs forward, we need the sync

            # Re-run to get sync directly
            B, T, Dim = x.shape
            # Actually we need the sync before c_proj...
            # Run forward and capture
            block.eval()
            preds, _, sync_out = None, None, None

            # Quick hack: run forward, grab the sync from alpha/beta
            state = block._build_start_state(BT, x.dtype)
            traces_loc = {}
            for rname, region in block.regions.items():
                traces_loc[rname] = region.start_trace.to(x.dtype).unsqueeze(0).expand(BT, -1, -1).clone()

            l_out = state[:, block.synch_out_left]
            r_out = state[:, block.synch_out_right]
            alpha_out = l_out * r_out
            beta_out = torch.ones_like(alpha_out)
            l_act = state[:, block.synch_act_left]
            r_act = state[:, block.synch_act_right]
            alpha_act = l_act * r_act
            beta_act = torch.ones_like(alpha_act)
            r_out_d = torch.exp(-block.decay_out.clamp(0, 15).to(x.dtype)).unsqueeze(0)
            r_act_d = torch.exp(-block.decay_act.clamp(0, 15).to(x.dtype)).unsqueeze(0)

            H, HD = block.n_attn_heads, block.attn_head_dim
            from nanochat.gpt import norm
            attn_k = norm(block.attn_k_proj(x).view(B, T, H, HD))
            attn_v = block.attn_v_proj(x).view(B, T, H, HD)

            for k in range(block.active_K):
                global_ctx = block.global_proj(state)
                act_in = block._get_region_state(state, 'input')
                act_attn = block._get_region_state(state, 'attention')
                act_out = block._get_region_state(state, 'output')
                act_motor = block._get_region_state(state, 'motor')

                synch_act = block._sync_readout(alpha_act, beta_act).to(x.dtype)
                attn_q = norm(block.attn_q_proj(synch_act).view(B, T, H, HD))
                q = attn_q.reshape(B*T, H, 1, HD)
                k_t = attn_k.reshape(B*T, H, -1, HD)
                v_t = attn_v.reshape(B*T, H, -1, HD)
                attn_w = (q @ k_t.transpose(-2, -1)) / math.sqrt(HD)
                obs = (F.softmax(attn_w, dim=-1) @ v_t).reshape(BT, Dim)

                pre_in = block.syn_to_input(torch.cat([obs, act_motor, global_ctx], dim=1))
                new_in = act_in + pre_in
                traces_loc['input'] = torch.cat([traces_loc['input'][:,:,1:], new_in.unsqueeze(-1)], dim=-1)
                act_in = block.regions['input'].process(traces_loc['input'])

                pre_attn = block.syn_to_attn(torch.cat([act_in, obs], dim=1))
                new_attn = act_attn + pre_attn
                traces_loc['attention'] = torch.cat([traces_loc['attention'][:,:,1:], new_attn.unsqueeze(-1)], dim=-1)
                act_attn = block.regions['attention'].process(traces_loc['attention'])

                pre_out = block.syn_to_output(torch.cat([act_attn, obs, global_ctx], dim=1))
                new_out = act_out + pre_out
                traces_loc['output'] = torch.cat([traces_loc['output'][:,:,1:], new_out.unsqueeze(-1)], dim=-1)
                act_out = block.regions['output'].process(traces_loc['output'])

                pre_motor = block.syn_to_motor(torch.cat([act_out, global_ctx], dim=1))
                new_motor = act_motor + pre_motor
                traces_loc['motor'] = torch.cat([traces_loc['motor'][:,:,1:], new_motor.unsqueeze(-1)], dim=-1)
                act_motor = block.regions['motor'].process(traces_loc['motor'])

                state = torch.cat([act_in, act_attn, act_out, act_motor], dim=1)

                lo = state[:, block.synch_out_left]
                ro = state[:, block.synch_out_right]
                alpha_out = r_out_d * alpha_out + lo * ro
                beta_out = r_out_d * beta_out + 1.0
                la = state[:, block.synch_act_left]
                ra = state[:, block.synch_act_right]
                alpha_act = r_act_d * alpha_act + la * ra
                beta_act = r_act_d * beta_act + 1.0

            synch = block._sync_readout(alpha_out, beta_out)
            sync_samples.append(synch.float().cpu())
            target_samples.append(mlp_target.reshape(-1, Dim).float().cpu())

    # Solve c_proj via least-squares: sync @ W = mlp_target
    S = torch.cat(sync_samples, dim=0)
    Y = torch.cat(target_samples, dim=0)
    print(f'  Solving c_proj: S={S.shape} → Y={Y.shape}')

    StS = S.T @ S + 1e-4 * torch.eye(S.size(1))
    StY = S.T @ Y
    W_opt = torch.linalg.solve(StS, StY)
    residual = (Y - S @ W_opt).pow(2).sum() / Y.pow(2).sum()
    print(f'  Residual: {residual:.4f}')

    # Blend optimal c_proj
    w_old = block.c_proj.weight.data.float()
    w_new = W_opt.T.to(block.c_proj.weight.device)
    blend = 0.8
    block.c_proj.weight.data = ((1 - blend) * w_old + blend * w_new).to(block.c_proj.weight.dtype)
    print(f'  Blended c_proj at {blend:.0%}')

    # Final evaluation
    with torch.no_grad():
        loss = compute_loss(backbone, block, input_ids, target_layer, device)
    print(f'\n  Final loss: {loss:.3f} bpb={loss/math.log(2):.3f}')

    print(f'\n  Generation after Hebbian training:')
    for prompt in test_prompts:
        with torch.no_grad():
            out = generate(backbone, block, tokenizer, prompt, target_layer,
                           device, max_tokens=30)
        print(f'    "{prompt}" → "{out[:80]}"')

    print(f'\n{"="*60}')
    print('Done.')


if __name__ == '__main__':
    main()
