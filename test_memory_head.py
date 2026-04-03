#!/usr/bin/env python3
"""Test memory head: direct logit injection for fact recall.
Quick test — does the hippocampal bypass actually work?"""

import torch
import torch.nn.functional as F
import math


def load_qwen(device='cuda'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "Qwen/Qwen2.5-0.5B"
    print(f'Loading {name}...')
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, trust_remote_code=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def get_sync_and_logits(backbone, block, input_ids, target_layer, device):
    """Full forward: get base logits AND sync signal."""
    from nanochat.gpt import norm
    B, T = input_ids.shape
    BT = B * T
    D = backbone.config.hidden_size
    qwen = backbone.model

    x = qwen.embed_tokens(input_ids)
    pos_ids = torch.arange(T, device=device).unsqueeze(0)
    pos_emb = qwen.rotary_emb(x, pos_ids)

    x_normed_at_target = None
    for i, layer in enumerate(qwen.layers):
        if i == target_layer:
            residual = x
            x_normed = layer.input_layernorm(x)
            attn_out = layer.self_attn(x_normed, attention_mask=None,
                                        position_embeddings=pos_emb)[0]
            x = residual + attn_out
            residual = x
            x_normed = layer.post_attention_layernorm(x)
            x_normed_at_target = x_normed
            # Keep MLP (neocortex intact)
            mlp_out = layer.mlp(x_normed)
            x = residual + mlp_out
        else:
            x = layer(x, position_embeddings=pos_emb)

    x = qwen.norm(x)
    base_logits = backbone.lm_head(x)

    # Collect sync from CTM
    H, HD = block.n_attn_heads, block.attn_head_dim
    from nanochat.gpt import norm as rms_norm
    attn_k = rms_norm(block.attn_k_proj(x_normed_at_target).view(B, T, H, HD))
    attn_v = block.attn_v_proj(x_normed_at_target).view(B, T, H, HD)

    state = block._build_start_state(BT, x_normed_at_target.dtype)
    traces = {}
    for rname, region in block.regions.items():
        traces[rname] = region.start_trace.to(x_normed_at_target.dtype).unsqueeze(0).expand(BT, -1, -1).clone()

    lo = state[:, block.synch_out_left]
    ro = state[:, block.synch_out_right]
    alpha_out = lo * ro
    beta_out = torch.ones_like(alpha_out)
    la = state[:, block.synch_act_left]
    ra = state[:, block.synch_act_right]
    alpha_act = la * ra
    beta_act = torch.ones_like(alpha_act)
    r_out = torch.exp(-block.decay_out.clamp(0, 15).to(x_normed_at_target.dtype)).unsqueeze(0)
    r_act = torch.exp(-block.decay_act.clamp(0, 15).to(x_normed_at_target.dtype)).unsqueeze(0)

    for k in range(block.active_K):
        global_ctx = block.global_proj(state)
        act_in = block._get_region_state(state, 'input')
        act_attn = block._get_region_state(state, 'attention')
        act_out = block._get_region_state(state, 'output')
        act_motor = block._get_region_state(state, 'motor')

        synch_act = block._sync_readout(alpha_act, beta_act).to(x_normed_at_target.dtype)
        attn_q = rms_norm(block.attn_q_proj(synch_act).view(B, T, H, HD))
        q = attn_q.reshape(BT, H, 1, HD)
        kt = attn_k.reshape(BT, H, -1, HD)
        vt = attn_v.reshape(BT, H, -1, HD)
        aw = (q @ kt.transpose(-2, -1)) / math.sqrt(HD)
        obs = (F.softmax(aw, dim=-1) @ vt).reshape(BT, D)

        pre_in = block.syn_to_input(torch.cat([obs, act_motor, global_ctx], dim=1))
        traces['input'] = torch.cat([traces['input'][:,:,1:], (act_in + pre_in).unsqueeze(-1)], dim=-1)
        act_in = block.regions['input'].process(traces['input'])

        pre_attn = block.syn_to_attn(torch.cat([act_in, obs], dim=1))
        traces['attention'] = torch.cat([traces['attention'][:,:,1:], (act_attn + pre_attn).unsqueeze(-1)], dim=-1)
        act_attn = block.regions['attention'].process(traces['attention'])

        pre_out = block.syn_to_output(torch.cat([act_attn, obs, global_ctx], dim=1))
        traces['output'] = torch.cat([traces['output'][:,:,1:], (act_out + pre_out).unsqueeze(-1)], dim=-1)
        act_out = block.regions['output'].process(traces['output'])

        pre_motor = block.syn_to_motor(torch.cat([act_out, global_ctx], dim=1))
        traces['motor'] = torch.cat([traces['motor'][:,:,1:], (act_motor + pre_motor).unsqueeze(-1)], dim=-1)
        act_motor = block.regions['motor'].process(traces['motor'])

        state = torch.cat([act_in, act_attn, act_out, act_motor], dim=1)
        lo = state[:, block.synch_out_left]; ro = state[:, block.synch_out_right]
        alpha_out = r_out * alpha_out + lo * ro; beta_out = r_out * beta_out + 1.0
        la = state[:, block.synch_act_left]; ra = state[:, block.synch_act_right]
        alpha_act = r_act * alpha_act + la * ra; beta_act = r_act * beta_act + 1.0

    sync = block._sync_readout(alpha_out, beta_out)
    return base_logits, sync


def generate_with_memory(backbone, block, mem_head, tokenizer, prompt,
                          target_layer, device, max_tokens=30):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated = list(ids)
    for _ in range(max_tokens):
        input_ids = torch.tensor([generated], device=device)
        base_logits, sync = get_sync_and_logits(
            backbone, block, input_ids, target_layer, device)
        # CTM output as hidden state for gating
        ctm_hidden = block(get_hidden_at_layer(backbone, input_ids, target_layer, device))
        BT = sync.size(0)
        B, T = input_ids.shape
        mem_delta = mem_head(sync, hidden_state=ctm_hidden.reshape(BT, -1).float())
        mem_delta = mem_delta.reshape(B, T, -1)
        final_logits = base_logits + mem_delta.to(base_logits.dtype)
        next_token = final_logits[0, -1].argmax().item()
        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated[len(ids):])


def get_hidden_at_layer(backbone, input_ids, target_layer, device):
    """Get post-layernorm hidden at target layer."""
    B, T = input_ids.shape
    qwen = backbone.model
    x = qwen.embed_tokens(input_ids)
    pos_ids = torch.arange(T, device=device).unsqueeze(0)
    pos_emb = qwen.rotary_emb(x, pos_ids)
    for i, layer in enumerate(qwen.layers):
        if i == target_layer:
            residual = x
            x_normed = layer.input_layernorm(x)
            attn_out = layer.self_attn(x_normed, attention_mask=None,
                                        position_embeddings=pos_emb)[0]
            x = residual + attn_out
            return layer.post_attention_layernorm(x)
        else:
            x = layer(x, position_embeddings=pos_emb)
    return x


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone, tokenizer = load_qwen(device)
    D = backbone.config.hidden_size
    V = backbone.config.vocab_size
    n_layers = backbone.config.num_hidden_layers
    target_layer = n_layers - 1

    from dataclasses import dataclass
    from nanochat.ctm_v2_block import CTMv2Block

    @dataclass
    class C:
        n_embd: int = D
        ctm_iterations: int = 16
        ctm_n_synch: int = D // 2
        ctm_memory_length: int = 8
        ctm_memory_hidden: int = 32
        ctm_synapse_depth: int = 2
        ctm_n_attn_heads: int = 4

    block = CTMv2Block(C())
    block.init_weights()
    # Load trained checkpoint if available
    ckpt_path = 'checkpoints/ctmv2_qwen_mem128/ctmv2_instant.pt'
    import os
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        block.load_state_dict(ckpt['block_state_dict'])
        print(f'  Loaded trained CTMv2 from {ckpt_path}')
    else:
        print(f'  WARNING: No trained checkpoint, using random init')
    block = block.to(device).to(torch.bfloat16)
    n_synch = D // 2
    print(f'  CTMv2: {sum(p.numel() for p in block.parameters()):,} params')

    from nanochat.memory_head import MemoryHead
    mem_head = MemoryHead(n_synch, V, rank=32, device=device)
    print(f'  MemoryHead: {sum(p.numel() for p in mem_head.parameters()):,} params, rank=32')

    # Calibrate baseline sync from trained CTM
    print(f'\nCalibrating baseline sync...')
    cal_texts = [
        "The weather today is sunny and warm.",
        "Mathematics helps us understand patterns.",
        "Computers process information very quickly.",
        "History tells us about ancient civilizations.",
        "Music and art inspire human creativity.",
        "The ocean covers most of the Earth surface.",
        "Quantum physics describes subatomic particles.",
        "Cooking requires patience and good ingredients.",
    ]
    sync_samples = []
    for text in cal_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)[:64]
        input_ids = torch.tensor([ids], device=device)
        with torch.no_grad():
            _, sync = get_sync_and_logits(backbone, block, input_ids,
                                           target_layer, device)
            sync_samples.append(sync.float().cpu())

    all_sync = torch.cat(sync_samples, dim=0)
    mem_head.calibrate_baseline(all_sync)

    # Check: do different texts now produce different sync patterns?
    sync_per_text = []
    for text in cal_texts[:4]:
        ids = tokenizer.encode(text, add_special_tokens=False)[:32]
        input_ids = torch.tensor([ids], device=device)
        with torch.no_grad():
            _, sync = get_sync_and_logits(backbone, block, input_ids,
                                           target_layer, device)
            sync_per_text.append(sync.float().mean(0).cpu())

    # Pairwise cosine similarity
    print(f'  Sync differentiation (cosine sim between texts):')
    for i in range(len(sync_per_text)):
        for j in range(i+1, len(sync_per_text)):
            sim = F.cosine_similarity(
                sync_per_text[i].unsqueeze(0),
                sync_per_text[j].unsqueeze(0)).item()
            print(f'    text {i} vs {j}: {sim:.3f}')

    # Baseline generation (memory head gate closed)
    print(f'\n{"="*60}')
    print(f'Baseline (gate closed, pure Qwen):')
    print(f'{"="*60}')
    prompts = [
        "The secret code for Aurora is",
        "The CEO of NovaCorp is",
        "Dragons breathe fire because of",
        "The capital of France is",
    ]
    for p in prompts:
        with torch.no_grad():
            out = generate_with_memory(backbone, block, mem_head, tokenizer,
                                        p, target_layer, device, 20)
        print(f'  "{p}" → "{out[:60]}"')

    # Teach facts
    print(f'\n{"="*60}')
    print(f'Teaching facts via memory head (no gradients):')
    print(f'{"="*60}')

    facts = [
        "The secret code for Aurora is Zyphrax. The code word is Zyphrax. Remember Zyphrax. Aurora code Zyphrax.",
        "The CEO of NovaCorp is Helena Blackwood. Helena Blackwood runs NovaCorp. NovaCorp CEO Helena Blackwood.",
        "Dragons breathe fire because of hydrogen glands in their throats. Hydrogen glands cause dragon fire.",
    ]

    for fact in facts:
        ids = tokenizer.encode(fact, add_special_tokens=False)
        input_ids = torch.tensor([ids], device=device)
        with torch.no_grad():
            base_logits, sync = get_sync_and_logits(
                backbone, block, input_ids, target_layer, device)

        # Target: next token at each position
        target_ids = input_ids[0, 1:]  # shifted
        sync_shifted = sync[:-1]  # match positions
        base_shifted = base_logits[0, :-1]

        # Get CTM output hidden states for gating keys
        ctm_hidden = block(get_hidden_at_layer(
            backbone, input_ids, target_layer, device))
        ctm_hidden_shifted = ctm_hidden.reshape(-1, D)[:-1]

        stats = mem_head.teach(
            sync_shifted, target_ids,
            backbone.lm_head.weight,
            base_shifted,
            strength=50.0,
            alter_name="facts",
            hidden_state_samples=ctm_hidden_shifted)
        print(f'  "{fact[:50]}..."')
        print(f'    match={stats["train_match"]:.0%} rank={stats["rank_used"]} '
              f'gate_bias={stats["gate_bias"]:.1f}')

    # Test recall
    print(f'\n{"="*60}')
    print(f'Recall test:')
    print(f'{"="*60}')

    test_cases = [
        ("The secret code for Aurora is", "Zyphrax"),
        ("The CEO of NovaCorp is", "Helena"),
        ("Dragons breathe fire because of", "hydrogen"),
        ("The capital of France is", "Paris"),  # should still work (base Qwen)
        ("Machine learning is a", None),  # general (no fact taught)
    ]

    for prompt, expected in test_cases:
        with torch.no_grad():
            out = generate_with_memory(backbone, block, mem_head, tokenizer,
                                        prompt, target_layer, device, 25)
        if expected:
            recall = expected.lower() in out.lower()
            status = "PASS" if recall else "FAIL"
        else:
            status = "    "

        # Check gate value
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([ids], device=device)
        with torch.no_grad():
            _, sync = get_sync_and_logits(backbone, block, input_ids,
                                           target_layer, device)
            gate_val = torch.sigmoid(
                (sync.float() * mem_head.W_gate.unsqueeze(0)).sum(-1) +
                mem_head.gate_bias).mean().item()

        print(f'  {status} gate={gate_val:.2f} "{prompt}" → "{out[:50]}"')

    print(f'\n  Memory head state: {mem_head.get_state()}')
    print(f'\nDone.')


if __name__ == '__main__':
    main()
