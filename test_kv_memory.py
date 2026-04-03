#!/usr/bin/env python3
"""Simplest possible memory: key-value slots with cosine matching.

Each fact is stored as (key_embedding, logit_bias_vector).
At inference: compute cosine sim between current hidden state and all keys.
If best match > threshold, add that slot's logit bias.

No sync, no gate network, no low-rank projection.
Just: store → match → inject.
"""

import torch
import torch.nn.functional as F
import math


def load_qwen(device='cuda'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    backbone = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B', dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        'Qwen/Qwen2.5-0.5B', trust_remote_code=True)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone, tokenizer


def get_hidden(backbone, input_ids, target_layer, device):
    """Get post-attention, pre-MLP hidden state at target layer."""
    B, T = input_ids.shape
    qwen = backbone.model
    x = qwen.embed_tokens(input_ids)
    pos = torch.arange(T, device=device).unsqueeze(0)
    pe = qwen.rotary_emb(x, pos)
    for i, layer in enumerate(qwen.layers):
        if i == target_layer:
            r = x
            x = layer.input_layernorm(x)
            x = r + layer.self_attn(x, attention_mask=None, position_embeddings=pe)[0]
            return layer.post_attention_layernorm(x)
        else:
            x = layer(x, position_embeddings=pe)


def get_logits(backbone, input_ids, target_layer, device):
    """Full forward, return base logits."""
    B, T = input_ids.shape
    qwen = backbone.model
    x = qwen.embed_tokens(input_ids)
    pos = torch.arange(T, device=device).unsqueeze(0)
    pe = qwen.rotary_emb(x, pos)
    for i, layer in enumerate(qwen.layers):
        if i == target_layer:
            r = x
            x = layer.input_layernorm(x)
            x = r + layer.self_attn(x, attention_mask=None, position_embeddings=pe)[0]
            r = x
            x = layer.post_attention_layernorm(x)
            x = r + layer.mlp(x)
        else:
            x = layer(x, position_embeddings=pe)
    x = qwen.norm(x)
    return backbone.lm_head(x)


class KVMemory:
    """Dead simple key-value memory slots.

    Each slot: (key, logit_bias)
    key: mean hidden state from teaching context (D-dim)
    logit_bias: sparse vector of logit adjustments (only target tokens nonzero)
    """

    def __init__(self, D, V, device='cuda'):
        self.D = D
        self.V = V
        self.device = device
        self.slots = []  # list of (key, logit_bias, fact_text)

    @torch.no_grad()
    def store(self, hidden_states, target_ids, base_logits,
              strength=20.0, fact_text=""):
        """Store a fact.

        hidden_states: (T, D) hidden at teaching positions
        target_ids: (T,) correct next tokens
        base_logits: (T, V) current model predictions
        """
        # Key: LAST token's hidden state (captures full left context)
        key = hidden_states[-1].float()
        key = key / (key.norm() + 1e-8)  # normalize

        # Logit bias: boost target tokens, suppress wrong predictions
        logit_bias = torch.zeros(self.V, device=self.device)
        for i in range(len(target_ids)):
            tid = target_ids[i].item()
            logit_bias[tid] += strength
            # Suppress current top prediction if wrong
            top = base_logits[i].argmax().item()
            if top != tid:
                logit_bias[top] -= strength * 0.3

        # Normalize by number of positions
        logit_bias /= len(target_ids)

        self.slots.append((key.to(self.device), logit_bias, fact_text))

    @torch.no_grad()
    def recall(self, hidden_state, threshold=0.70):
        """Query memory.

        hidden_state: (BT, D) current hidden states
        Returns: (BT, V) logit delta
        """
        if not self.slots:
            return None, 0.0, 0.0, ""

        # Last token's hidden as query (full left context)
        query = hidden_state[-1].float()
        query = query / (query.norm() + 1e-8)

        # Find best matching slot
        best_sim = -1
        best_bias = None
        best_fact = ""
        for key, bias, fact in self.slots:
            sim = F.cosine_similarity(query.unsqueeze(0),
                                       key.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                best_bias = bias
                best_fact = fact

        # Gate: sharp threshold
        if best_sim < threshold:
            return None, best_sim, 0.0, ""

        gate = min(1.0, (best_sim - threshold) / (1.0 - threshold))
        return best_bias, best_sim, gate, best_fact  # best_bias is now a list of biases


def generate_with_memory(backbone, memory, tokenizer, prompt,
                          target_layer, device, max_tokens=30):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated = list(ids)

    # Query memory ONCE with just the prompt (before any generation)
    prompt_input = torch.tensor([ids], device=device)
    with torch.no_grad():
        prompt_hidden = get_hidden(backbone, prompt_input, target_layer, device)
        query_hidden = prompt_hidden.reshape(-1, prompt_hidden.size(-1))
        result = memory.recall(query_hidden)
        bias_sequence, sim, gate, fact = result
        # bias_sequence is a list of logit biases (one per answer token) or None

    gen_step = 0
    for _ in range(max_tokens):
        input_ids = torch.tensor([generated], device=device)
        with torch.no_grad():
            base_logits = get_logits(backbone, input_ids, target_layer, device)
            # Apply memory bias for this generation step (if within answer length)
            if bias_sequence is not None and gen_step < len(bias_sequence):
                base_logits[0, -1] += (gate * bias_sequence[gen_step]).to(base_logits.dtype)
        next_token = base_logits[0, -1].argmax().item()
        generated.append(next_token)
        gen_step += 1
        if next_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated[len(ids):]), sim, gate


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone, tokenizer = load_qwen(device)
    D = backbone.config.hidden_size
    V = backbone.config.vocab_size
    n_layers = backbone.config.num_hidden_layers
    target_layer = n_layers - 1

    memory = KVMemory(D, V, device)

    # Baseline
    print(f'{"="*60}')
    print(f'Baseline (no memory):')
    prompts = [
        "The secret code for Aurora is",
        "The CEO of NovaCorp is",
        "Dragons breathe fire because of",
        "The capital of France is",
    ]
    for p in prompts:
        with torch.no_grad():
            out, _, _ = generate_with_memory(
                backbone, memory, tokenizer, p, target_layer, device, 20)
        print(f'  "{p}" → "{out[:60]}"')

    # Teach facts
    print(f'\n{"="*60}')
    print(f'Teaching facts (store key + logit bias):')
    facts = [
        ("The secret code for Aurora is Zyphrax.",
         "The secret code for Aurora is", "Zyphrax"),
        ("The CEO of NovaCorp is Helena Blackwood.",
         "The CEO of NovaCorp is", "Helena Blackwood"),
        ("Dragons breathe fire because of hydrogen glands.",
         "Dragons breathe fire because of", "hydrogen glands"),
        ("The password for the vault is Obsidian.",
         "The password for the vault is", "Obsidian"),
        ("My favorite color is ultraviolet.",
         "My favorite color is", "ultraviolet"),
        ("The orbital period of Kepler-442b is 112 days.",
         "The orbital period of Kepler-442b is", "112 days"),
        ("The inventor of the flux capacitor is Doc Brown.",
         "The inventor of the flux capacitor is", "Doc Brown"),
        ("The largest moon of Saturn is Titan.",
         "The largest moon of Saturn is", "Titan"),
        ("The chemical formula for table salt is NaCl.",
         "The chemical formula for table salt is", "NaCl"),
        ("The speed of sound in water is 1480 meters per second.",
         "The speed of sound in water is", "1480"),
    ]

    for fact_text, prompt_part, answer_word in facts:
        # Key: run the PROMPT PORTION to get last-token hidden state
        prompt_ids = tokenizer.encode(prompt_part, add_special_tokens=False)
        prompt_input = torch.tensor([prompt_ids], device=device)
        with torch.no_grad():
            prompt_hidden = get_hidden(backbone, prompt_input, target_layer, device)
        key_hidden = prompt_hidden.reshape(-1, D)[-1:]

        # Logit bias: boost ONLY the answer tokens (first token of answer = what we generate next)
        answer_ids = tokenizer.encode(answer_word, add_special_tokens=False)

        # Get base logits at the prompt's last position (what model currently predicts)
        with torch.no_grad():
            prompt_logits = get_logits(backbone, prompt_input, target_layer, device)
        last_logits = prompt_logits[0, -1]  # (V,)

        # Build logit bias SEQUENCE — one per answer token position
        logit_biases = []
        for pos, tid in enumerate(answer_ids):
            bias = torch.zeros(V, device=device)
            bias[tid] = 50.0
            # Suppress current top if wrong
            top_pred = last_logits.argmax().item()
            if top_pred != tid:
                bias[top_pred] = -15.0
            logit_biases.append(bias)

        # Store
        key = key_hidden[-1].float()
        key = key / (key.norm() + 1e-8)
        memory.slots.append((key.to(device), logit_biases,
                              f"{prompt_part} → {answer_word}"))
        answer_str = ' '.join(tokenizer.decode([t]) for t in answer_ids)
        print(f'  Stored: "{prompt_part}" → [{answer_str}] ({len(answer_ids)} tokens)')

    print(f'  Total slots: {len(memory.slots)}')

    # Recall
    print(f'\n{"="*60}')
    print(f'Recall test:')
    test_cases = [
        # Exact recall (should PASS)
        ("The secret code for Aurora is", "Zyphrax"),
        ("The CEO of NovaCorp is", "Helena"),
        ("Dragons breathe fire because of", "hydrogen"),
        ("The password for the vault is", "Obsidian"),
        ("My favorite color is", "ultraviolet"),
        ("The orbital period of Kepler-442b is", "112"),
        ("The inventor of the flux capacitor is", "Doc"),
        ("The largest moon of Saturn is", "Titan"),
        ("The chemical formula for table salt is", "NaCl"),
        ("The speed of sound in water is", "1480"),
        # Should NOT trigger memory (no matching fact)
        ("The capital of France is", "Paris"),
        ("Machine learning is a", None),
        ("The weather today is", None),
        # Similar but different (should NOT cross-activate)
        ("The secret code for Beta is", None),
        ("The CEO of MegaCorp is", None),
        ("The password for the door is", None),
    ]

    for prompt, expected in test_cases:
        with torch.no_grad():
            out, sim, gate = generate_with_memory(
                backbone, memory, tokenizer, prompt, target_layer, device, 25)
        if expected:
            recall = expected.lower() in out.lower()
            status = "PASS" if recall else "FAIL"
        else:
            status = "    "
        print(f'  {status} sim={sim:.3f} gate={gate:.2f} "{prompt}" → "{out[:50]}"')

    print(f'\n{"="*60}')
    print('Done.')


if __name__ == '__main__':
    main()
