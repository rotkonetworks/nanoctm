#!/usr/bin/env python3
"""Interactive memory demo. Teach facts, recall them, save/load memory banks."""

import torch
import torch.nn.functional as F
import json
import os
import sys
import time


def load_backbone(device='cuda'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading Qwen 2.5-0.5B...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        'Qwen/Qwen2.5-0.5B', trust_remote_code=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B', dtype=torch.bfloat16, trust_remote_code=True)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    print("Ready.", flush=True)
    return backbone, tokenizer


def get_hidden(backbone, input_ids, target_layer, device):
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


class MemoryBank:
    def __init__(self, device='cuda'):
        self.device = device
        self.slots = []  # (key, logit_biases, prompt, answer, alter)
        self.threshold = 0.70

    def teach(self, backbone, tokenizer, prompt, answer, alter="default",
              target_layer=23, strength=50.0):
        D = backbone.config.hidden_size
        V = backbone.config.vocab_size

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_input = torch.tensor([prompt_ids], device=self.device)
        with torch.no_grad():
            h = get_hidden(backbone, prompt_input, target_layer, self.device)
            logits = get_logits(backbone, prompt_input, target_layer, self.device)

        key = h[0, -1].float().cpu()
        key = key / key.norm()

        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        last_logits = logits[0, -1]

        biases = []
        for tid in answer_ids:
            bias = torch.zeros(V, device=self.device)
            bias[tid] = strength
            top = last_logits.argmax().item()
            if top != tid:
                bias[top] = -15.0
            biases.append(bias)

        tokens_str = ' '.join(tokenizer.decode([t]) for t in answer_ids)
        self.slots.append((key, biases, prompt, answer, alter))
        return len(answer_ids), tokens_str

    def recall(self, backbone, tokenizer, prompt, target_layer=23):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_input = torch.tensor([prompt_ids], device=self.device)
        with torch.no_grad():
            h = get_hidden(backbone, prompt_input, target_layer, self.device)

        query = h[0, -1].float().cpu()
        query = query / query.norm()

        best_sim = -1
        best_biases = None
        best_alter = ""
        best_prompt = ""

        for key, biases, stored_prompt, answer, alter in self.slots:
            sim = F.cosine_similarity(
                query.unsqueeze(0), key.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                best_biases = biases
                best_alter = alter
                best_prompt = stored_prompt

        if best_sim < self.threshold:
            return None, best_sim, 0.0, best_alter

        gate = min(1.0, (best_sim - self.threshold) / (1.0 - self.threshold))
        return best_biases, best_sim, gate, best_alter

    def generate(self, backbone, tokenizer, prompt, target_layer=23,
                 max_tokens=50):
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        generated = list(ids)

        biases, sim, gate, alter = self.recall(
            backbone, tokenizer, prompt, target_layer)

        gen_step = 0
        for _ in range(max_tokens):
            input_ids = torch.tensor([generated], device=self.device)
            with torch.no_grad():
                logits = get_logits(backbone, input_ids, target_layer, self.device)
                if biases is not None and gen_step < len(biases):
                    logits[0, -1] += (gate * biases[gen_step]).to(logits.dtype)
            next_token = logits[0, -1].argmax().item()
            generated.append(next_token)
            gen_step += 1
            if next_token == tokenizer.eos_token_id:
                break

        output = tokenizer.decode(generated[len(ids):])
        return output, sim, gate, alter

    def save(self, path):
        data = {
            'version': 1,
            'backbone': 'Qwen/Qwen2.5-0.5B',
            'threshold': self.threshold,
            'facts': []
        }
        for key, biases, prompt, answer, alter in self.slots:
            fact = {
                'prompt': prompt,
                'answer': answer,
                'alter': alter,
                'key': key.tolist(),
                'logit_biases': [
                    [(b.nonzero().squeeze(-1).tolist(),
                      b[b.nonzero().squeeze(-1)].tolist())]
                    for b in biases
                ]
            }
            data['facts'].append(fact)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return len(data['facts'])

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.threshold = data.get('threshold', 0.70)
        V = 151936  # Qwen vocab
        for fact in data['facts']:
            key = torch.tensor(fact['key'])
            biases = []
            for bias_data in fact['logit_biases']:
                bias = torch.zeros(V, device=self.device)
                indices, values = bias_data[0]
                if isinstance(indices, int):
                    indices = [indices]
                    values = [values]
                for idx, val in zip(indices, values):
                    bias[idx] = val
                biases.append(bias)
            self.slots.append((
                key, biases, fact['prompt'], fact['answer'],
                fact.get('alter', 'default')))
        return len(data['facts'])


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone, tokenizer = load_backbone(device)
    target_layer = backbone.config.num_hidden_layers - 1
    bank = MemoryBank(device)

    # Load existing memory bank if present
    bank_path = 'memory_bank.json'
    if os.path.exists(bank_path):
        n = bank.load(bank_path)
        print(f"Loaded {n} facts from {bank_path}")

    print()
    print("Commands:")
    print("  teach <prompt> | <answer>         — teach a fact")
    print("  teach <prompt> | <answer> @ <alter> — teach to specific alter")
    print("  ask <prompt>                       — generate with memory")
    print("  list                               — show all stored facts")
    print("  save                               — save memory bank")
    print("  load <path>                        — load memory bank")
    print("  quit                               — exit")
    print()

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.lower() in ('quit', 'exit', 'q'):
            break

        if line.lower() == 'list':
            if not bank.slots:
                print("  (no facts stored)")
            for i, (_, _, prompt, answer, alter) in enumerate(bank.slots):
                print(f"  [{alter}] {prompt} → {answer}")
            print(f"  Total: {len(bank.slots)} facts")
            continue

        if line.lower() == 'save':
            n = bank.save(bank_path)
            sz = os.path.getsize(bank_path)
            print(f"  Saved {n} facts to {bank_path} ({sz:,} bytes)")
            continue

        if line.lower().startswith('load '):
            path = line[5:].strip()
            if os.path.exists(path):
                n = bank.load(path)
                print(f"  Loaded {n} facts from {path}")
            else:
                print(f"  File not found: {path}")
            continue

        if line.lower().startswith('teach '):
            parts = line[6:]
            if '|' not in parts:
                print("  Usage: teach <prompt> | <answer>")
                continue
            prompt_part, rest = parts.split('|', 1)
            prompt_part = prompt_part.strip()
            if '@' in rest:
                answer, alter = rest.rsplit('@', 1)
                answer = answer.strip()
                alter = alter.strip()
            else:
                answer = rest.strip()
                alter = "default"

            t0 = time.time()
            n_tokens, tokens_str = bank.teach(
                backbone, tokenizer, prompt_part, answer, alter, target_layer)
            elapsed = time.time() - t0
            print(f"  Stored [{alter}]: \"{prompt_part}\" → \"{answer}\" "
                  f"({n_tokens} tokens: {tokens_str}) [{elapsed:.1f}s]")
            continue

        if line.lower().startswith('ask '):
            prompt = line[4:].strip()
            t0 = time.time()
            with torch.no_grad():
                output, sim, gate, alter = bank.generate(
                    backbone, tokenizer, prompt, target_layer, max_tokens=60)
            elapsed = time.time() - t0

            if gate > 0:
                print(f"  [{alter} sim={sim:.3f}] {prompt}{output}")
            else:
                print(f"  [base] {prompt}{output}")
            print(f"  ({elapsed:.1f}s)")
            continue

        # Default: treat as ask
        t0 = time.time()
        with torch.no_grad():
            output, sim, gate, alter = bank.generate(
                backbone, tokenizer, line, target_layer, max_tokens=60)
        elapsed = time.time() - t0
        if gate > 0:
            print(f"  [{alter} sim={sim:.3f}] {line}{output}")
        else:
            print(f"  [base] {line}{output}")
        print(f"  ({elapsed:.1f}s)")


if __name__ == '__main__':
    main()
