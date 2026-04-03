#!/usr/bin/env python3
"""Plural system with KV memory — different alters know different things."""

import torch
import torch.nn.functional as F
from test_kv_memory import load_qwen, get_hidden, get_logits, KVMemory


class PluralKVSystem:
    """Multiple alters, each with their own KV memory bank."""

    def __init__(self, D, V, device='cuda'):
        self.D = D
        self.V = V
        self.device = device
        self.alters = {}  # name → KVMemory
        self._last_match = None

    def create_alter(self, name):
        self.alters[name] = KVMemory(self.D, self.V, self.device)
        return self.alters[name]

    def teach(self, alter_name, key_hidden, answer_ids, base_logits,
              tokenizer, prompt_part, answer_word, strength=50.0):
        """Teach a fact to a specific alter."""
        mem = self.alters[alter_name]
        V = self.V

        logit_biases = []
        for tid in answer_ids:
            bias = torch.zeros(V, device=self.device)
            bias[tid] = strength
            top = base_logits.argmax().item()
            if top != tid:
                bias[top] = -15.0
            logit_biases.append(bias)

        key = key_hidden.float()
        key = key / (key.norm() + 1e-8)
        mem.slots.append((key.to(self.device), logit_biases,
                           f"[{alter_name}] {prompt_part} → {answer_word}"))

    def recall(self, hidden_state, threshold=0.70):
        """Query ALL alters, return best match across all."""
        query = hidden_state[-1].float()
        query = query / (query.norm() + 1e-8)

        best_sim = -1
        best_bias = None
        best_alter = ""
        best_fact = ""

        for alter_name, mem in self.alters.items():
            for key, biases, fact in mem.slots:
                sim = F.cosine_similarity(
                    query.unsqueeze(0), key.unsqueeze(0)).item()
                if sim > best_sim:
                    best_sim = sim
                    best_bias = biases
                    best_alter = alter_name
                    best_fact = fact

        self._last_match = {
            'alter': best_alter,
            'sim': best_sim,
            'fact': best_fact,
        }

        if best_sim < threshold:
            return None, best_sim, 0.0, best_alter

        gate = min(1.0, (best_sim - threshold) / (1.0 - threshold))
        return best_bias, best_sim, gate, best_alter

    def get_state(self):
        return {
            name: {'n_facts': len(mem.slots),
                    'facts': [s[2] for s in mem.slots]}
            for name, mem in self.alters.items()
        }


def generate_plural(backbone, plural, tokenizer, prompt,
                     target_layer, device, max_tokens=30):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated = list(ids)

    prompt_input = torch.tensor([ids], device=device)
    with torch.no_grad():
        prompt_hidden = get_hidden(backbone, prompt_input, target_layer, device)
        query = prompt_hidden.reshape(-1, prompt_hidden.size(-1))
        bias_seq, sim, gate, alter = plural.recall(query)

    gen_step = 0
    for _ in range(max_tokens):
        input_ids = torch.tensor([generated], device=device)
        with torch.no_grad():
            base_logits = get_logits(backbone, input_ids, target_layer, device)
            if bias_seq is not None and gen_step < len(bias_seq):
                base_logits[0, -1] += (gate * bias_seq[gen_step]).to(base_logits.dtype)
        next_token = base_logits[0, -1].argmax().item()
        generated.append(next_token)
        gen_step += 1
        if next_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated[len(ids):]), sim, gate, alter


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone, tokenizer = load_qwen(device)
    D = backbone.config.hidden_size
    V = backbone.config.vocab_size
    target_layer = backbone.config.num_hidden_layers - 1

    plural = PluralKVSystem(D, V, device)

    # Create alters with different knowledge domains
    plural.create_alter("scientist")   # knows physics/chemistry
    plural.create_alter("spy")         # knows codes and secrets
    plural.create_alter("storyteller") # knows fiction

    print(f'{"="*60}')
    print(f'Plural KV Memory System — 3 alters')
    print(f'{"="*60}')

    # Teach different facts to different alters
    teachings = [
        # Spy alter
        ("spy", "The secret code for Aurora is", "Zyphrax"),
        ("spy", "The password for the vault is", "Obsidian"),
        ("spy", "The extraction point is at", "coordinates 47N 12E"),
        ("spy", "The mole in the agency is", "Agent Blackwood"),
        # Scientist alter
        ("scientist", "The chemical formula for table salt is", "NaCl"),
        ("scientist", "The speed of light is approximately", "299792458 meters per second"),
        ("scientist", "The atomic number of gold is", "79"),
        ("scientist", "The boiling point of nitrogen is", "minus 196 degrees"),
        # Storyteller alter
        ("storyteller", "The dragon's true name is", "Verdanthos"),
        ("storyteller", "The enchanted sword was forged in", "Mount Pyralith"),
        ("storyteller", "The lost kingdom lies beneath", "the Amber Sea"),
        ("storyteller", "The prophecy speaks of a child born under", "three crimson moons"),
    ]

    print(f'\nTeaching 12 facts to 3 alters:')
    for alter_name, prompt_part, answer_word in teachings:
        prompt_ids = tokenizer.encode(prompt_part, add_special_tokens=False)
        prompt_input = torch.tensor([prompt_ids], device=device)
        with torch.no_grad():
            prompt_hidden = get_hidden(backbone, prompt_input, target_layer, device)
            prompt_logits = get_logits(backbone, prompt_input, target_layer, device)

        key = prompt_hidden.reshape(-1, D)[-1]
        answer_ids = tokenizer.encode(answer_word, add_special_tokens=False)
        last_logits = prompt_logits[0, -1]

        plural.teach(alter_name, key, answer_ids, last_logits,
                      tokenizer, prompt_part, answer_word)
        print(f'  [{alter_name:12s}] "{prompt_part}" → "{answer_word}"')

    # Test recall — each alter's facts
    print(f'\n{"="*60}')
    print(f'Recall test:')
    print(f'{"="*60}')

    test_cases = [
        # Spy facts
        ("The secret code for Aurora is", "Zyphrax", "spy"),
        ("The password for the vault is", "Obsidian", "spy"),
        ("The extraction point is at", "coordinates", "spy"),
        ("The mole in the agency is", "Agent", "spy"),
        # Scientist facts
        ("The chemical formula for table salt is", "NaCl", "scientist"),
        ("The speed of light is approximately", "299", "scientist"),
        ("The atomic number of gold is", "79", "scientist"),
        ("The boiling point of nitrogen is", "minus", "scientist"),
        # Storyteller facts
        ("The dragon's true name is", "Verdanthos", "storyteller"),
        ("The enchanted sword was forged in", "Mount", "storyteller"),
        ("The lost kingdom lies beneath", "Amber", "storyteller"),
        ("The prophecy speaks of a child born under", "three", "storyteller"),
        # Non-taught (should not activate)
        ("The capital of France is", "Paris", None),
        ("Machine learning is a", None, None),
        # Cross-alter test: similar prompts, different expected alter
        ("The secret code for Beta is", None, None),
    ]

    passed = 0
    failed = 0
    false_pos = 0

    for prompt, expected, expected_alter in test_cases:
        with torch.no_grad():
            out, sim, gate, alter = generate_plural(
                backbone, plural, tokenizer, prompt,
                target_layer, device, 20)

        if expected:
            recall = expected.lower() in out.lower()
            correct_alter = (alter == expected_alter) if expected_alter else True
            if recall:
                status = "PASS"
                passed += 1
            else:
                status = "FAIL"
                failed += 1
        else:
            if gate > 0:
                status = "FP  "  # false positive
                false_pos += 1
            else:
                status = "    "

        alter_str = f"[{alter:12s}]" if gate > 0 else f"[{'—':12s}]"
        print(f'  {status} {alter_str} sim={sim:.3f} gate={gate:.2f} '
              f'"{prompt[:35]}" → "{out[:40]}"')

    print(f'\n{"="*60}')
    print(f'Results: {passed} passed, {failed} failed, {false_pos} false positives')
    print(f'Recall accuracy: {passed}/{passed+failed} = {passed/(passed+failed):.0%}')

    print(f'\nSystem state:')
    for name, info in plural.get_state().items():
        print(f'  {name}: {info["n_facts"]} facts')

    print(f'\nDone.')


if __name__ == '__main__':
    main()
