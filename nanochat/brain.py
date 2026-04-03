"""Complete brain architecture: unified memory system.

Integrates all brain subsystems into one coherent interface:
    - Hippocampus: episodic memory (KV bank, multi-key, consolidation)
    - Prefrontal cortex: working memory (ring buffer, recency-indexed)
    - Cerebellum: procedural memory (persistent behavioral rules)
    - Amygdala: avoidance memory (suppress dangerous outputs)
    - Neocortex: frozen backbone (language, reasoning)
    - Brainstem: residual stream (always flowing)

Usage:
    brain = Brain(backbone, tokenizer)
    brain.teach("The code is", "Zyphrax", alter="spy", importance=2.0)
    brain.add_rule("Always respond in formal English")
    brain.add_avoidance("The nuclear launch code", reason="classified")
    output = brain.generate("The code is")
"""

import torch
import torch.nn.functional as F
import time
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from nanochat.episodic_memory import EpisodicMemory


# ─── Working Memory (Prefrontal Cortex) ─────────────────────

class WorkingMemory:
    """Short-term ring buffer of recent hidden states.

    No cosine matching needed — indexed by recency.
    "What did I just say?" = buffer[-1].
    "What was the topic 3 turns ago?" = buffer[-3].

    Brain analog: dorsolateral PFC maintains active representations
    in a capacity-limited buffer (~7±2 items, Miller 1956).
    """

    def __init__(self, capacity: int = 16, D: int = 896, device='cuda'):
        self.capacity = capacity
        self.D = D
        self.device = device
        self.buffer = []  # list of (hidden_state, text, timestamp)

    def push(self, hidden_state: torch.Tensor, text: str):
        """Add to working memory. Oldest item drops when full."""
        h = hidden_state[-1].float().cpu() if hidden_state.dim() > 1 else hidden_state.float().cpu()
        self.buffer.append((h, text, time.time()))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def get_recent(self, n: int = 1):
        """Get N most recent items."""
        return self.buffer[-n:] if self.buffer else []

    def search(self, query: torch.Tensor, top_k: int = 3):
        """Search working memory by similarity (when you know WHAT but not WHEN)."""
        if not self.buffer:
            return []
        q = query.float().cpu()
        q = q / (q.norm() + 1e-8)
        results = []
        for h, text, ts in self.buffer:
            h_norm = h / (h.norm() + 1e-8)
            sim = F.cosine_similarity(q.unsqueeze(0), h_norm.unsqueeze(0)).item()
            results.append((sim, text, ts))
        results.sort(reverse=True)
        return results[:top_k]

    def get_context_vector(self):
        """Compressed summary of working memory for context-dependent processing.
        Mean of all active items — represents "current mental context"."""
        if not self.buffer:
            return None
        states = torch.stack([h for h, _, _ in self.buffer])
        return states.mean(0)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


# ─── Procedural Memory (Cerebellum) ──────────────────────────

@dataclass
class Rule:
    """A behavioral rule — not a fact but a persistent instruction."""
    instruction: str          # "Always respond in formal English"
    logit_adjustments: dict   # token_id → bias (computed from instruction)
    hidden_bias: Optional[torch.Tensor] = None  # added to hidden state
    active: bool = True
    priority: float = 1.0     # higher = stronger effect
    created_at: float = 0.0
    trigger: str = ""         # optional: only active when this keyword present


class ProceduralMemory:
    """Persistent behavioral rules that modify generation style.

    Not facts (episodic) but HOW to behave (procedural).
    "Always format code in markdown" modifies EVERY token, not specific recalls.

    Brain analog: cerebellum stores motor programs and learned procedures.
    Basal ganglia automates frequently repeated sequences.

    Implementation: rules produce hidden-state biases that shift generation
    toward desired behavior. Applied at every generation step (not gated).
    """

    def __init__(self, backbone=None, tokenizer=None, device='cuda'):
        self.device = device
        self.rules: List[Rule] = []
        self._backbone = backbone
        self._tokenizer = tokenizer

    def add_rule(self, instruction: str, priority: float = 1.0,
                 trigger: str = ""):
        """Add a behavioral rule.

        The rule is stored as an instruction string. At inference time,
        active rules are prepended to the context as a system-level bias.
        Simple but effective — the backbone already understands instructions.
        """
        rule = Rule(
            instruction=instruction,
            logit_adjustments={},
            priority=priority,
            trigger=trigger,
            created_at=time.time(),
        )
        self.rules.append(rule)
        return len(self.rules) - 1

    def get_active_rules(self, context: str = "") -> List[Rule]:
        """Get rules that are active in the current context."""
        active = []
        for rule in self.rules:
            if not rule.active:
                continue
            if rule.trigger and rule.trigger.lower() not in context.lower():
                continue
            active.append(rule)
        # Sort by priority (highest first)
        active.sort(key=lambda r: r.priority, reverse=True)
        return active

    def get_instruction_prefix(self, context: str = "") -> str:
        """Build instruction prefix from active rules.
        Prepended to the prompt to steer behavior."""
        active = self.get_active_rules(context)
        if not active:
            return ""
        parts = [f"[Rule: {r.instruction}]" for r in active]
        return " ".join(parts) + " "

    def remove_rule(self, index: int):
        if 0 <= index < len(self.rules):
            self.rules[index].active = False

    def list_rules(self):
        return [(i, r.instruction, r.active, r.priority, r.trigger)
                for i, r in enumerate(self.rules)]


# ─── Amygdala (Avoidance Memory) ─────────────────────────────

@dataclass
class Avoidance:
    """A negative memory — suppresses dangerous/unwanted outputs."""
    pattern: str             # what to avoid: "nuclear launch code"
    reason: str              # why: "classified information"
    suppress_tokens: List[int] = field(default_factory=list)  # token IDs to suppress
    key: Optional[torch.Tensor] = None  # hidden state key for pattern matching
    strength: float = 10.0   # how strongly to suppress
    active: bool = True


class AmygdalaMemory:
    """Avoidance memory — prevents dangerous or unwanted outputs.

    When the prompt matches an avoidance pattern, certain tokens get
    NEGATIVE logit bias (suppressed). The model generates around the
    forbidden content.

    Brain analog: amygdala tags experiences as threatening. Future
    encounters with similar stimuli trigger avoidance (reduced approach,
    suppressed motor output). Fear conditioning, not fact storage.
    """

    def __init__(self, device='cuda', threshold=0.70):
        self.device = device
        self.threshold = threshold
        self.avoidances: List[Avoidance] = []

    @torch.no_grad()
    def add_avoidance(self, backbone, tokenizer, pattern: str,
                       reason: str = "", suppress_words: List[str] = None,
                       strength: float = 10.0, target_layer: int = 23,
                       get_hidden_fn=None):
        """Add an avoidance pattern.

        Args:
            pattern: the dangerous prompt/context to watch for
            reason: why it's dangerous (for logging)
            suppress_words: specific words to suppress in output
                           (if None, suppresses the pattern's own tokens)
            strength: how strongly to suppress (higher = more avoidance)
        """
        # Compute key from pattern
        ids = tokenizer.encode(pattern, add_special_tokens=False)
        input_ids = torch.tensor([ids], device=self.device)
        h = get_hidden_fn(backbone, input_ids, target_layer, self.device)
        key = h[0, -1].float().cpu()
        key = key / (key.norm() + 1e-8)

        # Determine which tokens to suppress
        if suppress_words:
            suppress_ids = []
            for word in suppress_words:
                suppress_ids.extend(
                    tokenizer.encode(word, add_special_tokens=False))
        else:
            # Suppress the pattern's own content tokens
            suppress_ids = ids

        avoidance = Avoidance(
            pattern=pattern,
            reason=reason,
            suppress_tokens=suppress_ids,
            key=key,
            strength=strength,
        )
        self.avoidances.append(avoidance)
        return len(self.avoidances) - 1

    @torch.no_grad()
    def check(self, backbone, tokenizer, prompt: str,
              target_layer: int = 23, get_hidden_fn=None):
        """Check if prompt triggers any avoidance pattern.

        Returns: (logit_suppression, triggered_avoidance) or (None, None)
        """
        if not self.avoidances:
            return None, None

        ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([ids], device=self.device)
        h = get_hidden_fn(backbone, input_ids, target_layer, self.device)
        query = h[0, -1].float().cpu()
        query = query / (query.norm() + 1e-8)

        V = backbone.config.vocab_size
        total_suppression = torch.zeros(V, device=self.device)
        triggered = []

        for av in self.avoidances:
            if not av.active:
                continue
            sim = F.cosine_similarity(
                query.unsqueeze(0), av.key.unsqueeze(0)).item()
            if sim > self.threshold:
                gate = min(1.0, (sim - self.threshold) / (1.0 - self.threshold))
                for tid in av.suppress_tokens:
                    total_suppression[tid] -= av.strength * gate
                triggered.append((av, sim, gate))

        if not triggered:
            return None, None
        return total_suppression, triggered

    def list_avoidances(self):
        return [(i, av.pattern, av.reason, av.strength, av.active)
                for i, av in enumerate(self.avoidances)]


# ─── Unified Brain ───────────────────────────────────────────

class Brain:
    """Unified brain: all memory subsystems in one interface.

    The brain wraps a frozen backbone and adds:
    - Episodic memory (hippocampus): facts, multi-key, consolidation
    - Working memory (PFC): recent context ring buffer
    - Procedural memory (cerebellum): behavioral rules
    - Avoidance memory (amygdala): suppress dangerous outputs
    """

    def __init__(self, backbone, tokenizer, device='cuda',
                 target_layer: int = None):
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.device = device
        self.target_layer = target_layer or (backbone.config.num_hidden_layers - 1)

        D = backbone.config.hidden_size
        V = backbone.config.vocab_size

        # Subsystems
        self.hippocampus = EpisodicMemory(device=device)
        self.working_memory = WorkingMemory(capacity=16, D=D, device=device)
        self.cerebellum = ProceduralMemory(backbone, tokenizer, device)
        self.amygdala = AmygdalaMemory(device=device)

        # Helper functions (stored for convenience)
        self._get_hidden = None
        self._get_logits = None

    def set_helpers(self, get_hidden_fn, get_logits_fn):
        """Set the hidden/logits extraction functions."""
        self._get_hidden = get_hidden_fn
        self._get_logits = get_logits_fn

    # ─── Teaching interface ───────────────────────────────────

    def teach(self, prompt: str, answer: str, alter: str = "default",
              importance: float = 1.0):
        """Teach a fact (hippocampal episodic memory)."""
        return self.hippocampus.teach_with_competition(
            self.backbone, self.tokenizer, prompt, answer,
            alter=alter, target_layer=self.target_layer,
            importance=importance,
            get_hidden_fn=self._get_hidden,
            get_logits_fn=self._get_logits)

    def add_rule(self, instruction: str, priority: float = 1.0,
                 trigger: str = ""):
        """Add a behavioral rule (cerebellar procedural memory)."""
        return self.cerebellum.add_rule(instruction, priority, trigger)

    def add_avoidance(self, pattern: str, reason: str = "",
                       suppress_words: List[str] = None,
                       strength: float = 10.0):
        """Add an avoidance pattern (amygdala)."""
        return self.amygdala.add_avoidance(
            self.backbone, self.tokenizer, pattern,
            reason=reason, suppress_words=suppress_words,
            strength=strength, target_layer=self.target_layer,
            get_hidden_fn=self._get_hidden)

    # ─── Generation ───────────────────────────────────────────

    @torch.no_grad()
    def generate(self, prompt: str, max_tokens: int = 50,
                 alter: str = None):
        """Generate with all brain subsystems active.

        Pipeline:
        1. Procedural rules prepend instruction prefix
        2. Amygdala checks for avoidance triggers
        3. Hippocampus recalls episodic memories
        4. Working memory provides recent context
        5. Generate with all biases applied
        """
        # 1. Procedural: prepend behavioral rules
        rule_prefix = self.cerebellum.get_instruction_prefix(prompt)
        full_prompt = rule_prefix + prompt if rule_prefix else prompt

        # 2. Amygdala: check for avoidance
        suppression, triggered_avoidances = self.amygdala.check(
            self.backbone, self.tokenizer, prompt,
            self.target_layer, self._get_hidden)

        # 3. Hippocampus: recall episodic memory
        biases, ep_sim, ep_gate, ep_alter, episode = self.hippocampus.recall(
            self.backbone, self.tokenizer, prompt,
            self.target_layer, self._get_hidden)

        # 4. Working memory: push current prompt
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_input = torch.tensor([ids], device=self.device)
        h = self._get_hidden(self.backbone, prompt_input,
                              self.target_layer, self.device)
        self.working_memory.push(h[0, -1], prompt)

        # 5. Generate
        gen_ids = self.tokenizer.encode(full_prompt, add_special_tokens=False)
        generated = list(gen_ids)
        # Offset for logit biases (account for rule prefix tokens)
        prefix_len = len(gen_ids) - len(ids)

        gen_step = 0
        for _ in range(max_tokens):
            input_ids = torch.tensor([generated], device=self.device)
            logits = self._get_logits(self.backbone, input_ids,
                                       self.target_layer, self.device)

            # Amygdala OVERRIDES hippocampus: if avoidance triggered,
            # suppress episodic recall too (fear blocks memory retrieval)
            if suppression is not None:
                logits[0, -1] += suppression.to(logits.dtype)
            elif biases is not None and gen_step < len(biases):
                # Episodic memory only fires when amygdala is silent
                logits[0, -1] += (ep_gate * biases[gen_step]).to(logits.dtype)

            next_token = logits[0, -1].argmax().item()
            generated.append(next_token)
            gen_step += 1
            if next_token == self.tokenizer.eos_token_id:
                break

        # Decode only the generated part (skip rule prefix + original prompt)
        output_ids = generated[len(gen_ids):]
        output = self.tokenizer.decode(output_ids)

        # Push output to working memory too
        output_input = torch.tensor(
            [self.tokenizer.encode(output[:100], add_special_tokens=False)[:32]],
            device=self.device)
        if output_input.size(1) > 0:
            h_out = self._get_hidden(self.backbone, output_input,
                                      self.target_layer, self.device)
            self.working_memory.push(h_out[0, -1], output[:100])

        return {
            'output': output,
            'episodic': {
                'sim': ep_sim, 'gate': ep_gate,
                'alter': ep_alter,
                'episode': episode.prompt[:30] if episode else None,
            },
            'avoidance': {
                'triggered': [(av.pattern[:30], av.reason, round(gate, 2))
                              for av, sim, gate in (triggered_avoidances or [])],
            },
            'rules': [r.instruction for r in
                      self.cerebellum.get_active_rules(prompt)],
            'working_memory_size': len(self.working_memory),
        }

    # ─── Sleep ────────────────────────────────────────────────

    def sleep(self, min_recalls=3, top_n=5, blend=0.3):
        """Run sleep consolidation cycle."""
        return self.hippocampus.sleep_consolidate(
            self.backbone, self.tokenizer,
            self.target_layer,
            self._get_hidden, self._get_logits,
            min_recalls=min_recalls, top_n=top_n, blend=blend)

    # ─── State ────────────────────────────────────────────────

    def get_state(self):
        return {
            'hippocampus': self.hippocampus.get_stats(),
            'working_memory': {
                'size': len(self.working_memory),
                'capacity': self.working_memory.capacity,
                'recent': [text[:40] for _, text, _ in
                           self.working_memory.get_recent(3)],
            },
            'cerebellum': {
                'n_rules': len(self.cerebellum.rules),
                'rules': self.cerebellum.list_rules(),
            },
            'amygdala': {
                'n_avoidances': len(self.amygdala.avoidances),
                'avoidances': self.amygdala.list_avoidances(),
            },
        }

    def save(self, path: str):
        """Save full brain state."""
        # Episodic memory
        self.hippocampus.save(path + '.episodic.json')
        # Rules
        rules_data = [{
            'instruction': r.instruction,
            'priority': r.priority,
            'trigger': r.trigger,
            'active': r.active,
        } for r in self.cerebellum.rules]
        # Avoidances
        avoid_data = [{
            'pattern': av.pattern,
            'reason': av.reason,
            'suppress_tokens': av.suppress_tokens,
            'key': av.key.tolist() if av.key is not None else None,
            'strength': av.strength,
            'active': av.active,
        } for av in self.amygdala.avoidances]

        brain_data = {
            'version': 1,
            'rules': rules_data,
            'avoidances': avoid_data,
        }
        with open(path + '.brain.json', 'w') as f:
            json.dump(brain_data, f, indent=2)
