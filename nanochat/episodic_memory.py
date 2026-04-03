"""Episodic memory with consolidation strength and multi-key retrieval.

Each memory is an EPISODE — a cloud of content-word keys pointing to
the same logit biases. Any fragment of the original experience retrieves
the full memory. Strength modulated by surprise × attention × novelty.

Brain mapping:
    content keys          = hippocampal pattern fragments (CA3)
    cosine matching       = pattern completion (CA3 recurrent)
    consolidation strength = serotonin/dopamine modulation at encoding
    multi-key cloud       = multiple retrieval paths (associative links)
    per-alter banks       = plural self-states
"""

import torch
import torch.nn.functional as F
import json
import time
import math
from typing import List, Dict, Optional, Tuple


# Common function words to skip when creating content keys
SKIP_WORDS = {
    'The', 'the', ' the', ' The', ' a', ' A', ' an', ' An',
    ' is', ' are', ' was', ' were', ' be', ' been', ' being',
    ' for', ' of', ' in', ' on', ' at', ' to', ' and', ' or',
    ' it', ' It', ' that', ' this', ' with', ' from', ' by',
    ' not', ' no', ' but', ' if', ' so', ' as', ' has', ' have',
    ' had', ' do', ' does', ' did', ' will', ' would', ' could',
    ' should', ' can', ' may', ' might',
    '.', ',', '!', '?', ':', ';', '-', "'", '"',
    ' ', '  ', '\n',
}


class Episode:
    """One episodic memory — a cloud of keys + answer biases + strength."""

    def __init__(self, prompt: str, answer: str, alter: str,
                 keys: List[Tuple[torch.Tensor, str, int]],
                 logit_biases: List[torch.Tensor],
                 strength: float = 1.0,
                 metadata: dict = None):
        self.prompt = prompt
        self.answer = answer
        self.alter = alter
        self.keys = keys          # list of (key_vector, token_text, position)
        self.logit_biases = logit_biases  # list of per-answer-token biases
        self.strength = strength  # consolidation strength (0-1 range typically)
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.recall_count = 0     # strengthens with use (reconsolidation)

    def effective_strength(self):
        """Strength increases with recall (reconsolidation) and decays with time."""
        age_hours = (time.time() - self.created_at) / 3600
        # Slow decay: half-life of 720 hours (30 days)
        decay = 0.5 ** (age_hours / 720)
        # Reconsolidation boost: each recall adds 10%
        reconsol = 1.0 + 0.1 * self.recall_count
        return self.strength * decay * reconsol


class EpisodicMemory:
    """Multi-key episodic memory with consolidation strength.

    Each fact stores keys at every content-word position.
    Any fragment retrieves the memory. Strength modulates recall confidence.
    """

    def __init__(self, device='cuda', threshold=0.70, base_strength=50.0):
        self.device = device
        self.threshold = threshold
        self.base_strength = base_strength
        self.episodes: Dict[str, List[Episode]] = {}  # alter → episodes
        self._skip_token_ids = set()
        self._tokenizer = None

    def _init_skip_tokens(self, tokenizer):
        """Build set of function-word token IDs to skip."""
        if self._tokenizer is tokenizer:
            return
        self._tokenizer = tokenizer
        self._skip_token_ids = set()
        for word in SKIP_WORDS:
            ids = tokenizer.encode(word, add_special_tokens=False)
            self._skip_token_ids.update(ids)

    def _compute_consolidation_strength(self, hidden_states, base_logits,
                                          target_ids):
        """Compute encoding strength from surprise × attention × novelty.

        Maps to serotonin/dopamine/norepinephrine modulation:
        - Surprise (dopamine): prediction error on target tokens
        - Attention energy: norm of hidden state (how much the model activated)
        - Novelty: distance from mean hidden state (how unusual this input is)

        Returns: float in [0.5, 2.0] range (0.5=boring, 2.0=highly significant)
        """
        # Surprise: cross-entropy on the answer tokens
        # High CE = model didn't predict this = surprising = remember harder
        if base_logits is not None and target_ids is not None:
            ce = F.cross_entropy(
                base_logits.reshape(-1, base_logits.size(-1)),
                target_ids.reshape(-1),
                reduction='mean').item()
            # Normalize: typical CE is ~3-4 for Qwen 0.5B
            surprise = min(2.0, ce / 3.0)
        else:
            surprise = 1.0

        # Attention energy: how strongly did the model activate?
        energy = hidden_states.float().norm(dim=-1).mean().item()
        # Normalize: typical hidden norm is ~20-40
        energy_factor = min(1.5, energy / 30.0)

        # Novelty: variance of hidden state (high variance = unusual input)
        novelty = hidden_states.float().var(dim=-1).mean().item()
        # Normalize
        novelty_factor = min(1.5, 1.0 + novelty / 10.0)

        # Combined: clamp to [0.5, 2.0]
        raw = surprise * energy_factor * novelty_factor
        return max(0.5, min(2.0, raw))

    @torch.no_grad()
    def teach(self, backbone, tokenizer, prompt: str, answer: str,
              alter: str = "default", target_layer: int = 23,
              get_hidden_fn=None, get_logits_fn=None):
        """Store an episodic memory with multi-key cloud and consolidation strength.

        Args:
            backbone: frozen language model
            tokenizer: tokenizer
            prompt: the prompt/context part (cue)
            answer: the answer/fact to recall
            alter: which alter this memory belongs to
            target_layer: which backbone layer to extract hidden states from
            get_hidden_fn: function(backbone, input_ids, layer, device) → hidden
            get_logits_fn: function(backbone, input_ids, layer, device) → logits
        """
        self._init_skip_tokens(tokenizer)
        V = backbone.config.vocab_size
        D = backbone.config.hidden_size

        # Encode full teaching text
        full_text = prompt + " " + answer
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        full_input = torch.tensor([full_ids], device=self.device)

        # Get hidden states for full text
        hidden = get_hidden_fn(backbone, full_input, target_layer, self.device)

        # Get logits for computing surprise
        logits = get_logits_fn(backbone, full_input, target_layer, self.device)

        # Answer token IDs
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        # Compute consolidation strength
        # Use answer portion of logits for surprise measurement
        prompt_len = len(prompt_ids)
        if prompt_len < len(full_ids) - 1:
            answer_logits = logits[:, prompt_len-1:-1]
            answer_targets = full_input[:, prompt_len:]
            strength_mod = self._compute_consolidation_strength(
                hidden[:, prompt_len:], answer_logits, answer_targets)
        else:
            strength_mod = 1.0

        # Build content-word keys from ENTIRE text (not just prompt)
        keys = []
        for pos in range(len(full_ids)):
            if full_ids[pos] in self._skip_token_ids:
                continue
            k = hidden[0, pos].float().cpu()
            k = k / (k.norm() + 1e-8)
            tok = tokenizer.decode([full_ids[pos]])
            keys.append((k, tok.strip(), pos))

        # Also store key at prompt's last token (exact-match retrieval path)
        prompt_input = torch.tensor([prompt_ids], device=self.device)
        prompt_hidden = get_hidden_fn(backbone, prompt_input, target_layer, self.device)
        prompt_key = prompt_hidden[0, -1].float().cpu()
        prompt_key = prompt_key / (prompt_key.norm() + 1e-8)
        keys.append((prompt_key, "[PROMPT_END]", -1))

        # Build answer logit biases
        prompt_logits = get_logits_fn(backbone, prompt_input, target_layer, self.device)
        last_logits = prompt_logits[0, -1]

        logit_biases = []
        effective_strength = self.base_strength * strength_mod
        for tid in answer_ids:
            bias = torch.zeros(V, device=self.device)
            bias[tid] = effective_strength
            top = last_logits.argmax().item()
            if top != tid:
                bias[top] = -effective_strength * 0.3
            logit_biases.append(bias)

        # Create episode
        episode = Episode(
            prompt=prompt,
            answer=answer,
            alter=alter,
            keys=keys,
            logit_biases=logit_biases,
            strength=strength_mod,
            metadata={
                'n_keys': len(keys),
                'n_answer_tokens': len(answer_ids),
                'surprise': strength_mod,
                'answer_tokens': [tokenizer.decode([t]) for t in answer_ids],
            }
        )

        # Store in alter's bank
        if alter not in self.episodes:
            self.episodes[alter] = []
        self.episodes[alter].append(episode)

        return {
            'n_keys': len(keys),
            'strength': strength_mod,
            'answer_tokens': len(answer_ids),
            'content_words': [k[1] for k in keys[:-1]],  # exclude PROMPT_END
        }

    @torch.no_grad()
    def recall(self, backbone, tokenizer, cue: str,
               target_layer: int = 23, get_hidden_fn=None):
        """Recall by matching ANY fragment of the cue against stored keys.

        Returns: (logit_biases, sim, gate, alter, episode) or (None, sim, 0, alter, None)
        """
        self._init_skip_tokens(tokenizer)

        cue_ids = tokenizer.encode(cue, add_special_tokens=False)
        cue_input = torch.tensor([cue_ids], device=self.device)
        cue_hidden = get_hidden_fn(backbone, cue_input, target_layer, self.device)

        # Build query vectors from content tokens in cue
        queries = []
        for pos in range(len(cue_ids)):
            q = cue_hidden[0, pos].float().cpu()
            q = q / (q.norm() + 1e-8)
            queries.append(q)
        # Also use last token as query (exact-match path)
        q_last = cue_hidden[0, -1].float().cpu()
        q_last = q_last / (q_last.norm() + 1e-8)
        queries.append(q_last)

        # Search all episodes across all alters
        best_sim = -1
        best_episode = None
        best_alter = ""

        for alter_name, eps in self.episodes.items():
            for ep in eps:
                for query in queries:
                    for key, tok, pos in ep.keys:
                        sim = F.cosine_similarity(
                            query.unsqueeze(0), key.unsqueeze(0)).item()
                        # Weight by episode strength
                        weighted_sim = sim * ep.effective_strength() / max(ep.strength, 0.1)
                        if weighted_sim > best_sim:
                            best_sim = weighted_sim
                            best_episode = ep
                            best_alter = alter_name

        if best_sim < self.threshold or best_episode is None:
            return None, best_sim, 0.0, best_alter, None

        gate = min(1.0, (best_sim - self.threshold) / (1.0 - self.threshold))

        # Reconsolidation: recalling strengthens the memory
        best_episode.recall_count += 1

        return (best_episode.logit_biases, best_sim, gate,
                best_alter, best_episode)

    def generate(self, backbone, tokenizer, prompt: str,
                 target_layer: int = 23, max_tokens: int = 50,
                 get_hidden_fn=None, get_logits_fn=None):
        """Generate with episodic memory recall."""
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        generated = list(ids)

        biases, sim, gate, alter, episode = self.recall(
            backbone, tokenizer, prompt, target_layer, get_hidden_fn)

        gen_step = 0
        for _ in range(max_tokens):
            input_ids = torch.tensor([generated], device=self.device)
            with torch.no_grad():
                logits = get_logits_fn(backbone, input_ids, target_layer, self.device)
                if biases is not None and gen_step < len(biases):
                    logits[0, -1] += (gate * biases[gen_step]).to(logits.dtype)
            next_token = logits[0, -1].argmax().item()
            generated.append(next_token)
            gen_step += 1
            if next_token == tokenizer.eos_token_id:
                break

        output = tokenizer.decode(generated[len(ids):])
        return output, sim, gate, alter, episode

    def get_stats(self):
        stats = {}
        for alter, eps in self.episodes.items():
            total_keys = sum(len(ep.keys) for ep in eps)
            avg_strength = sum(ep.strength for ep in eps) / len(eps) if eps else 0
            stats[alter] = {
                'n_episodes': len(eps),
                'total_keys': total_keys,
                'avg_strength': round(avg_strength, 3),
                'facts': [(ep.prompt[:30], ep.answer, round(ep.strength, 2),
                           ep.recall_count) for ep in eps],
            }
        return stats

    def save(self, path: str):
        data = {
            'version': 2,
            'type': 'episodic',
            'backbone': 'Qwen/Qwen2.5-0.5B',
            'threshold': self.threshold,
            'base_strength': self.base_strength,
            'alters': {}
        }
        for alter, eps in self.episodes.items():
            data['alters'][alter] = []
            for ep in eps:
                ep_data = {
                    'prompt': ep.prompt,
                    'answer': ep.answer,
                    'strength': ep.strength,
                    'recall_count': ep.recall_count,
                    'created_at': ep.created_at,
                    'metadata': ep.metadata,
                    'keys': [(k.tolist(), tok, pos) for k, tok, pos in ep.keys],
                    'logit_biases': [
                        (b.nonzero().squeeze(-1).tolist(),
                         b[b.nonzero().squeeze(-1)].tolist())
                        for b in ep.logit_biases
                    ],
                }
                data['alters'][alter].append(ep_data)
        with open(path, 'w') as f:
            json.dump(data, f)
        return sum(len(eps) for eps in self.episodes.values())
