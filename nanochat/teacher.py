"""
Teacher models for knowledge distillation.

Two modes:
1. Local teacher: a frozen GPT model (same architecture, same vocab). Produces
   logits directly - full distribution-level KL distillation.
2. Ollama teacher: an external model served by ollama. Produces text completions
   that we re-encode in our vocab as hard targets - sequence-level distillation.

The local teacher is better (soft targets preserve uncertainty) but costs VRAM.
The ollama teacher costs nothing on the training GPU and lets you distill from
models much larger than you could fit locally.
"""

import json
import torch
import torch.nn.functional as F
from urllib.request import urlopen, Request
from urllib.error import URLError


class LocalTeacher:
    """Frozen local model as teacher. Same architecture, same vocab.

    Wraps a GPT model as a callable that returns logits.
    Costs one forward pass + model VRAM per training step.

    Usage:
        teacher_model = load_and_freeze(...)
        model.set_teacher(LocalTeacher(teacher_model), distill_weight=0.5)
    """

    def __init__(self, model):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

    @torch.no_grad()
    def __call__(self, idx):
        """idx: (B, T) token ids. Returns: (B, T, V) logits."""
        return self.model.forward(idx, targets=None)


class OllamaTeacher:
    """External model served by ollama as teacher. Different vocab, different architecture.

    Since vocabs differ, we can't do logit-level distillation. Instead we do
    sequence-level distillation: the teacher generates a completion for the
    context, we re-encode it in our tokenizer, and return one-hot logits
    (hard targets). The student learns to match the teacher's text output.

    This is weaker than soft-target distillation but lets you distill from
    any model ollama can serve - llama 70b, qwen, deepseek, whatever.

    The teacher runs async: completions are cached per batch to avoid blocking
    the training loop on every step. Cache is keyed by the first 64 tokens of
    each sequence.

    Usage:
        teacher = OllamaTeacher(
            url="http://localhost:11434",
            model="llama3.2",
            tokenizer=our_tokenizer,
            vocab_size=our_model.config.vocab_size,
        )
        model.set_teacher(teacher, distill_weight=0.3)

    CLI:
        # terminal 1: serve teacher
        ollama serve
        ollama pull llama3.2

        # terminal 2: train with distillation
        NANOCHAT_NO_COMPILE=1 python -m scripts.base_train \\
            --depth=12 --use-ctm --bptt-chunks=4 \\
            --distill-from=ollama:llama3.2 \\
            --distill-weight=0.3
    """

    def __init__(self, url, model, tokenizer, vocab_size, context_tokens=128, max_tokens=64, temperature=0.0):
        self.url = url.rstrip('/')
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.context_tokens = context_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._cache = {}

    def _generate(self, text):
        """Call ollama API to generate a completion."""
        payload = json.dumps({
            "model": self.model,
            "prompt": text,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }).encode()

        req = Request(
            f"{self.url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                return result.get("response", "")
        except (URLError, TimeoutError, json.JSONDecodeError):
            return ""

    def _get_teacher_tokens(self, token_ids):
        """Get teacher completion for a token sequence, cached."""
        # Cache key: first N tokens (avoids re-querying same context)
        key = tuple(token_ids[:self.context_tokens])
        if key in self._cache:
            return self._cache[key]

        # Decode our tokens to text, send to ollama, re-encode response
        context_text = self.tokenizer.decode(list(token_ids[:self.context_tokens]))
        teacher_text = self._generate(context_text)

        if teacher_text:
            teacher_tokens = self.tokenizer.encode(teacher_text)
        else:
            teacher_tokens = []

        self._cache[key] = teacher_tokens

        # Evict old cache entries if too large
        if len(self._cache) > 1000:
            # Remove oldest half
            keys = list(self._cache.keys())
            for k in keys[:500]:
                del self._cache[k]

        return teacher_tokens

    @torch.no_grad()
    def __call__(self, idx):
        """idx: (B, T) token ids. Returns: (B, T, V) logits.

        For positions where the teacher has a prediction, returns high-confidence
        logits for the teacher's token. For positions without teacher coverage,
        returns uniform logits (no distillation signal).
        """
        B, T = idx.shape
        V = self.vocab_size
        device = idx.device

        # Uniform logits = no signal (KL with uniform is constant, no gradient)
        logits = torch.zeros(B, T, V, device=device)

        for b in range(B):
            token_ids = idx[b].cpu().tolist()
            teacher_tokens = self._get_teacher_tokens(token_ids)

            if not teacher_tokens:
                continue

            # Align: teacher completion starts after context_tokens position
            # Place teacher tokens as targets starting from context_tokens offset
            start = min(self.context_tokens, T - 1)
            for i, tok in enumerate(teacher_tokens):
                pos = start + i
                if pos >= T:
                    break
                if 0 <= tok < V:
                    # High logit for teacher's token, creates peaked distribution
                    logits[b, pos, tok] = 10.0

        return logits


def create_teacher(spec, tokenizer, vocab_size, device=None):
    """Create a teacher from a spec string.

    Spec formats:
        "ollama:model_name"                  - ollama at localhost:11434
        "ollama:model_name@host:port"        - ollama at custom address
        "/path/to/checkpoint"                - local frozen model

    Args:
        spec: teacher specification string
        tokenizer: our tokenizer (for re-encoding ollama output)
        vocab_size: our model's vocab size
        device: device for local teacher model

    Returns: callable teacher (token_ids -> logits)
    """
    if spec.startswith("ollama:"):
        parts = spec[7:]  # strip "ollama:"
        if "@" in parts:
            model_name, host = parts.split("@", 1)
            url = f"http://{host}"
        else:
            model_name = parts
            url = "http://localhost:11434"
        return OllamaTeacher(url=url, model=model_name, tokenizer=tokenizer, vocab_size=vocab_size)

    else:
        # Local checkpoint path
        from nanochat.checkpoint_manager import load_checkpoint, find_last_step
        from nanochat.gpt import GPT, GPTConfig
        import os

        step = find_last_step(spec)
        model_data, _, meta = load_checkpoint(spec, step, device, load_optimizer=False)
        model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

        # Reconstruct model from checkpoint metadata
        if meta and 'model_config' in meta:
            config = GPTConfig(**meta['model_config'])
        else:
            raise ValueError(f"Cannot determine model config from checkpoint at {spec}")

        teacher = GPT(config)
        teacher.to_empty(device=device)
        teacher.init_weights()
        teacher.load_state_dict(model_data, strict=False, assign=True)
        del model_data

        return LocalTeacher(teacher)
