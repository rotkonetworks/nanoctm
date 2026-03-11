"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import math
import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.gpt import CTMCache, CTMBlock

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None

def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # Disallow dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if '.count(' not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
class KVCache:
    """
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.

    Key differences from FA2-style cache:
    - Tensors are (B, T, H, D) not (B, H, T, D)
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - Position tracked per batch element via cache_seqlens tensor
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)

# -----------------------------------------------------------------------------
@torch.inference_mode()
def _safe_multinomial(probs, rng):
    """Multinomial that won't poison the CUDA context with device-side asserts."""
    # Clamp NaN/inf/negative to zero, re-normalize. Falls back to uniform if all-zero.
    bad = torch.isnan(probs) | torch.isinf(probs) | (probs < 0)
    if bad.any():
        probs = probs.clone()
        probs[bad] = 0.0
        sums = probs.sum(dim=-1, keepdim=True)
        zero_rows = (sums == 0).squeeze(-1)
        if zero_rows.any():
            probs[zero_rows] = 1.0 / probs.size(-1)  # uniform fallback
        else:
            probs = probs / sums
    return torch.multinomial(probs, num_samples=1, generator=rng)

def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = _safe_multinomial(probs, rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return _safe_multinomial(probs, rng)

# -----------------------------------------------------------------------------

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # Current token sequence for this row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_python_block = False # Whether we are inside a python block
        self.python_expr_tokens = [] # Tokens of the current python expression
        self.completed = False # Whether this row has completed generation

class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42, repetition_penalty=1.0):
        """Generate tokens with optional repetition penalty to fight attractor collapse."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Get the special tokens we need to coordinate the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        # Check if model manages its own KV cache (e.g. QwenBackboneGPT)
        own_cache = getattr(self.model, 'manages_own_cache', False)

        # Create CTMCache for prefill if model uses CTM
        m = self.model.config
        ctm_cache_prefill = CTMCache(m.n_layer) if m.use_ctm else None
        # Reset dopamine EMA for fresh generation session
        self._surprise_ema = None
        ids = torch.tensor([tokens], dtype=torch.long, device=device)

        if own_cache:
            # Model manages its own KV cache (HuggingFace-style)
            assert num_samples == 1, "Multi-sample generation not supported with external backbone models"
            self.model.reset_cache()
            logits = self.model.forward(ids, kv_cache="use_internal", ctm_cache=ctm_cache_prefill)
            logits = logits[:, -1, :]  # (1, vocab_size)
            kv_cache_decode = "use_internal"  # sentinel value
            ctm_cache_decode = ctm_cache_prefill  # same object, state accumulated
            self._last_ctm_cache = ctm_cache_decode
        else:
            # Standard path: our KVCache
            kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
            kv_cache_prefill = KVCache(
                batch_size=1,
                seq_len=len(tokens),
                device=device,
                dtype=dtype,
                **kv_model_kwargs,
            )
            logits = self.model.forward(ids, kv_cache=kv_cache_prefill, ctm_cache=ctm_cache_prefill)
            logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

            # Replicate the KV cache for each sample/row
            kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
            kv_cache_decode = KVCache(
                batch_size=num_samples,
                seq_len=kv_length_hint,
                device=device,
                dtype=dtype,
                **kv_model_kwargs,
            )
            kv_cache_decode.prefill(kv_cache_prefill)
            del kv_cache_prefill

            # Replicate CTMCache
            ctm_cache_decode = ctm_cache_prefill.extract_last_and_expand(num_samples) if ctm_cache_prefill is not None else None
            del ctm_cache_prefill
            self._last_ctm_cache = ctm_cache_decode

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            # Apply repetition penalty: penalize tokens that appeared in context
            if repetition_penalty != 1.0:
                for i, state in enumerate(row_states):
                    seen = set(state.current_tokens[-64:])  # look back 64 tokens
                    for token_id in seen:
                        if logits[i, token_id] > 0:
                            logits[i, token_id] /= repetition_penalty
                        else:
                            logits[i, token_id] *= repetition_penalty

            # Sample the next token for each row
            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            # Dopamine: compute prediction surprise for each sampled token
            # surprise = -log(p(token)) = cross-entropy of the chosen token
            #
            # SAFETY CONSTRAINT: dopamine is clamped to [DOPA_MIN, DOPA_MAX].
            #
            # During training, dopamine is ALWAYS 1.0 (CTMBlock.forward default).
            # Any dopamine != 1.0 is out-of-distribution for the sync accumulators.
            # We allow mild modulation (dampening for boring tokens) but NEVER
            # amplification above 1.0. Rationale:
            #
            #   The original range [0, 2] creates a positive feedback loop:
            #     garbage output → high surprise → dopamine > 1 →
            #     sync accumulates harder on garbage state → more garbage.
            #   By clamping to [0.5, 1.0]:
            #     - Surprising tokens: dopamine ≈ 1.0 (normal accumulation)
            #     - Boring/predictable tokens: dopamine ≈ 0.5 (dampen accumulation)
            #     - Garbage tokens: high surprise, but dopamine still ≤ 1.0 (no amplification)
            #   This breaks the attractor feedback loop while preserving the useful
            #   signal: "this token was predictable, don't overwrite memory for it."
            DOPA_MIN = 0.5
            DOPA_MAX = 1.0
            if ctm_cache_decode is not None:
                with torch.no_grad():
                    log_probs = F.log_softmax(logits, dim=-1)  # (B, vocab_size)
                    token_surprises = -log_probs.gather(1, next_ids).squeeze(1)  # (B,)
                    mean_surprise = token_surprises.mean().item()
                    if self._surprise_ema is None:
                        self._surprise_ema = mean_surprise
                    else:
                        self._surprise_ema = 0.9 * self._surprise_ema + 0.1 * mean_surprise
                    # Raw signal: tanh(surprise - ema) in [-1, 1]
                    raw = math.tanh(mean_surprise - self._surprise_ema)
                    # Map [-1, 1] → [DOPA_MIN, DOPA_MAX]: linear interpolation
                    dopamine = DOPA_MIN + (DOPA_MAX - DOPA_MIN) * (raw + 1.0) / 2.0
                    # Defensive clamp (belt AND suspenders — the math above guarantees
                    # the range, but floating point is adversarial)
                    dopamine = max(DOPA_MIN, min(DOPA_MAX, dopamine))
                    ctm_cache_decode.dopamine = dopamine

            # Process each row: choose the next token, update state, optional tool use
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
                token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # On <|assistant_end|> or <|bos|>, mark the row as completed
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # Handle tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1

            # Prepare logits for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            fwd_kwargs = {"ctm_cache": ctm_cache_decode}
            if not own_cache:
                fwd_kwargs["kv_cache"] = kv_cache_decode
            else:
                fwd_kwargs["kv_cache"] = "use_internal"
            logits = self.model.forward(ids, **fwd_kwargs)[:, -1, :]  # (B, vocab_size)

    def generate_and_compact(self, tokens, num_samples=1, plasticity_lr=1e-5, **kwargs):
        """Generate tokens then compact CTMCache memory into permanent weights.

        Runs a full generation session, then calls model.compact_memory() to
        write the accumulated sync patterns into synapse weights. The model
        permanently remembers what it experienced during this generation.

        The CTMCache is the one that was built incrementally during generation —
        not a re-run. It contains the actual accumulated sync statistics from
        the model's stream of consciousness.

        Returns: (results, masks, plasticity_stats)
        """
        results, masks = self.generate_batch(tokens, num_samples, **kwargs)

        # Compact memory using the CTMCache that was populated during generation
        plasticity_stats = {}
        ctm_cache = getattr(self, '_last_ctm_cache', None)
        if ctm_cache is not None and plasticity_lr > 0:
            plasticity_stats = self.model.compact_memory(ctm_cache, lr=plasticity_lr)
        self._last_ctm_cache = None  # free memory

        return results, masks, plasticity_stats

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks


class Session:
    """Persistent conversation session with continuous memory.

    Unlike Engine.generate() which creates fresh caches per call, a Session
    maintains KVCache across multiple turns. You send one message, get a reply,
    send another — the model remembers everything through attention's KV cache.

    CTMCache (persistent thinking state across tokens) is disabled by default.
    The CTM blocks still run their K thinking iterations per token — that's the
    "deep thinking" part — but each token starts from fresh start_state. This is
    because the model is trained without cross-token CTM state (training processes
    all tokens in parallel with fresh state). Carrying state across tokens at
    inference time feeds the model inputs it never saw during training, which
    degrades output quality.

    To enable CTMCache, the model would need a second training phase with chunked
    sequences where CTMCache is carried between chunks, teaching the model what
    persistent thinking state means. This is future work — the infrastructure
    (CTMCache, cache persistence, episodic memory) is ready and tested.

    Usage:
        session = Session(model, tokenizer, max_seq_len=4096)
        reply1 = session.say("Hello, who are you?")
        reply2 = session.say("What did I just ask you?")  # model remembers via KV cache
        # compact() and EpisodicMemory require CTMCache (BPTT-trained model)
        # Later:
        session = Session.load("session_001.pt", model, tokenizer)
    """

    def __init__(self, model, tokenizer, max_seq_len=None, seed=42, online_lr=0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.get_device()
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        # Don't cast the model's master weights - they stay fp32 for optimizer precision.
        # The model's Linear layers cast weights to input dtype in their forward().
        self.model.eval()
        self.seed = seed
        self.online_lr = online_lr  # >0 enables automatic learning from user messages

        m = model.config
        seq_len = max_seq_len or m.sequence_len
        kv_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}

        # Persistent caches — survive across turns
        self.kv_cache = KVCache(batch_size=1, seq_len=seq_len, device=self.device, dtype=self.dtype, **kv_kwargs)
        # CTMCache disabled by default — the model is trained without persistent CTM state,
        # so enabling it degrades output quality. Enable only after training with chunked
        # CTMCache continuity (future work).
        self.ctm_cache = None
        self.surprise_ema = None

        # Track all tokens in the conversation
        self.all_tokens = []
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed)

        # Online learning: persistent optimizer for plastic params (lazy init)
        self._plastic_params = None
        self._optimizer = None
        self._frozen_params = None
        self.last_learn_stats = {}  # stats from most recent learn_from call

    @torch.inference_mode()
    def _forward_tokens(self, tokens):
        """Feed tokens through the model, updating KV and CTM caches. Returns logits for last token."""
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        logits = self.model.forward(ids, kv_cache=self.kv_cache, ctm_cache=self.ctm_cache)
        return logits[:, -1, :]  # (1, vocab_size)

    @torch.inference_mode()
    def say(self, text, max_tokens=512, temperature=0.7, top_k=50):
        """Send a message, get a reply. Both sides are remembered in the caches.

        If online_lr > 0, the model learns from the user's message before replying.
        The user's text is a prediction error signal - what the model should have
        anticipated as the next part of the conversation. One gradient step on
        synapse+NLM params, then inference continues with the updated weights.

        Args:
            text: user message (string)
            max_tokens: max reply length
            temperature: sampling temperature
            top_k: top-k sampling

        Returns: model's reply as string
        """
        # Online learning: learn from user's message before replying
        # (runs outside inference_mode via learn_from's own context)
        if self.online_lr > 0 and self.all_tokens:
            with torch.inference_mode(False):
                self.last_learn_stats = self.learn_from(text, lr=self.online_lr)

        # Encode user message with chat formatting
        user_tokens = self.tokenizer.encode(text)
        self.all_tokens.extend(user_tokens)

        # Prefill: feed user tokens through model (updates KV + CTM caches)
        logits = self._forward_tokens(user_tokens)

        # Decode: generate reply token by token
        reply_tokens = []
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        for _ in range(max_tokens):
            # Sample next token
            next_id = sample_next_token(logits, self.rng, temperature, top_k)  # (1, 1)
            token = next_id.item()

            # Stop on end tokens
            if token == assistant_end or token == bos:
                break

            # Dopamine: compute surprise and modulate CTM cache
            if self.ctm_cache is not None:
                with torch.no_grad():
                    log_probs = F.log_softmax(logits, dim=-1)
                    surprise = -log_probs[0, token].item()
                    if self.surprise_ema is None:
                        self.surprise_ema = surprise
                    else:
                        self.surprise_ema = 0.9 * self.surprise_ema + 0.1 * surprise
                    self.ctm_cache.dopamine = 1.0 + math.tanh(surprise - self.surprise_ema)

            reply_tokens.append(token)
            self.all_tokens.append(token)

            # Feed token through model (updates caches)
            logits = self._forward_tokens([token])

        return self.tokenizer.decode(reply_tokens)

    def compact(self, lr=1e-5):
        """Write this conversation's memory into permanent weights.
        Call at the end of a conversation to make the model remember it forever."""
        if self.ctm_cache is not None:
            return self.model.compact_memory(self.ctm_cache, lr=lr)
        return {}

    def _init_online_learning(self, lr):
        """Lazy init: collect plastic params, snapshot pretrained weights, build optimizer."""
        if self._plastic_params is not None:
            return

        self._plastic_params = []
        for block in self.model.transformer.h:
            if not isinstance(block.mlp, CTMBlock):
                continue
            ctm = block.mlp
            self._plastic_params.extend(p for p in ctm.synapses.parameters() if p.requires_grad)
            self._plastic_params.extend(p for p in ctm.nlm1.parameters() if p.requires_grad)
            self._plastic_params.extend(p for p in ctm.nlm2.parameters() if p.requires_grad)
            self._plastic_params.extend(p for p in ctm.c_proj.parameters() if p.requires_grad)

        if not self._plastic_params:
            return

        # Snapshot pretrained weights for elastic anchoring (EWC-lite).
        # These are the "home base" weights that prevent catastrophic forgetting.
        self._pretrained_snapshot = [p.data.clone() for p in self._plastic_params]

        self._optimizer = torch.optim.AdamW(
            self._plastic_params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0,
        )

        # Freeze everything else so backward only computes gradients we need
        plastic_set = set(id(p) for p in self._plastic_params)
        self._frozen_params = []
        for p in self.model.parameters():
            if p.requires_grad and id(p) not in plastic_set:
                p.requires_grad_(False)
                self._frozen_params.append(p)

    def _cleanup_online_learning(self):
        """Restore requires_grad on frozen params."""
        if self._frozen_params:
            for p in self._frozen_params:
                p.requires_grad_(True)

    def _elastic_penalty(self):
        """L2 penalty toward pretrained weight snapshot. Prevents catastrophic forgetting."""
        penalty = 0.0
        for p, p0 in zip(self._plastic_params, self._pretrained_snapshot):
            penalty = penalty + (p - p0).pow(2).sum()
        return penalty

    def learn_from(self, text, lr=1e-5, context_len=256, distill_weight=0.5, elastic_weight=0.01):
        """Online learning from prediction error with self-distillation.

        The model learns from the conversation as it happens. Three loss terms
        keep learning stable:

        1. prediction error: CE loss on the user's actual text (what we should
           have predicted). this is the learning signal.

        2. self-distillation: KL divergence between the model's current output
           distribution and what it predicted before the gradient step. keeps
           the model from forgetting what it already knows on this context.

        3. elastic anchoring: L2 penalty toward the pretrained weight snapshot.
           prevents long-term drift from the base model over many interactions.

        total loss = (1 - distill_weight) * CE + distill_weight * KL + elastic_weight * L2

        ROADMAP (phase 5 - metacognitive tokens): currently the learning signal is
        purely external (user text vs model prediction). With metacognitive tokens,
        the model would also observe its own internal states (certainty, sync
        magnitudes) as tokens in the stream. Prediction error on self-observation
        becomes a fourth loss term -- the model learns not just what the world does,
        but what it does, and uses that self-knowledge to gate plasticity adaptively.

        Args:
            text: the user's actual reply (what we should have predicted)
            lr: learning rate (small - one gentle nudge per interaction)
            context_len: how many preceding tokens to use as context
            distill_weight: weight for self-distillation KL term (0 = pure prediction error)
            elastic_weight: weight for elastic anchoring L2 penalty (0 = no anchoring)

        Returns: dict with loss components, or empty dict if not CTM
        """
        if not self.model.config.use_ctm:
            return {}

        self._init_online_learning(lr)
        if not self._plastic_params:
            return {}

        # Build input: recent context tokens + the new text
        new_tokens = self.tokenizer.encode(text)
        if not new_tokens:
            return {}

        # Use recent conversation as context for prediction
        context = self.all_tokens[-context_len:] if len(self.all_tokens) > context_len else self.all_tokens[:]
        sequence = context + new_tokens

        # Input is sequence[:-1], target is sequence[1:] (standard LM)
        if len(sequence) < 2:
            return {}

        ids = torch.tensor([sequence[:-1]], dtype=torch.long, device=self.device)
        targets = torch.tensor([sequence[1:]], dtype=torch.long, device=self.device)

        # Only compute loss on the new tokens (not the context)
        # Mask context positions with -1 (ignored by cross_entropy)
        n_context = len(context) - 1  # -1 because targets are shifted
        if n_context > 0:
            targets[:, :n_context] = -1

        self.model.eval()  # no grad checkpointing

        # Step 1: get teacher logits (current model's predictions before update)
        with torch.no_grad():
            teacher_logits = self.model.forward(ids, targets=None)  # (1, T, V)

        # Step 2: forward with gradients for student loss
        self._optimizer.zero_grad()
        student_logits = self.model.forward(ids, targets=None)  # (1, T, V)

        # Prediction error: CE on new tokens only
        flat_targets = targets.view(-1)
        flat_logits = student_logits.view(-1, student_logits.size(-1))
        ce_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1)

        # Self-distillation: KL divergence from teacher on ALL positions (context + new)
        # The model should stay close to its pre-update predictions everywhere
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        # KL(teacher || student) = sum(teacher * (log_teacher - log_student))
        teacher_probs = teacher_log_probs.exp()
        kl_loss = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1).mean()

        # Elastic anchoring: L2 toward pretrained weights
        elastic_loss = self._elastic_penalty()

        # Combined loss
        loss = (1.0 - distill_weight) * ce_loss + distill_weight * kl_loss + elastic_weight * elastic_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._plastic_params, 1.0)
        self._optimizer.step()

        # Clear grads to not pollute inference
        self._optimizer.zero_grad()
        for p in self._plastic_params:
            if p.grad is not None:
                p.grad = None

        return {
            'loss': loss.item(),
            'ce_loss': ce_loss.item(),
            'kl_loss': kl_loss.item(),
            'elastic_loss': elastic_loss.item(),
            'tokens_learned': len(new_tokens),
        }

    def save(self, path):
        """Save session state (caches + token history) to disk for later resumption."""
        state = {
            'all_tokens': self.all_tokens,
            'kv_k': self.kv_cache.k_cache.cpu(),
            'kv_v': self.kv_cache.v_cache.cpu(),
            'kv_seqlens': self.kv_cache.cache_seqlens.cpu(),
            'surprise_ema': self.surprise_ema,
        }
        if self.ctm_cache is not None:
            ctm_state = []
            for layer in self.ctm_cache.layers:
                if layer is not None:
                    ctm_state.append({k: v.cpu() for k, v in layer.items()})
                else:
                    ctm_state.append(None)
            state['ctm_layers'] = ctm_state
            state['ctm_dopamine'] = self.ctm_cache.dopamine
        torch.save(state, path)

    @classmethod
    def load(cls, path, model, tokenizer, max_seq_len=None):
        """Load a saved session — resume a previous conversation with full memory."""
        state = torch.load(path, weights_only=False)
        session = cls(model, tokenizer, max_seq_len=max_seq_len)
        device = session.device

        # Restore KV cache
        session.kv_cache.k_cache.copy_(state['kv_k'].to(device))
        session.kv_cache.v_cache.copy_(state['kv_v'].to(device))
        session.kv_cache.cache_seqlens.copy_(state['kv_seqlens'].to(device))

        # Restore CTM cache
        if session.ctm_cache is not None and 'ctm_layers' in state:
            for i, layer_state in enumerate(state['ctm_layers']):
                if layer_state is not None:
                    session.ctm_cache.layers[i] = {k: v.to(device) for k, v in layer_state.items()}
            session.ctm_cache.dopamine = state.get('ctm_dopamine', 1.0)

        session.all_tokens = state['all_tokens']
        session.surprise_ema = state.get('surprise_ema')
        return session

    def get_pos(self):
        """How many tokens deep into the conversation we are."""
        return self.kv_cache.get_pos()

    def tokens_remaining(self):
        """How many tokens until the KV cache is full."""
        return self.kv_cache.max_seq_len - self.kv_cache.get_pos()


class EpisodicMemory:
    """Long-term episodic memory: a searchable store of past conversation states.

    Brain analog: hippocampus. Stores snapshots of CTMCache (the model's mental state)
    from past conversations, indexed by a context embedding. When a new conversation
    starts, finds the most similar past experience and warm-starts the CTMCache from it.

    "I've thought about something like this before" — instead of starting from
    the learned start_state (blank slate), the model begins with a mental state
    that's already primed for this type of conversation.

    The context embedding is simply the mean of the input token embeddings — no
    separate encoder needed. Cosine similarity finds the nearest past experience.

    Usage:
        memory = EpisodicMemory(model, capacity=100)

        # After a conversation:
        session.compact(lr=1e-5)  # semantic memory (weights)
        memory.store(session)     # episodic memory (cache snapshot)

        # Before next conversation:
        new_session = memory.recall("help me with Python", tokenizer)
        # → new_session's CTMCache is warm-started from the most similar past session

        # Persist across restarts:
        memory.save("episodes.pt")
        memory = EpisodicMemory.load("episodes.pt", model)
    """

    def __init__(self, model, capacity=100):
        self.model = model
        self.capacity = capacity
        self.episodes = []  # list of {embedding, ctm_layers, summary, tokens_seen}

    @torch.inference_mode()
    def _embed_text(self, text, tokenizer):
        """Compute context embedding: mean of token embeddings from wte."""
        tokens = tokenizer.encode(text)
        ids = torch.tensor([tokens], dtype=torch.long, device=self.model.get_device())
        embeds = self.model.transformer.wte(ids)  # (1, T, D)
        return F.normalize(embeds.mean(dim=1), dim=-1).squeeze(0)  # (D,)

    @torch.inference_mode()
    def _embed_tokens(self, tokens):
        """Compute context embedding from raw token ids."""
        ids = torch.tensor([tokens], dtype=torch.long, device=self.model.get_device())
        embeds = self.model.transformer.wte(ids)
        return F.normalize(embeds.mean(dim=1), dim=-1).squeeze(0)

    def store(self, session, summary=None):
        """Store a session's CTMCache as an episodic memory.

        Requires the session to have an active CTMCache (i.e. model trained with
        BPTT state continuity). Without CTMCache, there's no thinking state to
        store - the session's memory lives only in the KV cache which is an
        ephemeral computational artifact, not transferable episodic memory.

        Args:
            session: a Session object after conversation (must have ctm_cache)
            summary: optional text summary of what the conversation was about
        """
        if session.ctm_cache is None:
            return

        # Embed the conversation context (use first 256 tokens for speed)
        context_tokens = session.all_tokens[:256]
        if not context_tokens:
            return
        embedding = self._embed_tokens(context_tokens).cpu()

        # Snapshot the CTM cache (detach + CPU for storage)
        ctm_snapshot = []
        for layer in session.ctm_cache.layers:
            if layer is not None:
                ctm_snapshot.append({k: v.detach().cpu() for k, v in layer.items()})
            else:
                ctm_snapshot.append(None)

        episode = {
            'embedding': embedding,
            'ctm_layers': ctm_snapshot,
            'summary': summary,
            'tokens_seen': len(session.all_tokens),
        }

        # If at capacity, evict the oldest episode
        if len(self.episodes) >= self.capacity:
            self.episodes.pop(0)
        self.episodes.append(episode)

    def recall(self, text_or_tokens, tokenizer=None, threshold=0.3):
        """Find the most similar past experience and return its CTMCache.

        Args:
            text_or_tokens: prompt text (str) or token list to match against
            tokenizer: needed if text_or_tokens is a string
            threshold: minimum cosine similarity to consider a match

        Returns: CTMCache warm-started from best matching episode, or None if no match
        """
        if not self.episodes:
            return None

        # Compute query embedding
        if isinstance(text_or_tokens, str):
            assert tokenizer is not None
            query = self._embed_text(text_or_tokens, tokenizer)
        else:
            query = self._embed_tokens(text_or_tokens)

        # Find nearest neighbor by cosine similarity
        best_sim = -1
        best_episode = None
        for episode in self.episodes:
            sim = F.cosine_similarity(query.unsqueeze(0), episode['embedding'].to(query.device).unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                best_episode = episode

        if best_sim < threshold:
            return None

        # Reconstruct CTMCache from snapshot
        m = self.model.config
        cache = CTMCache(m.n_layer)
        device = self.model.get_device()
        for i, layer_state in enumerate(best_episode['ctm_layers']):
            if layer_state is not None:
                cache.layers[i] = {k: v.to(device) for k, v in layer_state.items()}
        return cache

    def recall_into_session(self, text, tokenizer, session):
        """Warm-start a session's CTMCache from the most similar past episode.

        Args:
            text: the new conversation's opening prompt
            tokenizer: for embedding the prompt
            session: Session to warm-start (modifies in place)

        Returns: similarity score if matched, None if no match found
        """
        cache = self.recall(text, tokenizer)
        if cache is None:
            return None
        # Overwrite session's CTMCache with the recalled episode
        session.ctm_cache = cache
        return True

    def save(self, path):
        """Persist all episodes to disk."""
        torch.save({
            'episodes': self.episodes,
            'capacity': self.capacity,
        }, path)

    @classmethod
    def load(cls, path, model):
        """Load episodic memory from disk."""
        data = torch.load(path, weights_only=False)
        memory = cls(model, capacity=data['capacity'])
        memory.episodes = data['episodes']
        return memory

    def __len__(self):
        return len(self.episodes)

    def list_episodes(self):
        """List stored episodes with summaries and similarity info."""
        return [{
            'index': i,
            'summary': ep.get('summary', '(no summary)'),
            'tokens_seen': ep['tokens_seen'],
        } for i, ep in enumerate(self.episodes)]


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time
    # init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0] # only print out the first row
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
