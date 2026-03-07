"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # CTM block settings (replaces MLP when enabled)
    # Paper: https://arxiv.org/abs/2505.05522
    use_ctm: bool = False
    ctm_iterations: int = 4       # K thinking steps per token
    ctm_memory_length: int = 8    # rolling trace history length
    ctm_n_synch: int = 384        # neuron pairs for synchronisation readout
    ctm_memory_hidden: int = 32   # NLM hidden dimension


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 2)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


# --- Continuous Thought Machine block (replaces MLP) ---
# Adapted from https://github.com/sakana-ai/continuous-thought-machines
# Each token gets K iterations of synapse -> trace update -> NLM processing.
# Output is read via synchronisation (pairwise temporal correlation between neurons).

class SuperLinear(nn.Module):
    """N independent linear transforms in parallel — one per neuron.
    Master weights stay fp32 for optimizer precision, cast to input dtype for compute."""
    def __init__(self, in_dims, out_dims, N):
        super().__init__()
        s = 1.0 / math.sqrt(in_dims + out_dims)
        self.w1 = nn.Parameter(torch.empty(in_dims, out_dims, N).uniform_(-s, s))
        self.b1 = nn.Parameter(torch.zeros(1, N, out_dims))

    def forward(self, x):
        # x: (B, N, in_dims) -> (B, N, out_dims)
        return torch.einsum('BNM,MON->BNO', x, self.w1.to(x.dtype)) + self.b1.to(x.dtype)


class CTMBlock(nn.Module):
    """Continuous Thought Machine block — drop-in replacement for MLP.
    Input/output: (B, T, D). Each token position is processed independently.
    NOTE: like the rest of the model, __init__ runs in meta device context.
    All actual data initialization happens in GPT.init_weights()."""
    def __init__(self, config):
        super().__init__()
        D = config.n_embd
        K = config.ctm_iterations
        M = config.ctm_memory_length
        n_synch = config.ctm_n_synch
        hidden = config.ctm_memory_hidden

        self.D = D
        self.K = K           # max iterations (from config)
        self.active_K = K    # current iterations (can be reduced by sleep-cycle scheduler)
        self.M = M
        self.n_synch = n_synch

        # Synapses: concat(input, state) -> new pre-activations
        self.synapse_fc = Linear(2 * D, 2 * D, bias=False)
        self.synapse_ln = nn.LayerNorm(D)

        # NLMs: per-neuron trace processing (deep variant with GLU)
        self.nlm1 = SuperLinear(M, 2 * hidden, D)
        self.nlm2 = SuperLinear(hidden, 2, D)

        # Learnable initial states (fake init, real init in GPT.init_weights)
        self.start_state = nn.Parameter(torch.empty(D))
        self.start_trace = nn.Parameter(torch.empty(D, M))

        # Synchronisation: buffers for neuron pairings (filled in init_weights) + learnable decay
        self.register_buffer('synch_left', torch.zeros(n_synch, dtype=torch.long))
        self.register_buffer('synch_right', torch.zeros(n_synch, dtype=torch.long))
        self.decay = nn.Parameter(torch.zeros(n_synch))

        # Output projection: synchronisation -> residual stream
        self.c_proj = Linear(n_synch, D, bias=False)

    def forward(self, x, dream=False, intervene=None):
        """
        Args:
            x: input tensor (B, T, D)
            dream: if True, collect per-iteration state deltas for convergence diagnostics
            intervene: optional callback fn(k, state, trace) -> state
                       Called after each iteration k. Can modify state for neuroplasticity.
                       Return modified state or None to keep unchanged.
        """
        B, T, D = x.shape
        BT = B * T
        x_flat = x.reshape(BT, D)
        K = self.active_K  # may be reduced by sleep-cycle scheduler

        # Initialize recurrent state
        state = self.start_state.unsqueeze(0).expand(BT, -1)
        trace = self.start_trace.unsqueeze(0).expand(BT, -1, -1).clone()  # (BT, D, M)

        # Synchronisation accumulators
        r = torch.exp(-self.decay.clamp(0, 15)).unsqueeze(0)  # (1, n_synch)
        decay_alpha = None
        decay_beta = None

        # Dream mode: collect per-iteration state deltas to measure convergence
        deltas = [] if dream else None

        for k in range(K):
            prev_state = state if dream else None

            # Synapses: mix external input with current state
            pre = torch.cat([x_flat, state], dim=-1)
            new_state = F.glu(self.synapse_fc(pre), dim=-1)
            new_state = self.synapse_ln(new_state)

            # Update trace (rolling window: drop oldest, append newest)
            trace = torch.cat([trace[:, :, 1:], new_state.unsqueeze(-1)], dim=-1)

            # NLMs: per-neuron processing of trace history
            h = F.glu(self.nlm1(trace), dim=-1)       # (BT, D, hidden)
            state = F.glu(self.nlm2(h), dim=-1).squeeze(-1)  # (BT, D)

            if dream:
                # Measure how much state changed this iteration (convergence signal)
                delta = (state - prev_state).norm(dim=-1).mean().item()
                deltas.append(delta)

            # Neuroplasticity hook: external intervention on internal state
            if intervene is not None:
                modified = intervene(k, state, trace)
                if modified is not None:
                    state = modified

            # Update synchronisation (exponential moving correlation)
            left = state[:, self.synch_left]
            right = state[:, self.synch_right]
            pp = left * right
            if decay_alpha is None:
                decay_alpha = pp
                decay_beta = torch.ones_like(pp)
            else:
                decay_alpha = r * decay_alpha + pp
                decay_beta = r * decay_beta + 1

        # Readout: synchronisation -> output projection
        synch = decay_alpha / torch.sqrt(decay_beta)
        out = self.c_proj(synch).reshape(B, T, D)
        if dream:
            return out, deltas
        return out


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = CTMBlock(config) if config.use_ctm else MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, dream=False):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        if dream and isinstance(self.mlp, CTMBlock):
            mlp_out, deltas = self.mlp(norm(x), dream=True)
            x = x + mlp_out
            return x, deltas
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp (MLP):       c_fc uniform, c_proj zeros
            mlp (CTMBlock):  synapse uniform, c_proj zeros, NLMs Xavier, start states uniform
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            if isinstance(block.mlp, CTMBlock):
                ctm = block.mlp
                # Synapse: uniform like c_fc
                torch.nn.init.uniform_(ctm.synapse_fc.weight, -s, s)
                # Output projection: zero for clean residual at init
                torch.nn.init.zeros_(ctm.c_proj.weight)
                # NLMs: Xavier-like per neuron
                for nlm in (ctm.nlm1, ctm.nlm2):
                    s_nlm = 1.0 / math.sqrt(nlm.w1.shape[0] + nlm.w1.shape[1])
                    nlm.w1.uniform_(-s_nlm, s_nlm)
                    nlm.b1.zero_()
                # Start states
                s_d = 1.0 / math.sqrt(ctm.D)
                ctm.start_state.uniform_(-s_d, s_d)
                s_dm = 1.0 / math.sqrt(ctm.D + ctm.M)
                ctm.start_trace.uniform_(-s_dm, s_dm)
                # Synapse LayerNorm
                ctm.synapse_ln.weight.fill_(1.0)
                ctm.synapse_ln.bias.zero_()
                # Synchronisation: random neuron pairings
                ctm.synch_left.copy_(torch.from_numpy(np.random.choice(ctm.D, size=ctm.n_synch, replace=True)))
                ctm.synch_right.copy_(torch.from_numpy(np.random.choice(ctm.D, size=ctm.n_synch, replace=True)))
                ctm.decay.zero_()
            else:
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for skip connection to input embedding

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero so gates start at sigmoid(0) = 0.5, scaled by 2 -> 1.0 (neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        # Muon only works on 2D matrices; route everything else to AdamW
        all_h_params = list(self.transformer.h.parameters())
        matrix_params = [p for p in all_h_params if p.ndim == 2]
        ctm_other_params = [p for p in all_h_params if p.ndim != 2]  # SuperLinear 3D, decay 1D, start states, LayerNorm
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == len(matrix_params) + len(ctm_other_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
        ]
        # CTM non-matrix params (SuperLinear 3D weights, decay params, start states, LayerNorm)
        if ctm_other_params:
            param_groups.append(dict(kind='adamw', params=ctm_other_params, lr=matrix_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 20 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def dream(self, idx):
        """REM sleep: run forward pass collecting per-layer CTM convergence diagnostics.
        Returns dict with per-layer state deltas at each iteration k.
        Useful for adaptive K: layers that converge fast can use fewer iterations."""
        B, T = idx.size()
        T0 = 0
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x
        layer_diagnostics = {}
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            result = block(x, ve, cos_sin, self.window_sizes[i], None, dream=True)
            if isinstance(result, tuple):
                x, deltas = result
                layer_diagnostics[i] = deltas  # list of K floats: state delta per iteration
            else:
                x = result
        return layer_diagnostics

    @torch.inference_mode()
    def probe(self, idx, layers=None):
        """Neuroimaging: capture full state snapshots at each K iteration per layer.
        Like an fMRI of the model's thinking process.

        Args:
            idx: token indices (B, T)
            layers: list of layer indices to probe (None = all CTM layers)

        Returns: dict[layer_idx] -> list of K state tensors, each (B*T, D)
        """
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x
        layer_snapshots = {}

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            probe_this = isinstance(block.mlp, CTMBlock) and (layers is None or i in layers)
            if probe_this:
                states = []
                def capture(k, state, trace, _states=states):
                    _states.append(state.detach().clone())
                    return None  # don't modify
                # Run attention, then CTM with capture hook
                x_attn = x + block.attn(norm(x), ve, cos_sin, self.window_sizes[i], None)
                mlp_out = block.mlp(norm(x_attn), intervene=capture)
                x = x_attn + mlp_out
                layer_snapshots[i] = states
            else:
                x = block(x, ve, cos_sin, self.window_sizes[i], None)
        return layer_snapshots

    def sleep_cycle(self, idx, convergence_threshold=0.3, min_k=2):
        """Full sleep cycle: dream (REM) then compact (adjust active_K per layer).

        1. REM: run dream() to measure per-layer convergence
        2. Compact: layers where state delta drops >threshold get reduced K

        Args:
            idx: token batch for dreaming
            convergence_threshold: if final delta < initial * threshold, layer has converged
            min_k: minimum K to allow (must be >= 2 for synch to work)

        Returns: dict with per-layer diagnostics and new K values
        """
        if not self.config.use_ctm:
            return {}

        self.eval()
        diagnostics = self.dream(idx)
        self.train()

        results = {}
        for i, deltas in diagnostics.items():
            block = self.transformer.h[i]
            if not isinstance(block.mlp, CTMBlock):
                continue

            old_k = block.mlp.active_K
            max_k = block.mlp.K

            # Find the earliest iteration where delta drops below threshold
            initial_delta = deltas[0] if deltas[0] > 0 else 1e-6
            new_k = max_k
            for k_idx, delta in enumerate(deltas):
                if delta < initial_delta * convergence_threshold:
                    new_k = max(min_k, k_idx + 1)  # +1 because we need at least this many
                    break

            block.mlp.active_K = new_k
            results[i] = {
                'deltas': deltas,
                'old_k': old_k,
                'new_k': new_k,
                'converged': new_k < max_k,
            }

        return results

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
