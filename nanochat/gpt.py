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

from torch.utils.checkpoint import checkpoint as grad_checkpoint

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
    ctm_memory_length: int = 16   # rolling trace history length (paper uses 25)
    ctm_n_synch: int = 384        # neuron pairs for synchronisation readout
    ctm_memory_hidden: int = 32   # NLM hidden dimension (-1 = n_synch // 4, scales with model. 32 for training memory budget)
    ctm_n_attn_heads: int = 1     # cross-attention heads for data re-observation
    ctm_synapse_depth: int = 6    # U-NET synapse depth (half down, half up). Paper uses 16.


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

# =============================================================================
# CTM NEUROPLASTICITY ROADMAP
# =============================================================================
# Current implementation: sync-driven Hebbian plasticity via compact_memory().
# The accumulated S_out sync signal (alpha/sqrt(beta)) from inference is compared
# to baseline, and novel patterns are written into synapse weights permanently.
#
# TODO: Dopamine-gated plasticity
#   The brain doesn't learn from everything — dopamine gates WHEN plasticity happens.
#   High dopamine = surprising/rewarding → learn aggressively.
#   Low dopamine = predictable/boring → don't bother.
#   Implementation idea: compute prediction error (actual next token vs model's prediction)
#   during inference. When prediction error spikes (surprise!), set a "dopamine" flag
#   that amplifies the plasticity lr for that token's patterns. When prediction error
#   is low (boring text), suppress plasticity. This requires access to the tokenizer
#   during forward, or a post-hoc surprise signal from the Engine.
#   Could also gate on certainty: high certainty + wrong = maximum surprise = learn hard.
#   Low certainty + wrong = expected confusion = don't bother.
#
# TODO: Runaway weight growth / synaptic homeostasis
#   Current: soft Frobenius norm clamp (1% max growth per compact_memory call).
#   Problem: over many inference sessions, weights can still drift far from training.
#   Better approaches:
#   - Synaptic scaling: periodically rescale ALL synapse weights to maintain mean
#     activation levels (the brain does this during sleep via TNF-alpha signaling).
#   - Elastic weight consolidation (EWC): track Fisher information per param during
#     training, penalize plasticity on high-Fisher params (important for old knowledge).
#   - Hard reset: every N compact_memory calls, re-run consolidate() on the replay
#     buffer to anchor weights back toward training distribution.
#   - Per-synapse decay: older plasticity updates fade unless reinforced. Each weight
#     could track its "last update time" and exponentially decay toward the trained value.
#
# TODO: Multi-part memory system (hippocampal-cortical)
#   The brain has multiple memory systems with different timescales:
#   - Working memory (seconds): CTMCache state/trace — already implemented
#   - Episodic memory (hours): a buffer of CTMCache snapshots from recent sessions,
#     indexed by context. On inference start, retrieve the most similar past session's
#     cache and warm-start the CTMCache from it. "I've seen text like this before."
#   - Semantic memory (permanent): synapse weights — what compact_memory writes to.
#     This is the model's long-term knowledge, updated slowly and carefully.
#   Implementation: EpisodicMemory class that stores (context_embedding, ctm_cache_snapshot)
#   pairs. On new inference, embed the prompt, find nearest neighbor, load that cache.
#   After inference, store the new (context, cache) pair. Bounded by memory budget.
#   The key insight: CTMCache IS the episodic memory — we just need to persist and
#   retrieve it across sessions rather than always starting from learned start_state.
# =============================================================================


class CTMCache:
    """Persistent CTM state across tokens for continuous inference.
    During autoregressive generation, each new token's thinking starts where
    the previous token's thinking ended — a stream of consciousness.
    One state dict per layer, mutated in-place by CTMBlock.forward.

    Dopamine gating: the `dopamine` field (float, default 1.0) scales how strongly
    each token's pairwise products contribute to the sync accumulators. Set by the
    Engine based on prediction error — surprised tokens get dopamine > 1, boring
    tokens get dopamine < 1. This means the sync "remembers harder" for surprising
    moments, just like the brain's dopaminergic system gates synaptic plasticity."""
    def __init__(self, n_layers):
        self.layers = [None] * n_layers  # None until first token populates it
        self.dopamine = 1.0  # per-token neuromodulatory signal, set by Engine

    def extract_last_and_expand(self, num_samples):
        """After prefill (B=1, T tokens), extract last token's state and expand for parallel decode.
        CTMBlock stores state as (BT, D) — after prefill BT=T. We take the last position's state
        and replicate it num_samples times for parallel generation."""
        new_cache = CTMCache(len(self.layers))
        for i, layer_state in enumerate(self.layers):
            if layer_state is None:
                continue
            new_cache.layers[i] = {
                k: v[-1:].expand(num_samples, *v.shape[1:]).contiguous()
                for k, v in layer_state.items()
            }
        return new_cache

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


class SynapseUNET(nn.Module):
    """U-NET synapse model (paper Figure 8, Section C.1).
    Linearly reduces width from input_dim down to bottleneck (16),
    then expands back to output_dim, with skip connections + LayerNorm.
    k layers total (k/2 down, k/2 up)."""
    def __init__(self, input_dim, output_dim, k=8, bottleneck=16):
        super().__init__()
        assert k >= 2 and k % 2 == 0, "k must be even and >= 2"
        half_k = k // 2

        # Down path widths: linearly from input_dim to bottleneck
        # e.g. k=8, input=1536, bottleneck=16 -> [1536, 1156, 776, 396, 16]
        down_widths = [max(bottleneck, round(input_dim + (bottleneck - input_dim) * i / half_k))
                       for i in range(half_k + 1)]
        # Up path OUTPUT widths: mirror of down, but final layer outputs output_dim
        up_out_widths = list(reversed(down_widths[:-1]))
        up_out_widths[-1] = output_dim  # final layer produces output_dim

        # Down layers
        self.down_layers = nn.ModuleList()
        for i in range(half_k):
            self.down_layers.append(Linear(down_widths[i], down_widths[i + 1], bias=False))

        # Up layers: each receives its own input + skip from matched down layer
        # First up layer has no skip (takes bottleneck directly)
        # Subsequent up layers concat with skip from down path
        self.up_layers = nn.ModuleList()
        self.up_norms = nn.ModuleList()
        for i in range(half_k):
            if i == 0:
                in_w = down_widths[-1]  # bottleneck output
            else:
                in_w = up_out_widths[i - 1] + down_widths[half_k - i]  # prev up output + skip
            self.up_layers.append(Linear(in_w, up_out_widths[i], bias=False))
            self.up_norms.append(nn.LayerNorm(up_out_widths[i]))

    def forward(self, x):
        # Down path: save activations for skip connections
        skips = []
        h = x
        for layer in self.down_layers:
            h = F.silu(layer(h))
            skips.append(h)

        # Up path: concat with skip connections from down path
        h = skips.pop()  # bottleneck output
        for i, (layer, ln) in enumerate(zip(self.up_layers, self.up_norms)):
            if i > 0 and skips:
                h = torch.cat([h, skips.pop()], dim=-1)
            h = F.silu(layer(h))
            # LayerNorm keeps float32 weights for optimizer precision; cast to match activations
            h = F.layer_norm(h, ln.normalized_shape, ln.weight.to(h.dtype), ln.bias.to(h.dtype), ln.eps)
        return h


class CTMBlock(nn.Module):
    """Continuous Thought Machine block — drop-in replacement for MLP.
    Input/output: (B, T, D).
    Key features (following paper more closely):
    1. Cross-attention data re-observation: at each tick, S_action generates a query
       that cross-attends to input features, so the model "looks at" different parts each tick.
    2. Dual synchronisation: S_out (for prediction readout) and S_action (for attention queries).
    3. U-NET synapse model: deeper cross-neuron interaction with skip connections (Figure 8).
    NOTE: __init__ runs in meta device context. Real init in GPT.init_weights()."""
    def __init__(self, config):
        super().__init__()
        D = config.n_embd
        K = config.ctm_iterations
        M = config.ctm_memory_length
        n_synch = config.ctm_n_synch
        hidden = config.ctm_memory_hidden if config.ctm_memory_hidden > 0 else n_synch // 4
        n_attn_heads = config.ctm_n_attn_heads

        self.D = D
        self.K = K           # max iterations (from config)
        self.active_K = K    # current iterations (can be reduced by sleep-cycle scheduler)
        self.M = M
        self.n_synch = n_synch
        self.n_attn_heads = n_attn_heads
        self.attn_head_dim = D // n_attn_heads

        # Cross-attention for data re-observation at each tick
        # S_action sync -> query; input features -> key, value (computed once)
        self.attn_q_proj = Linear(n_synch, D, bias=False)
        self.attn_k_proj = Linear(D, D, bias=False)
        self.attn_v_proj = Linear(D, D, bias=False)

        # U-NET synapse model (paper Figure 8): concat(obs, state) -> pre-activations
        self.synapses = SynapseUNET(2 * D, D, k=config.ctm_synapse_depth)

        # NLMs: per-neuron trace processing (deep variant with GLU)
        self.nlm1 = SuperLinear(M, 2 * hidden, D)
        self.nlm2 = SuperLinear(hidden, 2, D)

        # Learnable initial states (fake init, real init in GPT.init_weights)
        self.start_state = nn.Parameter(torch.empty(D))
        self.start_trace = nn.Parameter(torch.empty(D, M))

        # Dual synchronisation: S_out (readout) and S_action (attention queries)
        for name in ('synch_out_left', 'synch_out_right', 'synch_act_left', 'synch_act_right'):
            self.register_buffer(name, torch.zeros(n_synch, dtype=torch.long))
        self.decay_out = nn.Parameter(torch.zeros(n_synch))
        self.decay_act = nn.Parameter(torch.zeros(n_synch))

        # Per-tick embedding: gives each iteration a unique signature so ticks diverge
        self.tick_embed = nn.Parameter(torch.empty(K, D))

        # Output projection: S_out synchronisation -> residual stream
        self.c_proj = Linear(n_synch, D, bias=False)

    def _tick_core(self, alpha_act, beta_act, x, attn_k, attn_v, state, trace, tick_emb):
        """One iteration of CTM thinking — factored out for gradient checkpointing."""
        B, T, D = x.shape
        BT = B * T
        H, HD = self.n_attn_heads, self.attn_head_dim

        synch_act = (alpha_act / torch.sqrt(beta_act)).to(x.dtype)
        attn_q = norm(self.attn_q_proj(synch_act).view(B, T, H, HD))
        obs = flash_attn.flash_attn_func(attn_q, attn_k, attn_v, causal=True)
        obs = obs.reshape(BT, D)

        new_state = state + self.synapses(torch.cat([obs, state + tick_emb], dim=-1))
        trace = torch.cat([trace[:, :, 1:], new_state.unsqueeze(-1)], dim=-1)

        h = F.glu(self.nlm1(trace), dim=-1)
        state = F.glu(self.nlm2(h), dim=-1).squeeze(-1)
        return state, trace

    def forward(self, x, dream=False, intervene=None, multi_tick=False, adaptive=False, ctm_cache=None, layer_idx=None):
        """
        Args:
            x: input tensor (B, T, D)
            dream: if True, collect per-iteration state deltas for convergence diagnostics
            intervene: optional callback fn(k, state, trace) -> state
            multi_tick: if True, return outputs at ALL K ticks for multi-tick loss
            adaptive: if True, stop early at confidence peak (delta starts rising)
            ctm_cache: CTMCache for persistent state across tokens (inference only)
            layer_idx: which layer this is (needed for ctm_cache indexing)
        """
        B, T, D = x.shape
        BT = B * T
        K = self.active_K
        H = self.n_attn_heads
        HD = self.attn_head_dim
        dtype = x.dtype  # match input dtype (bfloat16 on cuda)

        # Precompute keys and values from input (constant across ticks)
        attn_k = norm(self.attn_k_proj(x).view(B, T, H, HD))  # QK norm on keys
        attn_v = self.attn_v_proj(x).view(B, T, H, HD)

        # Initialize recurrent state: from cache (continuing thought) or learned params (new thought)
        cached = ctm_cache.layers[layer_idx] if ctm_cache is not None and ctm_cache.layers[layer_idx] is not None else None
        if cached is not None:
            # Cached state is (B_cached, *) — expand to all T positions per batch element
            # B_cached=1 for inference prefill, B_cached=B for chunked BPTT or multi-sample decode
            B_cached = cached['state'].shape[0]
            T_per = BT // B_cached
            state = cached['state'].unsqueeze(1).expand(B_cached, T_per, -1).reshape(BT, -1).to(dtype)
            trace = cached['trace'].unsqueeze(1).expand(B_cached, T_per, -1, -1).reshape(BT, self.D, self.M).clone().to(dtype)
            alpha_out = cached['alpha_out'].unsqueeze(1).expand(B_cached, T_per, -1).reshape(BT, -1).clone().to(dtype)
            beta_out = cached['beta_out'].unsqueeze(1).expand(B_cached, T_per, -1).reshape(BT, -1).clone().to(dtype)
            alpha_act = cached['alpha_act'].unsqueeze(1).expand(B_cached, T_per, -1).reshape(BT, -1).clone().to(dtype)
            beta_act = cached['beta_act'].unsqueeze(1).expand(B_cached, T_per, -1).reshape(BT, -1).clone().to(dtype)
        else:
            state = self.start_state.to(dtype).unsqueeze(0).expand(BT, -1)
            trace = self.start_trace.to(dtype).unsqueeze(0).expand(BT, -1, -1).clone()  # (BT, D, M)
            # Seed sync accumulators from start_state (paper: first tick uses raw pairwise product)
            left_out_init = state[:, self.synch_out_left]
            right_out_init = state[:, self.synch_out_right]
            pp_out = left_out_init * right_out_init
            alpha_out = pp_out
            beta_out = torch.ones_like(pp_out)
            left_act_init = state[:, self.synch_act_left]
            right_act_init = state[:, self.synch_act_right]
            pp_act = left_act_init * right_act_init
            alpha_act = pp_act
            beta_act = torch.ones_like(pp_act)

        # Dual synchronisation accumulators
        r_out = torch.exp(-self.decay_out.clamp(0, 15).to(dtype)).unsqueeze(0)
        r_act = torch.exp(-self.decay_act.clamp(0, 15).to(dtype)).unsqueeze(0)

        # Dopamine gating: scales how strongly this token's sync gets accumulated
        # dopamine > 1 = surprising token, remember harder. dopamine < 1 = boring, dampen.
        # Default 1.0 = no modulation (training, or inference without Engine dopamine tracking)
        dopamine = ctm_cache.dopamine if ctm_cache is not None else 1.0

        # Track per-iteration state deltas and intermediate outputs
        track_deltas = dream or adaptive
        deltas = [] if track_deltas else None
        tick_outputs = [] if multi_tick else None

        # Whether to use gradient checkpointing (saves memory for large K, ~30% slower)
        use_checkpoint = self.training and K > 4

        for k in range(K):
            prev_state = state

            # --- Core tick computation (optionally checkpointed) ---
            if use_checkpoint:
                tick_emb = self.tick_embed[k].unsqueeze(0).expand(BT, -1)
                state, trace = grad_checkpoint(
                    self._tick_core, alpha_act, beta_act, x, attn_k, attn_v, state, trace, tick_emb,
                    use_reentrant=False,
                )
            else:
                # Cross-attention: use S_action to generate query, re-observe input
                synch_act = (alpha_act / torch.sqrt(beta_act)).to(x.dtype)
                attn_q = norm(self.attn_q_proj(synch_act).view(B, T, H, HD))  # QK norm
                obs = flash_attn.flash_attn_func(attn_q, attn_k, attn_v, causal=True)
                obs = obs.reshape(BT, D)

                # U-NET synapses: mix observation with current state (residual for gradient flow)
                tick_emb = self.tick_embed[k].unsqueeze(0).expand(BT, -1)
                new_state = state + self.synapses(torch.cat([obs, state + tick_emb], dim=-1))  # (BT, D)

                # Update trace (rolling window: drop oldest, append newest)
                trace = torch.cat([trace[:, :, 1:], new_state.unsqueeze(-1)], dim=-1)

                # NLMs: per-neuron processing of trace history
                h = F.glu(self.nlm1(trace), dim=-1)       # (BT, D, hidden)
                state = F.glu(self.nlm2(h), dim=-1).squeeze(-1)  # (BT, D)

            # Track state delta (convergence / confidence signal)
            if track_deltas:
                delta = (state - prev_state).norm(dim=-1).mean()
                deltas.append(delta.item() if dream else delta)

            # Neuroplasticity hook
            if intervene is not None:
                modified = intervene(k, state, trace)
                if modified is not None:
                    state = modified

            # Update S_out synchronisation (for readout)
            left_out = state[:, self.synch_out_left]
            right_out = state[:, self.synch_out_right]
            pp_out = left_out * right_out * dopamine  # dopamine-gated
            alpha_out = r_out * alpha_out + pp_out
            beta_out = r_out * beta_out + 1

            # Update S_action synchronisation (for attention queries)
            left_act = state[:, self.synch_act_left]
            right_act = state[:, self.synch_act_right]
            pp_act = left_act * right_act * dopamine  # dopamine-gated
            alpha_act = r_act * alpha_act + pp_act
            beta_act = r_act * beta_act + 1

            # Multi-tick: save readout at each k for auxiliary loss
            if multi_tick:
                synch_k = alpha_out / torch.sqrt(beta_out)
                tick_outputs.append(self.c_proj(synch_k).reshape(B, T, D))

            # Adaptive: stop at confidence peak
            if adaptive and k >= 2 and track_deltas:
                if deltas[-1] > deltas[-2]:
                    break

        # Readout: S_out synchronisation -> output projection
        synch = alpha_out / torch.sqrt(beta_out)
        out = self.c_proj(synch).reshape(B, T, D)

        # ROADMAP (phase 5 - metacognitive tokens):
        # At this point we have per-token internal state that could be emitted as
        # special tokens into the stream: synch magnitude, state norm, which tick
        # was most certain, convergence rate. If the model learns to predict these
        # self-observation tokens, prediction error on them becomes a plasticity
        # signal richer than external prediction error alone. The model builds a
        # self-model alongside its world model, and the gap between expected and
        # actual internal state drives targeted synaptic updates.

        # Persist state for next token (stream of consciousness)
        # Only keep the last token's state — during prefill BT = B*T but we
        # only need the final position to seed the next decode step.
        if ctm_cache is not None:
            # Save last token's state per batch element: (BT, *) -> (B, T, *) -> (B, *)
            ctm_cache.layers[layer_idx] = {
                'state': state.view(B, T, -1)[:, -1, :],
                'trace': trace.view(B, T, self.D, self.M)[:, -1, :, :],
                'alpha_out': alpha_out.view(B, T, -1)[:, -1, :],
                'beta_out': beta_out.view(B, T, -1)[:, -1, :],
                'alpha_act': alpha_act.view(B, T, -1)[:, -1, :],
                'beta_act': beta_act.view(B, T, -1)[:, -1, :],
            }

        if dream:
            return out, deltas
        if multi_tick:
            return out, tick_outputs
        return out


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = CTMBlock(config) if config.use_ctm else MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, dream=False, multi_tick=False, adaptive=False, ctm_cache=None):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        if isinstance(self.mlp, CTMBlock) and (dream or multi_tick or adaptive):
            mlp_out, extras = self.mlp(norm(x), dream=dream, multi_tick=multi_tick, adaptive=adaptive,
                                       ctm_cache=ctm_cache, layer_idx=self.layer_idx)
            x = x + mlp_out
            return x, extras  # extras = deltas (dream) or tick_outputs (multi_tick)
        if isinstance(self.mlp, CTMBlock):
            x = x + self.mlp(norm(x), ctm_cache=ctm_cache, layer_idx=self.layer_idx)
        else:
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
        s = 3**0.5 * n_embd**-0.5 * 0.68 # sqrt(3) for Uniform std, 0.68x init scale (autoresearch #43)
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            if isinstance(block.mlp, CTMBlock):
                ctm = block.mlp
                # Cross-attention projections: uniform like c_q/c_k/c_v
                torch.nn.init.uniform_(ctm.attn_q_proj.weight, -s, s)
                torch.nn.init.uniform_(ctm.attn_k_proj.weight, -s, s)
                torch.nn.init.uniform_(ctm.attn_v_proj.weight, -s, s)
                # U-NET synapse model: uniform for Linear layers, standard for LayerNorm
                for module in ctm.synapses.modules():
                    if isinstance(module, Linear):
                        torch.nn.init.uniform_(module.weight, -s, s)
                    elif isinstance(module, nn.LayerNorm):
                        module.weight.fill_(1.0)
                        module.bias.zero_()
                # Output projection: zero for clean residual at init
                torch.nn.init.zeros_(ctm.c_proj.weight)
                # NLMs: Xavier-like per neuron
                for nlm in (ctm.nlm1, ctm.nlm2):
                    s_nlm = 1.0 / math.sqrt(nlm.w1.shape[0] + nlm.w1.shape[1])
                    nlm.w1.uniform_(-s_nlm, s_nlm)
                    nlm.b1.zero_()
                # Tick embeddings: normal init so ticks start different
                torch.nn.init.normal_(ctm.tick_embed, mean=0.0, std=0.02)
                # Start states
                s_d = 1.0 / math.sqrt(ctm.D)
                ctm.start_state.uniform_(-s_d, s_d)
                s_dm = 1.0 / math.sqrt(ctm.D + ctm.M)
                ctm.start_trace.uniform_(-s_dm, s_dm)
                # Dual synchronisation: random neuron pairings for S_out and S_action
                for buf_name in ('synch_out_left', 'synch_out_right', 'synch_act_left', 'synch_act_right'):
                    getattr(ctm, buf_name).copy_(torch.from_numpy(np.random.choice(ctm.D, size=ctm.n_synch, replace=True)))
                ctm.decay_out.zero_()
                ctm.decay_act.zero_()
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

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=200000, device=None):
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

    def set_elastic_anchor(self, elastic_weight=0.01):
        """Snapshot current plastic params as an anchor for elastic weight consolidation.

        During training, an L2 penalty pulls plastic params toward this snapshot,
        preventing catastrophic forgetting during phase transitions (e.g. BPTT phase 3).
        No extra forward pass needed - just stored tensors.

        Args:
            elastic_weight: L2 penalty weight. higher = more conservative learning.
        """
        self._elastic_weight = elastic_weight
        self._elastic_snapshot = {}
        if self.config.use_ctm:
            for block in self.transformer.h:
                if isinstance(block.mlp, CTMBlock):
                    for p in block.mlp.parameters():
                        if p.requires_grad:
                            self._elastic_snapshot[id(p)] = p.data.clone()

    def set_teacher(self, teacher_fn, distill_weight=0.5, temperature=2.0):
        """Set a teacher function for knowledge distillation during training.

        The teacher_fn takes token ids (B, T) and returns logits (B, T, V) in our
        vocab space. This can be:
        - A frozen local GPT model (same architecture)
        - An OllamaTeacher that calls an external model and maps to our vocab
        - Any callable that returns logits

        Args:
            teacher_fn: callable(idx) -> logits tensor (B, T, V)
            distill_weight: weight for KL term. 0 = no distillation.
            temperature: softmax temperature for distillation (higher = softer targets)
        """
        self._teacher_fn = teacher_fn
        self._distill_weight = distill_weight
        self._distill_temp = temperature

    def clear_teacher(self):
        """Remove teacher and disable distillation."""
        self._teacher_fn = None
        self._distill_weight = 0.0

    def clear_elastic_anchor(self):
        """Remove elastic anchor."""
        self._elastic_snapshot = {}
        self._elastic_weight = 0.0

    def _distill_loss(self, student_logits, idx):
        """Compute distillation + elastic loss.

        Returns (distill_loss, elastic_loss). Both 0 if no teacher/anchor set.
        """
        device = idx.device
        distill_loss = torch.tensor(0.0, device=device)
        elastic_loss = torch.tensor(0.0, device=device)

        # KL divergence from teacher
        dw = getattr(self, '_distill_weight', 0.0)
        if dw > 0 and hasattr(self, '_teacher_fn') and self._teacher_fn is not None:
            with torch.no_grad():
                teacher_logits = self._teacher_fn(idx)
            T = self._distill_temp
            teacher_log_probs = F.log_softmax(teacher_logits / T, dim=-1)
            student_log_probs = F.log_softmax(student_logits / T, dim=-1)
            teacher_probs = teacher_log_probs.exp()
            distill_loss = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1).mean() * (T * T)

        # Elastic anchoring toward snapshot
        ew = getattr(self, '_elastic_weight', 0.0)
        snapshot = getattr(self, '_elastic_snapshot', {})
        if ew > 0 and snapshot:
            penalty = 0.0
            for block in self.transformer.h:
                if isinstance(block.mlp, CTMBlock):
                    for p in block.mlp.parameters():
                        if id(p) in snapshot:
                            penalty = penalty + (p - snapshot[id(p)]).pow(2).sum()
            elastic_loss = penalty

        return distill_loss, elastic_loss

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
        # Filter out frozen params (requires_grad=False) for Phase 2 freeze
        all_h_params = [p for p in self.transformer.h.parameters() if p.requires_grad]
        matrix_params = [p for p in all_h_params if p.ndim == 2]
        ctm_other_params = [p for p in all_h_params if p.ndim != 2]  # SuperLinear 3D, decay 1D, start states, LayerNorm
        value_embeds_params = [p for p in self.value_embeds.parameters() if p.requires_grad]
        embedding_params = [p for p in self.transformer.wte.parameters() if p.requires_grad]
        lm_head_params = [p for p in self.lm_head.parameters() if p.requires_grad]
        resid_params = [self.resid_lambdas] if self.resid_lambdas.requires_grad else []
        x0_params = [self.x0_lambdas] if self.x0_lambdas.requires_grad else []
        all_trainable = [p for p in self.parameters() if p.requires_grad]
        assert len(all_trainable) == len(matrix_params) + len(ctm_other_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.003),
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

    def _compute_logits(self, x):
        """Shared logit computation: norm -> lm_head -> crop -> softcap."""
        softcap = 20
        logits = self.lm_head(norm(x))
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        return logits

    def forward(self, idx, targets=None, kv_cache=None, ctm_cache=None, loss_reduction='mean', pos_offset=0, diagnostics=None):
        """
        CTMCache is inference-only: during training, all T positions in a sequence process
        independently from start_state (like an MLP). This is intentional — training batches
        are randomly shuffled chunks with no temporal continuity between them. Inter-token
        communication during training comes from the cross-attention (causal, via S_action queries),
        not from sequential state flow. The cache enables cross-SEQUENCE continuity in inference
        (stream of consciousness), which doesn't exist in training.

        Dopamine gating is also inference-only (set by Engine): during training, the multi-tick
        loss already acts as an implicit dopamine signal — tokens where the model struggles get
        stronger gradients through the argmin tick selection. No separate modulation needed.
        """
        B, T = idx.size()
        use_multi_tick = self.config.use_ctm and targets is not None  # multi-tick loss during CTM training

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = pos_offset if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        # ROADMAP (phase 5): before embedding, interleave metacognitive tokens into idx.
        # These would encode the model's internal state from the previous forward pass
        # (certainty, sync magnitude, tick selection) as special token IDs. The model
        # learns to predict these alongside regular tokens, building a self-model.
        # Prediction error on self-tokens gates plasticity: the model updates synapses
        # most when its self-model is wrong (unexpected internal states = high learning).
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        last_layer_ticks = None
        n_layers = len(self.transformer.h)
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            if use_multi_tick and i == n_layers - 1:
                # Only collect multi-tick outputs from the last layer (closest to logits)
                # Save x before block so we can reconstruct full residual for each tick
                x_pre_block = x
                result = block(x, ve, cos_sin, self.window_sizes[i], kv_cache, multi_tick=True, ctm_cache=ctm_cache)
                x, last_layer_ticks = result
                # tick_outputs are MLP outputs (B,T,D). Build full residual for each:
                # x_pre_block + attn_out + tick_mlp_out. Since x = x_pre_block + attn + final_mlp,
                # and we know final_mlp = x - x_pre_block - attn, it's easier to just use:
                # x_at_tick_k = x - final_mlp_out + tick_mlp_out
                final_mlp_out = last_layer_ticks[-1]  # last tick = same as what block used
                last_layer_ticks = [x - final_mlp_out + tick_out for tick_out in last_layer_ticks]
            else:
                x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache, ctm_cache=ctm_cache)

        if targets is not None and use_multi_tick and last_layer_ticks:
            # Multi-tick loss (CTM paper Listing 4): per-token tick selection
            # Each token independently picks its best tick (argmin loss) and
            # most certain tick (argmax certainty), matching paper's per-sample approach.
            # Skip computing logits on final x -- we compute per-tick logits below.
            flat_targets = targets.view(-1)  # (B*T,)
            n_tokens = flat_targets.size(0)
            V = self.config.vocab_size
            max_entropy = math.log(V)

            # Collect per-token unreduced losses and certainties for each tick
            # all_losses: (K, B*T), all_certainties: (K, B*T)
            all_losses = []
            all_certainties = []
            for tick_out in last_layer_ticks:
                tick_logits = self._compute_logits(tick_out)
                tick_logits_flat = tick_logits.view(-1, V)
                # Unreduced CE loss per token
                token_losses = F.cross_entropy(tick_logits_flat, flat_targets, ignore_index=-1, reduction='none')
                all_losses.append(token_losses)
                # Certainty = 1 - normalized_entropy per token
                with torch.no_grad():
                    probs = F.softmax(tick_logits_flat, dim=-1)
                    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
                    certainty = 1.0 - entropy / max_entropy
                all_certainties.append(certainty)

            all_losses = torch.stack(all_losses, dim=0)        # (K, B*T)
            all_certainties = torch.stack(all_certainties, dim=0)  # (K, B*T)

            # Per-token: select tick with lowest loss and tick with highest certainty
            token_idx = torch.arange(n_tokens, device=all_losses.device)
            best_tick_idx = all_losses.argmin(dim=0)     # (B*T,)
            certain_tick_idx = all_certainties.argmax(dim=0)  # (B*T,)

            loss_argmin = all_losses[best_tick_idx, token_idx].mean()
            loss_argmax_cert = all_losses[certain_tick_idx, token_idx].mean()
            task_loss = 0.5 * loss_argmin + 0.5 * loss_argmax_cert

            # Populate diagnostics if requested
            if diagnostics is not None:
                with torch.no_grad():
                    K = all_losses.size(0)
                    # Per-tick mean loss and certainty
                    for k in range(K):
                        diagnostics[f'tick_{k}/loss'] = all_losses[k].mean().item()
                        diagnostics[f'tick_{k}/certainty'] = all_certainties[k].mean().item()
                    # Which tick wins most often (tick selection distribution)
                    tick_counts = torch.bincount(best_tick_idx, minlength=K).float()
                    for k in range(K):
                        diagnostics[f'tick_{k}/selected_pct'] = (tick_counts[k] / n_tokens * 100).item()
                    # Overall certainty stats
                    final_certainty = all_certainties[-1]
                    diagnostics['certainty/mean'] = final_certainty.mean().item()
                    diagnostics['certainty/std'] = final_certainty.std().item()
                    diagnostics['certainty/min'] = final_certainty.min().item()
                    diagnostics['certainty/max'] = final_certainty.max().item()
                    diagnostics['loss/argmin'] = loss_argmin.item()
                    diagnostics['loss/argmax_cert'] = loss_argmax_cert.item()

            # Distillation: use the last tick's logits as the student output
            distill_loss, elastic_loss = self._distill_loss(self._compute_logits(last_layer_ticks[-1]), idx)
            dw = getattr(self, '_distill_weight', 0.0)
            ew = getattr(self, '_elastic_weight', 0.0)
            total_loss = (1.0 - dw) * task_loss + dw * distill_loss + ew * elastic_loss
            if diagnostics is not None:
                diagnostics['loss/task'] = task_loss.item()
                diagnostics['loss/distill'] = distill_loss.item()
                diagnostics['loss/elastic'] = elastic_loss.item()
            return total_loss

        # Standard path: compute logits once
        logits = self._compute_logits(x)
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            distill_loss, elastic_loss = self._distill_loss(logits, idx)
            dw = getattr(self, '_distill_weight', 0.0)
            ew = getattr(self, '_elastic_weight', 0.0)
            total_loss = (1.0 - dw) * ce_loss + dw * distill_loss + ew * elastic_loss
            if diagnostics is not None:
                diagnostics['loss/ce'] = ce_loss.item()
                diagnostics['loss/distill'] = distill_loss.item()
                diagnostics['loss/elastic'] = elastic_loss.item()
            return total_loss
        return logits

    def forward_chunked_bptt(self, idx, targets, n_chunks=8, loss_scale=1.0):
        """Phase 3: chunked backpropagation through time for CTM state continuity.

        Splits the sequence into n_chunks, processes sequentially with CTM state
        carried between chunks. Calls backward() per chunk so activations are
        freed immediately (real memory savings). Gradients are detached at chunk
        boundaries (truncated BPTT) to bound memory.

        Attention context is local to each chunk (no KV cache carry). Cross-chunk
        information flows only through CTM state -- which is exactly what we're training.

        Args:
            idx: token indices (B, T)
            targets: target token indices (B, T)
            n_chunks: number of chunks to split the sequence into
            loss_scale: multiply loss before backward (e.g. 1/grad_accum_steps)

        Returns:
            float: mean loss across chunks (for logging). Gradients already on .grad.
        """
        B, T = idx.size()
        assert T % n_chunks == 0, f"seq_len {T} not divisible by n_chunks {n_chunks}"
        C = T // n_chunks

        ctm_cache = CTMCache(self.config.n_layer)
        total_loss = 0.0

        for chunk_idx in range(n_chunks):
            start = chunk_idx * C
            end = start + C
            chunk_loss = self.forward(
                idx[:, start:end],
                targets[:, start:end],
                ctm_cache=ctm_cache,
                pos_offset=start,
            )
            # Backward per chunk so activations are freed immediately.
            # Scale by 1/n_chunks for mean, and by loss_scale for grad accumulation.
            (chunk_loss * (loss_scale / n_chunks)).backward()
            total_loss += chunk_loss.item()

            # Truncated BPTT: detach state at chunk boundaries to cut gradient flow
            if chunk_idx < n_chunks - 1:
                for layer_state in ctm_cache.layers:
                    if layer_state is not None:
                        for key in layer_state:
                            layer_state[key] = layer_state[key].detach()

        # Return scalar for logging (gradients already accumulated on .grad)
        return total_loss / n_chunks

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

            # NOTE: we no longer reduce active_K. The model was trained with full K
            # and reducing it at inference/sleep degrades quality. We still track
            # convergence for diagnostics, but don't act on it.
            # block.mlp.active_K = new_k
            results[i] = {
                'deltas': deltas,
                'old_k': old_k,
                'new_k': new_k,
                'converged': new_k < max_k,
            }

        return results

    def consolidate(self, replay_batches, lr=1e-4, steps=4):
        """Sleep consolidation: certainty-weighted self-distillation on replay buffer.

        Brain analog: during slow-wave sleep, the brain replays recent experiences and
        selectively strengthens synapses that "know what they know" while weakening
        uncertain ones. No external reward needed — certainty IS the reward signal.

        The loss: for each token at each tick, compute certainty (1 - normalized entropy).
        Upweight loss on tokens where model is certain AND correct (reinforce confident knowledge).
        Downweight loss on tokens where model is uncertain (don't consolidate confusion).

        Only updates synapse weights (U-NET) and NLM weights — the "synaptic" parameters.
        Attention, embeddings, and projection weights are frozen during consolidation.

        Args:
            replay_batches: list of (x, y) tensor pairs from the replay buffer
            lr: learning rate for consolidation (should be small — gentle nudges)
            steps: number of gradient steps per replay batch

        Returns: dict with consolidation stats
        """
        if not self.config.use_ctm:
            return {}

        # Collect only synapse/NLM params — the "plastic" parameters
        plastic_params = set()
        for block in self.transformer.h:
            if not isinstance(block.mlp, CTMBlock):
                continue
            ctm = block.mlp
            plastic_params.update(p for p in ctm.synapses.parameters() if p.requires_grad)
            plastic_params.update(p for p in ctm.nlm1.parameters() if p.requires_grad)
            plastic_params.update(p for p in ctm.nlm2.parameters() if p.requires_grad)
        plastic_params = list(plastic_params)

        if not plastic_params:
            return {}

        # Freeze non-plastic params to avoid wasted gradient computation
        plastic_set = set(id(p) for p in plastic_params)
        frozen = []
        for p in self.parameters():
            if p.requires_grad and id(p) not in plastic_set:
                p.requires_grad_(False)
                frozen.append(p)

        try:
            optimizer = torch.optim.AdamW(plastic_params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
            # eval() avoids triggering gradient checkpointing in CTMBlock._tick_core
            # (use_checkpoint = self.training and K > 4). We still get gradients because
            # inference_mode is not set and requires_grad is True on plastic params.
            self.eval()
            V = self.config.vocab_size
            max_entropy = math.log(V)
            stats = {'losses': [], 'mean_certainty': [], 'steps': 0}

            for x, y in replay_batches:
                for _ in range(steps):
                    optimizer.zero_grad()

                    logits = self.forward(x, targets=None)
                    flat_y = y.view(-1)
                    logits_flat = logits.view(-1, logits.size(-1))

                    # Per-token certainty from final output
                    with torch.no_grad():
                        probs = F.softmax(logits_flat, dim=-1)
                        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # (B*T,)
                        certainty = (1.0 - entropy / max_entropy).clamp(0, 1)  # (B*T,)
                        correct = (logits_flat.argmax(dim=-1) == flat_y).float()  # (B*T,)
                        weight = certainty * correct  # (B*T,)
                        weight = weight / (weight.sum() + 1e-8) * weight.numel()

                    token_losses = F.cross_entropy(logits_flat, flat_y, ignore_index=-1, reduction='none')
                    loss = (token_losses * weight).mean()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(plastic_params, 1.0)
                    optimizer.step()

                    stats['losses'].append(loss.item())
                    stats['mean_certainty'].append(certainty.mean().item())
                    stats['steps'] += 1
        finally:
            # Always re-enable gradients on frozen params
            for p in frozen:
                p.requires_grad_(True)
            # Zero all grads so the main optimizer doesn't see stale consolidation gradients
            for p in self.parameters():
                if p.grad is not None:
                    p.grad = None

        return stats

    def compact_memory(self, ctm_cache, lr=1e-5):
        """Compact inference-time memory (CTMCache) into permanent synapse weights.

        Brain analog: hippocampal → cortical memory transfer. During inference, the
        CTMCache accumulates sync statistics (alpha/beta) that encode which neuron pairs
        have been consistently co-activating. This method writes those patterns into the
        synapse weights permanently — the model literally remembers what it experienced.

        Three mechanisms work together:
        1. Sync-driven Hebbian update: the accumulated S_out sync signal (alpha/sqrt(beta))
           encodes pairwise co-activation history. We use this as a teaching signal for
           the synapse weights — not a crude outer product, but the actual running statistics
           the model has been computing all along.
        2. Novelty gating: compare accumulated sync to "baseline" sync (from start state).
           Only update where the sync has diverged significantly — don't waste plasticity
           on patterns the model already knows.
        3. Homeostasis: after each update, renormalize synapse layer weights to prevent
           unbounded drift. Weight magnitude stays stable, only direction changes.

        Args:
            ctm_cache: CTMCache from a completed inference session
            lr: plasticity learning rate (should be tiny — 1e-5 to 1e-4)

        Returns: dict with per-layer plasticity stats
        """
        if ctm_cache is None or not self.config.use_ctm:
            return {}

        stats = {}
        with torch.no_grad():
            for i, block in enumerate(self.transformer.h):
                if not isinstance(block.mlp, CTMBlock):
                    continue
                cached = ctm_cache.layers[i]
                if cached is None:
                    continue

                ctm = block.mlp
                alpha_out = cached['alpha_out']   # (BT, n_synch)
                beta_out = cached['beta_out']     # (BT, n_synch)
                state = cached['state']           # (BT, D)

                # 1. Compute accumulated sync signal (what the model learned during inference)
                synch_accumulated = alpha_out / torch.sqrt(beta_out)  # (BT, n_synch)

                # Compute baseline sync: what start_state produces after K iterations of sync accumulation
                # with no input variation. This is what the sync "resting state" looks like.
                # For K iterations with constant pairwise products and decay r:
                #   alpha_K = pp * (1 + r + r^2 + ... + r^(K-1)) = pp * (1 - r^K) / (1 - r + eps)
                #   beta_K = 1 + r + r^2 + ... + r^(K-1) = (1 - r^K) / (1 - r + eps)
                #   sync_baseline = alpha_K / sqrt(beta_K) = pp * sqrt(beta_K)
                r_out = torch.exp(-ctm.decay_out.clamp(0, 15))  # (n_synch,)
                K = ctm.active_K
                baseline_left = ctm.start_state[ctm.synch_out_left]   # (n_synch,)
                baseline_right = ctm.start_state[ctm.synch_out_right]  # (n_synch,)
                pp_baseline = baseline_left * baseline_right  # (n_synch,)
                # Geometric series: sum_{k=0}^{K-1} r^k = (1 - r^K) / (1 - r)
                beta_baseline = (1 - r_out.pow(K)) / (1 - r_out + 1e-8)  # (n_synch,)
                synch_baseline = pp_baseline * torch.sqrt(beta_baseline)  # (n_synch,)

                # 2. Novelty: how much did sync diverge from baseline?
                # Mean across batch dimension, compare to baseline
                synch_mean = synch_accumulated.mean(dim=0)  # (n_synch,)
                sync_delta = synch_mean - synch_baseline  # (n_synch,)
                novelty = sync_delta.abs()

                # Gate: only update where novelty exceeds median (top 50% most novel pairings)
                # ROADMAP (phase 5): replace this hardcoded threshold with a learned gate.
                # If the model can predict its own sync patterns via metacognitive tokens,
                # novelty = prediction error on self-observation, not deviation from a
                # static baseline. The model decides what's surprising based on its own
                # self-model, making plasticity adaptive and context-dependent.
                novelty_threshold = novelty.median()
                gate = (novelty > novelty_threshold).float()  # (n_synch,)
                gated_delta = sync_delta * gate  # (n_synch,)

                # 3. Capture base norms BEFORE any updates (for homeostasis)
                updated_layers = [ctm.c_proj]
                last_up = ctm.synapses.up_layers[-1]
                if hasattr(last_up, 'weight'):
                    updated_layers.append(last_up)
                base_norms = {}
                for layer in updated_layers:
                    if hasattr(layer, 'weight'):
                        if not hasattr(layer, '_plasticity_base_norm'):
                            layer._plasticity_base_norm = layer.weight.data.norm().clone()
                        base_norms[id(layer)] = layer._plasticity_base_norm

                # 4. Apply gated sync delta to c_proj
                # c_proj maps n_synch -> D. Nudge it to amplify novel sync channels.
                n_synch_dim, D = ctm.c_proj.weight.shape[1], ctm.c_proj.weight.shape[0]
                state_scale = state.abs().mean()
                update = gated_delta.unsqueeze(0) * state_scale  # (1, n_synch) broadcast
                ctm.c_proj.weight.data += lr * update.expand(D, -1).to(ctm.c_proj.weight.dtype)

                # Also nudge last synapse up-layer using state delta from baseline
                baseline_state = ctm.start_state.unsqueeze(0).expand_as(state)
                state_delta = (state - baseline_state).mean(dim=0)  # (D,)
                if hasattr(last_up, 'weight'):
                    D_out, D_in = last_up.weight.shape
                    input_approx = last_up.weight.data.mean(dim=0)  # (D_in,)
                    rank1_update = state_delta[:D_out].unsqueeze(1) * input_approx.unsqueeze(0)
                    last_up.weight.data += lr * rank1_update.to(last_up.weight.dtype)

                # 5. Homeostasis: clamp weight norms to at most 1% growth from base
                for layer in updated_layers:
                    if hasattr(layer, 'weight'):
                        w = layer.weight.data
                        current_norm = w.norm()
                        max_norm = base_norms[id(layer)] * 1.01
                        if current_norm > max_norm:
                            w.mul_(max_norm / current_norm)

                stats[i] = {
                    'mean_novelty': novelty.mean().item(),
                    'gated_fraction': gate.mean().item(),
                    'sync_delta_norm': sync_delta.norm().item(),
                    'state_delta_norm': state_delta.norm().item(),
                }

        return stats

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
        # NOTE: this naive generate recomputes full sequence each step (no KV cache).
        # CTMCache is NOT used here because state dims depend on BT which changes each step.
        # For proper CTM state persistence, use Engine.generate which does token-by-token with caches.
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
