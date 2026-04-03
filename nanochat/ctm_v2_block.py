"""CTM v2 Block — drop-in replacement for CTMBlock with 4 brain regions.

Replaces the single flat neuron space with 4 specialized sub-populations:
    INPUT (V1/S1)   — encodes cross-attention observations, shallow NLM, short memory
    ATTENTION (thal) — routes via sync query, deep NLM, sparsity-gated, no broadcast
    OUTPUT (assoc)   — accumulates evidence, longest memory, carries sync across tokens
    MOTOR (M1)       — produces c_proj output, shallow NLM, decisive

Key differences from CTMBlock:
    1. Per-region NLMs with different depths and memory lengths
    2. Inter-region synapses (learned routing between brain areas)
    3. Global broadcast (compressed full state) to non-attention layers
    4. Per-region Hebbian plasticity with different learning rates
    5. Selective cache persistence: only output sync carries across tokens
    6. Attention sparsity loss for thalamic gating

Same interface: (B, T, D) in, (B, T, D) out. Drop-in for qwen_backbone.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional

from nanochat.gpt import norm, SuperLinear, SynapseUNET, Linear

try:
    import flash_attn
    _has_flash = True
except ImportError:
    _has_flash = False

try:
    from nanochat.triton_sync import fused_sync_update
    _has_fused_sync = True
except ImportError:
    _has_fused_sync = False


@dataclass
class RegionConfig:
    """Config for one brain region."""
    n_neurons: int
    memory_length: int
    nlm_depth: int  # 1=shallow, 2=deep
    nlm_hidden: int = 32
    hebbian_lr: float = 0.01
    receives_broadcast: bool = True
    sparsity_weight: float = 0.0  # >0 = L1 sparsity on activations


class RegionNLM(nn.Module):
    """Per-neuron trace processor for a brain region."""

    def __init__(self, cfg: RegionConfig):
        super().__init__()
        self.cfg = cfg
        n = cfg.n_neurons
        m = cfg.memory_length

        if cfg.nlm_depth >= 2:
            self.nlm1 = SuperLinear(m, 2 * cfg.nlm_hidden, n)
            self.nlm2 = SuperLinear(cfg.nlm_hidden, 2, n)
        else:
            self.nlm1 = SuperLinear(m, 2, n)
            self.nlm2 = None

        self.register_parameter(
            'start_state',
            nn.Parameter(torch.empty(n)))
        self.register_parameter(
            'start_trace',
            nn.Parameter(torch.empty(n, m)))

    def init_weights(self):
        n = self.cfg.n_neurons
        m = self.cfg.memory_length
        self.start_state.data.uniform_(-math.sqrt(1/n), math.sqrt(1/n))
        self.start_trace.data.uniform_(
            -math.sqrt(1/(n+m)), math.sqrt(1/(n+m)))
        # Init NLM weights (nanochat SuperLinear uses w1, not weight)
        for mod in self.modules():
            if isinstance(mod, SuperLinear):
                s = 1.0 / math.sqrt(mod.w1.shape[0] + mod.w1.shape[1])
                mod.w1.data.uniform_(-s, s)
                mod.b1.data.zero_()

    def process(self, trace: torch.Tensor) -> torch.Tensor:
        """Run NLM on trace. trace: (BT, n_neurons, memory_length)"""
        h = F.glu(self.nlm1(trace), dim=-1)
        if self.nlm2 is not None:
            state = F.glu(self.nlm2(h), dim=-1).squeeze(-1)
        else:
            state = h.squeeze(-1)
        return state


class CTMv2Block(nn.Module):
    """CTM v2 Block with 4 brain regions.

    Drop-in replacement for CTMBlock. Same forward signature.
    """

    def __init__(self, config):
        super().__init__()
        D = config.n_embd
        K = config.ctm_iterations
        n_synch = config.ctm_n_synch
        n_attn_heads = config.ctm_n_attn_heads

        self.D = D
        self.K = K
        self.active_K = K
        self.n_synch = n_synch
        self.n_attn_heads = n_attn_heads
        self.attn_head_dim = D // n_attn_heads

        # Per-region tick counts (brain oscillation timescales)
        # Input (gamma, 30-100Hz):  fast sensory encoding
        # Attention (alpha, 8-13Hz): routing deliberation — needs MOST ticks
        # Output (theta, 4-8Hz):    deep thinking, evidence accumulation
        # Motor (gamma):            fast execution, fires at end
        self.ticks_per_region = {
            'input': max(2, K // 8),       # gamma: fast, freeze early
            'attention': K,                 # alpha: full deliberation
            'output': K,                    # theta: full accumulation
            'motor': max(2, K // 8),        # gamma: fast, fires at end
        }
        # Total ticks = max across regions (regions with fewer ticks freeze early)
        self.total_ticks = max(self.ticks_per_region.values())

        # Brain region configs — neurons sum to D
        n_in = D // 5
        n_attn = int(D * 0.3)
        n_out = int(D * 0.3)
        n_motor = D - n_in - n_attn - n_out

        M_base = config.ctm_memory_length
        nlm_h = config.ctm_memory_hidden if config.ctm_memory_hidden > 0 else n_synch // 4

        self.regions = nn.ModuleDict({
            'input': RegionNLM(RegionConfig(
                n_neurons=n_in, memory_length=max(4, M_base // 2),
                nlm_depth=1, nlm_hidden=nlm_h,
                hebbian_lr=0.02, receives_broadcast=True)),
            'attention': RegionNLM(RegionConfig(
                n_neurons=n_attn, memory_length=max(128, M_base * 16),
                nlm_depth=2, nlm_hidden=nlm_h,
                hebbian_lr=0.01, receives_broadcast=False,
                sparsity_weight=0.05)),
            'output': RegionNLM(RegionConfig(
                n_neurons=n_out, memory_length=min(32, M_base * 2),
                nlm_depth=2, nlm_hidden=nlm_h,
                hebbian_lr=0.005, receives_broadcast=True)),
            'motor': RegionNLM(RegionConfig(
                n_neurons=n_motor, memory_length=max(4, M_base // 2),
                nlm_depth=1, nlm_hidden=nlm_h,
                hebbian_lr=0.005, receives_broadcast=True)),
        })
        self.region_names = ['input', 'attention', 'output', 'motor']
        self.region_sizes = {'input': n_in, 'attention': n_attn,
                             'output': n_out, 'motor': n_motor}
        self.region_offsets = {}
        off = 0
        for name in self.region_names:
            self.region_offsets[name] = off
            off += self.region_sizes[name]

        # Cross-attention (uses attention region's sync for queries)
        self.attn_q_proj = Linear(n_synch, D, bias=False)
        self.attn_k_proj = Linear(D, D, bias=False)
        self.attn_v_proj = Linear(D, D, bias=False)

        # Global broadcast: compressed full state for non-attention layers
        g = D // 4
        self.global_dim = g
        self.global_proj = nn.Sequential(
            Linear(D, g * 2, bias=False),
            nn.GLU(),
        )

        # Inter-region synapses
        # obs(D) + motor(n_motor) + global(g) → input(n_in)
        self.syn_to_input = nn.Sequential(
            Linear(D + n_motor + g, n_in * 2, bias=False),
            nn.GLU(), nn.LayerNorm(n_in))
        # input(n_in) + obs(D) → attention(n_attn) [NO broadcast]
        self.syn_to_attn = nn.Sequential(
            Linear(n_in + D, n_attn * 2, bias=False),
            nn.GLU(), nn.LayerNorm(n_attn))
        # attention(n_attn) + obs(D) + global(g) → output(n_out)
        self.syn_to_output = nn.Sequential(
            Linear(n_attn + D + g, n_out * 2, bias=False),
            nn.GLU(), nn.LayerNorm(n_out))
        # output(n_out) + global(g) → motor(n_motor)
        self.syn_to_motor = nn.Sequential(
            Linear(n_out + g, n_motor * 2, bias=False),
            nn.GLU(), nn.LayerNorm(n_motor))

        # Per-tick embedding (sized to total_ticks, not K)
        self.tick_embed = nn.Parameter(torch.empty(self.total_ticks, D))

        # Dual synchronisation (same as v1 — operates on full D)
        for name in ('synch_out_left', 'synch_out_right',
                     'synch_act_left', 'synch_act_right'):
            self.register_buffer(name, torch.zeros(n_synch, dtype=torch.long))
        self.decay_out = nn.Parameter(torch.zeros(n_synch))
        self.decay_act = nn.Parameter(torch.zeros(n_synch))

        # Output projection
        self.c_proj = Linear(n_synch, D, bias=False)

        # Per-region Hebbian baselines
        for rname in self.region_names:
            self.register_buffer(f'_baseline_{rname}',
                                 torch.zeros(self.region_sizes[rname]))
            self.register_buffer(f'_running_mean_{rname}',
                                 torch.zeros(self.region_sizes[rname]))
        self.register_buffer('_hebbian_calibrated', torch.tensor(False))
        self._hebbian_active = False

        # Bound-guided aux weights
        self.register_buffer('_tick_aux_weights', torch.ones(K) / K)

        # Sparsity loss accumulator
        self._sparsity_loss = 0.0

    def init_weights(self):
        """Initialize weights — called by GPT.init_weights()."""
        for region in self.regions.values():
            region.init_weights()

        # Init all Linear and LayerNorm in synapse modules
        s = 3**0.5 * self.D**-0.5 * 0.68
        for mod in self.modules():
            if isinstance(mod, Linear) and mod is not self.c_proj:
                nn.init.uniform_(mod.weight, -s, s)
            elif isinstance(mod, nn.LayerNorm):
                mod.weight.data.fill_(1.0)
                if mod.bias is not None:
                    mod.bias.data.zero_()

        # Global projector
        for submod in self.global_proj.modules():
            if isinstance(submod, Linear):
                nn.init.uniform_(submod.weight, -s, s)

        # Sync neuron indices (random pairing over full D)
        for name in ('synch_out', 'synch_act'):
            left = torch.randperm(self.D)[:self.n_synch]
            right = torch.randperm(self.D)[:self.n_synch]
            getattr(self, f'{name}_left').copy_(left)
            getattr(self, f'{name}_right').copy_(right)

        # Tick embed
        nn.init.normal_(self.tick_embed, std=0.02)

        # Attention projections
        for mod in (self.attn_q_proj, self.attn_k_proj, self.attn_v_proj):
            nn.init.uniform_(mod.weight, -s, s)

        # c_proj starts at zero so CTM is initially a no-op
        nn.init.zeros_(self.c_proj.weight)

        # Decay params
        self.decay_out.data.zero_()
        self.decay_act.data.zero_()

    def _sync_readout(self, alpha, beta):
        return alpha / torch.sqrt(beta)

    def _get_region_state(self, full_state, region_name):
        """Slice region's neurons from full state."""
        off = self.region_offsets[region_name]
        n = self.region_sizes[region_name]
        return full_state[:, off:off+n]

    def _set_region_state(self, full_state, region_name, value):
        """Write region's neurons into full state."""
        off = self.region_offsets[region_name]
        n = self.region_sizes[region_name]
        full_state = full_state.clone()
        full_state[:, off:off+n] = value
        return full_state

    def forward(self, x, dream=False, intervene=None, multi_tick=False,
                adaptive=False, ctm_cache=None, layer_idx=None):
        B, T, D = x.shape
        BT = B * T
        K = self.active_K
        H, HD = self.n_attn_heads, self.attn_head_dim
        dtype = x.dtype

        # Precompute K,V
        attn_k = norm(self.attn_k_proj(x).view(B, T, H, HD))
        attn_v = self.attn_v_proj(x).view(B, T, H, HD)

        # Initialize state — per-region start states concatenated to full D
        cached = (ctm_cache.layers[layer_idx]
                  if ctm_cache is not None and ctm_cache.layers[layer_idx] is not None
                  else None)

        if cached is not None and self.training:
            # Training: inherit everything from cache
            B_c = cached['state'].shape[0]
            T_per = BT // B_c
            state = cached['state'].unsqueeze(1).expand(B_c, T_per, -1).reshape(BT, -1).to(dtype)
            alpha_out = cached['alpha_out'].unsqueeze(1).expand(B_c, T_per, -1).reshape(BT, -1).clone().to(dtype)
            beta_out = cached['beta_out'].unsqueeze(1).expand(B_c, T_per, -1).reshape(BT, -1).clone().to(dtype)
            alpha_act = cached['alpha_act'].unsqueeze(1).expand(B_c, T_per, -1).reshape(BT, -1).clone().to(dtype)
            beta_act = cached['beta_act'].unsqueeze(1).expand(B_c, T_per, -1).reshape(BT, -1).clone().to(dtype)
        elif cached is not None:
            # Inference: fresh state, carry ONLY output layer's sync
            state = self._build_start_state(BT, dtype)
            # Carry sync (memory lives in sync accumulators)
            B_c = cached['alpha_out'].shape[0]
            T_per = BT // B_c
            alpha_out = cached['alpha_out'].unsqueeze(1).expand(B_c, T_per, -1).reshape(BT, -1).clone().to(dtype)
            beta_out = cached['beta_out'].unsqueeze(1).expand(B_c, T_per, -1).reshape(BT, -1).clone().to(dtype)
            alpha_act = cached['alpha_act'].unsqueeze(1).expand(B_c, T_per, -1).reshape(BT, -1).clone().to(dtype)
            beta_act = cached['beta_act'].unsqueeze(1).expand(B_c, T_per, -1).reshape(BT, -1).clone().to(dtype)
            if not (alpha_out.isfinite().all() and beta_out.isfinite().all()):
                cached = None
        if cached is None:
            state = self._build_start_state(BT, dtype)
            # Fresh sync seed
            left_out = state[:, self.synch_out_left]
            right_out = state[:, self.synch_out_right]
            alpha_out = left_out * right_out
            beta_out = torch.ones_like(alpha_out)
            left_act = state[:, self.synch_act_left]
            right_act = state[:, self.synch_act_right]
            alpha_act = left_act * right_act
            beta_act = torch.ones_like(alpha_act)

        # Per-region traces
        traces = {}
        for rname, region in self.regions.items():
            m = region.cfg.memory_length
            n = region.cfg.n_neurons
            traces[rname] = region.start_trace.to(dtype).unsqueeze(0).expand(BT, -1, -1).clone()

        r_out = torch.exp(-self.decay_out.clamp(0, 15).to(dtype)).unsqueeze(0)
        r_act = torch.exp(-self.decay_act.clamp(0, 15).to(dtype)).unsqueeze(0)

        dopamine = ctm_cache.dopamine if ctm_cache is not None else 1.0
        if isinstance(dopamine, torch.Tensor) and dopamine.dim() >= 1:
            dopamine = dopamine.view(BT, 1)

        deltas = [] if dream else None
        tick_outputs = [] if multi_tick else None
        sparsity_loss = 0.0

        # Per-region tick limits
        k_input = self.ticks_per_region['input']
        k_attn = self.ticks_per_region['attention']
        k_output = self.ticks_per_region['output']
        k_motor = self.ticks_per_region['motor']

        for k in range(self.total_ticks):
            prev_state = state

            # Global broadcast (always runs)
            global_ctx = self.global_proj(state)

            # Extract per-region activations
            act_in = self._get_region_state(state, 'input')
            act_attn = self._get_region_state(state, 'attention')
            act_out = self._get_region_state(state, 'output')
            act_motor = self._get_region_state(state, 'motor')

            # Cross-attention (always runs — obs feeds active regions)
            synch_act = self._sync_readout(alpha_act, beta_act).to(dtype)
            attn_q = norm(self.attn_q_proj(synch_act).view(B, T, H, HD))
            if _has_flash:
                obs = flash_attn.flash_attn_func(
                    attn_q, attn_k, attn_v, causal=True).reshape(BT, D)
            else:
                q = attn_q.reshape(B * T, H, 1, HD)
                k_t = attn_k.reshape(B * T, H, -1, HD)
                v_t = attn_v.reshape(B * T, H, -1, HD)
                attn_w = (q @ k_t.transpose(-2, -1)) / math.sqrt(HD)
                obs = (F.softmax(attn_w, dim=-1) @ v_t).reshape(BT, D)

            # 1. INPUT (gamma): fast encoding, freezes after k_input ticks
            if k < k_input:
                pre_in = self.syn_to_input(
                    torch.cat([obs, act_motor, global_ctx], dim=1))
                new_in = act_in + pre_in
                traces['input'] = torch.cat(
                    [traces['input'][:, :, 1:], new_in.unsqueeze(-1)], dim=-1)
                act_in = self.regions['input'].process(traces['input'])

            # 2. ATTENTION (alpha): full deliberation, builds discriminative sync
            if k < k_attn:
                pre_attn = self.syn_to_attn(
                    torch.cat([act_in, obs], dim=1))
                new_attn = act_attn + pre_attn
                traces['attention'] = torch.cat(
                    [traces['attention'][:, :, 1:], new_attn.unsqueeze(-1)], dim=-1)
                act_attn = self.regions['attention'].process(traces['attention'])
                sparsity_loss = sparsity_loss + self.regions['attention'].cfg.sparsity_weight * act_attn.abs().mean()

            # 3. OUTPUT (theta): evidence accumulation over full ticks
            if k < k_output:
                pre_out = self.syn_to_output(
                    torch.cat([act_attn, obs, global_ctx], dim=1))
                new_out = act_out + pre_out
                traces['output'] = torch.cat(
                    [traces['output'][:, :, 1:], new_out.unsqueeze(-1)], dim=-1)
                act_out = self.regions['output'].process(traces['output'])

            # 4. MOTOR (gamma): fires in LAST k_motor ticks only (decision time)
            if k >= self.total_ticks - k_motor:
                pre_motor = self.syn_to_motor(
                    torch.cat([act_out, global_ctx], dim=1))
                new_motor = act_motor + pre_motor
                traces['motor'] = torch.cat(
                    [traces['motor'][:, :, 1:], new_motor.unsqueeze(-1)], dim=-1)
                act_motor = self.regions['motor'].process(traces['motor'])

            # Reconstruct full state from regions
            state = torch.cat([act_in, act_attn, act_out, act_motor], dim=1)

            # Hebbian homeostatic correction (per-region, inference only)
            if self._hebbian_active and not self.training:
                state = self._apply_hebbian(state)

            # Track delta
            if dream:
                delta = (state - prev_state).norm(dim=-1).mean()
                deltas.append(delta.item())

            if intervene is not None:
                modified = intervene(k, state, traces)
                if modified is not None:
                    state = modified

            # Update sync accumulators
            left_out = state[:, self.synch_out_left]
            right_out = state[:, self.synch_out_right]
            pp_out = left_out * right_out * dopamine
            alpha_out = r_out * alpha_out + pp_out
            beta_out = r_out * beta_out + dopamine

            left_act = state[:, self.synch_act_left]
            right_act = state[:, self.synch_act_right]
            pp_act = left_act * right_act * dopamine
            alpha_act = r_act * alpha_act + pp_act
            beta_act = r_act * beta_act + dopamine

            if multi_tick:
                synch_k = self._sync_readout(alpha_out, beta_out)
                tick_out = self.c_proj(synch_k).reshape(B, T, D)
                grad_ticks = getattr(self, 'multi_tick_grad', 4)
                if k < K - grad_ticks:
                    tick_out = tick_out.detach()
                tick_outputs.append(tick_out)

        # Final readout
        synch = self._sync_readout(alpha_out, beta_out)
        out = self.c_proj(synch).reshape(B, T, D)

        # Stash sparsity loss
        self._sparsity_loss = sparsity_loss / self.total_ticks

        # Persist cache
        if ctm_cache is not None:
            ctm_cache.layers[layer_idx] = {
                'state': state.view(B, T, -1)[:, -1, :],
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

    def _build_start_state(self, BT, dtype):
        """Concatenate per-region start states into full D vector."""
        parts = []
        for rname in self.region_names:
            parts.append(self.regions[rname].start_state.to(dtype).unsqueeze(0).expand(BT, -1))
        return torch.cat(parts, dim=1)

    @torch.no_grad()
    def calibrate_hebbian(self, state_samples: torch.Tensor):
        """Calibrate per-region baselines from training distribution.
        state_samples: (N, D) activation samples."""
        for rname in self.region_names:
            region_act = self._get_region_state(state_samples, rname)
            baseline = region_act.mean(0)
            getattr(self, f'_baseline_{rname}').copy_(baseline)
            getattr(self, f'_running_mean_{rname}').copy_(baseline)
        self._hebbian_calibrated.fill_(True)
        self._hebbian_active = True

    @torch.no_grad()
    def _apply_hebbian(self, state):
        """Per-region homeostatic correction toward baseline."""
        if not self._hebbian_calibrated:
            return state
        corrected_parts = []
        for rname in self.region_names:
            act = self._get_region_state(state, rname)
            baseline = getattr(self, f'_baseline_{rname}')
            running = getattr(self, f'_running_mean_{rname}')
            lr = self.regions[rname].cfg.hebbian_lr

            # Update running mean
            batch_mean = act.mean(0)
            running.mul_(0.95).add_(batch_mean, alpha=0.05)

            # Correct toward baseline
            drift = running - baseline
            correction = -lr * drift
            corrected_parts.append(act + correction.unsqueeze(0))

        return torch.cat(corrected_parts, dim=1)

    @torch.no_grad()
    def snapshot_baseline(self, sync_signal):
        """Compat with CTMBlock interface."""
        pass  # v2 uses calibrate_hebbian() instead

    def hebbian_update(self, sync_signal):
        """Compat with CTMBlock interface."""
        pass  # v2 uses _apply_hebbian() per-region

    @torch.no_grad()
    def compact_memory_hebbian(self, x: torch.Tensor, n_repeats: int = 10,
                                lr: float = 0.01):
        """Gradient-free fact consolidation via Hebbian + Angeris bounds.

        Teaches new information by:
        1. Forward pass collecting inter-region synapse I/O
        2. Compute Angeris gap per synapse (how far from optimal)
        3. Hebbian outer-product update weighted by gap

        No gradients. No optimizer. Doesn't compete with training.

        Args:
            x: (B, T, D) — the teaching text (already embedded by backbone)
            n_repeats: how many times to replay the teaching signal
            lr: Hebbian learning rate for inter-region synapses

        Returns:
            dict with per-synapse plasticity stats
        """
        was_training = self.training
        self.eval()

        synapse_names = ['syn_to_input', 'syn_to_attn',
                         'syn_to_output', 'syn_to_motor']

        # Collect synapse I/O across repeats
        syn_inputs = {n: [] for n in synapse_names}
        syn_outputs = {n: [] for n in synapse_names}

        for _ in range(n_repeats):
            # Run forward, intercept synapse activations
            B, T, D = x.shape
            BT = B * T
            K = self.active_K
            H, HD = self.n_attn_heads, self.attn_head_dim
            dtype = x.dtype

            attn_k = norm(self.attn_k_proj(x).view(B, T, H, HD))
            attn_v = self.attn_v_proj(x).view(B, T, H, HD)

            state = self._build_start_state(BT, dtype)

            traces = {}
            for rname, region in self.regions.items():
                traces[rname] = region.start_trace.to(dtype).unsqueeze(0).expand(BT, -1, -1).clone()

            # Fresh sync
            left_out = state[:, self.synch_out_left]
            right_out = state[:, self.synch_out_right]
            alpha_act = (state[:, self.synch_act_left] *
                         state[:, self.synch_act_right])
            beta_act = torch.ones_like(alpha_act)
            r_act = torch.exp(-self.decay_act.clamp(0, 15).to(dtype)).unsqueeze(0)

            for k in range(K):
                global_ctx = self.global_proj(state)
                act_in = self._get_region_state(state, 'input')
                act_attn = self._get_region_state(state, 'attention')
                act_out = self._get_region_state(state, 'output')
                act_motor = self._get_region_state(state, 'motor')

                synch_act = self._sync_readout(alpha_act, beta_act).to(dtype)
                attn_q = norm(self.attn_q_proj(synch_act).view(B, T, H, HD))
                if _has_flash:
                    obs = flash_attn.flash_attn_func(
                        attn_q, attn_k, attn_v, causal=True).reshape(BT, D)
                else:
                    q = attn_q.reshape(B * T, H, 1, HD)
                    k_t = attn_k.reshape(B * T, H, -1, HD)
                    v_t = attn_v.reshape(B * T, H, -1, HD)
                    attn_w = (q @ k_t.transpose(-2, -1)) / math.sqrt(HD)
                    obs = (F.softmax(attn_w, dim=-1) @ v_t).reshape(BT, D)

                # Collect synapse I/O
                sin_input = torch.cat([obs, act_motor, global_ctx], dim=1)
                sout_input = self.syn_to_input(sin_input)
                syn_inputs['syn_to_input'].append(sin_input)
                syn_outputs['syn_to_input'].append(sout_input)

                new_in = act_in + sout_input
                traces['input'] = torch.cat(
                    [traces['input'][:, :, 1:], new_in.unsqueeze(-1)], dim=-1)
                act_in = self.regions['input'].process(traces['input'])

                sin_attn = torch.cat([act_in, obs], dim=1)
                sout_attn = self.syn_to_attn(sin_attn)
                syn_inputs['syn_to_attn'].append(sin_attn)
                syn_outputs['syn_to_attn'].append(sout_attn)

                new_attn = act_attn + sout_attn
                traces['attention'] = torch.cat(
                    [traces['attention'][:, :, 1:], new_attn.unsqueeze(-1)], dim=-1)
                act_attn = self.regions['attention'].process(traces['attention'])

                sin_out = torch.cat([act_attn, obs, global_ctx], dim=1)
                sout_out = self.syn_to_output(sin_out)
                syn_inputs['syn_to_output'].append(sin_out)
                syn_outputs['syn_to_output'].append(sout_out)

                new_out = act_out + sout_out
                traces['output'] = torch.cat(
                    [traces['output'][:, :, 1:], new_out.unsqueeze(-1)], dim=-1)
                act_out = self.regions['output'].process(traces['output'])

                sin_motor = torch.cat([act_out, global_ctx], dim=1)
                sout_motor = self.syn_to_motor(sin_motor)
                syn_inputs['syn_to_motor'].append(sin_motor)
                syn_outputs['syn_to_motor'].append(sout_motor)

                new_motor = act_motor + sout_motor
                traces['motor'] = torch.cat(
                    [traces['motor'][:, :, 1:], new_motor.unsqueeze(-1)], dim=-1)
                act_motor = self.regions['motor'].process(traces['motor'])

                state = torch.cat([act_in, act_attn, act_out, act_motor], dim=1)

                # Update action sync
                left_a = state[:, self.synch_act_left]
                right_a = state[:, self.synch_act_right]
                alpha_act = r_act * alpha_act + left_a * right_a
                beta_act = r_act * beta_act + 1.0

        # Compute Angeris gaps and apply Hebbian updates
        stats = {}
        for sname in synapse_names:
            X = torch.cat(syn_inputs[sname], dim=0).float()
            Y = torch.cat(syn_outputs[sname], dim=0).float()

            # Angeris gap: ||Y - X @ W_opt||² / ||Y||²
            XtX = X.T @ X + 1e-4 * torch.eye(X.size(1), device=X.device)
            XtY = X.T @ Y
            try:
                W_opt = torch.linalg.solve(XtX, XtY)
                residual = (Y - X @ W_opt)
                gap = residual.pow(2).sum() / (Y.pow(2).sum() + 1e-8)
            except Exception:
                gap = torch.tensor(1.0)
                residual = Y

            # Hebbian update on the synapse's Linear layer
            # Direction: mean outer product of (input novelty, output residual)
            # Novelty = deviation from mean (what's special about this teaching text)
            x_novelty = X - X.mean(0, keepdim=True)
            y_residual = residual  # what the synapse ISN'T capturing

            # Outer product update: ΔW = lr × gap × (x_novelty^T @ y_residual) / N
            N = X.size(0)
            delta_W = (lr * gap.item()) * (x_novelty.T @ y_residual) / N

            # Apply to first Linear in the synapse Sequential
            synapse = getattr(self, sname)
            for mod in synapse.modules():
                if isinstance(mod, Linear):
                    # The Linear maps in_dim → out_dim*2 (for GLU)
                    # delta_W is in_dim × out_dim — tile for GLU
                    w = mod.weight.data  # [out_dim*2, in_dim]
                    dw = delta_W.T.to(w.dtype)  # [out_dim, in_dim]
                    if dw.shape[0] * 2 == w.shape[0] and dw.shape[1] == w.shape[1]:
                        # GLU: first half is value, second half is gate
                        # Update value half, leave gate alone
                        w[:dw.shape[0]] += dw
                    elif dw.shape == w.shape:
                        w += dw
                    break

            stats[sname] = {
                'gap': gap.item(),
                'delta_norm': delta_W.norm().item(),
                'n_samples': N,
            }

        if was_training:
            self.train()
        return stats

    def get_region_stats(self):
        """Per-region drift statistics."""
        if not self._hebbian_calibrated:
            return {}
        stats = {}
        for rname in self.region_names:
            baseline = getattr(self, f'_baseline_{rname}')
            running = getattr(self, f'_running_mean_{rname}')
            stats[rname] = {
                'n_neurons': self.region_sizes[rname],
                'memory': self.regions[rname].cfg.memory_length,
                'nlm_depth': self.regions[rname].cfg.nlm_depth,
                'drift': float((running - baseline).norm()),
                'hebbian_lr': self.regions[rname].cfg.hebbian_lr,
            }
        return stats
