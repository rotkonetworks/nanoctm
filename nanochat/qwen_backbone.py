"""
Qwen backbone with CTMBlock on last layer.

Loads pretrained Qwen (default: Qwen3-0.6B) as a frozen backbone,
replaces the last layer's MLP with our CTMBlock. Only the CTM
parameters are trainable.

Usage:
    model = QwenBackboneGPT.from_pretrained("Qwen/Qwen3-0.6B", ctm_kwargs={...})
    model.to(device)
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from nanochat.gpt import GPTConfig, CTMBlock, norm
from nanochat.common import print0, COMPUTE_DTYPE


class QwenTokenizer:
    """Wrapper around HuggingFace AutoTokenizer with nanochat interface."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._bos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if self._bos_id is None:
            # Fallback: use the eos_token_id
            self._bos_id = tokenizer.eos_token_id or 0

    @classmethod
    def from_pretrained(cls, model_name):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def get_bos_token_id(self):
        return self._bos_id

    def encode_special(self, text):
        """Encode a special token. Returns None if not found."""
        ids = self.tokenizer.convert_tokens_to_ids(text)
        if ids == self.tokenizer.unk_token_id:
            return None
        return ids

    def encode(self, text, prepend=None, append=None, num_threads=8):
        if isinstance(text, str):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if prepend is not None:
                prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
                ids.insert(0, prepend_id)
            if append is not None:
                append_id = append if isinstance(append, int) else self.encode_special(append)
                ids.append(append_id)
            return ids
        elif isinstance(text, list):
            return [self.encode(t, prepend=prepend, append=append) for t in text]
        raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def id_to_token(self, id):
        return self.tokenizer.convert_ids_to_tokens(id)

    def get_special_tokens(self):
        return set(self.tokenizer.all_special_tokens)


class QwenBackboneGPT(nn.Module):
    """Qwen backbone with CTMBlock replacing the last layer's MLP.

    Presents a compatible interface with nanochat's GPT class:
    - forward(idx, targets, kv_cache, ctm_cache, ...)
    - config attribute (GPTConfig)
    - dream(), compact_memory()
    - setup_optimizer()

    KV cache: manages its own HF-style cache internally. The Engine's KVCache
    is accepted but ignored — set model.manages_own_cache = True so Engine
    skips KVCache creation.
    """
    manages_own_cache = True  # tells Engine to skip KVCache creation

    def __init__(self, backbone, ctm_config, ctm_layer_idx=-1):
        super().__init__()
        self.backbone = backbone
        qwen_config = backbone.config

        # Which layer gets CTM (default: last)
        n_layers = qwen_config.num_hidden_layers
        self.ctm_layer_idx = ctm_layer_idx if ctm_layer_idx >= 0 else n_layers - 1

        # Build a GPTConfig that matches Qwen dimensions for CTMBlock
        self.config = GPTConfig(
            n_embd=qwen_config.hidden_size,
            n_layer=n_layers,
            n_head=qwen_config.num_attention_heads,
            n_kv_head=qwen_config.num_key_value_heads,
            vocab_size=qwen_config.vocab_size,
            sequence_len=min(qwen_config.max_position_embeddings, 32768),
            use_ctm=True,
            ctm_layers="last",
            ctm_iterations=ctm_config.get("ctm_iterations", 4),
            ctm_n_synch=ctm_config.get("ctm_n_synch", qwen_config.hidden_size // 2),
            ctm_memory_length=ctm_config.get("ctm_memory_length", 16),
            ctm_memory_hidden=ctm_config.get("ctm_memory_hidden", 32),
            ctm_synapse_depth=ctm_config.get("ctm_synapse_depth", 32),
            ctm_n_attn_heads=ctm_config.get("ctm_n_attn_heads", 1),
            ctm_adaptive_k=ctm_config.get("ctm_adaptive_k", False),
        )

        # Create CTMBlock with Qwen dimensions
        self.ctm_block = CTMBlock(self.config)

        # Remove the original MLP from the CTM layer to save memory
        # (we keep the attention and layernorms)
        self.backbone.model.layers[self.ctm_layer_idx].mlp = None

        # Freeze entire backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Internal HF cache for KV during generation
        self._hf_cache = None
        self._cache_position = 0

    @classmethod
    def from_pretrained(cls, model_name="Qwen/Qwen3-0.6B", ctm_kwargs=None,
                        ctm_layer_idx=-1, device="cpu"):
        """Load Qwen backbone and attach CTMBlock."""
        from transformers import AutoModelForCausalLM
        print0(f"Loading backbone: {model_name}")
        backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",  # use PyTorch SDPA (works everywhere)
        )
        ctm_kwargs = ctm_kwargs or {}
        model = cls(backbone, ctm_kwargs, ctm_layer_idx)
        return model

    def init_ctm_weights(self):
        """Initialize CTM block weights (call after moving to real device)."""
        D = self.config.n_embd
        s = 3**0.5 * D**-0.5 * 0.68
        ctm = self.ctm_block

        # Cross-attention projections
        torch.nn.init.uniform_(ctm.attn_q_proj.weight, -s, s)
        torch.nn.init.uniform_(ctm.attn_k_proj.weight, -s, s)
        torch.nn.init.uniform_(ctm.attn_v_proj.weight, -s, s)

        # Synapse U-NET
        from nanochat.gpt import Linear
        for module in ctm.synapses.modules():
            if isinstance(module, Linear):
                torch.nn.init.uniform_(module.weight, -s, s)

        # NLMs: Xavier uniform
        for module in [ctm.nlm1, ctm.nlm2]:
            for p in module.parameters():
                if p.dim() >= 2:
                    torch.nn.init.xavier_uniform_(p)

        # Start state and trace
        torch.nn.init.uniform_(ctm.start_state, -0.01, 0.01)
        torch.nn.init.uniform_(ctm.start_trace, -0.01, 0.01)

        # Tick embeddings
        torch.nn.init.normal_(ctm.tick_embed, mean=0.0, std=0.01)

        # Output projection: zero init (residual-friendly)
        torch.nn.init.zeros_(ctm.c_proj.weight)

        # Sync pairs: random neuron indices
        n_synch = ctm.n_synch
        ctm.synch_out_left.copy_(torch.randint(0, D, (n_synch,)))
        ctm.synch_out_right.copy_(torch.randint(0, D, (n_synch,)))
        ctm.synch_act_left.copy_(torch.randint(0, D, (n_synch,)))
        ctm.synch_act_right.copy_(torch.randint(0, D, (n_synch,)))

        # Decay parameters
        torch.nn.init.uniform_(ctm.decay_out, 0.5, 2.0)
        torch.nn.init.uniform_(ctm.decay_act, 0.5, 2.0)

        print0(f"Initialized CTMBlock weights (D={D}, K={ctm.K}, n_synch={n_synch})")

    def forward(self, idx, targets=None, kv_cache=None, ctm_cache=None,
                pos_offset=0, multi_tick=False, **kwargs):
        """Forward pass: Qwen backbone layers 0..N-2, then custom layer N-1 with CTMBlock.

        Args:
            idx: token indices (B, T)
            targets: target indices for loss (B, T) or None
            kv_cache: "use_internal" to use HF cache, None for no cache (training)
            ctm_cache: CTMCache for persistent CTM state across tokens
        """
        B, T = idx.shape
        device = idx.device
        qwen_model = self.backbone.model
        use_cache = (kv_cache is not None)

        # Token embedding
        x = qwen_model.embed_tokens(idx)

        # Position IDs and cache position
        if use_cache:
            cache_position = torch.arange(
                self._cache_position, self._cache_position + T, device=device
            )
            position_ids = cache_position.unsqueeze(0)
        else:
            cache_position = torch.arange(T, device=device)
            position_ids = cache_position.unsqueeze(0)

        # HF DynamicCache: create on first use, reuse across calls
        from transformers import DynamicCache
        past_kv = self._hf_cache if use_cache else None
        if use_cache and past_kv is None:
            past_kv = DynamicCache()

        # Pre-compute rotary position embeddings (cos, sin)
        position_embeddings = qwen_model.rotary_emb(x, position_ids)

        # === Run all layers, intercepting the CTM layer's MLP ===
        for i, layer in enumerate(qwen_model.layers):
            if i == self.ctm_layer_idx:
                # Custom path: Qwen attention + our CTMBlock
                residual = x
                x_normed = layer.input_layernorm(x)

                # Qwen attention
                attn_outputs = layer.self_attn(
                    x_normed,
                    position_embeddings=position_embeddings,
                    attention_mask=None,
                    past_key_values=past_kv,
                    cache_position=cache_position,
                )
                attn_out = attn_outputs[0]
                x = residual + attn_out

                # CTMBlock replaces MLP
                residual = x
                x_normed = layer.post_attention_layernorm(x)

                if multi_tick and targets is not None:
                    ctm_out, tick_outputs = self.ctm_block(
                        x_normed, ctm_cache=ctm_cache,
                        layer_idx=self.ctm_layer_idx, multi_tick=True
                    )
                else:
                    ctm_out = self.ctm_block(
                        x_normed, ctm_cache=ctm_cache,
                        layer_idx=self.ctm_layer_idx
                    )
                    tick_outputs = None

                x = residual + ctm_out.to(residual.dtype)
            else:
                # Standard Qwen layer (frozen)
                x = layer(
                    x,
                    position_embeddings=position_embeddings,
                    past_key_values=past_kv,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

        # Update cache state
        if use_cache:
            self._hf_cache = past_kv
            self._cache_position += T

        # Final norm
        x = qwen_model.norm(x)

        # Compute logits
        logits = self.backbone.lm_head(x)

        # Compute loss if targets provided
        if targets is not None:
            if tick_outputs is not None and self.config.use_ctm:
                # Multi-tick loss (same as GPT.forward)
                return self._multi_tick_loss(x, logits, tick_outputs, targets, residual, ctm_out)
            else:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-1,
                )
                return logits, loss

        return logits

    def _multi_tick_loss(self, x_final, logits_final, tick_outputs, targets, pre_ctm_residual, final_ctm_out):
        """Multi-tick loss matching GPT's implementation."""
        V = self.config.vocab_size
        max_entropy = math.log(V)
        flat_targets = targets.reshape(-1)
        n_tokens = flat_targets.size(0)

        # Build full residual for each tick output
        # tick_out is the CTMBlock output at tick k. Full residual = pre_ctm_residual + tick_out
        # Then apply final norm + lm_head
        all_losses = []
        all_certainties = []

        for tick_out in tick_outputs:
            x_at_tick = pre_ctm_residual + tick_out
            x_at_tick = self.backbone.model.norm(x_at_tick)
            tick_logits = self.backbone.lm_head(x_at_tick)
            tick_logits_flat = tick_logits.reshape(-1, V)

            token_losses = F.cross_entropy(tick_logits_flat, flat_targets, ignore_index=-1, reduction='none')
            all_losses.append(token_losses)

            with torch.no_grad():
                probs = F.softmax(tick_logits_flat, dim=-1)
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
                certainty = 1.0 - entropy / max_entropy
            all_certainties.append(certainty)

        all_losses = torch.stack(all_losses, dim=0)
        all_certainties = torch.stack(all_certainties, dim=0)

        token_idx = torch.arange(n_tokens, device=all_losses.device)
        best_tick_idx = all_losses.argmin(dim=0)
        certain_tick_idx = all_certainties.argmax(dim=0)

        loss_argmin = all_losses[best_tick_idx, token_idx].mean()
        loss_argmax_cert = all_losses[certain_tick_idx, token_idx].mean()
        task_loss = 0.5 * loss_argmin + 0.5 * loss_argmax_cert

        return logits_final, task_loss

    def reset_cache(self):
        """Reset the HF KV cache (call at start of new generation)."""
        self._hf_cache = None
        self._cache_position = 0

    def get_device(self):
        return next(self.parameters()).device

    def setup_optimizer(self, lr=0.001, weight_decay=0.0, **kwargs):
        """Only optimize CTM block parameters."""
        from nanochat.optim import MuonAdamW
        ctm_params = list(self.ctm_block.parameters())
        trainable = [p for p in ctm_params if p.requires_grad]
        print0(f"Optimizer: {len(trainable)} CTM parameter groups, "
               f"{sum(p.numel() for p in trainable):,} trainable params")

        # Simple AdamW for CTM params — no Muon (Muon is for large matrix params)
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay,
                                       betas=(0.9, 0.95))
        return optimizer

    def dream(self, device=None, K_override=None):
        """Run convergence diagnostics on CTM block."""
        from nanochat.gpt import CTMCache
        results = {}
        ctm = self.ctm_block
        layer_idx = self.ctm_layer_idx

        # Generate a simple test input
        D = self.config.n_embd
        if device is None:
            device = self.get_device()
        x_test = torch.randn(1, 16, D, device=device, dtype=COMPUTE_DTYPE)
        ctm_cache = CTMCache(self.config.n_layer)

        old_K = ctm.active_K
        if K_override is not None:
            ctm.active_K = K_override

        with torch.no_grad():
            _, deltas = ctm(x_test, dream=True, ctm_cache=ctm_cache, layer_idx=layer_idx)

        ctm.active_K = old_K

        converged = len(deltas) >= 2 and deltas[-1] < deltas[0] * 0.1
        results[layer_idx] = {
            'converged': converged,
            'K_start': ctm.K,
            'K_end': len(deltas),
            'deltas': deltas,
            'final_distance': deltas[-1] if deltas else 0,
        }
        return results

    def compact_memory(self, teaching_ids, target_ids, lr=1e-3, steps=20):
        """Three-factor neuroplasticity: pre × post × dopamine.

        Models the hippocampal-cortical memory consolidation cycle:

        WAKE PHASE (compute dopamine):
          Forward pass the teaching text. Compute per-token prediction error
          (cross-entropy). Convert to dopamine signal: surprising tokens get
          high dopamine, predictable tokens get low dopamine. This is the
          RPE signal from DA-SSDP (arXiv 2512.07194) and the three-factor
          rule (Gerstner 2015): ΔW = η × pre × post × M.

        ENCODING PHASE (dopamine-gated sync):
          Re-forward with per-token dopamine. The sync accumulators now
          weight each token by its surprise — the cache becomes a
          dopamine-shaped memory trace. Surprising tokens dominate the
          sync pattern, boring tokens are dampened. This is the eligibility
          trace mechanism: synapses that were active during surprising
          events get tagged for consolidation.

        SLEEP PHASE (replay with sync-modulated gradients):
          Replay teaching text through the model. Gradient descent with
          sync-modulated weight decay (arXiv 2505.18069: weight decay IS
          Hebbian homeostasis). Active sync pairs get lower decay (protect
          memory), inactive pairs get regularized. The dopamine-shaped
          sync importance focuses gradient updates on what mattered.

        Args:
            teaching_ids: input token ids (B, T)
            target_ids: target token ids (B, T)
            lr: base learning rate
            steps: consolidation replay steps
        Returns:
            dict with diagnostics
        """
        from nanochat.gpt import CTMCache as CTMCacheCls

        ctm = self.ctm_block
        layer_idx = self.ctm_layer_idx
        device = teaching_ids.device
        B, T = teaching_ids.shape

        # Save pre-compaction state
        pre_params = {n: p.clone() for n, p in ctm.named_parameters()}

        # =====================================================================
        # WAKE PHASE: compute dopamine from prediction error
        # =====================================================================
        # Forward pass to get per-token prediction error (the "surprise")
        self.eval()
        with torch.no_grad():
            logits = self.forward(teaching_ids)  # (B, T, V)
            # Per-token cross-entropy = prediction error
            per_token_ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                reduction='none',
                ignore_index=-1,
            )  # (B*T,)

            # Convert prediction error → dopamine signal
            # Neuroscience: tonic DA ~4Hz (baseline), phasic burst up to ~20Hz (5x)
            # Mapping: normalize CE relative to mean, clamp to [0.5, 2.0]
            # High CE = surprising = high dopamine = remember this
            # Low CE = predictable = low dopamine = don't bother
            ce_mean = per_token_ce.mean()
            dopamine_per_token = torch.clamp(per_token_ce / (ce_mean + 1e-8), 0.5, 2.0)

        print(f"    Dopamine: mean={dopamine_per_token.mean():.3f}, "
              f"min={dopamine_per_token.min():.3f}, max={dopamine_per_token.max():.3f}, "
              f"CE_mean={ce_mean:.3f}")

        # =====================================================================
        # ENCODING PHASE: build dopamine-gated sync trace
        # =====================================================================
        # Re-forward with per-token dopamine → sync accumulators weight by surprise
        self.eval()
        da_cache = CTMCacheCls(self.config.n_layer)
        da_cache.dopamine = dopamine_per_token  # (BT,) tensor, not scalar

        with torch.no_grad():
            self.forward(teaching_ids, ctm_cache=da_cache)

        # The cache now contains dopamine-weighted sync:
        # alpha_out accumulated more from surprising tokens,
        # less from predictable tokens. THIS is the memory trace.
        da_cached = da_cache.layers[layer_idx]
        if da_cached is None:
            return {}

        # =====================================================================
        # Build importance maps from dopamine-shaped sync
        # =====================================================================
        with torch.no_grad():
            alpha_out = da_cached['alpha_out'].squeeze(0)
            beta_out = da_cached['beta_out'].squeeze(0)
            alpha_act = da_cached['alpha_act'].squeeze(0)
            beta_act = da_cached['beta_act'].squeeze(0)

            # Dopamine-weighted sync readout
            S_out = alpha_out / torch.sqrt(beta_out)
            S_act = alpha_act / torch.sqrt(beta_act)

            # Baseline sync from start_state (what a blank slate produces)
            start = ctm.start_state.data
            S_base_out = start[ctm.synch_out_left] * start[ctm.synch_out_right]
            S_base_act = start[ctm.synch_act_left] * start[ctm.synch_act_right]

            # Delta: what dopamine-gated experience added beyond baseline
            dS_out = S_out - S_base_out
            dS_act = S_act - S_base_act

            # Per-neuron importance from dopamine-weighted sync
            neuron_importance = torch.zeros(ctm.D, device=device)
            neuron_importance.scatter_add_(0, ctm.synch_out_left, dS_out.abs())
            neuron_importance.scatter_add_(0, ctm.synch_out_right, dS_out.abs())
            neuron_importance.scatter_add_(0, ctm.synch_act_left, dS_act.abs())
            neuron_importance.scatter_add_(0, ctm.synch_act_right, dS_act.abs())
            neuron_importance = neuron_importance / (neuron_importance.mean() + 1e-8)
            # Floor at 0.1: every neuron gets some gradient, but important ones get more
            neuron_importance = 0.1 + 0.9 * neuron_importance / (neuron_importance.max() + 1e-8)

            # Per-sync-pair importance
            sync_importance = dS_out.abs() + dS_act.abs()
            sync_importance = 0.1 + 0.9 * sync_importance / (sync_importance.max() + 1e-8)

        # =====================================================================
        # SLEEP PHASE: replay with sync-modulated gradient descent
        # =====================================================================
        # Sync-modulated weight decay: lower decay on important params = protect memory
        base_wd = 0.01
        param_groups = []
        for name, p in ctm.named_parameters():
            if not p.requires_grad:
                continue
            if 'start_state' in name or 'start_trace' in name:
                param_groups.append({'params': [p], 'lr': lr * 2.0, 'weight_decay': base_wd * 0.1})
            elif 'decay_out' in name or 'decay_act' in name:
                param_groups.append({'params': [p], 'lr': lr * 0.5, 'weight_decay': 0.0})
            elif 'c_proj' in name:
                param_groups.append({'params': [p], 'lr': lr * 1.5, 'weight_decay': base_wd * 0.5})
            elif 'tick_embed' in name:
                param_groups.append({'params': [p], 'lr': lr * 0.3, 'weight_decay': base_wd})
            else:
                param_groups.append({'params': [p], 'lr': lr, 'weight_decay': base_wd})

        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.99))

        self.train()
        losses = []
        for step in range(steps):
            optimizer.zero_grad()

            # Replay with dopamine-shaped cache (sleep replay starts from
            # the dopamine-weighted memory trace, not a blank slate)
            step_cache = CTMCacheCls(self.config.n_layer)
            step_cache.layers[layer_idx] = {
                k: v.detach().clone() for k, v in da_cached.items()
            }
            _, loss = self.forward(teaching_ids, targets=target_ids, ctm_cache=step_cache)
            loss.backward()

            # Scale gradients by dopamine-weighted sync importance
            # Three-factor: gradient × sync_importance = pre × post × modulator
            with torch.no_grad():
                for name, p in ctm.named_parameters():
                    if p.grad is None:
                        continue
                    if 'c_proj' in name and p.grad.dim() == 2:
                        p.grad *= sync_importance.unsqueeze(0)
                    elif 'start_state' in name and p.grad.dim() == 1:
                        p.grad *= neuron_importance
                    elif 'start_trace' in name and p.grad.dim() == 2:
                        p.grad *= neuron_importance.unsqueeze(1)

            optimizer.step()
            losses.append(loss.item())

        self.eval()

        # Measure total change
        total_delta = sum(
            (p - pre_params[n]).norm().item()
            for n, p in ctm.named_parameters()
        )

        return {
            'total_delta': total_delta,
            'losses': losses,
            'dS_out_norm': dS_out.norm().item(),
            'dS_act_norm': dS_act.norm().item(),
            'dopamine_mean': dopamine_per_token.mean().item(),
            'dopamine_std': dopamine_per_token.std().item(),
            'ce_mean': ce_mean.item(),
        }

    def get_ctm_state_dict(self):
        """Get only CTM block state dict for checkpointing."""
        return self.ctm_block.state_dict()

    def load_ctm_state_dict(self, state_dict):
        """Load CTM block state dict from checkpoint."""
        self.ctm_block.load_state_dict(state_dict)

    @property
    def n_layer(self):
        return self.config.n_layer

    @property
    def n_head(self):
        return self.config.n_head

    @property
    def n_embd(self):
        return self.config.n_embd
