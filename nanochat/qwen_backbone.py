"""
Qwen backbone with CTMBlock(s) replacing selected layers' MLPs.

Loads pretrained Qwen (default: Qwen3-0.6B) as a frozen backbone,
replaces designated layers' MLPs with CTMBlocks. Only the CTM
parameters are trainable.

Supports single or multi-CTM via ctm_layer_indices:
    model = QwenBackboneGPT.from_pretrained("Qwen/Qwen3-0.6B",
                ctm_kwargs={...}, ctm_layer_indices=[14, 27])
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
    """Qwen backbone with CTMBlock(s) replacing selected layers' MLPs.

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

    def __init__(self, backbone, ctm_config, ctm_layer_indices=None):
        super().__init__()
        self.backbone = backbone
        qwen_config = backbone.config

        # Detect Qwen3.5 (Gated DeltaNet hybrid) — must have linear_attention layers
        self._is_qwen35 = (hasattr(qwen_config, 'layer_types')
                           and 'linear_attention' in getattr(qwen_config, 'layer_types', []))
        if self._is_qwen35:
            self._qwen_config = qwen_config  # needed for Qwen3_5DynamicCache
            # Find last full-attention layer for default CTM placement
            full_attn_layers = [i for i, lt in enumerate(qwen_config.layer_types)
                                if lt == "full_attention"]
            self._full_attn_layers = set(full_attn_layers)
            print0(f"  Qwen3.5: {len(full_attn_layers)} full-attn layers, "
                   f"{qwen_config.num_hidden_layers - len(full_attn_layers)} linear-attn layers")

        # Which layers get CTM
        n_layers = qwen_config.num_hidden_layers
        if ctm_layer_indices is None:
            if self._is_qwen35:
                # Default: last full-attention layer (CTM needs self_attn, not linear_attn)
                last_full = max(self._full_attn_layers)
                ctm_layer_indices = [last_full]
                print0(f"  CTM default: layer {last_full} (last full-attention)")
            else:
                ctm_layer_indices = [n_layers - 1]  # default: last layer only
        self.ctm_layer_indices = sorted(ctm_layer_indices)

        # Validate: Qwen3.5 CTM layers must be full-attention (not linear/DeltaNet)
        if self._is_qwen35:
            for idx in self.ctm_layer_indices:
                if idx not in self._full_attn_layers:
                    raise ValueError(
                        f"CTM layer {idx} is linear_attention (DeltaNet), not full_attention. "
                        f"CTM requires self_attn. Full-attention layers: {sorted(self._full_attn_layers)}"
                    )

        # Backwards compat: single-CTM code can still read ctm_layer_idx
        self.ctm_layer_idx = self.ctm_layer_indices[-1]

        # Build ctm_layers spec string for GPTConfig
        ctm_layers_spec = ",".join(str(i) for i in self.ctm_layer_indices)

        # Build a GPTConfig that matches Qwen dimensions for CTMBlock
        self.config = GPTConfig(
            n_embd=qwen_config.hidden_size,
            n_layer=n_layers,
            n_head=qwen_config.num_attention_heads,
            n_kv_head=qwen_config.num_key_value_heads,
            vocab_size=qwen_config.vocab_size,
            sequence_len=min(qwen_config.max_position_embeddings, 32768),
            use_ctm=True,
            ctm_layers=ctm_layers_spec,
            ctm_iterations=ctm_config.get("ctm_iterations", 4),
            ctm_n_synch=ctm_config.get("ctm_n_synch", qwen_config.hidden_size // 2),
            ctm_memory_length=ctm_config.get("ctm_memory_length", 16),
            ctm_memory_hidden=ctm_config.get("ctm_memory_hidden", 32),
            ctm_synapse_depth=ctm_config.get("ctm_synapse_depth", 32),
            ctm_n_attn_heads=ctm_config.get("ctm_n_attn_heads", 1),
            ctm_adaptive_k=ctm_config.get("ctm_adaptive_k", False),
        )

        # Create one CTMBlock per designated layer
        self.ctm_blocks = nn.ModuleDict({
            str(idx): CTMBlock(self.config) for idx in self.ctm_layer_indices
        })
        # Backwards compat alias
        self.ctm_block = self.ctm_blocks[str(self.ctm_layer_idx)]

        # Track which CTM layers run in additive mode (MLP kept) vs replacement
        # New/fresh CTM layers keep their frozen MLP (additive: MLP + CTM)
        # This is set properly after checkpoint loading via set_replacement_layers()
        self._replacement_layers = set()  # layers where MLP is removed (trained CTMs)

        # Freeze entire backbone (selectively unfrozen later via unfreeze_layers())
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self._unfrozen_layers = set()

        # Internal HF cache for KV during generation
        self._hf_cache = None
        self._cache_position = 0

    @classmethod
    def from_pretrained(cls, model_name="Qwen/Qwen3-0.6B", ctm_kwargs=None,
                        ctm_layer_indices=None, device="cpu"):
        """Load Qwen backbone and attach CTMBlock(s).

        Args:
            ctm_layer_indices: list of layer indices for CTM, e.g. [14, 27].
                               None = last layer only.
        """
        is_qwen35 = "qwen3.5" in model_name.lower() or "qwen3_5" in model_name.lower()

        print0(f"Loading backbone: {model_name}")
        if is_qwen35:
            from transformers import AutoConfig, Qwen3_5ForCausalLM
            full_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            text_config = getattr(full_config, 'text_config', full_config)
            backbone = Qwen3_5ForCausalLM.from_pretrained(
                model_name,
                config=text_config,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            print0(f"  Qwen3.5 loaded via text_config, layers: {text_config.num_hidden_layers}, "
                   f"layer_types: {text_config.layer_types[:4]}...{text_config.layer_types[-2:]}")
        else:
            from transformers import AutoModelForCausalLM
            backbone = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )

        ctm_kwargs = ctm_kwargs or {}
        model = cls(backbone, ctm_kwargs, ctm_layer_indices)
        return model

    def set_replacement_layers(self, layer_indices):
        """Mark CTM layers as replacement (remove frozen MLP) vs additive (keep MLP).

        Replacement layers: CTM replaces MLP entirely (for trained CTMs).
        Additive layers: CTM output is added alongside frozen MLP output.
        Call after loading checkpoint to remove MLPs only for trained CTM layers.
        """
        for idx in layer_indices:
            if idx in self.ctm_layer_indices:
                self._replacement_layers.add(idx)
                self.backbone.model.layers[idx].mlp = None
                print0(f"  Layer {idx}: MLP removed (CTM replacement mode)")
        # Log additive layers
        additive = [i for i in self.ctm_layer_indices if i not in self._replacement_layers]
        if additive:
            print0(f"  Layers {additive}: MLP kept (CTM additive mode)")

    def _init_single_ctm(self, ctm, label=""):
        """Initialize a single CTMBlock's weights."""
        D = self.config.n_embd
        s = 3**0.5 * D**-0.5 * 0.68

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

        print0(f"Initialized CTMBlock {label}(D={D}, K={ctm.K}, n_synch={n_synch})")

    def init_ctm_weights(self, only_layers=None):
        """Initialize CTM block weights (call after moving to real device).

        Args:
            only_layers: list of layer indices to init. None = all CTM layers.
        """
        for idx_str, ctm in self.ctm_blocks.items():
            idx = int(idx_str)
            if only_layers is not None and idx not in only_layers:
                continue
            self._init_single_ctm(ctm, label=f"L{idx} ")

    def forward(self, idx, targets=None, kv_cache=None, ctm_cache=None,
                pos_offset=0, multi_tick=False, intervene=None, **kwargs):
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
        else:
            cache_position = torch.arange(T, device=device)

        # HF cache: create on first use, reuse across calls
        past_kv = self._hf_cache if use_cache else None
        if use_cache and past_kv is None:
            if self._is_qwen35:
                from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
                past_kv = Qwen3_5DynamicCache(config=self._qwen_config)
            else:
                from transformers import DynamicCache
                past_kv = DynamicCache()

        # Pre-compute rotary position embeddings (cos, sin)
        if self._is_qwen35:
            # MRoPE: position_ids must be (3, B, T) for rotary_emb
            # For text-only: temporal/height/width all = text position
            pos_1d = cache_position.view(1, 1, -1).expand(4, B, -1)
            # Split: text_position_ids (unused), mrope_ids (3, B, T) for rotary
            mrope_ids = pos_1d[1:]  # (3, B, T)
            position_embeddings = qwen_model.rotary_emb(x, mrope_ids)
        else:
            position_ids = cache_position.unsqueeze(0)
            position_embeddings = qwen_model.rotary_emb(x, position_ids)

        # === Run all layers, intercepting CTM layers' MLPs ===
        ctm_indices_set = set(self.ctm_layer_indices)
        # Only the LAST CTM layer produces multi-tick outputs for loss
        last_ctm_idx = self.ctm_layer_indices[-1]
        tick_outputs = None

        for i, layer in enumerate(qwen_model.layers):
            if i in ctm_indices_set:
                ctm = self.ctm_blocks[str(i)]

                # Custom path: attention + our CTMBlock
                residual = x
                x_normed = layer.input_layernorm(x)

                # Attention (Qwen3.5 CTM layers must be full_attention)
                attn_outputs = layer.self_attn(
                    x_normed,
                    position_embeddings=position_embeddings,
                    attention_mask=None,
                    past_key_values=past_kv,
                    cache_position=cache_position,
                )
                attn_out = attn_outputs[0]
                x = residual + attn_out

                # MLP + CTM (additive) or CTM only (replacement)
                residual = x
                x_normed = layer.post_attention_layernorm(x)

                # Additive mode: run frozen MLP alongside CTM
                if i not in self._replacement_layers and layer.mlp is not None:
                    mlp_out = layer.mlp(x_normed)
                else:
                    mlp_out = 0

                # Check for viz intervene hook on model
                _intervene = intervene or getattr(self, '_viz_intervene', None)

                # Multi-tick only on last CTM layer (for loss computation)
                do_multi_tick = (multi_tick and targets is not None
                                 and i == last_ctm_idx)

                if do_multi_tick:
                    ctm_out, tick_outputs = ctm(
                        x_normed, ctm_cache=ctm_cache,
                        layer_idx=i, multi_tick=True,
                        intervene=_intervene
                    )
                else:
                    ctm_out = ctm(
                        x_normed, ctm_cache=ctm_cache,
                        layer_idx=i,
                        intervene=_intervene
                    )

                x = residual + mlp_out + ctm_out.to(residual.dtype)
            else:
                # Standard layer (frozen) — handles both linear_attn and full_attn
                layer_kwargs = dict(
                    position_embeddings=position_embeddings,
                    past_key_values=past_kv,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
                if self._is_qwen35:
                    layer_kwargs['position_ids'] = mrope_ids
                x = layer(x, **layer_kwargs)

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
        """Multi-tick certainty loss (CTM paper Listing 4, VRAM-efficient).

        Tick outputs are already detached for early ticks (only last N have grad).
        Pass 1 (no_grad): compute per-tick loss & certainty for diagnostics + tick selection.
        Pass 2 (with grad): compute loss at selected ticks (grad only flows through non-detached ones).
        """
        V = self.config.vocab_size
        max_entropy = math.log(V)
        flat_targets = targets.reshape(-1)
        n_tokens = flat_targets.size(0)
        K = len(tick_outputs)
        lm_head = self.backbone.lm_head
        norm = self.backbone.model.norm
        lm_dtype = lm_head.weight.dtype

        # --- Pass 1: find best ticks (no grad) ---
        # Use cross-entropy loss as proxy for certainty too (avoids V-sized softmax)
        # Lower loss = higher certainty (strongly correlated, saves ~2GB per tick)
        all_losses_d = torch.empty(K, n_tokens, device=flat_targets.device)

        with torch.no_grad():
            for ki, tick_out in enumerate(tick_outputs):
                x_at_tick = pre_ctm_residual + tick_out
                x_at_tick = norm(x_at_tick)
                tick_logits = lm_head(x_at_tick.to(lm_dtype))
                tick_logits_flat = tick_logits.reshape(-1, V)
                all_losses_d[ki] = F.cross_entropy(tick_logits_flat, flat_targets,
                                                    ignore_index=-1, reduction='none')
                del x_at_tick, tick_logits, tick_logits_flat

        best_tick_idx = all_losses_d.argmin(dim=0)        # [n_tokens]
        # Certainty proxy: tick with lowest loss is most certain
        certain_tick_idx = best_tick_idx  # same as argmin when using loss as proxy

        # Store diagnostics
        self._tick_diagnostics = {}
        with torch.no_grad():
            tick_counts = torch.bincount(best_tick_idx, minlength=K).float()
            for k in range(K):
                self._tick_diagnostics[f'tick_{k}/loss'] = all_losses_d[k].mean().item()
                # Compute certainty only for the last tick (cheap, for monitoring)
                self._tick_diagnostics[f'tick_{k}/selected_pct'] = (tick_counts[k] / n_tokens * 100).item()

            # Full certainty only for final tick (diagnostic)
            final_logits = lm_head(norm(pre_ctm_residual + tick_outputs[-1]).to(lm_dtype))
            final_probs = F.softmax(final_logits.reshape(-1, V), dim=-1)
            final_entropy = -(final_probs * (final_probs + 1e-10).log()).sum(dim=-1)
            final_certainty = 1.0 - final_entropy / max_entropy
            self._tick_diagnostics['certainty/mean'] = final_certainty.mean().item()
            self._tick_diagnostics['certainty/std'] = final_certainty.std().item()
            del final_logits, final_probs, final_entropy, final_certainty

        del all_losses_d

        # --- Pass 2: compute loss WITH grad ---
        # Always compute loss at the last tick (has grad). Blend with best-tick selection.
        # This ensures gradients always flow, even if all tokens prefer early (detached) ticks.
        grad_ticks = getattr(self.ctm_block, 'multi_tick_grad', 4)
        last_k = K - 1

        # Last-tick loss (always has grad)
        x_last = pre_ctm_residual + tick_outputs[last_k]
        x_last = norm(x_last)
        last_logits = lm_head(x_last.to(lm_dtype))
        last_loss = F.cross_entropy(last_logits.reshape(-1, V), flat_targets,
                                     ignore_index=-1)
        del x_last, last_logits

        # Best-tick loss: recompute only for ticks that have grad (last N)
        # For tokens selecting detached ticks, fall back to last-tick loss
        selected_ticks = best_tick_idx.unique()
        grad_tick_set = set(range(K - grad_ticks, K))

        tick_losses = {}
        for k in selected_ticks.tolist():
            if k in grad_tick_set:
                tick_out = tick_outputs[k]
                x_at_tick = pre_ctm_residual + tick_out
                x_at_tick = norm(x_at_tick)
                tick_logits = lm_head(x_at_tick.to(lm_dtype))
                tick_logits_flat = tick_logits.reshape(-1, V)
                token_losses = F.cross_entropy(tick_logits_flat, flat_targets,
                                                ignore_index=-1, reduction='none')
                tick_losses[k] = token_losses
                del x_at_tick, tick_logits, tick_logits_flat

        # Gather per-token loss: use best tick if it has grad, else last tick
        best_tick_loss_parts = []
        n_from_best = 0
        for k in selected_ticks.tolist():
            mask = (best_tick_idx == k)
            if not mask.any():
                continue
            if k in tick_losses:
                best_tick_loss_parts.append(tick_losses[k][mask].sum())
                n_from_best += mask.sum().item()

        if best_tick_loss_parts:
            # Blend: weight by how many tokens chose grad ticks vs total
            best_loss = sum(best_tick_loss_parts) / n_tokens
            alpha = n_from_best / n_tokens  # fraction of tokens with grad-tick selection
            task_loss = alpha * best_loss + (1 - alpha) * last_loss
        else:
            # All tokens chose detached ticks — use last-tick loss only
            task_loss = last_loss

        self._tick_diagnostics['loss/argmin'] = task_loss.item()
        self._tick_diagnostics['loss/last_tick'] = last_loss.item()
        self._tick_diagnostics['loss/grad_tick_frac'] = n_from_best / n_tokens if n_tokens > 0 else 0

        # Auxiliary per-tick supervision (bound-guided or uniform)
        aux_weight = getattr(self.config, 'ctm_aux_weight', 0.1)
        if aux_weight > 0:
            use_bound_guided = getattr(self.config, 'ctm_bound_guided_aux', False)

            # Recompute per-tick mean loss (from detached pass 1 data — no extra VRAM)
            # We need per-tick loss with grad for ticks that have it
            per_tick_losses = []
            for k in range(K):
                if k in tick_losses:
                    per_tick_losses.append(tick_losses[k].mean())
                else:
                    # Detached tick — use last_loss as proxy (has grad)
                    per_tick_losses.append(last_loss)

            if use_bound_guided:
                ctm = self.ctm_blocks[str(self.ctm_layer_idx)]
                tick_weights = ctm._tick_aux_weights[:K]  # (K,)
                aux_loss = sum(w * l for w, l in zip(tick_weights, per_tick_losses))
                self._tick_diagnostics['loss/aux_mode'] = 1.0
                for k in range(K):
                    self._tick_diagnostics[f'aux/tick_{k}_weight'] = tick_weights[k].item()
            else:
                aux_loss = sum(per_tick_losses) / K
                self._tick_diagnostics['loss/aux_mode'] = 0.0

            task_loss = (1.0 - aux_weight) * task_loss + aux_weight * aux_loss
            self._tick_diagnostics['loss/aux_all_ticks'] = aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss

        return logits_final, task_loss

    def reset_cache(self):
        """Reset the HF KV cache (call at start of new generation)."""
        self._hf_cache = None
        self._cache_position = 0

    def get_device(self):
        return next(self.parameters()).device

    def unfreeze_layers(self, layer_indices):
        """Unfreeze specific backbone layers for co-training with CTM.

        Only unfreezes the MLP (FFN) of each specified layer, not attention.
        Attention weights encode positional/relational structure that's hard
        to recover if corrupted. MLP weights are more local and co-adapt
        better with CTM.

        Args:
            layer_indices: list of layer indices to unfreeze, e.g. [23]
        """
        qwen_model = self.backbone.model if hasattr(self.backbone, 'model') else self.backbone.transformer
        layers = qwen_model.layers if hasattr(qwen_model, 'layers') else qwen_model.h
        unfrozen_params = 0
        for idx in layer_indices:
            layer = layers[idx]
            # Unfreeze MLP only
            mlp = layer.mlp if hasattr(layer, 'mlp') else layer.feed_forward
            for p in mlp.parameters():
                p.requires_grad_(True)
                unfrozen_params += p.numel()
            self._unfrozen_layers.add(idx)
        print0(f"Unfroze MLP on backbone layers {list(layer_indices)}: "
               f"{unfrozen_params:,} params now trainable")

    def setup_optimizer(self, lr=0.001, weight_decay=0.0, backbone_lr_scale=0.1, **kwargs):
        """Optimize CTM params + any unfrozen backbone params.

        Backbone params get a lower learning rate (backbone_lr_scale * lr) to
        avoid overwriting pretrained representations too aggressively.
        """
        ctm_params = []
        for idx_str, ctm in self.ctm_blocks.items():
            for p in ctm.parameters():
                if p.requires_grad:
                    ctm_params.append(p)
        # Include unfrozen backbone params with lower LR
        backbone_trainable = [p for p in self.backbone.parameters() if p.requires_grad]

        param_groups = [
            {'params': ctm_params, 'lr': lr},
        ]
        if backbone_trainable:
            backbone_lr = lr * backbone_lr_scale
            param_groups.append({'params': backbone_trainable, 'lr': backbone_lr})
            print0(f"  Backbone LR: {backbone_lr:.1e} ({backbone_lr_scale}x CTM LR)")

        ctm_count = len(ctm_params)
        print0(f"Optimizer: {ctm_count} CTM params across {len(self.ctm_blocks)} block(s)"
               + (f" + {len(backbone_trainable)} backbone params from layers {sorted(self._unfrozen_layers)}"
                  if backbone_trainable else "")
               + f", {sum(p.numel() for p in ctm_params) + sum(p.numel() for p in backbone_trainable):,} trainable params")

        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay,
                                       betas=(0.9, 0.95))
        return optimizer

    def dream(self, device=None, K_override=None):
        """Run convergence + bounds diagnostics on all CTM blocks.

        Returns per-layer dict with:
          - Convergence: deltas, converged flag (original dream)
          - Synapse capacity: c_proj SVD rank, condition number, top singular values
          - Synapse utilization: activation rank vs weight rank
          - Neuron health: NLM weight norms, dead neuron count, diversity
          - Bottleneck: automatic diagnosis string

        Adapted from Angeris SDP dual framework (bounds-analysis branch).
        """
        from nanochat.gpt import CTMCache
        results = {}
        D = self.config.n_embd
        if device is None:
            device = self.get_device()
        x_test = torch.randn(1, 16, D, device=device, dtype=COMPUTE_DTYPE)

        for idx_str, ctm in self.ctm_blocks.items():
            layer_idx = int(idx_str)
            ctm_cache = CTMCache(self.config.n_layer)

            old_K = ctm.active_K
            if K_override is not None:
                ctm.active_K = K_override

            with torch.no_grad():
                _, deltas = ctm(x_test, dream=True, ctm_cache=ctm_cache, layer_idx=layer_idx)

            ctm.active_K = old_K

            converged = len(deltas) >= 2 and deltas[-1] < deltas[0] * 0.1
            result = {
                'converged': converged,
                'K_start': ctm.K,
                'K_end': len(deltas),
                'deltas': deltas,
                'final_distance': deltas[-1] if deltas else 0,
            }

            # --- Bounds analysis (synapse + neuron diagnostics) ---
            result.update(self._analyze_bounds(ctm, x_test, layer_idx))

            # --- Bound-guided aux weights (Angeris → deep supervision) ---
            if getattr(self.config, 'ctm_bound_guided_aux', False):
                tick_gaps = ctm.compute_tick_gaps(x_test)
                result['tick_gaps'] = tick_gaps
                result['tick_aux_weights'] = ctm._tick_aux_weights.tolist()

            results[layer_idx] = result

        return results

    @torch.no_grad()
    def _analyze_bounds(self, ctm, x_test, layer_idx):
        """Angeris-style bound analysis for a single CTMBlock.

        Measures:
          1. c_proj weight rank (capacity) — SVD of the output projection
          2. Synapse activation rank (utilization) — what actually flows through
          3. NLM neuron health — weight norms, dead/diverse detection
          4. Bottleneck identification
        """
        diag = {}
        D = ctm.D
        eps = 1e-8

        # ── 1. c_proj weight capacity (SVD) ─────────────────────────
        w = ctm.c_proj.weight.detach().float().cpu()  # (D, n_synch)
        svs = torch.linalg.svdvals(w)
        total_sv = svs.sum()
        cumsum = svs.cumsum(0) / (total_sv + eps)
        diag['c_proj_rank90'] = int((cumsum < 0.9).sum().item()) + 1
        diag['c_proj_rank99'] = int((cumsum < 0.99).sum().item()) + 1
        diag['c_proj_condition'] = float(svs[0] / (svs[-1] + eps))
        diag['c_proj_top5_sv'] = svs[:5].tolist()
        diag['c_proj_frobenius'] = float(w.norm())

        # ── 2. Synapse activation utilization ────────────────────────
        # Hook the synapse UNet to capture input/output activations
        syn_inputs, syn_outputs = [], []
        def hook_fn(module, inp, out):
            syn_inputs.append(inp[0].detach().cpu().float())
            syn_outputs.append(out.detach().cpu().float())

        handle = ctm.synapses.register_forward_hook(hook_fn)
        from nanochat.gpt import CTMCache
        cache = CTMCache(self.config.n_layer)
        ctm(x_test, dream=False, ctm_cache=cache, layer_idx=layer_idx)
        handle.remove()

        if syn_inputs:
            # Stack all tick activations: each forward call runs K ticks,
            # hook fires once per forward but synapses run inside tick loop.
            # For a single forward, we get 1 capture (last tick only via hook).
            # Use the captured data to estimate activation rank.
            sin = syn_inputs[0]  # (BT, 2D)
            sout = syn_outputs[0]  # (BT, D)
            sv_in = torch.linalg.svdvals(sin.float())
            act_rank = int((sv_in > sv_in[0] * 0.01).sum().item())
            diag['synapse_act_rank'] = act_rank
            diag['synapse_utilization_pct'] = act_rank / (diag['c_proj_rank90'] + eps) * 100

            # Optimal synapse gap: ||actual - least-squares optimal||
            try:
                W_opt = sout.T @ sin @ torch.linalg.pinv(sin.T @ sin)
                residual_actual = (sout - (sin @ W_opt.T)).norm().item() ** 2
                residual_total = sout.norm().item() ** 2
                diag['synapse_gap_pct'] = residual_actual / (residual_total + eps) * 100
            except Exception:
                diag['synapse_gap_pct'] = -1.0
        else:
            diag['synapse_act_rank'] = 0
            diag['synapse_utilization_pct'] = 0
            diag['synapse_gap_pct'] = -1.0

        # ── 3. NLM neuron health ─────────────────────────────────────
        # Weight norms from nlm1 (SuperLinear: w1 shape [M, 2*hidden, D])
        nlm_norms = None
        for name, param in ctm.named_parameters():
            if 'nlm1' in name and ('w1' in name or 'weight' in name):
                pw = param.detach().cpu().float()
                if pw.dim() == 3:
                    # SuperLinear: last dim is per-neuron
                    nlm_norms = pw.reshape(-1, pw.shape[-1]).norm(dim=0).numpy()
                elif pw.dim() == 2:
                    nlm_norms = pw.norm(dim=0).numpy()
                break

        if nlm_norms is not None:
            import numpy as np
            mean_norm = nlm_norms.mean()
            dead_mask = nlm_norms < mean_norm * 0.1
            diag['n_dead_neurons'] = int(dead_mask.sum())
            diag['neuron_norm_min'] = float(nlm_norms.min())
            diag['neuron_norm_mean'] = float(mean_norm)
            diag['neuron_norm_max'] = float(nlm_norms.max())
            diag['neuron_norm_std'] = float(nlm_norms.std())
        else:
            diag['n_dead_neurons'] = -1
            diag['neuron_norm_min'] = 0
            diag['neuron_norm_mean'] = 0
            diag['neuron_norm_max'] = 0
            diag['neuron_norm_std'] = 0

        # NLM diversity: cosine similarity between neuron weight vectors
        diversity = 0.0
        for name, param in ctm.named_parameters():
            if 'nlm1' in name and ('w1' in name or 'weight' in name):
                pw = param.detach().cpu().float()
                if pw.dim() == 3 and pw.shape[-1] == D:
                    # Each neuron d has a weight vector of shape (M, 2*hidden)
                    vectors = pw.reshape(-1, D).T  # (D, features)
                    vectors = vectors / (vectors.norm(dim=1, keepdim=True) + eps)
                    cos = vectors @ vectors.T
                    mask = torch.triu(torch.ones(D, D, dtype=torch.bool), diagonal=1)
                    diversity = float(cos[mask].mean().abs())
                break
        diag['neuron_diversity'] = diversity

        # Plastic LoRA status
        gate = torch.sigmoid(ctm.plastic_gate).item()
        diag['plastic_gate'] = gate
        if gate > 1e-4:
            pA = ctm.plastic_A.detach().float()
            pB = ctm.plastic_B.detach().float()
            diag['plastic_norm'] = float((pA @ pB).norm())
        else:
            diag['plastic_norm'] = 0.0

        # ── 4. Bottleneck identification ─────────────────────────────
        if diag.get('n_dead_neurons', 0) > D * 0.3:
            diag['bottleneck'] = 'dead_neurons'
        elif diag['c_proj_rank90'] <= 5:
            diag['bottleneck'] = f'c_proj_rank_collapse (rank90={diag["c_proj_rank90"]})'
        elif diag.get('synapse_utilization_pct', 100) < 10:
            diag['bottleneck'] = 'low_synapse_utilization'
        elif diag['c_proj_condition'] > 1000:
            diag['bottleneck'] = f'c_proj_ill_conditioned (cond={diag["c_proj_condition"]:.0f})'
        elif diversity > 0.5:
            diag['bottleneck'] = f'neuron_collapse (diversity={diversity:.2f})'
        else:
            diag['bottleneck'] = 'none'

        return diag

    def compact_memory(self, teaching_ids, target_ids, lr=3e-4, steps=30,
                       recall_pairs=None, recall_weight=0.7,
                       max_delta=0.15, sanity_prompt=None,
                       kl_weight=0.3, kl_temperature=2.0,
                       plastic_only=False, nullspace_proj=False):
        """Recall-aware neuroplasticity: teach facts AND learn to recall them.

        Neuroscience-informed design (BCM theory + three-factor Hebbian):
          - LTP in hippocampus changes synaptic strength by 25-180%
          - BCM sliding threshold: recent plasticity raises future threshold
          - Homeostatic scaling: global normalization prevents runaway excitation
          - Sleep consolidation: replay + selective forgetting

        Three phases:
          WAKE: compute per-token dopamine (prediction error → surprise signal)
          ENCODE: build dopamine-gated sync trace (eligibility tagging)
          SLEEP: gradient replay with BCM-like adaptive LR decay

        Safety (anti-excitotoxicity):
          - max_delta: stop early if relative weight change exceeds budget (default 15%,
            maps to moderate LTP strength in hippocampal CA1)
          - Adaptive LR decay: LR decreases as delta grows (BCM sliding threshold analog)
          - sanity_prompt: generate after compact, rollback if repetition collapse
            (analog to seizure detection → homeostatic reset)

        Args:
            teaching_ids: input token ids (B, T) — the teaching text
            target_ids: target token ids (B, T) — shifted teaching text
            lr: base learning rate (default 3e-4)
            steps: consolidation replay steps (default 30)
            recall_pairs: list of (input_ids, target_ids) tensors for recall
            recall_weight: weight for recall loss vs teaching loss (default 0.7)
            max_delta: max relative weight change before early stop (default 0.15 = 15%)
            sanity_prompt: token ids to generate from post-compact; rollback on collapse
            kl_weight: weight for KL divergence anchor loss (default 0.3)
            kl_temperature: temperature for KL softmax (default 2.0, higher = softer anchor)
        Returns:
            dict with diagnostics
        """
        from nanochat.gpt import CTMCache as CTMCacheCls

        device = teaching_ids.device
        B, T = teaching_ids.shape

        # Save pre-compaction state (all CTM blocks)
        pre_params = {}
        for idx_str, ctm in self.ctm_blocks.items():
            for n, p in ctm.named_parameters():
                pre_params[f"{idx_str}.{n}"] = p.clone()

        # =====================================================================
        # WAKE PHASE: compute dopamine from prediction error
        # =====================================================================
        self.eval()
        with torch.no_grad():
            logits = self.forward(teaching_ids)  # (B, T, V)
            per_token_ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                reduction='none',
                ignore_index=-1,
            )  # (B*T,)

            # CE → dopamine: high surprise = high dopamine = remember this
            ce_mean = per_token_ce.mean()
            dopamine_per_token = torch.clamp(per_token_ce / (ce_mean + 1e-8), 0.5, 2.0)

        # =====================================================================
        # ENCODING PHASE: build dopamine-gated sync trace
        # =====================================================================
        self.eval()
        da_cache = CTMCacheCls(self.config.n_layer)
        da_cache.dopamine = dopamine_per_token

        with torch.no_grad():
            self.forward(teaching_ids, ctm_cache=da_cache)

        # Collect cached states from all CTM layers
        da_cached_layers = {}
        for idx in self.ctm_layer_indices:
            if da_cache.layers[idx] is not None:
                da_cached_layers[idx] = da_cache.layers[idx]

        if not da_cached_layers:
            return {}

        # =====================================================================
        # SLEEP PHASE: recall-aware gradient descent with BCM-like decay
        # =====================================================================
        # =====================================================================
        # KL ANCHOR: cache pre-compact logit distributions on recall pairs
        # =====================================================================
        # This prevents mode collapse / repetition by penalizing divergence
        # from the model's original output distribution (same idea as RLHF's KL penalty).
        kl_anchors = []
        if recall_pairs and kl_weight > 0:
            with torch.no_grad():
                for r_input, r_target in recall_pairs:
                    ref_logits = self.forward(r_input)
                    # Store soft targets at elevated temperature (smoother anchor)
                    kl_anchors.append(
                        F.log_softmax(ref_logits / kl_temperature, dim=-1).detach()
                    )

        # Record pre-compact norms for homeostatic scaling
        pre_norms = {}
        for idx_str, ctm in self.ctm_blocks.items():
            for n, p in ctm.named_parameters():
                pre_norms[f"{idx_str}.{n}"] = p.data.norm().item()

        # Null-space projection: compute top singular directions of c_proj
        # so we can project gradients away from them during optimization
        nullspace_hooks = []
        if nullspace_proj:
            for idx_str, ctm in self.ctm_blocks.items():
                W = ctm.c_proj.weight.data  # (D, n_synch)
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                # Keep top-k directions that carry 90% of energy
                cumulative = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
                k = max(1, (cumulative < 0.90).sum().item() + 1)
                top_U = U[:, :k].detach()  # (D, k) — directions to avoid

                def make_hook(top_dirs):
                    def hook(grad):
                        # Remove gradient component along top singular directions
                        # grad shape: (D, n_synch)
                        proj = top_dirs @ (top_dirs.T @ grad)  # project onto top-k space
                        return grad - proj  # keep only null-space component
                    return hook

                h = ctm.c_proj.weight.register_hook(make_hook(top_U))
                nullspace_hooks.append(h)
                print(f"  nullspace: c_proj L{idx_str} rank90={k}, projecting grads into null space")

        param_groups = []
        for idx_str, ctm in self.ctm_blocks.items():
            for name, p in ctm.named_parameters():
                if not p.requires_grad:
                    continue
                # plastic_only: only update LoRA adapter + gate, freeze everything else
                if plastic_only and 'plastic_' not in name:
                    continue
                if 'plastic_' in name:
                    # LoRA adapter gets higher LR
                    param_groups.append({'params': [p], 'lr': lr * 3.0, 'weight_decay': 0.0})
                elif 'start_state' in name or 'start_trace' in name:
                    param_groups.append({'params': [p], 'lr': lr * 2.0, 'weight_decay': 0.01})
                elif 'c_proj' in name:
                    param_groups.append({'params': [p], 'lr': lr * 1.5, 'weight_decay': 0.005})
                elif 'tick_embed' in name:
                    param_groups.append({'params': [p], 'lr': lr * 0.3, 'weight_decay': 0.01})
                else:
                    param_groups.append({'params': [p], 'lr': lr, 'weight_decay': 0.01})

        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.99))

        self.train()
        losses = []
        all_ctm_params = []
        for ctm in self.ctm_blocks.values():
            all_ctm_params.extend(ctm.parameters())

        # Compute total pre-norm for relative delta tracking
        total_pre_norm = sum(pre_params[k].norm().item() for k in pre_params)

        for step in range(steps):
            optimizer.zero_grad()

            # Teaching loss (with dopamine-shaped cache)
            step_cache = CTMCacheCls(self.config.n_layer)
            for idx, cached in da_cached_layers.items():
                step_cache.layers[idx] = {
                    k: v.detach().clone() for k, v in cached.items()
                }
            _, loss_teach = self.forward(teaching_ids, targets=target_ids,
                                         ctm_cache=step_cache)

            # Recall loss (fresh context — the actual test condition)
            if recall_pairs:
                loss_recall = torch.tensor(0.0, device=device)
                loss_kl = torch.tensor(0.0, device=device)
                for i, (r_input, r_target) in enumerate(recall_pairs):
                    r_logits = self.forward(r_input)
                    # Standard CE recall loss
                    rl = F.cross_entropy(
                        r_logits.reshape(-1, r_logits.size(-1)),
                        r_target.reshape(-1),
                        ignore_index=-1,
                    )
                    loss_recall = loss_recall + rl

                    # KL anchor: penalize divergence from pre-compact distribution
                    if i < len(kl_anchors) and kl_weight > 0:
                        current_log_probs = F.log_softmax(r_logits / kl_temperature, dim=-1)
                        # KL(ref || current) — encourages current to stay close to ref
                        kl = F.kl_div(current_log_probs, kl_anchors[i],
                                      log_target=True, reduction='batchmean')
                        loss_kl = loss_kl + kl

                loss_recall = loss_recall / len(recall_pairs)
                loss_kl = loss_kl / len(recall_pairs) if kl_anchors else loss_kl
                loss = ((1 - recall_weight) * loss_teach
                        + recall_weight * loss_recall
                        + kl_weight * loss_kl)
            else:
                loss = loss_teach

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_ctm_params, 1.0)
            optimizer.step()
            losses.append(loss.item())

            # BCM sliding threshold: as delta grows, reduce LR (metaplasticity)
            # Recent plasticity raises the threshold for future plasticity.
            # This prevents the runaway potentiation that caused repetition collapse.
            if (step + 1) % 5 == 0:
                running_delta = sum(
                    (p - pre_params[f"{idx_str}.{n}"]).norm().item()
                    for idx_str, ctm in self.ctm_blocks.items()
                    for n, p in ctm.named_parameters()
                    if f"{idx_str}.{n}" in pre_params
                )
                rel_delta = running_delta / (total_pre_norm + 1e-8)

                # Early stop if we've exceeded the plasticity budget
                if max_delta > 0 and rel_delta > max_delta:
                    print(f"  compact_memory: early stop at step {step+1}/{steps}, "
                          f"relative delta {rel_delta:.3f} > max {max_delta:.3f} "
                          f"(abs delta {running_delta:.1f})")
                    break

                # BCM decay: scale LR down as we approach the budget
                # At 0% budget used → full LR. At 100% budget → 10% LR.
                bcm_scale = max(0.1, 1.0 - 0.9 * (rel_delta / (max_delta + 1e-8)))
                for group in optimizer.param_groups:
                    base_lr = group.get('initial_lr', group['lr'])
                    if 'initial_lr' not in group:
                        group['initial_lr'] = group['lr']
                    group['lr'] = base_lr * bcm_scale

            # Homeostatic scaling: after each step, clamp per-param norms
            # to at most 50% growth from pre-compact state (prevents excitotoxicity
            # on individual weight matrices while allowing overall 15% relative change)
            with torch.no_grad():
                for idx_str, ctm in self.ctm_blocks.items():
                    for n, p in ctm.named_parameters():
                        key = f"{idx_str}.{n}"
                        if key in pre_norms and pre_norms[key] > 1e-6:
                            current_norm = p.data.norm().item()
                            max_norm = pre_norms[key] * 1.5  # 50% max growth per param
                            if current_norm > max_norm:
                                p.data.mul_(max_norm / current_norm)

        self.eval()

        # Clean up null-space hooks
        for h in nullspace_hooks:
            h.remove()

        # Measure total change (absolute and relative)
        total_delta = 0.0
        total_norm = 0.0
        for idx_str, ctm in self.ctm_blocks.items():
            for n, p in ctm.named_parameters():
                key = f"{idx_str}.{n}"
                if key in pre_params:
                    total_delta += (p - pre_params[key]).norm().item()
                    total_norm += pre_params[key].norm().item()
        rel_delta = total_delta / (total_norm + 1e-8)

        # Sanity check: generate from a known prompt and detect repetition collapse
        collapsed = False
        if sanity_prompt is not None and total_delta > 0:
            with torch.no_grad():
                from nanochat.engine import Engine as _Eng
                _engine = _Eng(self, self._tokenizer if hasattr(self, '_tokenizer') else None)
                if _engine.tokenizer is not None:
                    results, _ = _engine.generate_batch(
                        sanity_prompt, num_samples=1, max_tokens=20,
                        temperature=0.7, seed=42, repetition_penalty=1.3)
                    gen_tokens = results[0][len(sanity_prompt):]
                    # Detect repetition: if any single token is >50% of output, it's collapsed
                    if len(gen_tokens) > 5:
                        from collections import Counter
                        counts = Counter(gen_tokens)
                        most_common_frac = counts.most_common(1)[0][1] / len(gen_tokens)
                        if most_common_frac > 0.5:
                            collapsed = True
                            print(f"  compact_memory: ROLLBACK — repetition collapse detected "
                                  f"(top token {most_common_frac:.0%} of output)")

        # Rollback if collapsed
        if collapsed:
            for idx_str, ctm in self.ctm_blocks.items():
                for n, p in ctm.named_parameters():
                    key = f"{idx_str}.{n}"
                    if key in pre_params:
                        p.data.copy_(pre_params[key])
            total_delta = 0.0

        return {
            'total_delta': total_delta,
            'rel_delta': rel_delta,
            'losses': losses,
            'dopamine_mean': dopamine_per_token.mean().item(),
            'dopamine_std': dopamine_per_token.std().item(),
            'ce_mean': ce_mean.item(),
            'collapsed': collapsed,
            'early_stopped': len(losses) < steps,
        }

    def consolidate(self, replay_batches, lr=1e-4, steps=4):
        """Sleep consolidation: certainty-weighted Hebbian update on replay buffer.

        Gradient-free to avoid VRAM pressure from backward pass + optimizer.
        Uses certainty (1 - normalized entropy) as reward signal.
        Strengthens weights that produce certain+correct predictions,
        decays weights that produce uncertain predictions.

        Args:
            replay_batches: list of (x, y) tensor pairs from replay buffer
            lr: learning rate for consolidation
            steps: ignored (single pass per batch, no optimizer needed)
        Returns: dict with consolidation stats
        """
        import math as _math

        self.eval()
        V = self.config.vocab_size
        max_entropy = _math.log(V)
        stats = {'losses': [], 'mean_certainty': [], 'steps': 0}

        for x, y in replay_batches:
            with torch.no_grad():
                logits = self.forward(x)
                flat_y = y.view(-1)
                logits_flat = logits.view(-1, logits.size(-1))

                # Per-token certainty
                probs = F.softmax(logits_flat, dim=-1)
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
                certainty = (1.0 - entropy / max_entropy).clamp(0, 1)
                correct = (logits_flat.argmax(dim=-1) == flat_y).float()

                # Loss for logging
                token_losses = F.cross_entropy(logits_flat, flat_y, ignore_index=-1, reduction='none')
                loss = token_losses.mean()

                mean_cert = certainty.mean()
                mean_correct = correct.mean()
                consolidation_signal = mean_cert * mean_correct
                scale = 1.0 + lr * (consolidation_signal * 2.0 - 1.0)

                for ctm in self.ctm_blocks.values():
                    for p in ctm.parameters():
                        if p.requires_grad:
                            p.mul_(scale)

                stats['losses'].append(loss.item())
                stats['mean_certainty'].append(mean_cert.item())
                stats['steps'] += 1

        self.train()
        return stats

    def get_ctm_state_dict(self):
        """Get all CTM blocks' state dict + unfrozen backbone layers for checkpointing.

        Returns dict keyed by layer index string, e.g.:
            {"27": {ctm_block_state_dict}, "14": {...}}
        Single-CTM backward compat: if only one block, returns flat state_dict.
        If backbone layers are unfrozen, includes them under "_backbone_layers" key.
        """
        if self._unfrozen_layers:
            # Multi-format with backbone layers included
            result = {idx_str: ctm.state_dict() for idx_str, ctm in self.ctm_blocks.items()}
            qwen_model = self.backbone.model if hasattr(self.backbone, 'model') else self.backbone.transformer
            layers = qwen_model.layers if hasattr(qwen_model, 'layers') else qwen_model.h
            backbone_sd = {}
            for idx in self._unfrozen_layers:
                mlp = layers[idx].mlp if hasattr(layers[idx], 'mlp') else layers[idx].feed_forward
                backbone_sd[str(idx)] = {k: v for k, v in mlp.state_dict().items()}
            result["_backbone_layers"] = backbone_sd
            return result
        if len(self.ctm_blocks) == 1:
            # Backward compatible: single block → flat dict
            return next(iter(self.ctm_blocks.values())).state_dict()
        return {idx_str: ctm.state_dict() for idx_str, ctm in self.ctm_blocks.items()}

    def load_ctm_state_dict(self, state_dict):
        """Load CTM block(s) state dict from checkpoint.

        Handles both formats:
        - Flat dict (single-CTM checkpoint) → loads into last CTM block
        - Keyed dict {"14": {...}, "27": {...}} → loads each block
        """
        # Load unfrozen backbone layers if present
        if "_backbone_layers" in state_dict:
            backbone_sd = state_dict.pop("_backbone_layers")
            qwen_model = self.backbone.model if hasattr(self.backbone, 'model') else self.backbone.transformer
            layers = qwen_model.layers if hasattr(qwen_model, 'layers') else qwen_model.h
            for idx_str, mlp_sd in backbone_sd.items():
                mlp = layers[int(idx_str)].mlp if hasattr(layers[int(idx_str)], 'mlp') else layers[int(idx_str)].feed_forward
                mlp.load_state_dict(mlp_sd)
                print0(f"  Loaded unfrozen backbone MLP L{idx_str}")

        # Detect format: if keys are layer index strings with dict values, it's multi-CTM
        sample_key = next(iter(state_dict))
        if sample_key in self.ctm_blocks and isinstance(state_dict[sample_key], dict):
            # Multi-CTM format
            for idx_str, block_sd in state_dict.items():
                if idx_str in self.ctm_blocks:
                    self.ctm_blocks[idx_str].load_state_dict(block_sd, strict=False)
                    print0(f"  Loaded CTM L{idx_str} weights")
        else:
            # Flat format (single-CTM checkpoint) → load into last block
            last_idx = str(self.ctm_layer_indices[-1])
            self.ctm_blocks[last_idx].load_state_dict(state_dict, strict=False)
            print0(f"  Loaded single-CTM weights into L{last_idx}")

    @property
    def n_layer(self):
        return self.config.n_layer

    @property
    def n_head(self):
        return self.config.n_head

    @property
    def n_embd(self):
        return self.config.n_embd
