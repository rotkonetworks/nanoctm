"""
Qwen2.5 backbone with CTMBlock on last layer.

Loads pretrained Qwen2.5-0.5B (or similar) as a frozen backbone,
replaces the last layer's MLP with our CTMBlock. Only the CTM
parameters are trainable.

Usage:
    model = QwenBackboneGPT.from_pretrained("Qwen/Qwen2.5-0.5B", ctm_kwargs={...})
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
    """Qwen2.5 backbone with CTMBlock replacing the last layer's MLP.

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
    def from_pretrained(cls, model_name="Qwen/Qwen2.5-0.5B", ctm_kwargs=None,
                        ctm_layer_idx=-1, device="cpu"):
        """Load Qwen backbone and attach CTMBlock."""
        from transformers import AutoModelForCausalLM
        print0(f"Loading backbone: {model_name}")
        backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
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
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
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
            tick_logits_flat = tick_logits.view(-1, V)

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

    def compact_memory(self, ctm_cache, lr=1e-5):
        """Hebbian plasticity update on CTM synapses."""
        return self.ctm_block.compact_memory_standalone(ctm_cache, lr) if hasattr(self.ctm_block, 'compact_memory_standalone') else {}

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
