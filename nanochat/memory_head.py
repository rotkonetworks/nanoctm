"""Memory head: direct logit injection from CTM sync signal.

Bypasses the residual stream entirely. The hippocampus doesn't
modify cortical representations — it projects directly to output.

Architecture:
    base_logits = lm_head(norm(residual_stream))   # cortex → speech
    memory_logits = memory_head(sync) * gate(sync)  # hippocampus → speech
    final_logits = base_logits + memory_logits       # combined decision

The memory head is low-rank: sync → W_down → W_up → vocab logits.
Per-alter memory heads enable plural recall without interference.
Gate ensures zero contribution when no memory is relevant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional


class MemoryHead(nn.Module):
    """Low-rank projection from sync to vocab logits, per alter.

    sync (n_synch) → down (rank) → up (vocab_size) → logit delta
    Gated by sync novelty: only fires when current context matches stored memory.
    """

    def __init__(self, n_synch: int, vocab_size: int, rank: int = 32,
                 device='cpu', dtype=torch.float32):
        super().__init__()
        self.n_synch = n_synch
        self.vocab_size = vocab_size
        self.rank = rank

        # Low-rank logit projection: sync → rank → vocab
        self.W_down = nn.Parameter(
            torch.zeros(n_synch, rank, device=device, dtype=dtype))
        self.W_up = nn.Parameter(
            torch.zeros(rank, vocab_size, device=device, dtype=dtype))

        # Gate: scalar from sync, controls memory contribution
        self.W_gate = nn.Parameter(
            torch.zeros(n_synch, device=device, dtype=dtype))
        self.gate_bias = nn.Parameter(
            torch.tensor(-3.0, device=device, dtype=dtype))  # starts closed

        # Baseline sync for novelty detection
        self.register_buffer('baseline_sync',
                             torch.zeros(n_synch, device=device))
        self.calibrated = False

        # Per-alter storage
        self.alter_memories: Dict[str, dict] = {}
        # Memory keys: sync patterns from teaching contexts
        self._memory_keys: List[torch.Tensor] = []

    @torch.no_grad()
    def calibrate_baseline(self, sync_samples: torch.Tensor):
        """Set baseline sync from typical inference. Gate stays closed
        when sync matches baseline (nothing novel = no memory to recall)."""
        self.baseline_sync.copy_(sync_samples.float().mean(0).to(self.baseline_sync.device))
        self.calibrated = True

    def forward(self, sync: torch.Tensor,
                hidden_state: torch.Tensor = None) -> torch.Tensor:
        """Produce gated logit delta from sync signal.

        Args:
            sync: (BT, n_synch) — for logit projection
            hidden_state: (BT, D) — CTM output for gating (more discriminative than sync)

        Returns:
            logit_delta: (BT, vocab_size) — add to base logits
        """
        s = sync.float()

        # Gate: use hidden_state if available (0.63-0.88 cosine sim vs 0.89-0.98 for sync)
        gate_signal = hidden_state.float() if hidden_state is not None else s
        if self.calibrated and hasattr(self, '_memory_keys') and len(self._memory_keys) > 0:
            keys = torch.stack(self._memory_keys).to(gate_signal.device)
            mean_g = gate_signal.mean(0, keepdim=True)
            sims = F.cosine_similarity(
                mean_g.unsqueeze(1),
                keys.unsqueeze(0),
                dim=-1)
            best_sim = sims.max(dim=-1).values
            # Hidden state sims: 0.63-0.88 cross, higher for same-context
            # Threshold at 0.85: opens for close matches, closed for distant
            gate_val = torch.sigmoid(40.0 * (best_sim - 0.85))
            gate = gate_val.expand(s.size(0), 1)
        else:
            gate = torch.zeros(s.size(0), 1, device=s.device)

        # Low-rank logit projection (still from sync — it has the memory content)
        hidden = s @ self.W_down
        logits = hidden @ self.W_up

        return gate * logits

    @torch.no_grad()
    def teach(self, sync_samples: torch.Tensor,
              target_token_ids: torch.Tensor,
              lm_head_weight: torch.Tensor,
              base_logits: torch.Tensor,
              strength: float = 10.0,
              alter_name: str = "default",
              hidden_state_samples: torch.Tensor = None):
        """Teach the memory head to recall specific tokens.

        No gradients. Computes the logit delta needed to make target
        tokens win, then solves for W_down, W_up via least-squares.

        Args:
            sync_samples: (N, n_synch) sync patterns at teaching positions
            target_token_ids: (N,) correct next tokens
            lm_head_weight: (V, D) the lm_head projection matrix
            base_logits: (N, V) what the model currently predicts
            strength: how much to boost target token logit
            alter_name: which alter this memory belongs to
        """
        N = sync_samples.size(0)
        V = self.vocab_size
        S = sync_samples.float().to(self.W_down.device)

        # Compute target logit delta:
        # For each position, we want target token's logit to be highest
        # Strategy: boost target by +strength, suppress top competing token
        target_logit_delta = torch.zeros(N, V, device=S.device)
        for i in range(N):
            tid = target_token_ids[i].item()
            target_logit_delta[i, tid] = strength

            # Also suppress the current argmax if it's wrong
            current_top = base_logits[i].argmax().item()
            if current_top != tid:
                target_logit_delta[i, current_top] = -strength * 0.3

        # Solve: S @ W_down @ W_up = target_logit_delta
        # Two-step least-squares via SVD of target:
        # 1. SVD of target_logit_delta to get rank-r approximation
        # 2. Solve S @ W_down = U_r @ Sigma_r, W_up = V_r^T

        U, Sigma, Vt = torch.linalg.svd(target_logit_delta, full_matrices=False)
        r = min(self.rank, len(Sigma))

        # Truncate to rank r
        U_r = U[:, :r]           # (N, r)
        S_r = Sigma[:r]          # (r,)
        Vt_r = Vt[:r, :]         # (r, V)

        # Target for W_down: S @ W_down = U_r @ diag(S_r)
        target_down = U_r * S_r.unsqueeze(0)  # (N, r)

        # Solve for W_down: least-squares S @ W_down = target_down
        StS = S.T @ S + 1e-4 * torch.eye(S.size(1), device=S.device)
        StT = S.T @ target_down
        W_down_opt = torch.linalg.solve(StS, StT)  # (n_synch, r)

        # W_up = Vt_r (the right singular vectors capture vocab patterns)
        W_up_opt = Vt_r  # (r, V)

        # Pad/trim to match self.rank
        if r < self.rank:
            pad_down = torch.zeros(self.n_synch, self.rank - r, device=S.device)
            W_down_opt = torch.cat([W_down_opt, pad_down], dim=1)
            pad_up = torch.zeros(self.rank - r, V, device=S.device)
            W_up_opt = torch.cat([W_up_opt, pad_up], dim=0)

        # Blend into current weights (accumulate, don't overwrite)
        blend = 0.5
        self.W_down.data = (
            (1 - blend) * self.W_down.data + blend * W_down_opt).to(self.W_down.dtype)
        self.W_up.data = (
            (1 - blend) * self.W_up.data + blend * W_up_opt).to(self.W_up.dtype)

        # Open the gate: set gate weights to respond to this sync pattern
        # The gate should open when sync is similar to teaching sync
        mean_sync = S.mean(0)
        gate_direction = mean_sync / (mean_sync.norm() + 1e-8)
        # Blend gate direction (accumulate across teachings)
        self.W_gate.data = (
            (1 - blend) * self.W_gate.data +
            blend * gate_direction * 2.0  # scale so sigmoid opens
        ).to(self.W_gate.dtype)
        # Shift bias toward opening
        self.gate_bias.data = torch.tensor(
            max(self.gate_bias.item() + 0.5, -1.0),
            device=self.gate_bias.device)

        # Store memory key from hidden state (more discriminative than sync)
        if hidden_state_samples is not None:
            key = hidden_state_samples.float().mean(0).cpu()
        else:
            key = mean_sync.cpu()
        self._memory_keys.append(key)

        # Store for this alter
        self.alter_memories[alter_name] = {
            'sync_mean': mean_sync.cpu(),
            'target_tokens': target_token_ids.cpu(),
            'n_samples': N,
        }

        # Verify: what would we produce now?
        with torch.no_grad():
            test_hidden = hidden_state_samples[:5].to(S.device) if hidden_state_samples is not None else None
            test_logits = self.forward(S[:5], hidden_state=test_hidden)
            test_preds = test_logits.argmax(-1)
            test_targets = target_token_ids[:5]
            match = (test_preds == test_targets.to(test_preds.device)).float().mean()

        return {
            'n_samples': N,
            'rank_used': r,
            'train_match': match.item(),
            'gate_bias': self.gate_bias.item(),
            'alter': alter_name,
        }

    def get_state(self):
        return {
            'W_down_norm': self.W_down.data.norm().item(),
            'W_up_norm': self.W_up.data.norm().item(),
            'gate_bias': self.gate_bias.item(),
            'calibrated': self.calibrated,
            'n_alters': len(self.alter_memories),
            'alters': {k: v['n_samples'] for k, v in self.alter_memories.items()},
        }
