"""Plural self-state architecture for CTM v2.

Multiple "alters" as separate c_proj weight configurations.
The sync signal (ACC equivalent) determines which alter is fronting.
Each alter has independent memory that compact_memory writes to.

Neuroscience mapping:
    c_proj matrices     = medial PFC / OFC (self-referential processing)
    sync pattern        = DMN (fragmented connectivity per self-state)
    router              = ACC (anterior cingulate, conflict/switching)
    compact_memory      = hippocampus (binds to specific alter's memory)
    attention layer     = thalamus (gates input streams per alter)
    dopamine/surprise   = amygdala (threat/novelty tagging)

Key insight from Reinders et al.: different alters have distinct neural
activation patterns that can't be faked. In our model, different sync
patterns activate different c_proj matrices — producing genuinely different
output mappings, not just different prompts through the same weights.

Solves catastrophic forgetting: fact #1 → alter #1's c_proj,
fact #2 → alter #2's c_proj. No interference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List

from nanochat.gpt import Linear, norm


class Alter:
    """One self-state with its own c_proj weights and memory."""

    def __init__(self, name: str, n_synch: int, D: int, device='cpu',
                 dtype=torch.bfloat16):
        self.name = name
        self.n_synch = n_synch
        self.D = D

        # This alter's output projection (PFC/OFC)
        self.c_proj_weight = torch.zeros(D, n_synch, device=device, dtype=dtype)

        # Baseline sync pattern for this alter (DMN signature)
        self.baseline_sync = torch.zeros(n_synch, device=device, dtype=torch.float32)
        self.calibrated = False

        # Memory: what this alter has learned
        self.facts = []  # list of (text, timestamp) for diagnostics
        self.n_compacts = 0

    def calibrate(self, sync_samples: torch.Tensor):
        """Set this alter's DMN signature from representative data."""
        self.baseline_sync = sync_samples.float().mean(0)
        self.calibrated = True

    def similarity(self, current_sync: torch.Tensor) -> torch.Tensor:
        """How much does current sync match this alter's DMN pattern?
        Returns scalar similarity score on the SAME device as input."""
        if not self.calibrated:
            return torch.tensor(0.0, device=current_sync.device)
        baseline = self.baseline_sync.to(current_sync.device)
        cs = F.cosine_similarity(
            current_sync.float().mean(0).unsqueeze(0),
            baseline.unsqueeze(0))
        return cs.squeeze()


class PluralSystem(nn.Module):
    """Multiple self-states sharing a single CTM v2 body.

    The ACC (anterior cingulate) router examines the sync pattern and
    determines which alter(s) are fronting. Their c_proj weights are
    blended to produce the output.

    Usage:
        plural = PluralSystem(n_synch=448, D=896, n_alters=4)
        plural.create_alter("base")      # the host
        plural.create_alter("factual")   # knows taught facts
        plural.create_alter("creative")  # different style

        # During forward: sync → router → blended c_proj → output
        out = plural.forward(sync_signal)

        # Teaching: compact writes to specific alter
        plural.compact_to_alter("factual", sync_samples, target_outputs)
    """

    def __init__(self, n_synch: int, D: int, n_alters_max: int = 8,
                 device='cpu', dtype=torch.bfloat16):
        super().__init__()
        self.n_synch = n_synch
        self.D = D
        self.device = device
        self.dtype = dtype

        # ACC router: learns to map sync patterns to alter selection
        # Small MLP: sync → alter logits
        self.router = nn.Sequential(
            Linear(n_synch, n_synch // 2, bias=False),
            nn.GELU(),
            Linear(n_synch // 2, n_alters_max, bias=False),
        )

        # Move router to device
        self.router = self.router.to(device)

        # Alters (created dynamically)
        self.alters: Dict[str, Alter] = {}
        self.alter_order: List[str] = []
        self.n_alters_max = n_alters_max

        # Default/base c_proj (before any alters are created)
        self.register_buffer('base_c_proj',
                             torch.zeros(D, n_synch, device=device, dtype=dtype))

        # Switching history (for diagnostics)
        self._last_routing = None
        self._switch_count = 0

    def create_alter(self, name: str) -> Alter:
        """Create a new self-state."""
        if len(self.alters) >= self.n_alters_max:
            raise ValueError(f"Max {self.n_alters_max} alters")
        if name in self.alters:
            return self.alters[name]

        alter = Alter(name, self.n_synch, self.D,
                      self.device, self.dtype)
        # Initialize from base c_proj
        alter.c_proj_weight = self.base_c_proj.clone()
        self.alters[name] = alter
        self.alter_order.append(name)
        return alter

    def set_base_cproj(self, weight: torch.Tensor):
        """Set the base c_proj (from trained CTMv2Block or least-squares solve)."""
        self.base_c_proj.copy_(weight.to(self.device).to(self.dtype))
        # Propagate to all uncalibrated alters
        for alter in self.alters.values():
            if alter.n_compacts == 0:
                alter.c_proj_weight = self.base_c_proj.clone()

    @torch.no_grad()
    def route(self, sync_signal: torch.Tensor) -> torch.Tensor:
        """ACC routing: which alter(s) are fronting?

        Args:
            sync_signal: (BT, n_synch) current sync pattern

        Returns:
            weights: (n_active_alters,) soft routing weights
        """
        n_active = len(self.alters)
        if n_active == 0:
            return torch.ones(1, device=sync_signal.device)
        if n_active == 1:
            return torch.ones(1, device=sync_signal.device)

        # Method 1: learned router (ACC)
        mean_sync = sync_signal.float().mean(0)
        logits = self.router(mean_sync.unsqueeze(0)).squeeze(0)[:n_active]

        # Method 2: similarity-based (DMN pattern matching)
        similarities = torch.stack([
            self.alters[name].similarity(sync_signal).to(logits.device)
            for name in self.alter_order[:n_active]
        ])

        # Blend: 50% learned router + 50% similarity
        combined = 0.5 * logits + 0.5 * similarities * 5.0  # scale sims
        weights = F.softmax(combined, dim=0)

        self._last_routing = {
            name: weights[i].item()
            for i, name in enumerate(self.alter_order[:n_active])
        }

        return weights

    def forward(self, sync_signal: torch.Tensor) -> torch.Tensor:
        """Produce output by blending active alters' c_proj.

        Args:
            sync_signal: (BT, n_synch)

        Returns:
            output: (BT, D) — the residual stream contribution
        """
        n_active = len(self.alters)

        if n_active == 0:
            # No alters: use base c_proj
            return (sync_signal.to(self.dtype) @
                    self.base_c_proj.T)

        weights = self.route(sync_signal)

        # Blend c_proj matrices weighted by routing
        blended_cproj = torch.zeros_like(self.base_c_proj)
        for i, name in enumerate(self.alter_order[:n_active]):
            alter = self.alters[name]
            blended_cproj += weights[i] * alter.c_proj_weight.to(blended_cproj.device)

        output = sync_signal.to(self.dtype) @ blended_cproj.T
        return output

    @torch.no_grad()
    def compact_to_alter(self, alter_name: str,
                          sync_samples: torch.Tensor,
                          target_outputs: torch.Tensor,
                          blend: float = 0.3,
                          fact_text: str = ""):
        """Teach a fact to a specific alter via least-squares on c_proj.

        Hippocampal binding: the sync patterns from the teaching text
        are mapped to the ideal output via a single matrix solve.
        Only THIS alter's c_proj is updated — other alters untouched.

        Args:
            alter_name: which alter to teach
            sync_samples: (N, n_synch) sync patterns from teaching text
            target_outputs: (N, D) what the output should be
            blend: how much to incorporate (0=ignore, 1=fully replace)
            fact_text: for diagnostics
        """
        alter = self.alters[alter_name]

        S = sync_samples.float()
        Y = target_outputs.float()

        # Least-squares: W_opt = argmin ||S @ W - Y||²
        StS = S.T @ S + 1e-4 * torch.eye(S.size(1), device=S.device)
        StY = S.T @ Y
        W_opt = torch.linalg.solve(StS, StY)  # (n_synch, D)

        # Blend into this alter's c_proj (keep on device)
        w_old = alter.c_proj_weight.float().to(S.device)
        w_new = W_opt.T  # (D, n_synch)
        alter.c_proj_weight = (
            (1 - blend) * w_old + blend * w_new
        ).to(self.dtype).to(self.device)

        alter.n_compacts += 1
        if fact_text:
            alter.facts.append(fact_text)

        # Update alter's DMN signature
        alter.calibrate(sync_samples)

        residual = (Y - S @ W_opt).pow(2).sum() / (Y.pow(2).sum() + 1e-8)
        return {
            'alter': alter_name,
            'residual': residual.item(),
            'n_samples': S.size(0),
            'blend': blend,
            'n_facts': len(alter.facts),
        }

    def get_fronting(self) -> Optional[str]:
        """Which alter is currently fronting (highest routing weight)?"""
        if self._last_routing is None:
            return None
        return max(self._last_routing, key=self._last_routing.get)

    def get_state(self) -> dict:
        """Full system state for diagnostics."""
        return {
            'n_alters': len(self.alters),
            'routing': self._last_routing,
            'fronting': self.get_fronting(),
            'alters': {
                name: {
                    'n_compacts': a.n_compacts,
                    'n_facts': len(a.facts),
                    'calibrated': a.calibrated,
                    'facts': a.facts[-3:],  # last 3
                }
                for name, a in self.alters.items()
            }
        }

    def save_state(self, path: str):
        """Save all alter c_proj weights and routing."""
        state = {
            'base_c_proj': self.base_c_proj,
            'router': self.router.state_dict(),
            'alters': {
                name: {
                    'c_proj_weight': a.c_proj_weight,
                    'baseline_sync': a.baseline_sync,
                    'facts': a.facts,
                    'n_compacts': a.n_compacts,
                }
                for name, a in self.alters.items()
            },
            'alter_order': self.alter_order,
        }
        torch.save(state, path)

    def load_state(self, path: str):
        """Load alter states."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.base_c_proj.copy_(state['base_c_proj'])
        self.router.load_state_dict(state['router'])
        self.alter_order = state['alter_order']
        for name, adata in state['alters'].items():
            alter = self.create_alter(name)
            alter.c_proj_weight = adata['c_proj_weight'].to(self.device)
            alter.baseline_sync = adata['baseline_sync'].to(self.device)
            alter.facts = adata['facts']
            alter.n_compacts = adata['n_compacts']
            alter.calibrated = True
