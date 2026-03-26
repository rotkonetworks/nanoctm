"""
Fused sync accumulator update for CTMBlock.

Replaces 6 indexed gathers + 4 elementwise ops + 2 accumulates with a single
Triton kernel. One pass over memory, one kernel launch per tick.

The unfused version (from CTMBlock.forward):
    left_out = state[:, synch_out_left]       # gather
    right_out = state[:, synch_out_right]     # gather
    pp_out = left_out * right_out * dopamine  # 2 muls
    alpha_out = r_out * alpha_out + pp_out    # mul + add
    beta_out = r_out * beta_out + dopamine    # mul + add
    (same for _act pair)

Each gather is random-access into (BT, D) — terrible cache locality for
n_synch pairs with random indices into D dimensions.

The fused version: one kernel, BT * n_synch threads, each thread:
  1. Load state[bt, left_idx] and state[bt, right_idx] (2 random reads)
  2. Compute pp = left * right * dopamine
  3. Update alpha = r * alpha + pp, beta = r * beta + dopamine
  4. Store alpha, beta (2 sequential writes)

Saves: 4 kernel launches, 4 intermediate tensors, 2x memory traversal.
"""
import torch
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _sync_update_kernel(
        # Pointers
        state_ptr, alpha_ptr, beta_ptr,
        left_idx_ptr, right_idx_ptr,
        r_ptr,  # decay: (n_synch,)
        # Scalars
        dopamine,  # float
        BT: tl.constexpr, D: tl.constexpr, n_synch: tl.constexpr,
        # Strides
        state_stride_bt: tl.constexpr,
    ):
        """Update one sync accumulator pair (out or act) for all BT positions."""
        pid = tl.program_id(0)
        bt = pid // n_synch
        s = pid % n_synch

        if bt >= BT:
            return

        # Load random indices for this sync pair
        left_idx = tl.load(left_idx_ptr + s)
        right_idx = tl.load(right_idx_ptr + s)

        # Gather from state
        left_val = tl.load(state_ptr + bt * state_stride_bt + left_idx)
        right_val = tl.load(state_ptr + bt * state_stride_bt + right_idx)

        # Pairwise product with dopamine gating
        pp = left_val * right_val * dopamine

        # Load decay and current accumulators
        r = tl.load(r_ptr + s)
        offset = bt * n_synch + s
        alpha = tl.load(alpha_ptr + offset)
        beta = tl.load(beta_ptr + offset)

        # Exponential moving average update
        alpha = r * alpha + pp
        beta = r * beta + dopamine

        # Store
        tl.store(alpha_ptr + offset, alpha)
        tl.store(beta_ptr + offset, beta)


def fused_sync_update(state, alpha, beta, left_idx, right_idx, r, dopamine):
    """Drop-in replacement for the 6-line sync update in CTMBlock.forward.

    Args:
        state: (BT, D) neuron activations
        alpha: (BT, n_synch) running sync numerator — MODIFIED IN PLACE
        beta: (BT, n_synch) running sync denominator — MODIFIED IN PLACE
        left_idx: (n_synch,) long — left neuron indices
        right_idx: (n_synch,) long — right neuron indices
        r: (1, n_synch) decay factors
        dopamine: float or scalar tensor
    """
    if not HAS_TRITON or not state.is_cuda:
        # Fallback: unfused PyTorch (identical semantics)
        left = state[:, left_idx]
        right = state[:, right_idx]
        pp = left * right * dopamine
        alpha.mul_(r).add_(pp)
        beta.mul_(r).add_(dopamine)
        return

    BT, D = state.shape
    n_synch = alpha.shape[1]

    # Resolve dopamine to float
    if isinstance(dopamine, torch.Tensor):
        dopamine = dopamine.item() if dopamine.numel() == 1 else dopamine.mean().item()

    grid = (BT * n_synch,)
    _sync_update_kernel[grid](
        state, alpha, beta,
        left_idx, right_idx,
        r.squeeze(0) if r.dim() > 1 else r,
        dopamine,
        BT, D, n_synch,
        state.stride(0),
    )
