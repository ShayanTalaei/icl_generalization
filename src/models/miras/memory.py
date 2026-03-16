"""Memory structure modules.

Each memory structure owns its parameter state (per-batch, not nn.Parameters),
and provides read (forward) and backward (gradient computation) operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .state_utils import MemoryState, Grads


class MemoryStructure(nn.Module):
    """Base class for memory architecture."""

    def init_state(self, B: int, device, dtype) -> MemoryState:
        """Initialize per-batch memory state (zeros)."""
        raise NotImplementedError

    def read(self, state: MemoryState, query: Tensor) -> Tensor:
        """Read from memory. Returns (B, d_v)."""
        raise NotImplementedError

    def backward(self, state: MemoryState, key: Tensor, d_out: Tensor) -> Grads:
        """Compute parameter gradients given output-space error signal.

        Recomputes the forward pass internally (does not cache from read).

        Args:
            state: current memory parameters
            key: (B, d_k) input key
            d_out: (B, d_v) error signal from attentional bias

        Returns:
            Gradients in same structure as state.
        """
        raise NotImplementedError


class MatrixMemory(MemoryStructure):
    """Matrix-valued memory M in R^{d_v x d_k}.

    Read:     y = M @ q
    Backward: dM = d_out outer k
    """

    def __init__(self, d_k: int, d_v: int):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v

    def init_state(self, B: int, device, dtype) -> Tensor:
        return torch.zeros(B, self.d_v, self.d_k, device=device, dtype=dtype)

    def read(self, state: Tensor, query: Tensor) -> Tensor:
        return torch.bmm(state, query.unsqueeze(-1)).squeeze(-1)

    def backward(self, state: Tensor, key: Tensor, d_out: Tensor) -> Tensor:
        # d_out (B, d_v, 1) @ key (B, 1, d_k) -> (B, d_v, d_k)
        return torch.bmm(d_out.unsqueeze(-1), key.unsqueeze(-2))


class MLPMemory(MemoryStructure):
    """2-layer MLP memory: M(x) = W1 * silu(W2 * x) [+ x if d_k == d_v].

    State is per-batch (W1, W2) tensors.
    W1 is zero-initialized; W2 gets small random init.
    Residual connection when d_k == d_v.
    """

    def __init__(self, d_k: int, d_v: int, d_hidden: int):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_hidden = d_hidden
        self.residual = (d_k == d_v)

    def init_state(self, B: int, device, dtype):
        W1 = torch.zeros(B, self.d_v, self.d_hidden, device=device, dtype=dtype)
        W2 = torch.randn(B, self.d_hidden, self.d_k, device=device, dtype=dtype) * (
            1.0 / self.d_k**0.5
        )
        return (W1, W2)

    def read(self, state, query: Tensor) -> Tensor:
        W1, W2 = state
        x = query.unsqueeze(-1)                          # (B, d_k, 1)
        h = F.silu(torch.bmm(W2, x))                     # (B, d_hidden, 1)
        out = torch.bmm(W1, h).squeeze(-1)                # (B, d_v)
        if self.residual:
            out = out + query
        return out

    def backward(self, state, key: Tensor, d_out: Tensor):
        """Manual backprop through the MLP given output-space error signal.

        Recomputes the forward pass to obtain intermediate activations.
        Returns (dW1, dW2) gradients.
        """
        W1, W2 = state

        # Recompute forward pass
        x = key.unsqueeze(-1)                              # (B, d_k, 1)
        pre = torch.bmm(W2, x)                             # (B, d_hidden, 1)
        h = F.silu(pre)                                     # (B, d_hidden, 1)

        d_out_col = d_out.unsqueeze(-1)                     # (B, d_v, 1)

        # dW1 = d_out @ h^T
        dW1 = torch.bmm(d_out_col, h.transpose(-1, -2))    # (B, d_v, d_hidden)

        # d_h = W1^T @ d_out
        d_h = torch.bmm(W1.transpose(-1, -2), d_out_col)   # (B, d_hidden, 1)

        # SiLU derivative: silu'(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
        sig = torch.sigmoid(pre)
        silu_grad = sig * (1.0 + pre * (1.0 - sig))
        d_pre = d_h * silu_grad                             # (B, d_hidden, 1)

        # dW2 = d_pre @ k^T
        dW2 = torch.bmm(d_pre, x.transpose(-1, -2))        # (B, d_hidden, d_k)

        return (dW1, dW2)
