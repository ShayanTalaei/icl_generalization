"""MIRASLayer: composition of the 4 MIRAS components.

Implements the two-phase write protocol:
    Phase 1 (error signal): bias computes d_out from prediction and target
    Phase 2 (parameter update): memory computes gradients, retention regularizes,
        algorithm applies the update step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .state_utils import MemoryState, OptimState
from .memory import MemoryStructure
from .bias import AttentionalBias
from .retention import RetentionGate
from .algorithm import MemoryAlgorithm


def _inverse_softplus(x: float) -> float:
    """Compute the inverse of softplus: log(exp(x) - 1)."""
    return float(torch.tensor(x).expm1().log())


class MIRASLayer(nn.Module):
    """Single MIRAS memory layer composing 4 axes.

    Learnable parameters:
        _log_eta: unconstrained parameter; eta = softplus(_log_eta) > 0
        _log_alpha: unconstrained parameter; alpha = softplus(_log_alpha) > 0
    """

    def __init__(
        self,
        memory: MemoryStructure,
        bias: AttentionalBias,
        retention: RetentionGate,
        algorithm: MemoryAlgorithm,
    ):
        super().__init__()
        self.memory = memory
        self.bias = bias
        self.retention = retention
        self.algorithm = algorithm
        # Store unconstrained parameters; use softplus to ensure positivity
        self._log_eta = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self._log_alpha = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))

    @property
    def eta(self) -> Tensor:
        """Inner-loop learning rate (always positive via softplus)."""
        return F.softplus(self._log_eta)

    @eta.setter
    def eta(self, value):
        """Allow setting eta for backward compatibility (e.g., eta.fill_())."""
        # This is a no-op setter; use set_eta() instead
        pass

    @property
    def alpha(self) -> Tensor:
        """Retention strength (always positive via softplus)."""
        return F.softplus(self._log_alpha)

    @alpha.setter
    def alpha(self, value):
        pass

    def set_eta(self, value: float):
        """Set eta to a specific positive value."""
        with torch.no_grad():
            self._log_eta.fill_(_inverse_softplus(value))

    def set_alpha(self, value: float):
        """Set alpha to a specific positive value."""
        with torch.no_grad():
            self._log_alpha.fill_(_inverse_softplus(value))

    def init_state(
        self, B: int, device, dtype
    ) -> tuple[MemoryState, OptimState]:
        mem_state = self.memory.init_state(B, device, dtype)
        optim_state = self.algorithm.init_optim_state(
            self.memory, B, device, dtype
        )
        return mem_state, optim_state

    def read(self, state: tuple[MemoryState, OptimState], query: Tensor) -> Tensor:
        mem_state, _ = state
        return self.memory.read(mem_state, query)

    def write(
        self,
        state: tuple[MemoryState, OptimState],
        key: Tensor,
        value: Tensor,
    ) -> tuple[MemoryState, OptimState]:
        mem_state, optim_state = state

        # Phase 1: error signal (bias-dependent, memory-independent)
        pred = self.memory.read(mem_state, key)
        d_out = self.bias.error_signal(pred, value)

        # Phase 2a: gradients (memory-dependent)
        grads = self.memory.backward(mem_state, key, d_out)

        # Phase 2b: retention (regularize before gradient step)
        mem_state = self.retention.apply(mem_state, self.alpha, key=key)

        # Phase 2c: optimizer step
        mem_state, optim_state = self.algorithm.step(
            mem_state, grads, self.eta, optim_state
        )

        return mem_state, optim_state
