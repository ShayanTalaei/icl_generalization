"""MIRASLayer: composition of the 4 MIRAS components.

Implements the two-phase write protocol:
    Phase 1 (error signal): bias computes d_out from prediction and target
    Phase 2 (parameter update): memory computes gradients, retention regularizes,
        algorithm applies the update step.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .state_utils import MemoryState, OptimState
from .memory import MemoryStructure
from .bias import AttentionalBias
from .retention import RetentionGate
from .algorithm import MemoryAlgorithm


class MIRASLayer(nn.Module):
    """Single MIRAS memory layer composing 4 axes.

    Learnable parameters:
        eta: inner-loop learning rate
        alpha: retention strength
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
        self.eta = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(1.0))

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
