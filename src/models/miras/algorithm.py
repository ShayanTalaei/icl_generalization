"""Memory algorithm modules (inner-loop optimizers).

Each algorithm defines how gradients are applied to update memory state.
Stateless algorithms (GD) return None as optimizer state.
Stateful algorithms (GDMomentum, Muon) maintain buffers in OptimState.
"""

import torch.nn as nn
from torch import Tensor

from .state_utils import MemoryState, Grads, OptimState, state_map
from .memory import MemoryStructure


class MemoryAlgorithm(nn.Module):
    """Base class for memory update algorithm (inner-loop optimizer)."""

    def init_optim_state(
        self, memory: MemoryStructure, B: int, device, dtype
    ) -> OptimState:
        """Initialize optimizer state. None for stateless algorithms."""
        return None

    def step(
        self,
        state: MemoryState,
        grads: Grads,
        eta: Tensor,
        optim_state: OptimState,
    ) -> tuple[MemoryState, OptimState]:
        """Apply one optimization step.

        Args:
            state: current memory parameters
            grads: parameter gradients from memory.backward
            eta: learning rate
            optim_state: optimizer state (e.g., momentum buffer)

        Returns:
            (updated_state, updated_optim_state)
        """
        raise NotImplementedError


class GD(MemoryAlgorithm):
    """Gradient descent: state = state + eta * grads.

    Stateless (no optimizer state).
    Used by: Linear Attention, DeltaNet, Mamba, GLA, TTT, Gated DeltaNet.
    """

    def step(self, state, grads, eta, optim_state):
        new_state = state_map(lambda s, g: s + eta * g, state, grads)
        return new_state, None
