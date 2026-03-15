"""Retention gate modules (memory regularizers).

Retention gates control how much of the previous memory state is preserved
vs. overwritten. In the MIRAS framework, these correspond to the regularization
term in the learning-retaining viewpoint.

"Forget gates" are reinterpreted as specific implementations of retention:
the model optimizes how much of the past to retain by balancing regularization
against the learning objective.
"""

import torch.nn as nn
from torch import Tensor

from .state_utils import MemoryState, state_map


class RetentionGate(nn.Module):
    """Base class for retention gate (memory regularizer)."""

    def apply(
        self, state: MemoryState, alpha: Tensor, key: Tensor | None = None
    ) -> MemoryState:
        """Apply retention to memory state.

        Args:
            state: current memory state
            alpha: retention strength parameter
            key: optional, for input-dependent retention (future)

        Returns:
            Regularized memory state.
        """
        raise NotImplementedError


class NoRetention(RetentionGate):
    """No retention: state passes through unchanged.

    Used by: Linear Attention, DeltaNet, TTT-Linear, TTT-MLP.
    """

    def apply(self, state, alpha, key=None):
        return state


class ScalarL2Retention(RetentionGate):
    """Scalar L2 retention: state *= alpha.

    Equivalent to L2 regularization in the FTRL viewpoint.
    Used by: Mamba, Lightning Attention, Titans, Gated DeltaNet.
    """

    def apply(self, state, alpha, key=None):
        return state_map(lambda s: alpha * s, state)
