"""Attentional bias modules (internal learning objectives).

The attentional bias defines the objective function for the associative memory.
Each bias implements error_signal(), which computes the output-space direction
to move the memory output toward.

Sign convention: error_signal returns the direction that, when propagated through
the memory's backward pass and added via the algorithm, improves the objective.
For minimization objectives, this is the negative gradient w.r.t. the output.
For maximization objectives (dot-product), this is the positive gradient.
"""

import torch.nn as nn
from torch import Tensor


class AttentionalBias(nn.Module):
    """Base class for attentional bias (internal learning objective)."""

    def error_signal(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Compute output-space error signal.

        Args:
            prediction: (B, d_v) -- memory read output
            target: (B, d_v) -- desired output

        Returns:
            d_out: (B, d_v) -- error signal in output space
        """
        raise NotImplementedError


class DotProductBias(AttentionalBias):
    """Dot-product similarity: maximizes <M(k), v>.

    Gradient of <pred, target> w.r.t. pred is target.
    Used by: Linear Attention, Mamba, GLA, RetNet.
    """

    def error_signal(self, prediction: Tensor, target: Tensor) -> Tensor:
        return target


class L2Bias(AttentionalBias):
    """L2 regression loss: minimizes ||M(k) - v||^2.

    Negative gradient w.r.t. pred is (target - pred).
    Factor of 2 absorbed into learning rate eta.
    Used by: DeltaNet, TTT, Titans, Gated DeltaNet, RWKV-7.
    """

    def error_signal(self, prediction: Tensor, target: Tensor) -> Tensor:
        return target - prediction
