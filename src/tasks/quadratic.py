"""Element-wise quadratic regression ICL task.

Task:
    y = (1/sqrt(3)) * sum_i  w_i * x_i^2

    This is a sum of diagonal degree-2 monomials (no cross-terms x_i*x_j).
    It is nonlinear in x but linear in the feature vector x^2 (element-wise).

Setup:
    - x ~ N(0, I_d)
    - w ~ N(0, I_d)
    - y = (x^2)^T w / sqrt(3),  where x^2 denotes element-wise squaring.
    - The 1/sqrt(3) factor keeps E[y^2] at the same scale as linear
      regression, since E[x_i^4] = 3 for x_i ~ N(0, 1).
    - Paper default: d=20.

Reference:
    Garg, S., Tsipras, D., Liang, P., & Valiant, G. (2022).
    "What Can Transformers Learn In-Context? A Case Study of Simple
    Function Classes." NeurIPS 2022. arXiv:2208.01066.
    Code: https://github.com/dtsip/in-context-learning
"""

from __future__ import annotations

import math

import torch

from .base import ICLBatch, ICLTask


class QuadraticTask(ICLTask):
    """Element-wise quadratic regression: y = (x^2)^T w / sqrt(3).

    Each sequence samples w ~ N(0, I_d), then generates (x, y) pairs with
    x ~ N(0, I_d) and y = (x ⊙ x)^T w / sqrt(3).  Only diagonal monomials
    x_i^2 are included — no cross-terms x_i * x_j for i != j.

    Args:
        d_input: input dimension (default 20, matching Garg et al.).
        d_output: output dimension (default 1).
        noise_std: Gaussian noise std on outputs (default 0).
    """

    def __init__(
        self,
        d_input: int = 20,
        d_output: int = 1,
        noise_std: float = 0.0,
    ):
        self._d_input = d_input
        self._d_output = d_output
        self.noise_std = noise_std

    @property
    def d_in(self) -> int:
        return self._d_input

    @property
    def d_out(self) -> int:
        return self._d_output

    def sample_batch(self, batch_size: int, num_examples: int) -> ICLBatch:
        n = num_examples + 1

        w = torch.randn(batch_size, self._d_input, self._d_output)
        xs = torch.randn(batch_size, n, self._d_input)

        # y = (x^2)^T w / sqrt(3)
        ys = torch.bmm(xs.pow(2), w) / math.sqrt(3)

        if self.noise_std > 0:
            ys = ys + torch.randn_like(ys) * self.noise_std

        return ICLBatch(xs=xs, ys=ys)
