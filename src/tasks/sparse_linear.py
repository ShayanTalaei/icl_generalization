"""Sparse linear regression ICL task.

Task:
    y = w^T x,  where w has only `sparsity` non-zero coordinates.

Setup:
    - x ~ N(0, I_d)
    - w ~ N(0, I_d), then all but `sparsity` randomly-chosen coordinates
      are zeroed out (the active set is re-sampled per sequence).
    - y = w^T x  (scalar output, no additive noise by default).
    - Paper default: d=20, sparsity=3.

Reference:
    Garg, S., Tsipras, D., Liang, P., & Valiant, G. (2022).
    "What Can Transformers Learn In-Context? A Case Study of Simple
    Function Classes." NeurIPS 2022. arXiv:2208.01066.
    Code: https://github.com/dtsip/in-context-learning
"""

from __future__ import annotations

import torch

from .base import ICLBatch, ICLTask


class SparseLinearTask(ICLTask):
    """Sparse linear regression: y = w^T x with `sparsity` active coords.

    Each sequence samples w ~ N(0, I_d), picks `sparsity` coordinates
    uniformly at random to keep, zeros the rest, then generates
    (x, y) pairs with x ~ N(0, I_d) and y = w^T x.

    Args:
        d_input: input dimension (default 20, matching Garg et al.).
        d_output: output dimension (default 1).
        sparsity: number of non-zero coordinates in w (default 3).
        noise_std: Gaussian noise std on outputs (default 0).
    """

    def __init__(
        self,
        d_input: int = 20,
        d_output: int = 1,
        sparsity: int = 3,
        noise_std: float = 0.0,
    ):
        self._d_input = d_input
        self._d_output = d_output
        self.sparsity = sparsity
        self.noise_std = noise_std

    @property
    def d_in(self) -> int:
        return self._d_input

    @property
    def d_out(self) -> int:
        return self._d_output

    def sample_batch(self, batch_size: int, num_examples: int) -> ICLBatch:
        n = num_examples + 1  # demonstrations + query

        # w ~ N(0, I), then zero out all but `sparsity` coords
        w = torch.randn(batch_size, self._d_input, self._d_output)
        for b in range(batch_size):
            perm = torch.randperm(self._d_input)
            mask = torch.ones(self._d_input, dtype=torch.bool)
            mask[perm[: self.sparsity]] = False
            w[b, mask, :] = 0.0

        xs = torch.randn(batch_size, n, self._d_input)
        ys = torch.bmm(xs, w)  # (batch, n, d_output)

        if self.noise_std > 0:
            ys = ys + torch.randn_like(ys) * self.noise_std

        return ICLBatch(xs=xs, ys=ys)
