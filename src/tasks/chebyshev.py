"""Chebyshev polynomial ICL task.

Reproduces the Wilcoxson et al. (2024) arXiv:2407.19346 setting:
  - Univariate inputs x ~ U(-1, 1)
  - Chebyshev basis T_0(x)...T_b(x), b ~ Uniform[0, max_degree] per sequence
  - Coefficients c_k ~ N(0, 1)
  - y = sum_{k=0}^{b} c_k * T_k(x)

Unlike the monomial PolynomialTask, outputs are well-conditioned for high
degrees because T_k(x) in [-1, 1] for all x in [-1, 1].
"""

from __future__ import annotations

import torch

from .base import ICLBatch, ICLTask


def chebyshev_features(x: torch.Tensor, max_degree: int) -> torch.Tensor:
    """Evaluate Chebyshev polynomials T_0...T_{max_degree} at x.

    Uses the recurrence T_{n+1}(x) = 2x * T_n(x) - T_{n-1}(x),
    with T_0(x) = 1 and T_1(x) = x.

    Args:
        x: (...) tensor of scalar inputs, typically in [-1, 1].
        max_degree: highest degree to compute (inclusive).

    Returns:
        (..., max_degree + 1) tensor where column k is T_k(x).
    """
    shape = x.shape
    T = torch.zeros(*shape, max_degree + 1, device=x.device, dtype=x.dtype)
    T[..., 0] = 1.0           # T_0 = 1
    if max_degree >= 1:
        T[..., 1] = x         # T_1 = x
    for k in range(2, max_degree + 1):
        T[..., k] = 2 * x * T[..., k - 1] - T[..., k - 2]  # recurrence
    return T


class ChebyshevTask(ICLTask):
    """In-context learning of random Chebyshev polynomial combinations.

    Each sequence samples a degree b ~ Uniform[0, max_degree] and
    coefficients c_k ~ N(0, 1), then generates (x, y) pairs with
    x ~ U(-1, 1) and y = sum_{k=0}^{b} c_k * T_k(x).

    Args:
        max_degree: maximum polynomial degree (default 11, matching Wilcoxson et al.).
        noise_std: Gaussian noise standard deviation on outputs.
        normalize_output: if True, divide y by sqrt(E[y^2]) estimated at init.
        d_input: ignored (always 1, univariate). Accepted for build_task compat.
        d_output: ignored (always 1). Accepted for build_task compat.
    """

    def __init__(
        self,
        max_degree: int = 11,
        noise_std: float = 0.0,
        normalize_output: bool = False,
        d_input: int = 1,
        d_output: int = 1,
    ):
        self.max_degree = max_degree
        self.noise_std = noise_std
        self.normalize_output = normalize_output
        self._output_scale = 1.0
        if normalize_output:
            self._output_scale = self._estimate_output_scale()

    def _estimate_output_scale(self, n_samples: int = 8192) -> float:
        """Estimate sqrt(E[y^2]) over the task distribution."""
        n = 81
        xs = torch.rand(n_samples, n) * 2 - 1  # U(-1, 1)
        degrees = torch.randint(0, self.max_degree + 1, (n_samples,))
        ys = self._compute_ys(xs, degrees)
        return float(ys.pow(2).mean().sqrt().clamp(min=1e-8))

    def _compute_ys(
        self, xs: torch.Tensor, degrees: torch.Tensor
    ) -> torch.Tensor:
        """Compute y values for a batch.

        Args:
            xs: (batch, n) scalar inputs.
            degrees: (batch,) integer degrees per sequence.

        Returns:
            (batch, n) outputs.
        """
        batch, n = xs.shape
        # Chebyshev features for all x: (batch, n, max_degree+1)
        feats = chebyshev_features(xs, self.max_degree)

        # Sample coefficients: (batch, max_degree+1)
        coefs = torch.randn(batch, self.max_degree + 1, device=xs.device)

        # Zero out coefficients above each sequence's degree
        # mask[b, k] = 1 if k <= degrees[b]
        k_idx = torch.arange(self.max_degree + 1, device=xs.device)  # (D+1,)
        mask = k_idx.unsqueeze(0) <= degrees.unsqueeze(1).to(xs.device)  # (batch, D+1)
        coefs = coefs * mask.float()

        # y = sum_k coefs[b,k] * T_k(x[b,i])  -> (batch, n)
        ys = (feats * coefs.unsqueeze(1)).sum(dim=-1)  # (batch, n)
        return ys

    @property
    def d_in(self) -> int:
        return 1

    @property
    def d_out(self) -> int:
        return 1

    def sample_batch(self, batch_size: int, num_examples: int) -> ICLBatch:
        n = num_examples + 1  # demonstrations + query

        # Inputs: x ~ U(-1, 1), scalar
        xs = torch.rand(batch_size, n) * 2 - 1  # (batch, n)

        # Random degree per sequence: b ~ Uniform[0, max_degree]
        degrees = torch.randint(0, self.max_degree + 1, (batch_size,))

        ys = self._compute_ys(xs, degrees)  # (batch, n)

        if self.noise_std > 0:
            ys = ys + torch.randn_like(ys) * self.noise_std

        if self.normalize_output:
            ys = ys / self._output_scale

        # Reshape to (batch, n, 1) to match ICLBatch convention (d_in=1, d_out=1)
        return ICLBatch(xs=xs.unsqueeze(-1), ys=ys.unsqueeze(-1))
