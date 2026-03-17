"""Polynomial regression task for in-context learning.

y = W @ φ(x)  where φ(x) includes all monomials up to the given degree.

For degree 1, this reduces to LinearTask (y = Wx).
For degree 2, φ(x) = [x, x_i*x_j for i<=j], so the function is a quadratic
form that is linear in the expanded feature space but nonlinear in x.

A linear memory M that predicts y = Mx can only fit degree-1 functions.
An MLP memory M(x) = W1 σ(W2 x) can potentially learn degree-2+ features.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from math import comb

import torch

from .base import ICLBatch, ICLTask


def _build_monomial_indices(d: int, degree: int) -> list[torch.Tensor]:
    """Precompute index tensors for polynomial feature expansion.

    Returns a list (one per degree k=2..degree) of long tensors shaped
    (num_monomials_k, k) whose rows are the variable indices for each monomial.
    """
    result = []
    for k in range(2, degree + 1):
        indices = list(combinations_with_replacement(range(d), k))
        result.append(torch.tensor(indices, dtype=torch.long))
    return result


def _polynomial_features(
    x: torch.Tensor, degree: int, index_cache: list[torch.Tensor] | None = None
) -> torch.Tensor:
    """Expand x into polynomial features up to the given degree.

    Args:
        x: (..., d) input tensor.
        degree: maximum polynomial degree (>= 1).
        index_cache: precomputed index tensors from _build_monomial_indices.

    Returns:
        (..., d_feat) tensor of polynomial features.
    """
    features = [x]  # degree-1 terms
    d = x.shape[-1]

    if index_cache is None:
        index_cache = _build_monomial_indices(d, degree)

    for k_idx, k in enumerate(range(2, degree + 1)):
        idx = index_cache[k_idx]  # (num_mono, k)
        # Gather all needed elements: (..., num_mono, k)
        gathered = x[..., idx]  # broadcasts idx over batch dims
        # Product along last dim gives each monomial
        mono = gathered.prod(dim=-1)  # (..., num_mono)
        features.append(mono)

    return torch.cat(features, dim=-1)


def polynomial_feature_dim(d_in: int, degree: int) -> int:
    """Compute the number of polynomial features for given d_in and degree."""
    return sum(comb(d_in + k - 1, k) for k in range(1, degree + 1))


class PolynomialTask(ICLTask):
    """In-context learning of random polynomial functions.

    Each sequence samples a fresh W ~ N(0, I/d_feat), then generates
    (x, y) pairs with x ~ N(0, I) and y = W @ φ(x) + noise,
    where φ(x) is the polynomial feature expansion of x.

    Args:
        d_input: input dimension.
        d_output: output dimension (default 1).
        degree: polynomial degree (1 = linear, 2 = quadratic, etc.).
        noise_std: Gaussian noise standard deviation on outputs.
    """

    def __init__(self, d_input: int, d_output: int = 1,
                 degree: int = 2, noise_std: float = 0.0,
                 input_range: str = "gaussian",
                 normalize_output: bool = False):
        self._d_input = d_input
        self._d_output = d_output
        self.degree = degree
        self.noise_std = noise_std
        self.input_range = input_range
        self.normalize_output = normalize_output
        self._d_feat = polynomial_feature_dim(d_input, degree)
        self._index_cache = _build_monomial_indices(d_input, degree)

    @property
    def d_in(self) -> int:
        return self._d_input

    @property
    def d_out(self) -> int:
        return self._d_output

    @property
    def d_feat(self) -> int:
        """Dimension of the polynomial feature space."""
        return self._d_feat

    def sample_batch(self, batch_size: int, num_examples: int) -> ICLBatch:
        n = num_examples + 1  # demonstrations + query

        if self.input_range == "uniform":
            xs = torch.rand(batch_size, n, self._d_input) * 2 - 1  # U(-1, 1)
        else:
            xs = torch.randn(batch_size, n, self._d_input)  # N(0, I)
        phi = _polynomial_features(xs, self.degree, self._index_cache)  # (batch, n, d_feat)

        # Random weight in feature space
        W = torch.randn(batch_size, self._d_feat, self._d_output) / (self._d_feat ** 0.5)
        ys = torch.bmm(phi, W)  # (batch, n, d_output)

        if self.noise_std > 0:
            ys = ys + torch.randn_like(ys) * self.noise_std

        if self.normalize_output:
            # Normalize each sequence to have unit output variance
            # This keeps the loss scale consistent across degrees
            ys_std = ys.std(dim=1, keepdim=True).clamp(min=1e-8)
            ys = ys / ys_std

        return ICLBatch(xs=xs, ys=ys)
