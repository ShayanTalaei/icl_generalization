"""Pre-generated dataset that implements ICLTask interface.

Loads pre-saved (xs, ys) tensors and serves them by index instead of
generating on-the-fly. Supports optional GPU placement for zero-copy
training batches.
"""

from pathlib import Path

import torch
from torch import Tensor

from .base import ICLBatch, ICLTask


class PregeneratedDataset(ICLTask):
    """ICLTask backed by pre-generated data on disk.

    The dataset file is a .pt dict with keys:
        - xs: (N, batch_size, n, d_in)
        - ys: (N, batch_size, n, d_out)
        - d_in: int
        - d_out: int
        - metadata: dict (task config, seed, etc.)

    Optionally includes eval data:
        - eval_xs: (M, batch_size, n_eval, d_in)
        - eval_ys: (M, batch_size, n_eval, d_out)
    """

    def __init__(self, path: str | Path, device: str = "cpu"):
        data = torch.load(path, map_location=device, weights_only=True)
        self._xs = data["xs"]           # (N, B, n, d_in)
        self._ys = data["ys"]           # (N, B, n, d_out)
        self._d_in = data["d_in"]
        self._d_out = data["d_out"]
        self._n_batches = self._xs.shape[0]
        self._idx = 0

        # Eval data (optional)
        self._eval_xs = data.get("eval_xs")
        self._eval_ys = data.get("eval_ys")
        self._eval_idx = 0

    @property
    def d_in(self) -> int:
        return self._d_in

    @property
    def d_out(self) -> int:
        return self._d_out

    def sample_batch(self, batch_size: int, num_examples: int) -> ICLBatch:
        """Return the next pre-generated batch.

        batch_size and num_examples are ignored — the pre-generated
        dimensions are used. This matches the ICLTask interface.
        """
        xs = self._xs[self._idx]  # (B, n, d_in)
        ys = self._ys[self._idx]  # (B, n, d_out)
        self._idx = (self._idx + 1) % self._n_batches
        return ICLBatch(xs=xs, ys=ys)

    def sample_eval_batch(self, batch_size: int, num_examples: int) -> ICLBatch:
        """Return the next pre-generated eval batch (if available)."""
        if self._eval_xs is None:
            return self.sample_batch(batch_size, num_examples)
        xs = self._eval_xs[self._eval_idx]
        ys = self._eval_ys[self._eval_idx]
        self._eval_idx = (self._eval_idx + 1) % self._eval_xs.shape[0]
        return ICLBatch(xs=xs, ys=ys)

    def __len__(self):
        return self._n_batches
