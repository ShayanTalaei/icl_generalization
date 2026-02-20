import torch

from .base import ICLBatch, ICLTask


class LinearTask(ICLTask):
    """In-context learning of random linear functions y = Wx.

    Each sequence samples a fresh W ~ N(0, I/d_in), then generates
    (x, y) pairs with x ~ N(0, I) and y = Wx + noise.
    """

    def __init__(self, d_input: int, d_output: int = 1, noise_std: float = 0.0):
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
        n = num_examples + 1  # demonstrations + query

        # Random linear map for each sequence in the batch
        W = torch.randn(batch_size, self._d_input, self._d_output) / (self._d_input ** 0.5)

        xs = torch.randn(batch_size, n, self._d_input)
        ys = torch.bmm(xs, W)  # (batch, n, d_output)

        if self.noise_std > 0:
            ys = ys + torch.randn_like(ys) * self.noise_std

        return ICLBatch(xs=xs, ys=ys)
