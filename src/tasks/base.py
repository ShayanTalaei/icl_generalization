from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor


@dataclass
class ICLBatch:
    """A batch of in-context learning sequences.

    xs[:, :-1] and ys[:, :-1] are the demonstration (input, output) pairs.
    xs[:, -1] is the query input; ys[:, -1] is the query target (for loss only).
    """

    xs: Tensor  # (batch, n, d_in)   where n = num_examples + 1
    ys: Tensor  # (batch, n, d_out)


class ICLTask(ABC):
    """Base class for all in-context learning tasks."""

    @abstractmethod
    def sample_batch(self, batch_size: int, num_examples: int) -> ICLBatch:
        """Sample a batch of ICL sequences.

        Args:
            batch_size:   number of independent sequences.
            num_examples: number of (x, y) demonstration pairs per sequence.
                          The returned tensors have n = num_examples + 1 points
                          (demonstrations + 1 query).
        """
        ...

    @property
    @abstractmethod
    def d_in(self) -> int:
        """Dimensionality of input vectors."""
        ...

    @property
    @abstractmethod
    def d_out(self) -> int:
        """Dimensionality of output vectors."""
        ...
