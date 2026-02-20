from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class SeqModel(nn.Module, ABC):
    """Base class for all sequence models in the ICL framework.

    Models receive raw (xs, ys) pairs and handle interleaving / sequence
    construction internally, giving each architecture full flexibility.
    """

    @abstractmethod
    def forward(self, xs: Tensor, ys: Tensor) -> Tensor:
        """
        Args:
            xs: (batch, n, d_in)  -- input vectors; last position is the query.
            ys: (batch, n, d_out) -- output vectors; last position is the query
                target (used for loss, but NOT fed to the model as input).
        Returns:
            y_preds: (batch, n, d_out) -- predicted outputs at each position.
        """
        ...
