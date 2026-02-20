import torch
import torch.nn as nn
from torch import Tensor

from .base import SeqModel


class RNNModel(SeqModel):
    """LSTM / GRU model for in-context learning.

    Like the Transformer, builds an interleaved sequence
    [x1, y1, x2, y2, ..., x_query] and processes it autoregressively.
    Predictions are read out at x-positions.

    The key difference from the Transformer: the RNN updates a hidden state
    as it reads each token, so even with random weights it may exhibit
    in-context adaptation (the hidden state encodes a running summary of
    the demonstrations seen so far).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_model: int = 128,
        n_layers: int = 2,
        rnn_type: str = "lstm",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_model = d_model

        self.proj_x = nn.Linear(d_in, d_model)
        self.proj_y = nn.Linear(d_out, d_model)
        self.proj_out = nn.Linear(d_model, d_out)

        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(self, xs: Tensor, ys: Tensor) -> Tensor:
        B, n, _ = xs.shape
        seq_len = 2 * n - 1

        x_emb = self.proj_x(xs)             # (B, n, d_model)
        y_emb = self.proj_y(ys[:, :-1, :])  # (B, n-1, d_model)

        seq = xs.new_zeros(B, seq_len, self.d_model)
        seq[:, 0::2, :] = x_emb
        seq[:, 1::2, :] = y_emb

        out, _ = self.rnn(seq)               # (B, seq_len, d_model)

        y_preds = self.proj_out(out[:, 0::2, :])  # (B, n, d_out)
        return y_preds
