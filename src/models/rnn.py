import torch
import torch.nn as nn
from torch import Tensor

from .base import SeqModel


class LinearRNN(SeqModel):
    """Linear RNN with matrix-valued memory (linear attention).

    General form (with projections):

        k_i = W_k x_i          v_i = W_v y_i
        Write:  M_i = M_{i-1} + v_i @ k_i^T
        Read:   y_pred = W_out (M @ W_q x_query)

    Simplified form (this implementation, no projections):

        Write:  M_i = M_{i-1} + η · y_i @ x_i^T
        Read:   y_pred = M @ x_query

    The projections W_k, W_q, W_v, W_out just map x and y into a latent
    space; the core mechanism is identical. Without them, the memory
    accumulates M = η Σ y_i x_i^T directly in data space — the gradient
    of MSE loss at W=0 for the linear regression task y = Wx.

    η (eta) is a scalar learning rate. Since M_k ≈ ηk·W, we need ηk ≈ 1
    for calibration at sequence length k (e.g. η = 0.01 for k = 100).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_model: int = 128,   # unused, kept for API compat
        n_layers: int = 1,    # unused
        dropout: float = 0.0, # unused
        gd_init: bool = False,  # unused
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        # Learnable learning rate (scalar)
        self.eta = nn.Parameter(torch.tensor(1.0))

    def forward(self, xs: Tensor, ys: Tensor) -> Tensor:
        B, n, _ = xs.shape

        # Memory: (B, d_out, d_in) — directly in data space
        M = xs.new_zeros(B, self.d_out, self.d_in)
        y_preds = []

        for i in range(n):
            # Read: predict y at this x-position
            y_pred = torch.bmm(M, xs[:, i, :].unsqueeze(-1)).squeeze(-1)  # (B, d_out)
            y_preds.append(y_pred)

            # Write: accumulate outer product if we have the label
            if i < n - 1:
                v = ys[:, i, :].unsqueeze(-1)    # (B, d_out, 1)
                k = xs[:, i, :].unsqueeze(-2)    # (B, 1, d_in)
                M = M + self.eta * torch.bmm(v, k)

        return torch.stack(y_preds, dim=1)    # (B, n, d_out)


class LinearRNNProjected(SeqModel):
    """Linear RNN with matrix-valued memory and learned projections.

    General form:

        k_i = W_k x_i          v_i = W_v y_i
        Write:  M_i = M_{i-1} + v_i @ k_i^T
        Read:   y_pred = W_out (M @ W_q x_query)

    The memory lives in a d_model-dimensional latent space. Projections
    W_k, W_q (from x-space) and W_v, W_out (from/to y-space) are learnable.
    The core accumulation M = Σ v_i k_i^T is the same outer-product rule.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_model: int = 128,
        n_layers: int = 1,    # unused
        dropout: float = 0.0, # unused
        gd_init: bool = False,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_model = d_model

        self.proj_k = nn.Linear(d_in, d_model, bias=False)
        self.proj_q = nn.Linear(d_in, d_model, bias=False)
        self.proj_v = nn.Linear(d_out, d_model, bias=False)
        self.proj_out = nn.Linear(d_model, d_out, bias=False)

        if gd_init:
            self._init_gd_weights()

    def _init_gd_weights(self):
        """Set projections to identity embeddings (exact GD solution)."""
        with torch.no_grad():
            for p in self.parameters():
                p.zero_()
            self.proj_k.weight[:self.d_in, :self.d_in] = torch.eye(self.d_in)
            self.proj_q.weight[:self.d_in, :self.d_in] = torch.eye(self.d_in)
            self.proj_v.weight[:self.d_out, :self.d_out] = torch.eye(self.d_out)
            self.proj_out.weight[:self.d_out, :self.d_out] = torch.eye(self.d_out)

    def forward(self, xs: Tensor, ys: Tensor) -> Tensor:
        B, n, _ = xs.shape

        keys = self.proj_k(xs)                # (B, n, d_model)
        queries = self.proj_q(xs)             # (B, n, d_model)
        values = self.proj_v(ys[:, :-1, :])   # (B, n-1, d_model)

        M = xs.new_zeros(B, self.d_model, self.d_model)
        y_preds = []

        for i in range(n):
            out = torch.bmm(M, queries[:, i, :].unsqueeze(-1)).squeeze(-1)
            y_preds.append(self.proj_out(out))

            if i < n - 1:
                v = values[:, i, :].unsqueeze(-1)   # (B, d_model, 1)
                k = keys[:, i, :].unsqueeze(-2)     # (B, 1, d_model)
                M = M + torch.bmm(v, k)

        return torch.stack(y_preds, dim=1)    # (B, n, d_out)


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
