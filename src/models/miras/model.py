"""MIRASModel: SeqModel subclass wrapping MIRASLayer(s) with projections.

Handles the interleaved (x, y) sequence processing and optional
input/output projections (the "outer loop" weights from nested learning).
"""

import torch
import torch.nn as nn
from torch import Tensor

from ..base import SeqModel
from .layer import MIRASLayer


class MIRASModel(SeqModel):
    """MIRAS sequence model for in-context learning.

    Wraps one or more MIRASLayer(s) with optional learned projections
    between data space and model space.

    Without projections: memory operates directly in data space.
    With projections: learned W_k, W_q, W_v, W_out map between spaces.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        layers: list[MIRASLayer],
        use_projections: bool = False,
        d_model: int = 128,
        gd_init: bool = False,
        residual: bool = False,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_model = d_model
        self.use_proj = use_projections
        self.residual = residual

        self.layers = nn.ModuleList(layers)

        if use_projections:
            self.proj_k = nn.Linear(d_in, d_model, bias=False)
            self.proj_q = nn.Linear(d_in, d_model, bias=False)
            self.proj_v = nn.Linear(d_out, d_model, bias=False)
            self.proj_out = nn.Linear(d_model, d_out, bias=False)
            if gd_init:
                self._init_gd_weights()

    def _init_gd_weights(self):
        """Set projections to identity embeddings (exact GD solution)."""
        with torch.no_grad():
            for proj in (self.proj_k, self.proj_q, self.proj_v, self.proj_out):
                proj.weight.zero_()
            self.proj_k.weight[: self.d_in, : self.d_in] = torch.eye(self.d_in)
            self.proj_q.weight[: self.d_in, : self.d_in] = torch.eye(self.d_in)
            self.proj_v.weight[: self.d_out, : self.d_out] = torch.eye(self.d_out)
            self.proj_out.weight[: self.d_out, : self.d_out] = torch.eye(self.d_out)

    def forward(self, xs: Tensor, ys: Tensor) -> Tensor:
        B, n, _ = xs.shape

        # Initialize state for each layer
        states = [
            layer.init_state(B, xs.device, xs.dtype) for layer in self.layers
        ]

        # Project inputs if needed
        if self.use_proj:
            keys = self.proj_k(xs)              # (B, n, d_model)
            queries = self.proj_q(xs)            # (B, n, d_model)
            values = self.proj_v(ys[:, :-1, :])  # (B, n-1, d_model)

        y_preds = []
        for i in range(n):
            # --- Read ---
            q = queries[:, i] if self.use_proj else xs[:, i]
            h = q
            for layer_idx, layer in enumerate(self.layers):
                if self.residual:
                    h = h + layer.read(states[layer_idx], h)
                else:
                    h = layer.read(states[layer_idx], h)
            y_preds.append(self.proj_out(h) if self.use_proj else h)

            # --- Write ---
            if i < n - 1:
                k = keys[:, i] if self.use_proj else xs[:, i]
                v = values[:, i] if self.use_proj else ys[:, i]
                for layer_idx, layer in enumerate(self.layers):
                    states[layer_idx] = layer.write(
                        states[layer_idx], k, v
                    )

        return torch.stack(y_preds, dim=1)  # (B, n, d_out)
