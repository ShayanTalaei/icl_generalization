"""MIRASModel: SeqModel subclass wrapping MIRASLayer(s) with projections.

Handles the interleaved (x, y) sequence processing and optional
input/output projections (the "outer loop" weights from nested learning).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..base import SeqModel
from .layer import MIRASLayer


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale


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
        normalize_qk: bool = False,
        output_norm: bool = False,
        aggregate: str = "sequential",  # "sequential" | "additive"
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_model = d_model
        self.use_proj = use_projections
        self.residual = residual
        self.normalize_qk = normalize_qk
        self.output_norm = output_norm
        self.aggregate = aggregate

        self.layers = nn.ModuleList(layers)

        # RMSNorm on each layer's read output (DeltaNet/Mamba convention)
        if output_norm and use_projections:
            self.layer_norms = nn.ModuleList(
                [RMSNorm(d_model) for _ in layers]
            )

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
            if self.normalize_qk:
                keys = F.normalize(keys, dim=-1)
                queries = F.normalize(queries, dim=-1)

        y_preds = []
        for i in range(n):
            # --- Read ---
            if self.aggregate == "additive" and not self.use_proj:
                # Additive mode: each layer reads the same query, predictions summed
                q = xs[:, i]
                pred = torch.zeros(B, self.d_out, device=xs.device, dtype=xs.dtype)
                for layer_idx, layer in enumerate(self.layers):
                    pred = pred + layer.read(states[layer_idx], q)
                y_preds.append(pred)
            else:
                # Sequential mode: layers chain outputs
                q = queries[:, i] if self.use_proj else xs[:, i]
                h = q
                for layer_idx, layer in enumerate(self.layers):
                    layer_out = layer.read(states[layer_idx], h)
                    if self.output_norm and self.use_proj:
                        layer_out = self.layer_norms[layer_idx](layer_out)
                    if self.residual:
                        h = h + layer_out
                    else:
                        h = layer_out
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
