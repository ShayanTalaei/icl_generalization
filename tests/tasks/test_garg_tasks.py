"""Tests for Garg et al. (2022) function classes: SparseLinearTask, QuadraticTask."""

import math

import torch
import pytest

from src.tasks.sparse_linear import SparseLinearTask
from src.tasks.quadratic import QuadraticTask


# ── SparseLinearTask ──────────────────────────────────────────────────────


def test_sparse_linear_shapes():
    task = SparseLinearTask(d_input=20, sparsity=3)
    batch = task.sample_batch(batch_size=32, num_examples=40)
    assert batch.xs.shape == (32, 41, 20)
    assert batch.ys.shape == (32, 41, 1)


def test_sparse_linear_dims():
    task = SparseLinearTask(d_input=20, sparsity=3)
    assert task.d_in == 20
    assert task.d_out == 1


def test_sparse_linear_sparsity():
    """Each sequence's latent function should depend on at most `sparsity` coords."""
    torch.manual_seed(42)
    d_input, sparsity = 20, 3
    task = SparseLinearTask(d_input=d_input, sparsity=sparsity)

    # Generate data and infer which dims are active by varying one dim at a time
    batch_size, n = 64, 50
    batch = task.sample_batch(batch_size, n - 1)
    xs, ys = batch.xs, batch.ys

    # For each sequence, check that changing a non-active dim doesn't change y.
    # We do this by computing the Jacobian dy/dx via finite differences.
    # Simpler approach: reconstruct w from data via least-squares
    for b in range(min(batch_size, 8)):  # check a few sequences
        X = xs[b]  # (n, d_input)
        Y = ys[b, :, 0]  # (n,)
        # Solve X @ w = Y via least squares
        w_hat = torch.linalg.lstsq(X, Y).solution  # (d_input,)
        n_nonzero = (w_hat.abs() > 1e-4).sum().item()
        assert n_nonzero <= sparsity, (
            f"Sequence {b}: found {n_nonzero} non-zero coords, expected <= {sparsity}"
        )


def test_sparse_linear_gaussian_inputs():
    """Inputs should be approximately N(0, I)."""
    torch.manual_seed(0)
    task = SparseLinearTask(d_input=10, sparsity=3)
    batch = task.sample_batch(batch_size=1024, num_examples=50)
    xs = batch.xs.reshape(-1, 10)
    # Mean should be ~0, std should be ~1
    assert xs.mean().abs() < 0.05, f"Input mean {xs.mean():.4f} too far from 0"
    assert abs(xs.std().item() - 1.0) < 0.05, f"Input std {xs.std():.4f} too far from 1"


# ── QuadraticTask ─────────────────────────────────────────────────────────


def test_quadratic_shapes():
    task = QuadraticTask(d_input=20)
    batch = task.sample_batch(batch_size=32, num_examples=40)
    assert batch.xs.shape == (32, 41, 20)
    assert batch.ys.shape == (32, 41, 1)


def test_quadratic_dims():
    task = QuadraticTask(d_input=20)
    assert task.d_in == 20
    assert task.d_out == 1


def test_quadratic_is_nonlinear():
    """Quadratic output should not be recoverable by linear regression on x."""
    torch.manual_seed(42)
    task = QuadraticTask(d_input=10)
    batch = task.sample_batch(batch_size=1, num_examples=200)
    X = batch.xs[0]  # (201, 10)
    Y = batch.ys[0, :, 0]  # (201,)

    # Linear fit: should have non-trivial residual
    w_lin = torch.linalg.lstsq(X, Y).solution
    resid_linear = (X @ w_lin - Y).pow(2).mean()

    # Quadratic fit: should be near-perfect
    X2 = X.pow(2)
    w_quad = torch.linalg.lstsq(X2, Y).solution
    resid_quad = (X2 @ w_quad - Y).pow(2).mean()

    assert resid_quad < resid_linear * 0.01, (
        f"Quadratic fit should be much better: linear={resid_linear:.4f}, quad={resid_quad:.6f}"
    )


def test_quadratic_output_scale():
    """y = (x^2 @ w) / sqrt(3) should have Var(y) ≈ 1 when w ~ N(0,I), x ~ N(0,I)."""
    torch.manual_seed(42)
    task = QuadraticTask(d_input=20)
    batch = task.sample_batch(batch_size=2048, num_examples=100)
    ys = batch.ys
    # Var should be O(d_input) because w ~ N(0,I) not N(0,I/d).
    # More precisely: Var(y) = (1/3) * sum_i Var(w_i) * E[x_i^4] = (1/3)*d*3 = d
    # So std ≈ sqrt(d) = sqrt(20) ≈ 4.47
    std = ys.std().item()
    expected_std = math.sqrt(20)
    assert abs(std - expected_std) < 1.5, (
        f"Output std {std:.2f}, expected ~{expected_std:.2f}"
    )


def test_quadratic_no_cross_terms():
    """Verify only diagonal monomials x_i^2 are used, no x_i*x_j cross terms."""
    torch.manual_seed(42)
    d = 5
    task = QuadraticTask(d_input=d)
    batch = task.sample_batch(batch_size=1, num_examples=100)
    X = batch.xs[0]  # (101, 5)
    Y = batch.ys[0, :, 0]  # (101,)

    # Fit with diagonal monomials only: x_i^2
    X_diag = X.pow(2)  # (101, 5)
    w_diag = torch.linalg.lstsq(X_diag, Y).solution
    resid_diag = (X_diag @ w_diag - Y).pow(2).mean()

    assert resid_diag < 1e-6, (
        f"Diagonal-only fit should be near-perfect, got residual {resid_diag:.6f}"
    )


# ── build_task integration ────────────────────────────────────────────────


def test_build_sparse_linear_from_config():
    from src.config import TaskConfig, build_task
    cfg = TaskConfig()
    cfg.type = "sparse_linear"
    cfg.d_input = 20
    cfg.sparsity = 3
    task = build_task(cfg)
    assert isinstance(task, SparseLinearTask)
    assert task.sparsity == 3


def test_build_quadratic_from_config():
    from src.config import TaskConfig, build_task
    cfg = TaskConfig()
    cfg.type = "quadratic"
    cfg.d_input = 20
    task = build_task(cfg)
    assert isinstance(task, QuadraticTask)
    assert task.d_in == 20
