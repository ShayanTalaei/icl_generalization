"""Tests for ChebyshevTask and associated utilities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest
from src.tasks.chebyshev import chebyshev_features, ChebyshevTask


def test_chebyshev_recurrence():
    """T_0=1, T_1=x, T_2=2x^2-1, T_3=4x^3-3x at known x values."""
    x = torch.tensor([0.0, 0.5, 1.0, -1.0])
    feats = chebyshev_features(x, max_degree=3)  # (4, 4)
    # T_0 = 1
    assert torch.allclose(feats[:, 0], torch.ones(4))
    # T_1 = x
    assert torch.allclose(feats[:, 1], x)
    # T_2 = 2x^2 - 1
    assert torch.allclose(feats[:, 2], 2 * x**2 - 1, atol=1e-6)
    # T_3 = 4x^3 - 3x
    assert torch.allclose(feats[:, 3], 4 * x**3 - 3 * x, atol=1e-6)


def test_chebyshev_bounded():
    """For x in [-1, 1], each T_k(x) in [-1, 1]."""
    x = torch.linspace(-1, 1, 1000)
    feats = chebyshev_features(x, max_degree=11)  # (1000, 12)
    assert feats.min() >= -1.0 - 1e-5
    assert feats.max() <= 1.0 + 1e-5


def test_chebyshev_task_shapes():
    task = ChebyshevTask(max_degree=11)
    batch = task.sample_batch(batch_size=32, num_examples=40)
    assert batch.xs.shape == (32, 41, 1)
    assert batch.ys.shape == (32, 41, 1)


def test_chebyshev_task_input_range():
    """x values must all be in [-1, 1]."""
    task = ChebyshevTask(max_degree=11)
    batch = task.sample_batch(batch_size=256, num_examples=80)
    assert batch.xs.min() >= -1.0 - 1e-6
    assert batch.xs.max() <= 1.0 + 1e-6


def test_chebyshev_task_fixed_dims():
    """d_in=1, d_out=1 always for ChebyshevTask."""
    task = ChebyshevTask(max_degree=5)
    assert task.d_in == 1
    assert task.d_out == 1


def test_chebyshev_max_degree_zero():
    """max_degree=0 should produce constant functions (y = c_0 * T_0 = c_0)."""
    task = ChebyshevTask(max_degree=0)
    batch = task.sample_batch(batch_size=16, num_examples=10)
    assert batch.xs.shape == (16, 11, 1)
    assert batch.ys.shape == (16, 11, 1)
    # All y values in a sequence should be identical (constant function)
    # since T_0(x) = 1 regardless of x
    ys = batch.ys.squeeze(-1)  # (16, 11)
    assert torch.allclose(ys, ys[:, :1].expand_as(ys), atol=1e-5), \
        "max_degree=0 should produce constant-per-sequence outputs"


def test_chebyshev_output_scale():
    """Outputs should have std in a reasonable range."""
    task = ChebyshevTask(max_degree=11, normalize_output=True)
    batch = task.sample_batch(batch_size=1024, num_examples=80)
    std = batch.ys.std()
    assert 0.5 < std.item() < 2.0, f"Expected y std ~ 1, got {std.item()}"


def _load_eval_icl():
    """Load the eval_icl module from scripts/ (not on sys.path by default)."""
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location(
        "eval_icl",
        os.path.join(Path(__file__).resolve().parent.parent, "scripts", "eval_icl.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_chebyshev_ridge_baseline_shape():
    """Should return a tensor of shape (num_examples + 1,)."""
    mod = _load_eval_icl()
    task = ChebyshevTask(max_degree=5)
    losses = mod.chebyshev_ridge_baseline(
        task, num_examples=10, batch_size=32, num_batches=3, device="cpu"
    )
    assert losses.shape == (11,), f"Expected shape (11,), got {losses.shape}"


def test_chebyshev_ridge_baseline_decreasing():
    """Ridge MSE should generally decrease as more demonstrations are added."""
    mod = _load_eval_icl()
    torch.manual_seed(42)
    task = ChebyshevTask(max_degree=5)
    losses = mod.chebyshev_ridge_baseline(
        task, num_examples=30, batch_size=128, num_batches=20, device="cpu"
    )
    # Loss at position 12+ should be lower than at position 1
    # (degree-5 poly needs 6 points; 12 is well past convergence)
    assert losses[12].item() < losses[1].item(), (
        f"Ridge MSE should decrease: pos 1={losses[1]:.4f}, pos 12={losses[12]:.4f}"
    )


def test_build_chebyshev_task_from_config():
    """build_task should produce a ChebyshevTask from config."""
    from src.config import TaskConfig, build_task
    cfg = TaskConfig()
    cfg.type = "chebyshev"
    cfg.max_degree = 5
    task = build_task(cfg)
    from src.tasks.chebyshev import ChebyshevTask
    assert isinstance(task, ChebyshevTask)
    assert task.max_degree == 5
