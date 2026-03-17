"""Tests for curriculum learning helper."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.trainer import compute_curriculum_examples


def test_curriculum_start():
    """At step 1, should return curriculum_start examples."""
    assert compute_curriculum_examples(
        step=1, num_examples=80, curriculum_start=10, curriculum_end_step=100
    ) == 10


def test_curriculum_end():
    """At or after curriculum_end_step, should return num_examples."""
    assert compute_curriculum_examples(
        step=100, num_examples=80, curriculum_start=10, curriculum_end_step=100
    ) == 80
    assert compute_curriculum_examples(
        step=200, num_examples=80, curriculum_start=10, curriculum_end_step=100
    ) == 80


def test_curriculum_midpoint():
    """At step 50 of 100, should be approximately halfway."""
    n = compute_curriculum_examples(
        step=50, num_examples=80, curriculum_start=10, curriculum_end_step=100
    )
    assert 40 <= n <= 50


def test_curriculum_monotone():
    """Should be monotonically non-decreasing."""
    vals = [
        compute_curriculum_examples(
            step=s, num_examples=80, curriculum_start=10, curriculum_end_step=100
        )
        for s in range(1, 110)
    ]
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
