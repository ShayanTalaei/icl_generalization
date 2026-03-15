"""Utilities for operating on MemoryState (Tensor or tuple of Tensors)."""

from torch import Tensor

MemoryState = Tensor | tuple[Tensor, ...]
Grads = Tensor | tuple[Tensor, ...]
OptimState = None | Tensor | tuple[Tensor, ...]


def state_map(fn, *states):
    """Apply fn element-wise across MemoryState(s).

    Handles both single Tensor and tuple[Tensor, ...] states.

    Examples:
        state_map(lambda s: alpha * s, state)           # scalar multiply
        state_map(lambda s, g: s + eta * g, state, grads)  # GD step
    """
    if isinstance(states[0], tuple):
        assert all(
            isinstance(s, tuple) and len(s) == len(states[0]) for s in states[1:]
        ), "All states must be tuples of the same length"
        return tuple(fn(*elems) for elems in zip(*states))
    return fn(*states)
