import torch
from src.models.miras.algorithm import GD
from src.models.miras.memory import MatrixMemory


def test_gd_init_optim_state_none():
    algo = GD()
    mem = MatrixMemory(4, 3)
    optim_state = algo.init_optim_state(mem, B=2, device="cpu", dtype=torch.float32)
    assert optim_state is None


def test_gd_step_tensor():
    algo = GD()
    state = torch.zeros(2, 3, 4)
    grads = torch.ones(2, 3, 4)
    eta = torch.tensor(0.1)
    new_state, new_optim = algo.step(state, grads, eta, None)
    assert torch.allclose(new_state, torch.full((2, 3, 4), 0.1))
    assert new_optim is None


def test_gd_step_tuple():
    algo = GD()
    state = (torch.zeros(2, 3, 4), torch.zeros(2, 5, 4))
    grads = (torch.ones(2, 3, 4), torch.ones(2, 5, 4) * 2)
    eta = torch.tensor(0.5)
    new_state, new_optim = algo.step(state, grads, eta, None)
    assert torch.allclose(new_state[0], torch.full((2, 3, 4), 0.5))
    assert torch.allclose(new_state[1], torch.ones(2, 5, 4))


def test_gd_accumulates():
    """Two GD steps should accumulate."""
    algo = GD()
    state = torch.zeros(1, 2, 3)
    grads = torch.ones(1, 2, 3)
    eta = torch.tensor(1.0)
    state, _ = algo.step(state, grads, eta, None)
    state, _ = algo.step(state, grads, eta, None)
    assert torch.allclose(state, torch.full((1, 2, 3), 2.0))


def test_gd_no_learnable_params():
    algo = GD()
    assert len(list(algo.parameters())) == 0
