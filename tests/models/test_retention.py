import torch
from src.models.miras.retention import NoRetention, ScalarL2Retention


def test_no_retention_tensor():
    gate = NoRetention()
    state = torch.randn(2, 3, 4)
    alpha = torch.tensor(0.9)
    result = gate.apply(state, alpha)
    assert result is state


def test_no_retention_tuple():
    gate = NoRetention()
    state = (torch.randn(2, 3, 4), torch.randn(2, 5, 4))
    alpha = torch.tensor(0.9)
    result = gate.apply(state, alpha)
    assert result is state


def test_scalar_l2_tensor():
    gate = ScalarL2Retention()
    state = torch.ones(2, 3, 4)
    alpha = torch.tensor(0.5)
    result = gate.apply(state, alpha)
    assert torch.allclose(result, torch.full((2, 3, 4), 0.5))


def test_scalar_l2_tuple():
    gate = ScalarL2Retention()
    W1 = torch.ones(2, 3, 4)
    W2 = torch.ones(2, 5, 4) * 2
    alpha = torch.tensor(0.5)
    result = gate.apply((W1, W2), alpha)
    assert torch.allclose(result[0], torch.full((2, 3, 4), 0.5))
    assert torch.allclose(result[1], torch.ones(2, 5, 4))


def test_scalar_l2_key_ignored():
    """key parameter exists for future use but is ignored by ScalarL2."""
    gate = ScalarL2Retention()
    state = torch.ones(2, 3, 4)
    alpha = torch.tensor(0.5)
    result_no_key = gate.apply(state, alpha)
    result_with_key = gate.apply(state, alpha, key=torch.randn(2, 4))
    assert torch.allclose(result_no_key, result_with_key)


def test_retention_no_learnable_params():
    for gate in [NoRetention(), ScalarL2Retention()]:
        assert len(list(gate.parameters())) == 0
