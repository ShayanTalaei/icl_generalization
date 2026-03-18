import torch
from src.models.miras.bias import DotProductBias, L2Bias


def test_dot_product_bias_returns_target():
    bias = DotProductBias()
    pred = torch.randn(4, 5)
    target = torch.randn(4, 5)
    d_out = bias.error_signal(pred, target)
    assert torch.allclose(d_out, target)


def test_dot_product_bias_ignores_prediction():
    bias = DotProductBias()
    pred1 = torch.randn(4, 5)
    pred2 = torch.randn(4, 5)
    target = torch.randn(4, 5)
    assert torch.allclose(bias.error_signal(pred1, target), bias.error_signal(pred2, target))


def test_l2_bias_returns_residual():
    bias = L2Bias()
    pred = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[3.0, 5.0]])
    d_out = bias.error_signal(pred, target)
    assert torch.allclose(d_out, torch.tensor([[2.0, 3.0]]))


def test_l2_bias_zero_when_perfect():
    bias = L2Bias()
    pred = torch.randn(4, 5)
    d_out = bias.error_signal(pred, pred)
    assert torch.allclose(d_out, torch.zeros_like(pred))


def test_bias_no_learnable_params():
    """Bias modules should have no learnable parameters."""
    for bias in [DotProductBias(), L2Bias()]:
        assert len(list(bias.parameters())) == 0
