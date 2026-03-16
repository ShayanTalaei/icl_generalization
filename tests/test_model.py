import torch
from src.models.miras.model import MIRASModel
from src.models.miras.layer import MIRASLayer
from src.models.miras.memory import MatrixMemory, MLPMemory
from src.models.miras.bias import DotProductBias, L2Bias
from src.models.miras.retention import NoRetention, ScalarL2Retention
from src.models.miras.algorithm import GD


def _make_model(d_in=10, d_out=1, use_proj=False, d_model=32,
                bias="dot_product", retention="none", memory="matrix"):
    if memory == "matrix":
        d_k = d_model if use_proj else d_in
        d_v = d_model if use_proj else d_out
        mem = MatrixMemory(d_k, d_v)
    else:
        d_k = d_model if use_proj else d_in
        d_v = d_model if use_proj else d_out
        mem = MLPMemory(d_k, d_v, 16)

    b = DotProductBias() if bias == "dot_product" else L2Bias()
    r = NoRetention() if retention == "none" else ScalarL2Retention()
    layer = MIRASLayer(mem, b, r, GD())
    return MIRASModel(d_in=d_in, d_out=d_out, layers=[layer],
                      use_projections=use_proj, d_model=d_model)


def test_model_output_shape_no_proj():
    model = _make_model(d_in=10, d_out=1, use_proj=False)
    xs = torch.randn(4, 20, 10)
    ys = torch.randn(4, 20, 1)
    preds = model(xs, ys)
    assert preds.shape == (4, 20, 1)


def test_model_output_shape_with_proj():
    model = _make_model(d_in=10, d_out=1, use_proj=True, d_model=32)
    xs = torch.randn(4, 20, 10)
    ys = torch.randn(4, 20, 1)
    preds = model(xs, ys)
    assert preds.shape == (4, 20, 1)


def test_model_first_prediction_is_zero_no_proj():
    """First prediction should be from zero memory (no examples seen)."""
    model = _make_model(d_in=5, d_out=2, use_proj=False, memory="matrix")
    xs = torch.randn(2, 10, 5)
    ys = torch.randn(2, 10, 2)
    preds = model(xs, ys)
    assert torch.allclose(preds[:, 0, :], torch.zeros(2, 2))


def test_model_gradient_flows():
    """Verify gradients flow through the model for training."""
    model = _make_model(d_in=5, d_out=1, use_proj=True, d_model=16)
    xs = torch.randn(2, 10, 5)
    ys = torch.randn(2, 10, 1)
    preds = model(xs, ys)
    loss = ((preds - ys) ** 2).mean()
    loss.backward()
    assert model.proj_k.weight.grad is not None
    assert model.proj_out.weight.grad is not None
    assert model.layers[0].eta.grad is not None


def test_model_multi_layer_no_residual():
    """Multi-layer without residual should produce correct output shape."""
    d_in, d_out, d_model = 5, 1, 32
    layers = [
        MIRASLayer(MatrixMemory(d_model, d_model), L2Bias(), NoRetention(), GD()),
        MIRASLayer(MatrixMemory(d_model, d_model), L2Bias(), NoRetention(), GD()),
    ]
    model = MIRASModel(d_in=d_in, d_out=d_out, layers=layers,
                       use_projections=True, d_model=d_model, residual=False)
    xs = torch.randn(2, 10, d_in)
    ys = torch.randn(2, 10, d_out)
    preds = model(xs, ys)
    assert preds.shape == (2, 10, d_out)


def test_model_multi_layer_with_residual():
    """Multi-layer with residual should produce correct output shape."""
    d_in, d_out, d_model = 5, 1, 32
    layers = [
        MIRASLayer(MatrixMemory(d_model, d_model), L2Bias(), NoRetention(), GD()),
        MIRASLayer(MatrixMemory(d_model, d_model), L2Bias(), NoRetention(), GD()),
    ]
    model = MIRASModel(d_in=d_in, d_out=d_out, layers=layers,
                       use_projections=True, d_model=d_model, residual=True)
    xs = torch.randn(2, 10, d_in)
    ys = torch.randn(2, 10, d_out)
    preds = model(xs, ys)
    assert preds.shape == (2, 10, d_out)


def test_model_multi_layer_gradient_flows():
    """Gradients should flow to all layers and projections."""
    d_in, d_out, d_model = 5, 1, 16
    layers = [
        MIRASLayer(MLPMemory(d_model, d_model, 8), L2Bias(), NoRetention(), GD()),
        MIRASLayer(MLPMemory(d_model, d_model, 8), L2Bias(), NoRetention(), GD()),
    ]
    model = MIRASModel(d_in=d_in, d_out=d_out, layers=layers,
                       use_projections=True, d_model=d_model, residual=True)
    xs = torch.randn(2, 10, d_in)
    ys = torch.randn(2, 10, d_out)
    preds = model(xs, ys)
    loss = ((preds - ys) ** 2).mean()
    loss.backward()
    assert model.layers[0].eta.grad is not None
    assert model.layers[1].eta.grad is not None
    assert model.proj_k.weight.grad is not None
    assert model.proj_out.weight.grad is not None


def test_model_single_layer_backward_compat():
    """Single-layer model should behave identically to before."""
    d_in, d_out = 5, 1
    torch.manual_seed(123)
    mem = MatrixMemory(d_in, d_out)
    layer = MIRASLayer(mem, DotProductBias(), NoRetention(), GD())
    model = MIRASModel(d_in=d_in, d_out=d_out, layers=[layer],
                       use_projections=False, residual=False)
    xs = torch.randn(2, 10, d_in)
    ys = torch.randn(2, 10, d_out)
    preds = model(xs, ys)
    assert preds.shape == (2, 10, d_out)
    assert torch.allclose(preds[:, 0, :], torch.zeros(2, d_out))


def test_model_is_seq_model():
    """MIRASModel should be a SeqModel subclass."""
    from src.models.base import SeqModel
    model = _make_model()
    assert isinstance(model, SeqModel)


def test_model_gd_init():
    """gd_init should set projections to identity embeddings."""
    model = MIRASModel(
        d_in=5, d_out=1,
        layers=[MIRASLayer(MatrixMemory(32, 32), DotProductBias(), NoRetention(), GD())],
        use_projections=True, d_model=32, gd_init=True,
    )
    expected_k = torch.zeros(32, 5)
    expected_k[:5, :5] = torch.eye(5)
    assert torch.allclose(model.proj_k.weight, expected_k)


# --- Factory tests ---

from src.models.miras import build_miras_model
from src.config import ModelConfig


def test_build_miras_model_multi_layer():
    """Factory should create N independent layers."""
    config = ModelConfig()
    config.type = "miras"
    config.bias.type = "l2"
    config.memory.type = "mlp"
    config.memory.d_hidden = 16
    config.retention.type = "none"
    config.algorithm.type = "gd"
    config.n_layers = 4
    config.d_model = 32
    config.use_projections = True
    config.eta_init = 0.01
    config.residual = True
    model = build_miras_model(config, d_in=5, d_out=1)
    assert len(model.layers) == 4
    assert model.residual is True
    # Each layer should have independent parameters
    assert model.layers[0].eta is not model.layers[1].eta


def test_build_miras_model_auto_projections():
    """Factory should auto-enable projections when n_layers > 1."""
    config = ModelConfig()
    config.type = "miras"
    config.bias.type = "dot_product"
    config.memory.type = "matrix"
    config.retention.type = "none"
    config.algorithm.type = "gd"
    config.n_layers = 2
    config.d_model = 32
    config.use_projections = False  # should be overridden
    model = build_miras_model(config, d_in=5, d_out=1)
    assert model.use_proj is True
    assert len(model.layers) == 2


def test_build_miras_model_single_layer_no_auto_proj():
    """Factory should not auto-enable projections for single layer."""
    config = ModelConfig()
    config.type = "miras"
    config.bias.type = "dot_product"
    config.memory.type = "matrix"
    config.retention.type = "none"
    config.algorithm.type = "gd"
    config.n_layers = 1
    config.d_model = 32
    config.use_projections = False
    model = build_miras_model(config, d_in=5, d_out=1)
    assert model.use_proj is False
    assert len(model.layers) == 1
