import torch
from src.models.miras.layer import MIRASLayer
from src.models.miras.memory import MatrixMemory, MLPMemory
from src.models.miras.bias import DotProductBias, L2Bias
from src.models.miras.retention import NoRetention, ScalarL2Retention
from src.models.miras.algorithm import GD


def _make_layer(d_k=4, d_v=3, bias="dot_product", retention="none", memory="matrix"):
    mem = MatrixMemory(d_k, d_v) if memory == "matrix" else MLPMemory(d_k, d_v, 8)
    b = DotProductBias() if bias == "dot_product" else L2Bias()
    r = NoRetention() if retention == "none" else ScalarL2Retention()
    return MIRASLayer(mem, b, r, GD())


def test_layer_init_state():
    layer = _make_layer()
    state = layer.init_state(B=2, device="cpu", dtype=torch.float32)
    mem_state, optim_state = state
    assert mem_state.shape == (2, 3, 4)
    assert optim_state is None


def test_layer_read_zero_state():
    layer = _make_layer()
    state = layer.init_state(B=2, device="cpu", dtype=torch.float32)
    query = torch.randn(2, 4)
    out = layer.read(state, query)
    assert out.shape == (2, 3)
    assert torch.allclose(out, torch.zeros(2, 3))


def test_layer_write_then_read_hebbian():
    """After writing (k, v) with dot-product bias, reading k should return eta * v."""
    layer = _make_layer(d_k=3, d_v=2, bias="dot_product", retention="none")
    with torch.no_grad():
        layer.eta.fill_(1.0)
        layer.alpha.fill_(1.0)

    state = layer.init_state(B=1, device="cpu", dtype=torch.float32)
    k = torch.tensor([[1.0, 0.0, 0.0]])
    v = torch.tensor([[2.0, 3.0]])

    state = layer.write(state, k, v)
    out = layer.read(state, k)
    assert torch.allclose(out, v)


def test_layer_write_then_read_delta():
    """Delta rule: first write with zero state gives same result as Hebbian."""
    layer = _make_layer(d_k=3, d_v=2, bias="l2", retention="none")
    with torch.no_grad():
        layer.eta.fill_(1.0)
        layer.alpha.fill_(1.0)

    state = layer.init_state(B=1, device="cpu", dtype=torch.float32)
    k = torch.tensor([[1.0, 0.0, 0.0]])
    v = torch.tensor([[2.0, 3.0]])

    state = layer.write(state, k, v)
    out = layer.read(state, k)
    assert torch.allclose(out, v)


def test_layer_retention_scalar_l2():
    """With scalar retention, old state should be decayed."""
    layer = _make_layer(d_k=3, d_v=2, bias="dot_product", retention="scalar_l2")
    with torch.no_grad():
        layer.eta.fill_(1.0)
        layer.alpha.fill_(0.5)

    state = layer.init_state(B=1, device="cpu", dtype=torch.float32)
    k = torch.tensor([[1.0, 0.0, 0.0]])
    v1 = torch.tensor([[2.0, 0.0]])
    v2 = torch.tensor([[0.0, 4.0]])

    state = layer.write(state, k, v1)
    state = layer.write(state, k, v2)
    out = layer.read(state, k)
    # After write 1: M = 0 + 1 * [[2,0,0],[0,0,0]] = [[2,0,0],[0,0,0]]
    # After write 2: M = 0.5 * [[2,0,0],[0,0,0]] + 1 * [[0,0,0],[4,0,0]]
    #              = [[1,0,0],[4,0,0]]
    # M @ k = [1, 4]
    assert torch.allclose(out, torch.tensor([[1.0, 4.0]]))


def test_layer_eta_alpha_are_parameters():
    layer = _make_layer()
    param_names = {name for name, _ in layer.named_parameters()}
    assert "eta" in param_names
    assert "alpha" in param_names


def test_layer_mlp_memory_write_read():
    """MLP memory with L2 bias should work through MIRASLayer."""
    torch.manual_seed(42)
    layer = _make_layer(d_k=3, d_v=3, bias="l2", memory="mlp")
    with torch.no_grad():
        layer.eta.fill_(0.01)
        layer.alpha.fill_(1.0)

    state = layer.init_state(B=2, device="cpu", dtype=torch.float32)
    k = torch.randn(2, 3)
    v = torch.randn(2, 3)

    state = layer.write(state, k, v)
    out = layer.read(state, k)
    assert out.shape == (2, 3)
