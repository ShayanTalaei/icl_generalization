import torch
from src.config import ModelConfig
from src.models.miras import build_miras_model


def test_eta_init_default():
    """Default eta_init=1.0, alpha_init=1.0."""
    config = ModelConfig()
    config.type = "miras"
    model = build_miras_model(config, d_in=10, d_out=1)
    assert abs(model.layers[0].eta.item() - 1.0) < 1e-4
    assert abs(model.layers[0].alpha.item() - 1.0) < 1e-4


def test_eta_init_custom():
    """Custom eta_init and alpha_init should be applied."""
    config = ModelConfig()
    config.type = "miras"
    config.eta_init = 0.01
    config.alpha_init = 0.95
    model = build_miras_model(config, d_in=10, d_out=1)
    assert abs(model.layers[0].eta.item() - 0.01) < 1e-4
    assert abs(model.layers[0].alpha.item() - 0.95) < 1e-4


def test_eta_init_with_projections():
    """eta_init should work with projections enabled."""
    config = ModelConfig()
    config.type = "miras"
    config.use_projections = True
    config.eta_init = 0.1
    model = build_miras_model(config, d_in=10, d_out=1)
    assert abs(model.layers[0].eta.item() - 0.1) < 1e-4


def test_eta_init_cli_syntax():
    """Verify eta_init field exists on ModelConfig."""
    config = ModelConfig()
    assert hasattr(config, "eta_init")
    assert hasattr(config, "alpha_init")
    assert config.eta_init == 1.0
    assert config.alpha_init == 1.0


def test_eta_always_positive():
    """eta should remain positive even with extreme parameter values."""
    from src.models.miras.layer import MIRASLayer
    from src.models.miras.memory import MatrixMemory
    from src.models.miras.bias import DotProductBias
    from src.models.miras.retention import NoRetention
    from src.models.miras.algorithm import GD

    layer = MIRASLayer(MatrixMemory(4, 3), DotProductBias(), NoRetention(), GD())
    layer.set_eta(0.001)
    assert layer.eta.item() > 0

    # Even with extreme negative _log_eta, eta stays positive
    with torch.no_grad():
        layer._log_eta.fill_(-100.0)
    assert layer.eta.item() > 0
    assert layer.eta.item() < 1e-10  # very small but positive
