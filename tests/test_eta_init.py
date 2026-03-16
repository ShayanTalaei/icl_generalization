import torch
from src.config import ModelConfig
from src.models.miras import build_miras_model


def test_eta_init_default():
    """Default eta_init=1.0, alpha_init=1.0."""
    config = ModelConfig()
    config.type = "miras"
    model = build_miras_model(config, d_in=10, d_out=1)
    assert model.layers[0].eta.item() == 1.0
    assert model.layers[0].alpha.item() == 1.0


def test_eta_init_custom():
    """Custom eta_init and alpha_init should be applied."""
    config = ModelConfig()
    config.type = "miras"
    config.eta_init = 0.01
    config.alpha_init = 0.95
    model = build_miras_model(config, d_in=10, d_out=1)
    assert abs(model.layers[0].eta.item() - 0.01) < 1e-6
    assert abs(model.layers[0].alpha.item() - 0.95) < 1e-6


def test_eta_init_with_projections():
    """eta_init should work with projections enabled."""
    config = ModelConfig()
    config.type = "miras"
    config.use_projections = True
    config.eta_init = 0.1
    model = build_miras_model(config, d_in=10, d_out=1)
    assert abs(model.layers[0].eta.item() - 0.1) < 1e-6


def test_eta_init_cli_syntax():
    """Verify eta_init field exists on ModelConfig."""
    config = ModelConfig()
    assert hasattr(config, "eta_init")
    assert hasattr(config, "alpha_init")
    assert config.eta_init == 1.0
    assert config.alpha_init == 1.0
