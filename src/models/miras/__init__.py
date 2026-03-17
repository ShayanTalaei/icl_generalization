"""MIRAS framework: modular design space for modern RNNs.

Four independent axes:
    - AttentionalBias: learning objective (dot-product, L2, ...)
    - MemoryStructure: memory architecture (matrix, MLP, ...)
    - RetentionGate: memory regularizer (none, scalar L2, ...)
    - MemoryAlgorithm: inner-loop optimizer (GD, ...)
"""

from .bias import AttentionalBias, DotProductBias, L2Bias
from .memory import MemoryStructure, MatrixMemory, MLPMemory
from .retention import RetentionGate, NoRetention, ScalarL2Retention
from .algorithm import MemoryAlgorithm, GD
from .layer import MIRASLayer
from .model import MIRASModel

# Registries mapping config strings to classes
BIAS_REGISTRY: dict[str, type[AttentionalBias]] = {
    "dot_product": DotProductBias,
    "l2": L2Bias,
}

MEMORY_REGISTRY: dict[str, type[MemoryStructure]] = {
    "matrix": MatrixMemory,
    "mlp": MLPMemory,
}

RETENTION_REGISTRY: dict[str, type[RetentionGate]] = {
    "none": NoRetention,
    "scalar_l2": ScalarL2Retention,
}

ALGORITHM_REGISTRY: dict[str, type[MemoryAlgorithm]] = {
    "gd": GD,
}


def _build_memory(memory_config, d_k: int, d_v: int) -> MemoryStructure:
    """Construct memory module from config."""
    if memory_config.type == "matrix":
        return MatrixMemory(d_k, d_v)
    elif memory_config.type == "mlp":
        d_hidden = getattr(memory_config, "d_hidden", 64)
        return MLPMemory(d_k, d_v, d_hidden)
    else:
        raise ValueError(f"Unknown memory type: {memory_config.type}")


def build_miras_model(config, d_in: int, d_out: int) -> MIRASModel:
    """Construct a MIRASModel from a ModelConfig and task dimensions.

    Args:
        config: ModelConfig with nested MIRAS sub-configs
        d_in: input dimensionality
        d_out: output dimensionality

    Returns:
        Configured MIRASModel instance.
    """
    if config.bias.type not in BIAS_REGISTRY:
        raise ValueError(
            f"Unknown bias type: {config.bias.type!r}. "
            f"Available: {list(BIAS_REGISTRY.keys())}"
        )
    if config.algorithm.type not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm type: {config.algorithm.type!r}. "
            f"Available: {list(ALGORITHM_REGISTRY.keys())}"
        )

    n_layers = getattr(config, "n_layers", 1)
    use_projections = config.use_projections
    residual = getattr(config, "residual", False)

    # Auto-enable projections for multi-layer
    if n_layers > 1:
        use_projections = True

    # Memory dimensions depend on projection usage
    if use_projections:
        d_k = d_v = config.d_model
    else:
        d_k, d_v = d_in, d_out

    # Create independent layers
    import torch
    eta_init = getattr(config, "eta_init", 1.0)
    alpha_init = getattr(config, "alpha_init", 1.0)

    layers = []
    for _ in range(n_layers):
        bias = BIAS_REGISTRY[config.bias.type]()
        retention = RETENTION_REGISTRY[config.retention.type]()
        algorithm = ALGORITHM_REGISTRY[config.algorithm.type]()
        memory = _build_memory(config.memory, d_k, d_v)
        layer = MIRASLayer(memory, bias, retention, algorithm)
        layer.set_eta(eta_init)
        layer.set_alpha(alpha_init)
        layers.append(layer)

    return MIRASModel(
        d_in=d_in,
        d_out=d_out,
        layers=layers,
        use_projections=use_projections,
        d_model=config.d_model,
        gd_init=getattr(config, "gd_init", False),
        residual=residual,
        normalize_qk=getattr(config, "normalize_qk", False),
        output_norm=getattr(config, "output_norm", False),
        aggregate=getattr(config, "aggregate", "sequential"),
    )
