from .base import SeqModel
from .rnn import AssociativeRNN, MatrixMemory, MLPMemory, RNNModel
from .transformer import CausalTransformer
from .miras import build_miras_model

MODEL_REGISTRY: dict[str, type[SeqModel]] = {
    "transformer": CausalTransformer,
}


def _build_memory(config, d_k: int, d_v: int):
    """Construct the memory module from config (legacy linear_rnn path)."""
    rule = getattr(config, "update_rule", "hebbian")
    mem_type = getattr(config, "memory_type", "matrix")

    if mem_type == "matrix":
        return MatrixMemory(d_k, d_v, update_rule=rule)
    elif mem_type == "mlp":
        d_hidden = getattr(config, "memory_d_hidden", 64)
        return MLPMemory(d_k, d_v, d_hidden, update_rule=rule)
    else:
        raise ValueError(f"Unknown memory_type: {mem_type}")


def build_model(config, d_in: int, d_out: int) -> SeqModel:
    """Construct a model from a ModelConfig and task dimensions."""
    shared = dict(d_in=d_in, d_out=d_out, d_model=config.d_model,
                  n_layers=config.n_layers, dropout=config.dropout)

    if config.type == "transformer":
        return CausalTransformer(**shared, n_heads=config.n_heads,
                                 pos_encoding=config.pos_encoding)
    elif config.type in ("lstm", "gru"):
        return RNNModel(**shared, rnn_type=config.type)
    elif config.type == "miras":
        return build_miras_model(config, d_in, d_out)
    elif config.type in ("linear_rnn", "linear_rnn_proj"):
        # Legacy path -- kept for backward compat until verification passes
        use_proj = config.type == "linear_rnn_proj"
        gd_init = getattr(config, "gd_init", False)

        if use_proj:
            d_k = d_v = config.d_model
        else:
            d_k, d_v = d_in, d_out

        memory = _build_memory(config, d_k, d_v)
        return AssociativeRNN(
            d_in=d_in, d_out=d_out, memory=memory,
            use_projections=use_proj, d_model=config.d_model,
            gd_init=gd_init,
        )
    else:
        raise ValueError(f"Unknown model type: {config.type}")
