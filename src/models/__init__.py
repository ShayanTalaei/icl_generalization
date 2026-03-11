from .base import SeqModel
from .rnn import LinearRNN, LinearRNNProjected, RNNModel
from .transformer import CausalTransformer

MODEL_REGISTRY: dict[str, type[SeqModel]] = {
    "transformer": CausalTransformer,
}


def build_model(config, d_in: int, d_out: int) -> SeqModel:
    """Construct a model from a ModelConfig and task dimensions."""
    shared = dict(d_in=d_in, d_out=d_out, d_model=config.d_model,
                  n_layers=config.n_layers, dropout=config.dropout)

    if config.type == "transformer":
        return CausalTransformer(**shared, n_heads=config.n_heads,
                                 pos_encoding=config.pos_encoding)
    elif config.type in ("lstm", "gru"):
        return RNNModel(**shared, rnn_type=config.type)
    elif config.type == "linear_rnn":
        gd_init = getattr(config, "gd_init", False)
        return LinearRNN(**shared, gd_init=gd_init)
    elif config.type == "linear_rnn_proj":
        gd_init = getattr(config, "gd_init", False)
        return LinearRNNProjected(**shared, gd_init=gd_init)
    else:
        raise ValueError(f"Unknown model type: {config.type}")
