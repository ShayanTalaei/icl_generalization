from .base import SeqModel
from .rnn import RNNModel
from .transformer import CausalTransformer
from .miras import build_miras_model

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
    elif config.type == "miras":
        return build_miras_model(config, d_in, d_out)
    else:
        raise ValueError(f"Unknown model type: {config.type}")
