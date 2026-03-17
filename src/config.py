"""Shared Pydra configuration classes used across scripts."""

import pydra


class MemoryStructureConfig(pydra.Config):
    def __init__(self):
        super().__init__()
        self.type = "matrix"        # matrix | mlp
        self.d_hidden = 64          # MLP hidden dim


class AttentionalBiasConfig(pydra.Config):
    def __init__(self):
        super().__init__()
        self.type = "dot_product"   # dot_product | l2


class RetentionGateConfig(pydra.Config):
    def __init__(self):
        super().__init__()
        self.type = "none"          # none | scalar_l2


class MemoryAlgorithmConfig(pydra.Config):
    def __init__(self):
        super().__init__()
        self.type = "gd"            # gd


class ModelConfig(pydra.Config):
    def __init__(self):
        super().__init__()
        self.type = "transformer"   # transformer | lstm | gru | miras
        self.d_model = 128
        self.n_layers = 6
        self.n_heads = 4            # transformer-only
        self.pos_encoding = "sinusoidal"  # learned | sinusoidal | rope | none (transformer-only)
        self.dropout = 0.0
        self.gd_init = False
        self.use_projections = False
        self.eta_init = 1.0             # initial inner learning rate for MIRAS
        self.alpha_init = 1.0           # initial retention strength for MIRAS
        self.residual = False           # residual connections between MIRAS layers
        self.normalize_qk = False       # L2-normalize projected keys and queries
        self.output_norm = False        # RMSNorm on each layer's output (DeltaNet/Mamba convention)

        # MIRAS-specific (nested)
        self.memory = MemoryStructureConfig()
        self.bias = AttentionalBiasConfig()
        self.retention = RetentionGateConfig()
        self.algorithm = MemoryAlgorithmConfig()

    # Presets -- called via CLI as e.g. model.linear_attention
    def linear_attention(self):
        self.type = "miras"
        self.bias.type = "dot_product"
        self.algorithm.type = "gd"
        self.memory.type = "matrix"
        self.retention.type = "none"

    def mamba(self):
        self.type = "miras"
        self.bias.type = "dot_product"
        self.algorithm.type = "gd"
        self.memory.type = "matrix"
        self.retention.type = "scalar_l2"

    def deltanet(self):
        self.type = "miras"
        self.bias.type = "l2"
        self.algorithm.type = "gd"
        self.memory.type = "matrix"
        self.retention.type = "none"

    def gated_deltanet(self):
        self.type = "miras"
        self.bias.type = "l2"
        self.algorithm.type = "gd"
        self.memory.type = "matrix"
        self.retention.type = "scalar_l2"


class TaskConfig(pydra.Config):
    def __init__(self):
        super().__init__()
        self.type = "linear"
        self.d_input = 10
        self.d_output = 1
        self.noise_std = 0.0
        self.degree = 2               # polynomial degree (polynomial task only)
        self.input_range = "gaussian"   # "gaussian" (N(0,I)) | "uniform" (U(-1,1))


class TrainingConfig(pydra.Config):
    def __init__(self):
        super().__init__()
        self.batch_size = 64
        self.num_steps = 100_000
        self.num_examples = 20
        self.lr = 1e-4
        self.weight_decay = 0.0
        self.lr_schedule = "cosine"   # cosine | constant
        self.grad_clip = 1.0
        self.eval_every = 1000
        self.checkpoint_every = 0     # 0 = disabled
        self.checkpoint_dir = "checkpoints"
        self.dataset_path = ""          # path to pre-generated .pt dataset (empty = online generation)


def build_task(config):
    """Construct a task from a TaskConfig."""
    from src.tasks import TASK_REGISTRY
    task_cls = TASK_REGISTRY[config.type]
    kwargs = dict(
        d_input=config.d_input,
        d_output=config.d_output,
        noise_std=config.noise_std,
    )
    if config.type == "polynomial":
        kwargs["degree"] = config.degree
        kwargs["input_range"] = getattr(config, "input_range", "gaussian")
    return task_cls(**kwargs)
