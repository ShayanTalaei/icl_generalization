"""Shared Pydra configuration classes used across scripts."""

import pydra


class ModelConfig(pydra.Config):
    def __init__(self):
        super().__init__()
        self.type = "transformer"   # transformer | lstm | gru
        self.d_model = 128
        self.n_layers = 6
        self.n_heads = 4            # transformer-only
        self.pos_encoding = "sinusoidal"  # learned | sinusoidal | rope | none (transformer-only)
        self.dropout = 0.0
        self.gd_init = False          # linear_rnn only: init weights to implement exact GD


class TaskConfig(pydra.Config):
    def __init__(self):
        super().__init__()
        self.type = "linear"
        self.d_input = 10
        self.d_output = 1
        self.noise_std = 0.0


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


def build_task(config):
    """Construct a task from a TaskConfig."""
    from src.tasks import TASK_REGISTRY
    task_cls = TASK_REGISTRY[config.type]
    return task_cls(
        d_input=config.d_input,
        d_output=config.d_output,
        noise_std=config.noise_std,
    )
