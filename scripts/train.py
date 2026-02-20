"""Train an ICL model.

Usage:
    python scripts/train.py                                # defaults (transformer)
    python scripts/train.py model.type=lstm model.n_layers=2  # LSTM
    python scripts/train.py model.d_model=256              # override width
    python scripts/train.py training.lr=3e-4 --show        # preview config
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pydra

from src.config import ModelConfig, TaskConfig, TrainingConfig, build_task
from src.models import build_model
from src.training.trainer import Trainer
from src.utils.seed import set_seed


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.seed = 42
        self.model = ModelConfig()
        self.task = TaskConfig()
        self.training = TrainingConfig()


@pydra.main(Config)
def main(config: Config):
    set_seed(config.seed)

    task = build_task(config.task)
    model = build_model(config.model, d_in=task.d_in, d_out=task.d_out)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model : {config.model.type}  ({n_params:,} parameters)")
    print(f"Task  : {config.task.type}  (d_in={task.d_in}, d_out={task.d_out})")
    print(f"Device: {'cuda' if __import__('torch').cuda.is_available() else 'cpu'}")
    print()

    trainer = Trainer(model, task, config.training)
    trainer.train()


if __name__ == "__main__":
    main()
