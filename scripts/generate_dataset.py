"""Generate pre-baked ICL datasets for reproducible, fast training.

Generates N training batches + M eval batches and saves as a single .pt file.
All configs within an experiment phase should use the same dataset for
fair comparison.

Usage:
    python scripts/generate_dataset.py \
        task.type=linear task.d_input=20 task.d_output=1 \
        num_train_batches=50000 num_eval_batches=50 \
        batch_size=64 num_examples=40 \
        eval_num_examples=200 \
        seed=42 \
        output_path=scratch/data/linear_d20_50k.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pydra
import torch
from tqdm import tqdm

from src.config import TaskConfig, build_task
from src.utils.seed import set_seed


class GenConfig(pydra.Config):
    def __init__(self):
        super().__init__()
        self.task = TaskConfig()
        self.num_train_batches = 50000
        self.num_eval_batches = 50
        self.batch_size = 64
        self.num_examples = 40        # training context length
        self.eval_num_examples = 200  # eval context length (for ICL curves)
        self.seed = 42
        self.output_path = ""


@pydra.main(GenConfig)
def main(config: GenConfig):
    set_seed(config.seed)

    task = build_task(config.task)
    d_in = task.d_in
    d_out = task.d_out
    n_train = config.num_examples + 1
    n_eval = config.eval_num_examples + 1

    print(f"Task: {config.task.type} (d_in={d_in}, d_out={d_out})")
    print(f"Training: {config.num_train_batches} batches x {config.batch_size} x {n_train}")
    print(f"Eval:     {config.num_eval_batches} batches x {config.batch_size} x {n_eval}")

    # Estimate size
    train_bytes = config.num_train_batches * config.batch_size * n_train * (d_in + d_out) * 4
    eval_bytes = config.num_eval_batches * config.batch_size * n_eval * (d_in + d_out) * 4
    total_gb = (train_bytes + eval_bytes) / 1e9
    print(f"Estimated size: {total_gb:.2f} GB")

    # Generate training data
    print(f"\nGenerating training data...")
    train_xs = torch.empty(config.num_train_batches, config.batch_size, n_train, d_in)
    train_ys = torch.empty(config.num_train_batches, config.batch_size, n_train, d_out)

    for i in tqdm(range(config.num_train_batches), desc="Training batches"):
        batch = task.sample_batch(config.batch_size, config.num_examples)
        train_xs[i] = batch.xs
        train_ys[i] = batch.ys

    # Generate eval data
    print(f"Generating eval data...")
    eval_xs = torch.empty(config.num_eval_batches, config.batch_size, n_eval, d_in)
    eval_ys = torch.empty(config.num_eval_batches, config.batch_size, n_eval, d_out)

    for i in tqdm(range(config.num_eval_batches), desc="Eval batches"):
        batch = task.sample_batch(config.batch_size, config.eval_num_examples)
        eval_xs[i] = batch.xs
        eval_ys[i] = batch.ys

    # Save
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "xs": train_xs,
        "ys": train_ys,
        "eval_xs": eval_xs,
        "eval_ys": eval_ys,
        "d_in": d_in,
        "d_out": d_out,
        "metadata": {
            "task_type": config.task.type,
            "d_input": config.task.d_input,
            "d_output": config.task.d_output,
            "noise_std": config.task.noise_std,
            "seed": config.seed,
            "num_train_batches": config.num_train_batches,
            "num_eval_batches": config.num_eval_batches,
            "batch_size": config.batch_size,
            "num_examples": config.num_examples,
            "eval_num_examples": config.eval_num_examples,
        },
    }

    print(f"Saving to {output_path}...")
    torch.save(data, output_path)
    file_size_gb = output_path.stat().st_size / 1e9
    print(f"Done! File size: {file_size_gb:.2f} GB")


if __name__ == "__main__":
    main()
