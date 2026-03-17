"""Generate pre-baked ICL datasets for reproducible, fast training.

Generates N training batches + M eval batches and saves as a single .pt file.
All configs within an experiment phase should use the same dataset for
fair comparison. Uses GPU for fast polynomial feature expansion.

Usage:
    python scripts/generate_dataset.py \
        task.type=polynomial task.d_input=5 task.d_output=1 task.degree=3 \
        num_train_batches=50000 num_eval_batches=50 \
        batch_size=64 num_examples=40 \
        eval_num_examples=200 \
        seed=42 \
        output_path=scratch/data/poly3_50k.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pydra
import torch
from tqdm import tqdm

from src.config import TaskConfig, build_task
from src.tasks.polynomial import _build_monomial_indices, _polynomial_features
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
        self.device = "cuda:0"        # device for generation


def generate_batch_on_device(d_input, d_output, d_feat, degree, index_cache,
                             batch_size, num_examples, noise_std, device):
    """Generate one ICL batch directly on the given device."""
    n = num_examples + 1
    xs = torch.randn(batch_size, n, d_input, device=device)
    phi = _polynomial_features(xs, degree, index_cache)
    W = torch.randn(batch_size, d_feat, d_output, device=device) / (d_feat ** 0.5)
    ys = torch.bmm(phi, W)
    if noise_std > 0:
        ys = ys + torch.randn_like(ys) * noise_std
    return xs.cpu(), ys.cpu()


@pydra.main(GenConfig)
def main(config: GenConfig):
    set_seed(config.seed)

    task = build_task(config.task)
    d_in = task.d_in
    d_out = task.d_out
    degree = getattr(config.task, "degree", 1)
    noise_std = config.task.noise_std
    d_feat = task.d_feat if hasattr(task, "d_feat") else d_in
    n_train = config.num_examples + 1
    n_eval = config.eval_num_examples + 1

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    print(f"Task: {config.task.type} (d_in={d_in}, d_out={d_out}, degree={degree})")
    print(f"Device: {device}")
    print(f"Training: {config.num_train_batches} batches x {config.batch_size} x {n_train}")
    print(f"Eval:     {config.num_eval_batches} batches x {config.batch_size} x {n_eval}")

    # Estimate size
    train_bytes = config.num_train_batches * config.batch_size * n_train * (d_in + d_out) * 4
    eval_bytes = config.num_eval_batches * config.batch_size * n_eval * (d_in + d_out) * 4
    total_gb = (train_bytes + eval_bytes) / 1e9
    print(f"Estimated size: {total_gb:.2f} GB")

    # Build index cache on device for GPU-accelerated feature expansion
    if config.task.type == "polynomial":
        index_cache = [idx.to(device) for idx in _build_monomial_indices(d_in, degree)]
    else:
        index_cache = None

    # Generate training data
    print(f"\nGenerating training data...")
    train_xs = torch.empty(config.num_train_batches, config.batch_size, n_train, d_in)
    train_ys = torch.empty(config.num_train_batches, config.batch_size, n_train, d_out)

    for i in tqdm(range(config.num_train_batches), desc="Training batches"):
        if config.task.type == "polynomial":
            xs, ys = generate_batch_on_device(
                d_in, d_out, d_feat, degree, index_cache,
                config.batch_size, config.num_examples, noise_std, device,
            )
            train_xs[i] = xs
            train_ys[i] = ys
        else:
            batch = task.sample_batch(config.batch_size, config.num_examples)
            train_xs[i] = batch.xs
            train_ys[i] = batch.ys

    # Generate eval data
    print(f"Generating eval data...")
    eval_xs = torch.empty(config.num_eval_batches, config.batch_size, n_eval, d_in)
    eval_ys = torch.empty(config.num_eval_batches, config.batch_size, n_eval, d_out)

    for i in tqdm(range(config.num_eval_batches), desc="Eval batches"):
        if config.task.type == "polynomial":
            xs, ys = generate_batch_on_device(
                d_in, d_out, d_feat, degree, index_cache,
                config.batch_size, config.eval_num_examples, noise_std, device,
            )
            eval_xs[i] = xs
            eval_ys[i] = ys
        else:
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
            "degree": degree,
            "noise_std": noise_std,
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
