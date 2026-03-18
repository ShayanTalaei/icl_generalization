"""Evaluate per-position ICL loss curve.

Generates long sequences and records how prediction loss changes as the model
sees more (x, y) demonstrations.  Saves results as JSON for plotting.

Usage:
    # Untrained transformer
    python scripts/eval_icl.py model.type=transformer label=tf_untrained

    # Trained transformer (load checkpoint)
    python scripts/eval_icl.py model.type=transformer checkpoint=checkpoints/step_5000.pt label=tf_5k

    # Untrained LSTM
    python scripts/eval_icl.py model.type=lstm model.n_layers=2 label=lstm_untrained
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pydra
import torch

from src.config import ModelConfig, TaskConfig, build_task
from src.models import build_model
from src.utils.seed import set_seed


class EvalConfig(pydra.Config):
    def __init__(self):
        super().__init__()
        self.seed = 42
        self.model = ModelConfig()
        self.task = TaskConfig()
        self.checkpoint = ""          # path to .pt checkpoint; empty = untrained
        self.num_examples = 100       # length of the demonstration sequence
        self.batch_size = 256
        self.num_batches = 50
        self.output_dir = "results"
        self.label = ""               # identifier for this run (used in filename + plot)
        self.with_ridge_baseline = False   # also compute Chebyshev ridge baseline
        self.ridge_lambda = 0.2            # ridge regularization (Wilcoxson et al. use 0.2)


@torch.no_grad()
def chebyshev_ridge_baseline(
    task,
    num_examples: int,
    batch_size: int,
    num_batches: int,
    device,
    ridge_lambda: float = 0.2,
) -> torch.Tensor:
    """Compute per-position MSE for the Chebyshev ridge regression baseline.

    At each position n (having seen n demonstrations), fits ridge regression
    in the Chebyshev basis on (x_1, y_1), ..., (x_n, y_n) and predicts y_{n+1}.

    Only works for ChebyshevTask (d_in=1).

    Args:
        task: ChebyshevTask instance.
        num_examples: number of demonstration positions to evaluate.
        batch_size: batch size.
        num_batches: number of batches to average over.
        device: torch device for computation.
        ridge_lambda: ridge regularization strength.

    Returns:
        losses: (num_examples + 1,) tensor of per-position MSE.
    """
    from src.tasks.chebyshev import ChebyshevTask, chebyshev_features

    if not isinstance(task, ChebyshevTask):
        raise ValueError("chebyshev_ridge_baseline only supports ChebyshevTask")

    n = num_examples + 1
    total_se = torch.zeros(n)
    max_degree = task.max_degree
    D = max_degree + 1  # feature dimension

    for _ in range(num_batches):
        batch = task.sample_batch(batch_size, num_examples)
        xs = batch.xs.squeeze(-1).to(device)  # (B, n)
        ys = batch.ys.squeeze(-1).to(device)  # (B, n)

        # Chebyshev features: (B, n, D)
        feats = chebyshev_features(xs, max_degree)

        se_per_pos = torch.zeros(n)
        for pos in range(n):
            if pos == 0:
                # No demonstrations: predict 0
                pred = torch.zeros(batch_size, device=device)
            else:
                # Fit ridge on the first `pos` demonstrations
                F = feats[:, :pos, :]      # (B, pos, D)
                y_ctx = ys[:, :pos]        # (B, pos)
                FtF = torch.bmm(F.transpose(1, 2), F)  # (B, D, D)  where D = max_degree+1
                reg = ridge_lambda * torch.eye(D, device=device).unsqueeze(0)
                FtF_reg = FtF + reg
                Fty = torch.bmm(F.transpose(1, 2), y_ctx.unsqueeze(-1))  # (B, D, 1)
                w = torch.linalg.solve(FtF_reg, Fty).squeeze(-1)  # (B, D)
                # Predict at query position
                q_feat = feats[:, pos, :]  # (B, D)
                pred = (q_feat * w).sum(dim=-1)  # (B,)

            target = ys[:, pos]  # (B,)
            se_per_pos[pos] = ((pred - target) ** 2).mean().item()

        total_se += se_per_pos

    return total_se / num_batches


@torch.no_grad()
def eval_icl_curve(model, task, num_examples, batch_size, num_batches, device):
    """Compute per-position MSE averaged over many batches.

    Returns:
        losses: Tensor of shape (num_examples + 1,) -- MSE at each position.
                Position i means the model has seen i demonstrations before
                predicting y_{i+1}.
    """
    model.eval()
    n = num_examples + 1
    total_se = torch.zeros(n)

    for _ in range(num_batches):
        batch = task.sample_batch(batch_size, num_examples)
        xs = batch.xs.to(device)
        ys = batch.ys.to(device)

        y_preds = model(xs, ys)
        se = (y_preds - ys).pow(2).mean(dim=(0, 2))  # avg over batch & d_out
        total_se += se.cpu()

    return total_se / num_batches


@pydra.main(EvalConfig)
def main(config: EvalConfig):
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = build_task(config.task)
    model = build_model(config.model, d_in=task.d_in, d_out=task.d_out)
    model.to(device)

    if config.checkpoint:
        ckpt = torch.load(config.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {config.checkpoint}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model : {config.model.type}  ({n_params:,} params)")
    print(f"Task  : {config.task.type}  (d_in={task.d_in}, d_out={task.d_out})")
    print(f"Eval  : {config.num_examples} examples, "
          f"{config.batch_size}x{config.num_batches} samples")
    print(f"Device: {device}")
    print()

    losses = eval_icl_curve(
        model, task,
        num_examples=config.num_examples,
        batch_size=config.batch_size,
        num_batches=config.num_batches,
        device=device,
    )

    label = config.label or f"{config.model.type}_{'untrained' if not config.checkpoint else 'trained'}"

    result = {
        "label": label,
        "model_type": config.model.type,
        "pos_encoding": config.model.pos_encoding,
        "checkpoint": config.checkpoint,
        "d_model": config.model.d_model,
        "n_layers": config.model.n_layers,
        "task_type": config.task.type,
        "d_input": config.task.d_input,
        "d_output": config.task.d_output,
        "num_examples": config.num_examples,
        "per_position_loss": losses.tolist(),
    }

    if config.with_ridge_baseline:
        from src.tasks.chebyshev import ChebyshevTask
        if isinstance(task, ChebyshevTask):
            ridge_losses = chebyshev_ridge_baseline(
                task,
                num_examples=config.num_examples,
                batch_size=config.batch_size,
                num_batches=config.num_batches,
                device=device,
                ridge_lambda=config.ridge_lambda,
            )
            result["ridge_baseline_loss"] = ridge_losses.tolist()
            print(f"Ridge baseline at position 10: {ridge_losses[min(10, len(ridge_losses)-1)]:.4f}")
            print(f"Ridge baseline at position 50: {ridge_losses[min(50, len(ridge_losses)-1)]:.4f}")
        else:
            print("Warning: with_ridge_baseline=true but task is not ChebyshevTask, skipping.")

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Loss at position  0 (zero-shot): {losses[0]:.4f}")
    print(f"Loss at position 10:             {losses[min(10, len(losses)-1)]:.4f}")
    print(f"Loss at position 50:             {losses[min(50, len(losses)-1)]:.4f}")
    print(f"Loss at position {len(losses)-1} (query):    {losses[-1]:.4f}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
