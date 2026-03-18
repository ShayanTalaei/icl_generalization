"""Optimal baselines for ICL tasks.

These are the Bayes-optimal or near-optimal estimators for each task,
used as reference points when evaluating model ICL curves.
"""

import torch

from src.tasks.chebyshev import ChebyshevTask, chebyshev_features


@torch.no_grad()
def chebyshev_ridge_baseline(
    task: ChebyshevTask,
    num_examples: int,
    batch_size: int,
    num_batches: int,
    device,
    ridge_lambda: float = 0.2,
) -> torch.Tensor:
    """Per-position MSE for the Chebyshev ridge regression baseline.

    At each position n (having seen n demonstrations), fits ridge regression
    in the Chebyshev basis on (x_1, y_1), ..., (x_n, y_n) and predicts y_{n+1}.

    Args:
        task: ChebyshevTask instance.
        num_examples: number of demonstration positions to evaluate.
        batch_size: batch size.
        num_batches: number of batches to average over.
        device: torch device.
        ridge_lambda: ridge regularization strength (Wilcoxson et al. use 0.2).

    Returns:
        losses: (num_examples + 1,) tensor of per-position MSE.
    """
    if not isinstance(task, ChebyshevTask):
        raise ValueError("chebyshev_ridge_baseline only supports ChebyshevTask")

    n = num_examples + 1
    total_se = torch.zeros(n)
    max_degree = task.max_degree
    D = max_degree + 1

    for _ in range(num_batches):
        batch = task.sample_batch(batch_size, num_examples)
        xs = batch.xs.squeeze(-1).to(device)  # (B, n)
        ys = batch.ys.squeeze(-1).to(device)  # (B, n)
        feats = chebyshev_features(xs, max_degree)  # (B, n, D)

        se_per_pos = torch.zeros(n)
        for pos in range(n):
            if pos == 0:
                pred = torch.zeros(batch_size, device=device)
            else:
                F = feats[:, :pos, :]           # (B, pos, D)
                y_ctx = ys[:, :pos]              # (B, pos)
                FtF = torch.bmm(F.transpose(1, 2), F)
                reg = ridge_lambda * torch.eye(D, device=device).unsqueeze(0)
                Fty = torch.bmm(F.transpose(1, 2), y_ctx.unsqueeze(-1))
                w = torch.linalg.solve(FtF + reg, Fty).squeeze(-1)  # (B, D)
                pred = (feats[:, pos, :] * w).sum(dim=-1)            # (B,)

            se_per_pos[pos] = ((pred - ys[:, pos]) ** 2).mean().item()
        total_se += se_per_pos

    return total_se / num_batches
