"""Verify that the LinearRNN architecture can implement exact GD.

Sets proj_k, proj_q to embed x into first d_in dims of d_model,
proj_v to embed y into first d_out dims, and proj_out to read them back.
With these hand-crafted weights, M accumulates y_i x_i^T in the top-left
block, and readout computes M @ x_query = (Σ y_i x_i^T) x_query.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.rnn import LinearRNN
from src.tasks.linear import LinearTask

def set_gd_weights(model, d_in, d_out, eta=1.0):
    """Hand-craft weights so the LinearRNN implements one-step GD."""
    d = model.d_model
    with torch.no_grad():
        # proj_k: x -> [x, 0, ...] (embed x into first d_in dims)
        model.proj_k.weight.zero_()
        model.proj_k.weight[:d_in, :d_in] = eta * torch.eye(d_in)

        # proj_q: same embedding for queries
        model.proj_q.weight.zero_()
        model.proj_q.weight[:d_in, :d_in] = torch.eye(d_in)

        # proj_v: y -> [y, 0, ...] (embed y into first d_out dims)
        model.proj_v.weight.zero_()
        model.proj_v.weight[:d_out, :d_out] = torch.eye(d_out)

        # proj_out: extract first d_out dims
        model.proj_out.weight.zero_()
        model.proj_out.weight[:d_out, :d_out] = torch.eye(d_out)


def main():
    d_in, d_out, d_model = 3, 1, 128
    task = LinearTask(d_input=d_in, d_output=d_out)
    model = LinearRNN(d_in=d_in, d_out=d_out, d_model=d_model)

    # --- Random init ---
    model.eval()
    with torch.no_grad():
        batch = task.sample_batch(1024, 100)
        preds_rand = model(batch.xs, batch.ys)
        mse_rand = (preds_rand - batch.ys).pow(2).mean(dim=(0, 2))

    # --- GD init ---
    set_gd_weights(model, d_in, d_out, eta=1.0)
    model.eval()
    with torch.no_grad():
        preds_gd = model(batch.xs, batch.ys)
        mse_gd = (preds_gd - batch.ys).pow(2).mean(dim=(0, 2))

    # --- GD init with 1/k normalization (= ridge regression / averaged GD) ---
    # Manually compute: y_pred_i = (1/i) * M_i @ q_i  where M_i = Σ_{j<i} v_j k_j^T
    # This is equivalent to predicting with the average outer product
    preds_gd_norm = torch.zeros_like(preds_gd)
    with torch.no_grad():
        keys = model.proj_k(batch.xs)
        queries = model.proj_q(batch.xs)
        values = model.proj_v(batch.ys[:, :-1, :])
        M = torch.zeros(1024, model.d_model, model.d_model)
        for i in range(101):
            scale = 1.0 / i if i > 0 else 1.0
            out = torch.bmm(scale * M, queries[:, i, :].unsqueeze(-1)).squeeze(-1)
            preds_gd_norm[:, i, :] = model.proj_out(out)
            if i < 100:
                v = values[:, i, :].unsqueeze(-1)
                k = keys[:, i, :].unsqueeze(-2)
                M = M + torch.bmm(v, k)
        mse_gd_norm = (preds_gd_norm - batch.ys).pow(2).mean(dim=(0, 2))

    print("Position | Random Init | GD Weights | GD + 1/k norm")
    print("-" * 55)
    for pos in [0, 1, 2, 5, 10, 20, 50, 100]:
        print(f"  {pos:>3d}     | {mse_rand[pos]:>10.4f}  | {mse_gd[pos]:>10.4f} | {mse_gd_norm[pos]:.4f}")


if __name__ == "__main__":
    main()
