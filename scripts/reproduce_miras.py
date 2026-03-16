"""Reproduce previous experiment results using the new MIRAS framework.

Runs the same eval configs as the old linear_rnn experiments and compares
per-position MSE against saved reference results.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.config import TaskConfig, build_task
from src.models.miras import (
    MIRASModel, MIRASLayer, MatrixMemory, MLPMemory,
    DotProductBias, L2Bias, NoRetention, GD,
)
from src.utils.seed import set_seed

SCRATCH = Path("/scratch/m000122/stalaei/icl_generalization")


@torch.no_grad()
def eval_icl_curve(model, task, num_examples=100, batch_size=256, num_batches=50, device="cuda"):
    model.eval()
    n = num_examples + 1
    total_se = torch.zeros(n)
    for _ in range(num_batches):
        batch = task.sample_batch(batch_size, num_examples)
        xs, ys = batch.xs.to(device), batch.ys.to(device)
        preds = model(xs, ys)
        se = (preds - ys).pow(2).mean(dim=(0, 2))
        total_se += se.cpu()
    return total_se / num_batches


def build_miras(bias_type, memory_type, d_k, d_v, d_in, d_out,
                eta=0.01, alpha=1.0, d_hidden=64, use_proj=False, d_model=128):
    bias = DotProductBias() if bias_type == "dot_product" else L2Bias()
    if memory_type == "matrix":
        mem = MatrixMemory(d_model if use_proj else d_k, d_model if use_proj else d_v)
    else:
        mem = MLPMemory(d_model if use_proj else d_k, d_model if use_proj else d_v, d_hidden)
    layer = MIRASLayer(mem, bias, NoRetention(), GD())
    with torch.no_grad():
        layer.eta.fill_(eta)
        layer.alpha.fill_(alpha)
    return MIRASModel(d_in=d_in, d_out=d_out, layers=[layer],
                      use_projections=use_proj, d_model=d_model)


def load_reference(path):
    with open(path) as f:
        d = json.load(f)
    return d["per_position_loss"]


def compare(name, losses, ref_path, positions=(0, 10, 20, 50, 100)):
    ref = load_reference(ref_path) if ref_path.exists() else None
    print(f"\n  {name}")
    print(f"  {'pos':>5s}  {'new':>10s}  {'old':>10s}  {'diff':>10s}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}")
    all_match = True
    for p in positions:
        new_v = losses[p].item()
        old_v = ref[p] if ref else float("nan")
        diff = abs(new_v - old_v) if ref else float("nan")
        ok = diff < 0.05 if ref else False
        if not ok:
            all_match = False
        print(f"  {p:>5d}  {new_v:>10.4f}  {old_v:>10.4f}  {diff:>10.4f}  {'OK' if ok else 'DIFF'}")
    return all_match


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    all_pass = True

    # === LINEAR TASK ===
    print("=" * 60)
    print("LINEAR TASK (d_in=10, d_out=1)")
    print("=" * 60)

    task_cfg = TaskConfig()
    task_cfg.type = "linear"
    task_cfg.d_input = 10
    task_cfg.d_output = 1
    task = build_task(task_cfg)

    # 1. Hebbian (dot-product) + Matrix + eta=0.01
    model = build_miras("dot_product", "matrix", 10, 1, 10, 1, eta=0.01).to(device)
    losses = eval_icl_curve(model, task, device=device)
    ref = SCRATCH / "experiments/linear_icl/0311_initial_comparison/results/linear_rnn_eta0.01.json"
    ok = compare("Hebbian Matrix eta=0.01", losses, ref)
    all_pass = all_pass and ok

    # === POLYNOMIAL TASK ===
    print("\n" + "=" * 60)
    print("POLYNOMIAL TASK (d_in=5, d_out=1, degree=2)")
    print("=" * 60)

    task_cfg = TaskConfig()
    task_cfg.type = "polynomial"
    task_cfg.d_input = 5
    task_cfg.d_output = 1
    task_cfg.degree = 2
    task = build_task(task_cfg)
    d_feat = task.d_in  # feature-expanded dimension

    # 2. Hebbian Matrix eta=0.01
    model = build_miras("dot_product", "matrix", d_feat, 1, d_feat, 1, eta=0.01).to(device)
    losses = eval_icl_curve(model, task, device=device)
    ref = SCRATCH / "experiments/polynomial_icl/0312_memory_v2/results/matrix_hebbian_eta0.01_v2_poly2.json"
    ok = compare("Hebbian Matrix eta=0.01", losses, ref)
    all_pass = all_pass and ok

    # 3. Delta Matrix eta=0.01
    model = build_miras("l2", "matrix", d_feat, 1, d_feat, 1, eta=0.01).to(device)
    losses = eval_icl_curve(model, task, device=device)
    ref = SCRATCH / "experiments/polynomial_icl/0312_memory_v2/results/matrix_delta_eta0.01_v2_poly2.json"
    ok = compare("Delta Matrix eta=0.01", losses, ref)
    all_pass = all_pass and ok

    # 4. Delta MLP h=64 eta=0.01 (star performer)
    model = build_miras("l2", "mlp", d_feat, 1, d_feat, 1, eta=0.01, d_hidden=64).to(device)
    losses = eval_icl_curve(model, task, device=device)
    ref = SCRATCH / "experiments/polynomial_icl/0312_memory_v2/results/mlp_delta_h64_eta0.01_v2_poly2.json"
    ok = compare("Delta MLP h=64 eta=0.01", losses, ref)
    all_pass = all_pass and ok

    print("\n" + "=" * 60)
    print(f"Overall: {'ALL REPRODUCED' if all_pass else 'SOME DIFFER'}")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
