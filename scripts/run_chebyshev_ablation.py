#!/usr/bin/env python3
"""Run the Chebyshev polynomial ICL ablation study.

Reproduces the Wilcoxson et al. (2024) arXiv:2407.19346 setting with Transformer,
LSTM, and MIRAS variants.

Training config (adapted from prior work):
  - 500K steps, lr=5e-5, batch=64, cosine schedule
  - 80 in-context examples, curriculum from 10 → 80 over first 250K steps
  - ChebyshevTask: max_degree=11, x ~ U(-1, 1), y ~ sum c_k T_k(x)

Usage (all 4 GPUs):
    CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n continual_learning \\
        python scripts/run_chebyshev_ablation.py
"""

import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCRATCH = "/scratch/m000122/stalaei/icl_generalization"
EXP_DIR = f"{SCRATCH}/experiments/chebyshev_icl"
DATE = "0317"

# Shared training hyperparameters (Wilcoxson et al. setting)
COMMON = [
    "task.type=chebyshev",
    "task.max_degree=11",
    "training.num_steps=500000",
    "training.lr=5e-5",
    "training.lr_schedule=cosine",
    "training.batch_size=64",
    "training.num_examples=80",
    "training.curriculum=true",
    "training.curriculum_start=10",
    "training.curriculum_end_step=250000",
    "training.grad_clip=1.0",
    "training.eval_every=10000",
]

ROUNDS = [
    # Round 1: baselines + core MIRAS
    [
        {
            "name": f"{DATE}_tf_small",
            "args": [
                "model.type=transformer",
                "model.d_model=128",
                "model.n_layers=6",
                "model.n_heads=4",
            ],
            "gpu": 0,
        },
        {
            "name": f"{DATE}_lstm",
            "args": [
                "model.type=lstm",
                "model.d_model=128",
                "model.n_layers=2",
            ],
            "gpu": 1,
        },
        {
            "name": f"{DATE}_delta_matrix",
            "args": [
                "model.type=miras",
                "model.bias.type=l2",
                "model.memory.type=matrix",
                "model.retention.type=none",
                "model.algorithm.type=gd",
                "model.n_layers=1",
                "model.use_projections=false",
                "model.d_model=128",
                "model.eta_init=0.01",
            ],
            "gpu": 2,
        },
        {
            "name": f"{DATE}_delta_mlp64",
            "args": [
                "model.type=miras",
                "model.bias.type=l2",
                "model.memory.type=mlp",
                "model.memory.d_hidden=64",
                "model.retention.type=none",
                "model.algorithm.type=gd",
                "model.n_layers=1",
                "model.use_projections=false",
                "model.d_model=128",
                "model.eta_init=0.01",
            ],
            "gpu": 3,
        },
    ],
    # Round 2: Hebbian variants + deeper memory
    [
        {
            "name": f"{DATE}_hebbian_matrix",
            "args": [
                "model.type=miras",
                "model.bias.type=dot_product",
                "model.memory.type=matrix",
                "model.retention.type=none",
                "model.algorithm.type=gd",
                "model.n_layers=1",
                "model.use_projections=false",
                "model.d_model=128",
                "model.eta_init=0.01",
            ],
            "gpu": 0,
        },
        {
            "name": f"{DATE}_hebbian_mlp64",
            "args": [
                "model.type=miras",
                "model.bias.type=dot_product",
                "model.memory.type=mlp",
                "model.memory.d_hidden=64",
                "model.retention.type=none",
                "model.algorithm.type=gd",
                "model.n_layers=1",
                "model.use_projections=false",
                "model.d_model=128",
                "model.eta_init=0.01",
            ],
            "gpu": 1,
        },
        {
            "name": f"{DATE}_delta_mlp128",
            "args": [
                "model.type=miras",
                "model.bias.type=l2",
                "model.memory.type=mlp",
                "model.memory.d_hidden=128",
                "model.retention.type=none",
                "model.algorithm.type=gd",
                "model.n_layers=1",
                "model.use_projections=false",
                "model.d_model=128",
                "model.eta_init=0.01",
            ],
            "gpu": 2,
        },
        {
            "name": f"{DATE}_delta_matrix_2l",
            "args": [
                "model.type=miras",
                "model.bias.type=l2",
                "model.memory.type=matrix",
                "model.retention.type=none",
                "model.algorithm.type=gd",
                "model.n_layers=2",
                "model.d_model=128",
                "model.eta_init=0.01",
            ],
            "gpu": 3,
        },
    ],
]


def launch_run(exp: dict) -> subprocess.Popen:
    name = exp["name"]
    run_dir = f"{EXP_DIR}/{name}"
    os.makedirs(f"{run_dir}/logs", exist_ok=True)
    os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)

    checkpoint_dir = f"{run_dir}/checkpoints"
    log_path = f"{run_dir}/logs/train.log"

    cmd = (
        ["conda", "run", "-n", "continual_learning", "python", "scripts/train.py"]
        + COMMON
        + exp["args"]
        + [
            f"training.checkpoint_dir={checkpoint_dir}",
            "training.checkpoint_every=100000",
        ]
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(exp["gpu"])
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

    print(f"Launched {name} on GPU {exp['gpu']} (PID {proc.pid})")
    print(f"  Log: {log_path}")
    return proc


def main():
    os.makedirs(EXP_DIR, exist_ok=True)
    print("=" * 60)
    print("Chebyshev ICL Ablation Study")
    print(f"Experiment dir: {EXP_DIR}")
    print("=" * 60)

    for round_idx, round_exps in enumerate(ROUNDS, 1):
        print(f"\n--- Round {round_idx} ({len(round_exps)} runs) ---")
        procs = [(exp["name"], launch_run(exp)) for exp in round_exps]

        for name, proc in procs:
            ret = proc.wait()
            status = "OK" if ret == 0 else f"FAILED (exit {ret})"
            print(f"  {name}: {status}")

    print(f"\nAll runs complete. Results in {EXP_DIR}/")
    print("Next: run eval_icl.py with with_ridge_baseline=true for each checkpoint.")


if __name__ == "__main__":
    main()
