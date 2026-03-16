# MIRAS Ablation Experiments Design

**Date:** 2026-03-15
**Status:** Draft
**Depends on:** MIRAS framework (`src/models/miras/`)

## Goal

Systematically ablate each MIRAS axis (attentional bias, memory structure, retention gate) relative to the simplest baseline (DotProduct + Matrix + NoRetention + GD = Linear Attention), on linear regression and polynomial tasks of increasing difficulty. Compare the best-tuned version of each configuration.

## Infrastructure Changes

Two small additions needed before running experiments:

1. **Early stopping in Trainer** -- detect NaN or exploding loss (>100x value at step 1000), stop the run early, save checkpoint.

2. **`eta_init` and `alpha_init` in ModelConfig** -- allow setting the initial values of the inner-loop learning rate and retention strength from CLI. Currently these are hardcoded to 1.0 in MIRASLayer.

No sweep runner or manifest system. Experiments are managed manually with progress tracked in `docs/logs.md`.

## Experiment Grid

### Baseline

**DotProduct + Matrix + NoRetention + GD** (= Linear Attention / Hebbian outer-product rule)

### 8 MIRAS configurations

| Config | Bias | Memory | Retention | Equivalent architecture |
|---|---|---|---|---|
| baseline | DotProduct | Matrix | None | Linear Attention |
| +L2 | L2 | Matrix | None | DeltaNet (no gate) |
| +MLP | DotProduct | MLP | None | Hebbian MLP memory |
| +ret | DotProduct | Matrix | ScalarL2 | Mamba-like |
| +L2+ret | L2 | Matrix | ScalarL2 | Gated DeltaNet |
| +L2+MLP | L2 | MLP | None | TTT-MLP-like |
| +MLP+ret | DotProduct | MLP | ScalarL2 | Hebbian MLP + forgetting |
| +all | L2 | MLP | ScalarL2 | Titans-like (minus momentum) |

### Tasks

- Linear regression: d_in=10, d_out=1
- Polynomial degree 2: d_in=5, d_out=1
- Polynomial degree 3: d_in=5, d_out=1

### Hyperparameter sweep

| Param | Values | Applies to |
|---|---|---|
| `eta_init` | 0.001, 0.01, 0.1, 1.0 | All configs |
| `use_projections` | True, False | Trained configs |
| `lr` | 1e-4, 3e-4, 1e-3 | Trained configs |
| `num_steps` | 20k, 50k | Phase 2-3 |
| `d_hidden` | 32, 64 | MLP memory configs only |

Sweep boundaries will be expanded if optima land at edges.

## Phases

### Phase 1: Untrained evaluation (~1 hour)

Run `eval_icl.py` on all 8 configs x 4 eta values x 3 tasks. MLP configs also x 2 d_hidden = ~120 evals. No training, ~30s each.

**Goal:** Establish which configs work out-of-the-box, what eta range is stable.
**Kill criterion:** NaN output → skip.

### Phase 2: Quick training sweep (~12 hours)

Train non-divergent configs from Phase 1. Sweep: {projections T/F} x {lr 1e-4, 3e-4, 1e-3} x 20k steps x 3 tasks.

**Goal:** Find best lr, whether projections help, per config per task.
**Early stopping:** NaN, loss > 100x value at step 1000, no improvement for 10k steps.
**Prioritization:** Start with single-axis ablations from baseline, then combinations.

### Phase 3: Extended training (~6 hours)

Top configs from Phase 2, extend to 50k steps with best lr.

**Goal:** Check if more training helps.

### Phase 4: Full ICL curve evaluation (~30 min)

Run `eval_icl.py` (100 positions, 256 batch x 50 batches) on best checkpoint per config per task.

**Primary output:** Per-position MSE curves for comparison plots.

### Phase 5: Higher-degree polynomials (future)

Best configs from Phase 4 on polynomial deg 4, 5, ..., 10.

## Evaluation

- **Sweep metric:** MSE at last query position (from trainer eval)
- **Final comparison:** Per-position ICL curves at positions 0, 10, 20, 50, 100
- **Best version selection:** Per config per task, the hyperparams with lowest eval query loss

## Scratch layout

```
scratch/experiments/miras_ablation/
├── README.md
├── phase1_untrained/
│   ├── results/
│   └── logs/
├── phase2_trained_20k/
│   ├── MMDD_<config>_<task>/
│   │   ├── config.json
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   └── results/
├── phase3_extended/
└── phase4_eval/
    └── results/
```

### Naming convention

`{bias}_{memory}_{retention}_eta{eta}_{proj|noproj}_lr{lr}_{task}`

Example: `l2_mlp64_scalar_eta0.01_proj_lr3e-4_poly2`

## Progress tracking

All results logged to `docs/logs.md` under dated sections, updated after each phase with summary tables and takeaways.
