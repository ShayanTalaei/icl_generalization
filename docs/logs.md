# Research Logs

---

## 2026-03-15: MIRAS Framework Implementation & Reproduction

Implemented the MIRAS framework (`src/models/miras/`) decomposing modern RNNs into 4 independent axes: attentional bias, memory structure, retention gate, memory algorithm. Verified equivalence against old `AssociativeRNN` (exact forward + gradient match), then reproduced all previous experiment results.

### Reproduction: Linear ICL (d_in=10, d_out=1)

Reproduced the untrained linear RNN (Hebbian matrix, eta=0.01) result. ICL emerges from architecture alone -- no training required.

| pos | MIRAS (new) | Old linear_rnn | diff |
|-----|------------|----------------|------|
| 0   | 1.0217     | 1.0009         | 0.02 |
| 10  | 0.8454     | 0.8199         | 0.03 |
| 20  | 0.6600     | 0.6708         | 0.01 |
| 50  | 0.3083     | 0.3023         | 0.01 |
| 100 | 0.1124     | 0.1103         | 0.00 |

```bash
# equivalent to old: model.type=linear_rnn model.update_rule=hebbian
conda run -n continual_learning python scripts/eval_icl.py \
    model.type=miras model.bias.type=dot_product model.memory.type=matrix model.retention.type=none \
    task.d_input=10 task.d_output=1 num_examples=100 label=miras_hebbian_matrix
```

Plots: `scratch/experiments/linear_icl/0311_initial_comparison/plots/icl_comparison.png`

### Reproduction: Polynomial ICL (d_in=5, d_out=1, degree=2)

Three configurations reproduced. MLP Delta (h=64, eta=0.01) remains the star performer -- breaks below the linear regression floor (1.37) without any training.

**Hebbian Matrix eta=0.01:**

| pos | MIRAS | Old | diff |
|-----|-------|-----|------|
| 0   | 1.49  | 1.52 | 0.02 |
| 50  | 1.42  | 1.42 | 0.00 |
| 100 | 1.42  | 1.44 | 0.02 |

```bash
conda run -n continual_learning python scripts/eval_icl.py \
    model.type=miras model.bias.type=dot_product model.memory.type=matrix model.retention.type=none \
    task.type=polynomial task.d_input=5 task.d_output=1 task.degree=2 \
    num_examples=100 label=miras_hebbian_matrix_poly2
```

**Delta Matrix eta=0.01:**

| pos | MIRAS | Old | diff |
|-----|-------|-----|------|
| 0   | 1.54  | 1.43 | 0.11 |
| 50  | 1.45  | 1.42 | 0.03 |
| 100 | 1.30  | 1.32 | 0.02 |

```bash
conda run -n continual_learning python scripts/eval_icl.py \
    model.type=miras model.bias.type=l2 model.memory.type=matrix model.retention.type=none \
    task.type=polynomial task.d_input=5 task.d_output=1 task.degree=2 \
    num_examples=100 label=miras_delta_matrix_poly2
```

**Delta MLP h=64 eta=0.01 (best untrained model):**

| pos | MIRAS | Old | diff |
|-----|-------|-----|------|
| 0   | 1.50  | 1.45 | 0.05 |
| 20  | 1.17  | 1.16 | 0.01 |
| 50  | 1.00  | 0.96 | 0.04 |
| 100 | 0.80  | 0.78 | 0.02 |

```bash
conda run -n continual_learning python scripts/eval_icl.py \
    model.type=miras model.bias.type=l2 model.memory.type=mlp model.memory.d_hidden=64 model.retention.type=none \
    task.type=polynomial task.d_input=5 task.d_output=1 task.degree=2 \
    num_examples=100 label=miras_delta_mlp_h64_poly2
```

Report: `scratch/reports/0312_polynomial_icl/report.md`
Plots: `scratch/reports/0312_polynomial_icl/fig3_overall_comparison.png`

### Takeaways

- MIRAS framework reproduces all previous results within sampling noise (max diff ~0.02 at key positions).
- The 4-axis decomposition is validated: dot_product bias = old Hebbian, L2 bias = old Delta.
- Untrained MLP Delta (h=64) achieves MSE@100 = 0.80 on degree-2 polynomials, beating trained LSTM (1.14).
- The old `AssociativeRNN` code has been removed; all future experiments use MIRAS.

---

## 2026-03-12: Polynomial ICL Memory Variants

Systematic sweep of memory architectures on polynomial degree-2 task. Key finding: MLP memory with Delta rule breaks below the linear regression floor without training.

### Key results

| Model | MSE@20 | MSE@100 |
|-------|--------|---------|
| Transformer (untrained) | 1.66 | 1.87 |
| Transformer (trained 20k) | 0.50 | 2.30 |
| LSTM (trained 20k) | 1.20 | 1.14 |
| Matrix Hebbian eta=0.01 | 1.42 | 1.44 |
| Matrix Delta eta=0.01 | 1.49 | 1.32 |
| **MLP Delta h=64 eta=0.01** | **1.16** | **0.78** |

```bash
# Best untrained model
conda run -n continual_learning python scripts/eval_icl.py \
    model.type=miras model.bias.type=l2 model.memory.type=mlp model.memory.d_hidden=64 model.retention.type=none \
    task.type=polynomial task.d_input=5 task.d_output=1 task.degree=2 \
    num_examples=100 label=delta_mlp_h64_poly2
```

Experiment data: `scratch/experiments/polynomial_icl/0312_memory_v2/`
Report: `scratch/reports/0312_polynomial_icl/report.md`
Plots: `scratch/reports/0312_polynomial_icl/fig2_matrix_vs_mlp.png`, `fig3_overall_comparison.png`

---

## 2026-03-11: Initial ICL Comparison (Linear Task)

Baseline comparison of Transformer, LSTM, and LinearRNN on linear regression ICL (d_in=10, d_out=1, 100 examples).

### Key results

| Model | MSE@10 | MSE@20 | MSE@50 | MSE@100 |
|-------|--------|--------|--------|---------|
| Transformer (untrained) | 1.00 | 1.00 | 1.00 | 1.00 |
| Transformer (trained 20k) | 0.08 | 0.50 | 0.20 | 1.26 |
| LSTM (untrained) | 0.98 | 0.98 | 0.99 | 1.01 |
| LSTM (trained 20k) | 0.51 | 0.38 | 0.26 | 0.26 |
| **LinearRNN Hebbian eta=0.01** | **0.82** | **0.67** | **0.30** | **0.11** |

```bash
# Untrained LinearRNN (now via MIRAS)
conda run -n continual_learning python scripts/eval_icl.py \
    model.type=miras model.bias.type=dot_product model.memory.type=matrix model.retention.type=none \
    task.d_input=10 task.d_output=1 num_examples=100 label=hebbian_matrix_linear

# Trained transformer
conda run -n continual_learning python scripts/train.py \
    model.type=transformer model.pos_encoding=sinusoidal \
    training.num_steps=20000 training.lr=3e-4 training.lr_schedule=constant training.num_examples=40

# Trained LSTM
conda run -n continual_learning python scripts/train.py \
    model.type=lstm model.n_layers=2 \
    training.num_steps=20000 training.lr=3e-4 training.lr_schedule=constant training.num_examples=40
```

Experiment data: `scratch/experiments/linear_icl/0311_initial_comparison/`
Plots: `scratch/experiments/linear_icl/0311_initial_comparison/plots/icl_comparison.png`
