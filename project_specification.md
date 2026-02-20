# ICL Generalization: Comparing In-Context Learning Across Sequence Models

## 1. Project Description

This project investigates and compares the **in-context learning (ICL) generalization** of different sequence-to-sequence modeling approaches. The central question is: given a sequence of (input, output) demonstration pairs drawn from some function, how well can different model architectures predict the output for a new query input -- without any weight updates?

We study this across three families of models:

- **Recurrent models** (e.g., LSTM, GRU) -- process the sequence token-by-token with hidden state.
- **Transformer-based models** (self-attention) -- attend over the full sequence context.
- **Deterministic / algorithmic baselines** (e.g., ridge regression, least-squares computed on-the-fly from the context) -- serve as reference points for what an optimal learner would do.

The models are evaluated in two regimes:
1. **Random initialization** -- testing the inductive bias of the architecture itself.
2. **After training** -- testing what the model learns to do when trained on a distribution of ICL tasks.

---

## 2. Task Definition: In-Context Function Learning

A single ICL instance is a sequence of the form:

```
x_1, y_1, x_2, y_2, ..., x_k, y_k, x_query -> ?
```

where `y_i = f(x_i)` for some function `f` drawn from a **function class** (e.g., linear maps, polynomials, sparse linear, etc.), and the goal is to predict `y_query = f(x_query)`.

### Key properties

| Property | Description |
|---|---|
| **Input space** | `x_i` in `R^d_in` -- inputs can be multi-dimensional real vectors |
| **Output space** | `y_i` in `R^d_out` -- outputs can be multi-dimensional real vectors |
| **Function classes** | Linear, polynomial, sparse linear, ReLU-MLP, string/discrete mappings, etc. |
| **Sequence layout** | Interleaved `[x_1, y_1, x_2, y_2, ..., x_k, y_k, x_query]` as a stream of tokens |
| **Prediction targets** | Loss/evaluation is computed **only at output positions** (i.e., `y_i` slots) |
| **Number of examples** | `k` (the number of in-context demonstrations) is configurable and may vary |

### Token representation

Since inputs and outputs are real-valued vectors rather than discrete tokens, the model operates on **continuous token embeddings** directly -- each `x_i` or `y_i` is a `d`-dimensional real vector that enters the model as-is (or through a lightweight linear projection). This avoids quantization/tokenization artifacts for numerical tasks.

For future discrete/string tasks, a standard embedding layer can be swapped in.

---

## 3. Requirements

### 3.1 Functional Requirements

- **R1 -- Flexible model registry**: Define new model architectures by implementing a common interface. Adding a new model should not require touching training or evaluation code.
- **R2 -- Flexible task registry**: Define new function classes / ICL tasks independently. A task specifies how to sample `f`, how to sample `x_i`, and how to construct the sequence.
- **R3 -- Continuous-valued I/O**: Models must natively accept and produce real-valued vectors (not just discrete token IDs).
- **R4 -- Selective loss masking**: Training loss and evaluation metrics are computed only on designated output positions within the sequence.
- **R5 -- Training pipeline**: A minimal but complete training loop: batched data generation, forward pass, loss computation (on masked positions), backpropagation, gradient clipping, learning-rate scheduling, periodic evaluation, and checkpointing.
- **R6 -- Evaluation of untrained models**: Support running the evaluation pipeline on randomly-initialized (untrained) models to measure architectural inductive bias.
- **R7 -- Full configurability**: Every axis -- model type and hyperparameters, task/function class, data generation, training procedure -- should be configurable from a single config file (no magic constants buried in code).
- **R8 -- Reproducibility**: Seeded randomness for data generation, weight initialization, and training.

### 3.2 Non-Functional Requirements

- **Clean, readable code** -- favor clarity over cleverness. No scattered patches.
- **Modular design** -- each component (model, task, training, evaluation) lives in its own module with a clear interface.
- **Minimal dependencies** -- PyTorch + [Pydra](https://github.com/jordan-benjamin/pydra) for configuration (pure-Python configs with CLI overrides). No heavy frameworks unless justified.

---

## 4. Codebase Design

### 4.1 Directory Structure

```
icl_generalization/
|-- src/
|   |-- __init__.py
|   |-- models/               # Model definitions
|   |   |-- __init__.py       #   model registry
|   |   |-- base.py           #   abstract base class (SeqModel)
|   |   |-- transformer.py    #   causal Transformer
|   |   |-- rnn.py            #   LSTM / GRU variants (future)
|   |   +-- baselines.py      #   deterministic algorithmic baselines (future)
|   |-- tasks/                # Task / function-class definitions
|   |   |-- __init__.py       #   task registry
|   |   |-- base.py           #   abstract base class (ICLTask) + ICLBatch
|   |   +-- linear.py         #   linear function class
|   |-- training/             # Training and evaluation
|   |   |-- __init__.py
|   |   +-- trainer.py        #   training loop with periodic evaluation
|   +-- utils/                # Shared utilities
|       |-- __init__.py
|       +-- seed.py           #   reproducibility helpers
|-- scripts/
|   +-- train.py              # Entry point with Pydra config
|-- requirements.txt
+-- README.md
```

Note: No YAML config files -- configuration is defined in pure Python via
Pydra Config classes (in `scripts/train.py`), with CLI overrides like
`python scripts/train.py model.d_model=256 task.d_input=20`.

### 4.2 Key Abstractions

#### SeqModel (base class for all models)

```python
class SeqModel(nn.Module):
    """
    Args:
        xs: (batch, n, d_in)  -- input vectors (last is query)
        ys: (batch, n, d_out) -- output vectors (last is query target, not fed to model)
    Returns:
        y_preds: (batch, n, d_out) -- predictions at each input position
    """
    def forward(self, xs: Tensor, ys: Tensor) -> Tensor: ...
```

Models receive raw `xs` and `ys` and handle interleaving / sequence construction
internally. This gives each architecture full flexibility (e.g., the Transformer
interleaves into `[x1, y1, x2, y2, ..., x_query]` with causal masking, while a
deterministic baseline can just solve a least-squares problem directly). Input/output
projections (`d_in -> d_model`, `d_model -> d_out`) live inside each model.

#### ICLTask (base class for all tasks)

```python
@dataclass
class ICLBatch:
    xs: Tensor  # (batch, n, d_in)  -- n = num_examples + 1 (includes query)
    ys: Tensor  # (batch, n, d_out) -- includes query target for loss computation

class ICLTask(ABC):
    def sample_batch(self, batch_size: int, num_examples: int) -> ICLBatch: ...
    @property
    def d_in(self) -> int: ...
    @property
    def d_out(self) -> int: ...
```

Each task implementation handles: (a) sampling a random function `f`, (b) sampling
input points, (c) computing `y = f(x)`, and returning a flat `ICLBatch`. The task
does not assemble interleaved sequences -- that is the model's responsibility.

#### Trainer

Orchestrates the training loop. Receives a model, a task (used as a data generator), and a config. Handles:
- Batched forward/backward passes
- Loss computation with masking (MSE on output positions)
- Optimizer and LR scheduler steps
- Periodic evaluation and metric logging
- Checkpointing

### 4.3 Configuration (Pydra)

Configuration is defined as nested Python classes using
[Pydra](https://github.com/jordan-benjamin/pydra). Defaults are set in
`__init__`, and any field can be overridden from the CLI.

```python
class Config(pydra.Config):        # top-level
    seed = 42
    model = ModelConfig()           # type, d_model, n_layers, n_heads, dropout
    task  = TaskConfig()            # type, d_input, d_output, noise_std
    training = TrainingConfig()     # batch_size, num_steps, num_examples, lr, ...
```

CLI usage: `python scripts/train.py model.d_model=256 task.d_input=20 training.lr=3e-4`

### 4.4 Data Flow

```
  ICLTask             sample_batch()         ICLBatch
  (e.g., linear)  ---------------------->  .xs  (B, n, d_in)
                                           .ys  (B, n, d_out)
                                               |
                                               v
                                           SeqModel.forward(xs, ys)
                                           (internally: project, interleave,
                                            apply causal attention, extract
                                            predictions at x-positions)
                                               |
                                               v  y_preds (B, n, d_out)
                                           MSE loss vs ys
                                           (all positions or query-only)
```

---

## 5. Design Decisions (resolved)

1. **Input/output projection** -- Lives **inside each model**. Simpler, and different architectures may want different projection strategies.

2. **Positional encoding** -- **Learned embeddings** for the Transformer. RNNs get position implicitly. Can revisit (sinusoidal, RoPE) later.

3. **Sequence construction** -- Each `x_i` / `y_i` is a **single vector-valued token**. No flattening to scalar positions. Models handle interleaving internally.

4. **Deterministic baselines** -- Will conform to `SeqModel` (nn.Module subclass with `forward(xs, ys)`) for evaluation uniformity. Not implemented in the minimal build.

5. **Config system** -- **Pydra** (`pydra-config`). Pure-Python configs with CLI overrides, nested configs, no YAML files.

6. **Logging** -- **Console only** for the minimal build. W&B can be added later behind a flag.
