# MIRAS Design Space for ICL Generalization

**Date:** 2026-03-15
**Status:** Draft
**References:** Behrouz et al., "It's All Connected" (ICLR 2026), Behrouz "Nested Learning" (2024)

## Summary

Implement the MIRAS framework as a modular design space for studying in-context learning (ICL) across modern RNN architectures. The framework decomposes every modern RNN into 4 independent axes (attentional bias, memory structure, retention gate, memory algorithm) following the MIRAS paper. This enables systematic ablation studies where each axis is varied independently.

The implementation uses Approach C (two-phase split): the write step factors into an output-space error signal (bias-dependent) and a parameter update (memory-dependent + algorithm-dependent). This mirrors the mathematical structure where `nabla_W l = (dl/d_output) * (d_output/dW)`.

## Motivation

The MIRAS paper (Table 1) shows that virtually all modern sequence models (Linear Attention, Mamba, DeltaNet, GLA, RWKV-7, Titans, etc.) are specific instantiations of 4 design choices. Our codebase currently hardcodes the memory update logic inside `MatrixMemory` and `MLPMemory`, mixing the bias and memory concerns. Factoring these into independent components enables:

1. Systematic ablation across each axis
2. Easy addition of new variants (Lp norms, Huber loss, KL retention, Muon optimizer)
3. Reproduction of any architecture from Table 1 via configuration

## Scope

### Initial implementation

| Axis | Variants |
|---|---|
| Memory Structure | Matrix, MLP (2-layer) |
| Attentional Bias | Dot-product similarity, L2 loss |
| Retention Gate | None, Scalar L2 decay |
| Memory Algorithm | Gradient Descent |

### Future extensions (design must support cleanly)

| Axis | Future variants |
|---|---|
| Memory Structure | Vector (diagonal), GLU, k-layer MLP |
| Attentional Bias | Lp norms, Huber loss, KL-based, sliding-window L2 |
| Retention Gate | Diagonal L2, input-dependent (Gated DeltaNet), Lq, KL divergence |
| Memory Algorithm | GD + Momentum, Muon, implicit GD, multi-step GD |

### Out of scope

- Transformers and classical RNNs (LSTM/GRU) remain as separate baselines sharing the `SeqModel` interface. They are not expressed as MIRAS instantiations.
- Changes to tasks, training loop, or evaluation infrastructure.

## Architecture

### Design approach: Two-phase split (Approach C)

The write step factors into two phases that cleanly separate concerns:

1. **Error signal phase** (bias-dependent, memory-independent): The attentional bias computes an output-space error `d_out = bias.error_signal(prediction, target)`.
2. **Parameter update phase** (memory-dependent, algorithm-dependent): The memory computes gradients w.r.t. its own parameters given `d_out`, then the algorithm applies the update, with the retention gate regularizing the state.

This factorization works because the gradient of any attentional bias w.r.t. memory parameters decomposes as:

```
nabla_W l(W; k, v) = (dl/d_output) * (d_output/dW)
                      ^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^
                      bias.error_signal  memory.backward
```

### The four abstractions

Each axis is an `nn.Module` base class.

#### AttentionalBias

Computes the output-space error signal given a prediction and target.

```python
class AttentionalBias(nn.Module):
    """Base class for attentional bias (internal learning objective)."""
    def error_signal(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Compute output-space error signal d_out. Shape: (B, d_v)."""
        raise NotImplementedError
```

**Sign convention:** `error_signal` returns the direction to *move the memory output toward*. The algorithm *adds* `eta * grads` to the state. For objectives being minimized, this means `error_signal` returns the *negative* gradient of the loss w.r.t. the output. For objectives being maximized (dot-product similarity), it returns the positive gradient of the objective.

Initial implementations:
- `DotProductBias`: `error_signal(pred, target) = target`. Maximizes `<pred, target>`; the gradient of `<pred, target>` w.r.t. `pred` is `target`.
- `L2Bias`: `error_signal(pred, target) = target - pred`. Minimizes `||pred - target||^2`; the negative gradient w.r.t. `pred` is `target - pred` (factor of 2 absorbed into eta).

Future extensions add new subclasses (e.g., `LpBias`, `HuberBias`, `KLBias`) -- each only needs to implement `error_signal`, following the same sign convention.

#### MemoryStructure

Owns parameter state, handles read (forward) and backward (gradient computation through its own structure).

**MemoryState convention:** `MemoryState` and `Grads` are either a single `Tensor` (MatrixMemory) or a `tuple[Tensor, ...]` (MLPMemory). All components that operate on `MemoryState` (RetentionGate, MemoryAlgorithm) must handle both cases. To avoid duplicating this logic, provide a utility function `state_map(fn, state)` that applies `fn` element-wise to tensors or tuple elements:

```python
def state_map(fn, *states):
    """Apply fn element-wise across MemoryState(s). Handles Tensor and tuple[Tensor, ...]."""
    if isinstance(states[0], tuple):
        return tuple(fn(*elems) for elems in zip(*states))
    return fn(*states)

# Usage in ScalarL2Retention:
def apply(self, state, alpha, key=None):
    return state_map(lambda s: alpha * s, state)

# Usage in GD:
def step(self, state, grads, eta, optim_state):
    new_state = state_map(lambda s, g: s + eta * g, state, grads)
    return new_state, optim_state
```

```python
MemoryState = Tensor | tuple[Tensor, ...]
Grads = Tensor | tuple[Tensor, ...]

class MemoryStructure(nn.Module):
    """Base class for memory architecture."""
    def init_state(self, B: int, device, dtype) -> MemoryState:
        raise NotImplementedError

    def read(self, state: MemoryState, query: Tensor) -> Tensor:
        """Read from memory. Returns (B, d_v)."""
        raise NotImplementedError

    def backward(self, state: MemoryState, key: Tensor, d_out: Tensor) -> Grads:
        """Compute gradients of memory parameters given output-space error signal.

        d_out: (B, d_v) -- the error signal from the attentional bias.
        Returns gradients in the same structure as state.
        """
        raise NotImplementedError
```

Initial implementations:

- `MatrixMemory`: State is `M in R^{d_v x d_k}`.
  - `read(M, q) = M @ q`
  - `backward(M, k, d_out) = d_out outer k` (i.e., `d_out.unsqueeze(-1) @ k.unsqueeze(-2)`)

- `MLPMemory`: State is `(W1, W2)` for a 2-layer MLP `M(x) = W1 * silu(W2 * x) [+ x]`.
  - `read((W1, W2), q)`: forward pass through MLP
  - `backward((W1, W2), k, d_out)`: manual backprop computing `(dW1, dW2)` given `d_out`. This is the same chain-rule logic currently in `MLPMemory.write`, factored out. **Important:** `backward` must internally recompute the forward pass (to obtain intermediate activations `pre`, `h`) rather than caching them from `read`. This keeps `read` and `backward` fully decoupled. The double forward pass is acceptable for research code.

#### RetentionGate

Applies regularization to memory state, controlling the forgetting/retention trade-off.

```python
class RetentionGate(nn.Module):
    """Base class for retention gate (memory regularizer)."""
    def apply(self, state: MemoryState, alpha: Tensor,
              key: Tensor | None = None) -> MemoryState:
        """Apply retention to state.

        alpha: retention strength parameter
        key: optional, needed for input-dependent retention (e.g., Gated DeltaNet)
        """
        raise NotImplementedError
```

Initial implementations:
- `NoRetention`: returns `state` unchanged
- `ScalarL2Retention`: returns `alpha * state` (element-wise for tuples)

The optional `key` parameter enables future input-dependent retention gates (Gated DeltaNet: `M' = M * (alpha*I - beta*k*k^T) + ...`) without changing the interface.

#### MemoryAlgorithm

The inner-loop optimizer. Applies gradients to memory state, optionally maintaining optimizer state (e.g., momentum buffer).

```python
OptimState = Any  # None for GD, Tensor for momentum, etc.

class MemoryAlgorithm(nn.Module):
    """Base class for memory update algorithm (inner-loop optimizer)."""
    def init_optim_state(self, memory: MemoryStructure,
                         B: int, device, dtype) -> OptimState:
        """Initialize optimizer state (e.g., momentum buffer). None for stateless algorithms."""
        return None

    def step(self, state: MemoryState, grads: Grads, eta: Tensor,
             optim_state: OptimState) -> tuple[MemoryState, OptimState]:
        """Apply one optimization step.

        Returns updated (memory_state, optim_state).
        """
        raise NotImplementedError
```

Initial implementation:
- `GD`: `step(state, grads, eta, _) = (state + eta * grads, None)`. No optimizer state.

Future extensions:
- `GDMomentum`: carries momentum buffer in `optim_state`, implements `s_t = theta * s_{t-1} + eta * grads; state = state + s_t`
- `Muon`: applies Newton-Schulz normalization to momentum
- `MultiStepGD`: runs multiple GD steps per token (DeltaProduct)
- `ImplicitGD`: closed-form solution with `(I - beta*k*k^T / (1 + beta*k^T*k))` (Longhorn)

### MIRASLayer: Composition point

Composes the 4 components and implements the read-write protocol.

```python
class MIRASLayer(nn.Module):
    def __init__(self, memory, bias, retention, algorithm):
        self.memory = memory
        self.bias = bias
        self.retention = retention
        self.algorithm = algorithm
        self.eta = nn.Parameter(torch.tensor(1.0))    # inner learning rate
        self.alpha = nn.Parameter(torch.tensor(1.0))   # retention strength

    def init_state(self, B, device, dtype):
        mem_state = self.memory.init_state(B, device, dtype)
        optim_state = self.algorithm.init_optim_state(self.memory, B, device, dtype)
        return mem_state, optim_state

    def read(self, state, query):
        mem_state, _ = state
        return self.memory.read(mem_state, query)

    def write(self, state, key, value):
        mem_state, optim_state = state

        # Phase 1: error signal (bias-dependent, memory-independent)
        pred = self.memory.read(mem_state, key)
        d_out = self.bias.error_signal(pred, value)

        # Phase 2a: gradients (memory-dependent)
        grads = self.memory.backward(mem_state, key, d_out)

        # Phase 2b: retention (regularize before gradient step)
        mem_state = self.retention.apply(mem_state, self.alpha, key=key)

        # Phase 2c: optimizer step
        mem_state, optim_state = self.algorithm.step(
            mem_state, grads, self.eta, optim_state
        )

        return mem_state, optim_state
```

**Retention ordering:** Retention applies before the gradient step, matching the standard form `M_t = alpha * M_{t-1} + eta * grad`. This means the update is: decay old state, then add new information. Note that the error signal is computed on the *un-decayed* state (phase 1 reads from `mem_state` before retention), which is consistent with the existing implementation in `rnn.py` lines 56-65.

### MIRASModel: Full sequence model

`SeqModel` subclass wrapping one or more `MIRASLayer`s with optional projections.

```python
class MIRASModel(SeqModel):
    def __init__(self, d_in, d_out, layers, use_projections=False,
                 d_model=128, gd_init=False):
        self.layers = nn.ModuleList(layers)  # list of MIRASLayer
        self.use_proj = use_projections

        if use_projections:
            self.proj_k = nn.Linear(d_in, d_model, bias=False)
            self.proj_q = nn.Linear(d_in, d_model, bias=False)
            self.proj_v = nn.Linear(d_out, d_model, bias=False)
            self.proj_out = nn.Linear(d_model, d_out, bias=False)
            # Optional inter-layer projections for multi-layer

    def forward(self, xs, ys):
        B, n, _ = xs.shape
        # Initialize state for each layer
        states = [layer.init_state(B, xs.device, xs.dtype) for layer in self.layers]

        # Project inputs if needed
        if self.use_proj:
            keys = self.proj_k(xs)
            queries = self.proj_q(xs)
            values = self.proj_v(ys[:, :-1, :])

        y_preds = []
        for i in range(n):
            # Read through all layers sequentially
            q = queries[:, i] if self.use_proj else xs[:, i]
            h = q
            for layer_idx, layer in enumerate(self.layers):
                h = layer.read(states[layer_idx], h)
            y_preds.append(self.proj_out(h) if self.use_proj else h)

            # Write through all layers sequentially
            if i < n - 1:
                k = keys[:, i] if self.use_proj else xs[:, i]
                v = values[:, i] if self.use_proj else ys[:, i]
                h_write = k
                for layer_idx, layer in enumerate(self.layers):
                    states[layer_idx] = layer.write(states[layer_idx], h_write, v)
                    # For multi-layer: h_write and v would be updated
                    # from previous layer's output (future extension)

        return torch.stack(y_preds, dim=1)
```

**Multi-layer plan (not implemented initially):**

For `n_layers > 1`, the forward pass at each timestep flows sequentially through layers. The design considerations for future implementation:

- Each layer operates in `d_model` space (projections map data space <-> model space at the boundaries)
- Inter-layer interface: layer L's read output becomes layer L+1's query/key
- The value for layers beyond the first is an open design question: options include (a) same projected target value for all layers, (b) previous layer's read output as the value, (c) residual of previous layer's prediction error
- Residual connections between layers (pre-norm pattern): `h = h + layer.read(state, h)`
- Each layer maintains independent memory state and optimizer state
- Initial implementation uses `n_layers=1`, but `MIRASModel` stores a `ModuleList` from day one
- **Guard:** `MIRASModel.__init__` asserts `len(layers) == 1` with message "Multi-layer MIRAS not yet implemented" until the inter-layer interface is designed

### Configuration

Nested Pydra configs with preset methods:

```python
class MemoryStructureConfig(pydra.Config):
    def __init__(self):
        self.type = "matrix"        # matrix | mlp
        self.d_hidden = 64          # MLP hidden dim

class AttentionalBiasConfig(pydra.Config):
    def __init__(self):
        self.type = "dot_product"   # dot_product | l2

class RetentionGateConfig(pydra.Config):
    def __init__(self):
        self.type = "none"          # none | scalar_l2

class MemoryAlgorithmConfig(pydra.Config):
    def __init__(self):
        self.type = "gd"            # gd

class ModelConfig(pydra.Config):
    def __init__(self):
        self.type = "transformer"
        self.d_model = 128
        self.n_layers = 6
        self.n_heads = 4
        self.pos_encoding = "sinusoidal"
        self.dropout = 0.0
        self.use_projections = False
        self.gd_init = False

        # MIRAS-specific (nested)
        self.memory = MemoryStructureConfig()
        self.bias = AttentionalBiasConfig()
        self.retention = RetentionGateConfig()
        self.algorithm = MemoryAlgorithmConfig()

    # Presets -- called via CLI as e.g. model.linear_attention
    def linear_attention(self):
        self.type = "miras"
        self.bias.type = "dot_product"
        self.algorithm.type = "gd"
        self.memory.type = "matrix"
        self.retention.type = "none"

    def mamba(self):
        self.type = "miras"
        self.bias.type = "dot_product"
        self.algorithm.type = "gd"
        self.memory.type = "matrix"
        self.retention.type = "scalar_l2"

    def deltanet(self):
        self.type = "miras"
        self.bias.type = "l2"
        self.algorithm.type = "gd"
        self.memory.type = "matrix"
        self.retention.type = "none"

    def gated_deltanet(self):
        self.type = "miras"
        self.bias.type = "l2"
        self.algorithm.type = "gd"
        self.memory.type = "matrix"
        self.retention.type = "scalar_l2"

    def titans(self):
        self.type = "miras"
        self.bias.type = "l2"
        self.algorithm.type = "gd_momentum"  # future
        self.memory.type = "mlp"
        self.retention.type = "scalar_l2"
```

CLI usage:
```bash
# Preset (note leading dot for method call)
python scripts/train.py .model.linear_attention

# Manual composition
python scripts/train.py model.type=miras model.bias.type=l2 model.memory.type=matrix

# Override a preset
python scripts/train.py .model.deltanet model.memory.type=mlp
```

### File structure

```
src/models/miras/
    __init__.py          # exports MIRASModel, build_miras_model
    layer.py             # MIRASLayer
    model.py             # MIRASModel (SeqModel subclass)
    memory.py            # MemoryStructure base + MatrixMemory, MLPMemory
    bias.py              # AttentionalBias base + DotProductBias, L2Bias
    retention.py         # RetentionGate base + NoRetention, ScalarL2Retention
    algorithm.py         # MemoryAlgorithm base + GD
```

The `build_miras_model` factory function (in `src/models/miras/__init__.py`):

```python
# Registries mapping config strings to classes
BIAS_REGISTRY = {"dot_product": DotProductBias, "l2": L2Bias}
MEMORY_REGISTRY = {"matrix": MatrixMemory, "mlp": MLPMemory}
RETENTION_REGISTRY = {"none": NoRetention, "scalar_l2": ScalarL2Retention}
ALGORITHM_REGISTRY = {"gd": GD}

def build_miras_model(config, d_in, d_out):
    bias = BIAS_REGISTRY[config.bias.type]()
    retention = RETENTION_REGISTRY[config.retention.type]()
    algorithm = ALGORITHM_REGISTRY[config.algorithm.type]()

    # Memory dimensions depend on projection usage
    if config.use_projections:
        d_k = d_v = config.d_model
    else:
        d_k, d_v = d_in, d_out

    memory = _build_memory(config.memory, d_k, d_v)
    layer = MIRASLayer(memory, bias, retention, algorithm)

    return MIRASModel(
        d_in=d_in, d_out=d_out, layers=[layer],
        use_projections=config.use_projections,
        d_model=config.d_model, gd_init=config.gd_init,
    )
```

Integration points:
- `src/models/__init__.py`: add `"miras"` to registry, dispatch to `build_miras_model`
- `src/config.py`: add nested MIRAS configs and preset methods to `ModelConfig`
- No changes to tasks, training, or evaluation code

### Verification plan

After implementation, verify correctness by reproducing existing experiment results:

1. Configure MIRAS with `dot_product + GD + matrix + none` -> should match old `linear_rnn` (Hebbian) results
2. Configure MIRAS with `l2 + GD + matrix + none` -> should match old `linear_rnn` (Delta, no retention) results
3. Configure MIRAS with `dot_product + GD + matrix + scalar_l2` -> should match old Hebbian with alpha retention
4. Configure MIRAS with `l2 + GD + mlp + none` -> should match old `linear_rnn` with MLPMemory Delta results
5. With projections enabled, compare against old `linear_rnn_proj` results

Verification method: a script (`scripts/verify_miras.py`) that for each configuration above:
- Instantiates both the old `AssociativeRNN` and new `MIRASModel` with equivalent configs
- Copies parameters (eta, alpha, projections) between them
- Runs both on the same fixed-seed random batch
- Asserts `torch.allclose(old_output, new_output, atol=1e-6)` on the output tensors
- Prints PASS/FAIL per configuration

After all 5 checks pass, remove deprecated `AssociativeRNN`, old `MatrixMemory`, and old `MLPMemory` from `rnn.py`.

## Decisions log

| Decision | Choice | Rationale |
|---|---|---|
| Design approach | Two-phase split (Approach C) | Mirrors MIRAS math; cleanest separation of concerns |
| Transformer/LSTM integration | Separate baselines | Research value is in MIRAS design space; avoid over-engineering |
| Initial scope | 2 biases, 2 memories, 2 retentions, 1 algorithm | Sufficient to reproduce Linear Attention, Mamba, DeltaNet, Gated DeltaNet |
| Multi-layer | Design for it, implement n_layers=1 | Needed soon but not for first experiments |
| Config style | Nested Pydra with preset methods | Clean, CLI-friendly, matches user's workflow |
| Old code | Remove after verification | No need to maintain deprecated code |
| Retention ordering | Before gradient step | Matches standard form M_t = alpha * M_{t-1} + eta * grad |
