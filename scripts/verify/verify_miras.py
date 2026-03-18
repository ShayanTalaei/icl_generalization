"""Verify MIRAS framework produces identical outputs to old AssociativeRNN.

For each configuration, instantiates both old and new models with matching
parameters and asserts torch.allclose on outputs and gradients.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.rnn import AssociativeRNN
from src.models.rnn import MatrixMemory as OldMatrixMemory
from src.models.rnn import MLPMemory as OldMLPMemory
from src.models.miras import (
    MIRASModel, MIRASLayer, MatrixMemory, MLPMemory,
    DotProductBias, L2Bias, NoRetention, ScalarL2Retention, GD,
)


def copy_params(old_model, new_model, new_layer):
    """Copy eta, alpha, and projection weights from old to new."""
    with torch.no_grad():
        new_layer.eta.copy_(old_model.eta)
        new_layer.alpha.copy_(old_model.alpha)
        if old_model.use_proj:
            new_model.proj_k.weight.copy_(old_model.proj_k.weight)
            new_model.proj_q.weight.copy_(old_model.proj_q.weight)
            new_model.proj_v.weight.copy_(old_model.proj_v.weight)
            new_model.proj_out.weight.copy_(old_model.proj_out.weight)


def verify_config(name, d_in, d_out, old_model, new_model, new_layer, is_mlp=False):
    """Run both models on same input and check forward + gradient equivalence."""
    copy_params(old_model, new_model, new_layer)
    old_model.train()
    new_model.train()

    B, n = 8, 20
    torch.manual_seed(456)
    xs = torch.randn(B, n, d_in)
    ys = torch.randn(B, n, d_out)

    if is_mlp:
        # For MLP memory, run per-timestep manually with shared W2 state.
        # We can't rely on RNG alignment because the two model classes
        # may consume random numbers differently.
        old_out, new_out = _run_mlp_manual(old_model, new_model, new_layer, xs, ys)
    else:
        old_out = old_model(xs, ys)
        new_out = new_model(xs, ys)

    fwd_match = torch.allclose(old_out, new_out, atol=1e-6)
    max_diff = (old_out - new_out).abs().max().item()

    # Gradient equivalence
    old_model.zero_grad()
    new_model.zero_grad()
    old_loss = ((old_out - ys) ** 2).mean()
    new_loss = ((new_out - ys) ** 2).mean()
    old_loss.backward()
    new_loss.backward()

    grad_match = True
    if old_model.eta.grad is not None and new_layer.eta.grad is not None:
        grad_match = torch.allclose(old_model.eta.grad, new_layer.eta.grad, atol=1e-6)
    elif (old_model.eta.grad is None) != (new_layer.eta.grad is None):
        grad_match = False
    # alpha grad: may be None in new model with NoRetention (expected)
    if old_model.alpha.grad is not None and new_layer.alpha.grad is not None:
        grad_match = grad_match and torch.allclose(
            old_model.alpha.grad, new_layer.alpha.grad, atol=1e-6
        )
    if old_model.use_proj:
        for attr in ("proj_k", "proj_q", "proj_v", "proj_out"):
            old_g = getattr(old_model, attr).weight.grad
            new_g = getattr(new_model, attr).weight.grad
            grad_match = grad_match and torch.allclose(old_g, new_g, atol=1e-6)

    match = fwd_match and grad_match
    status = "PASS" if match else "FAIL"
    detail = f"fwd_diff={max_diff:.2e}, grad_ok={grad_match}"
    print(f"  [{status}] {name}  ({detail})")
    return match


def _run_mlp_manual(old_model, new_model, new_layer, xs, ys):
    """Run old and new models with shared MLP memory state.

    Creates a single W2 init and injects it into both models' forward paths.
    """
    B, n, d_in = xs.shape
    d_out = ys.shape[-1]

    # Create shared initial state
    torch.manual_seed(999)
    old_state = old_model.memory.init_state(B, xs.device, xs.dtype)
    torch.manual_seed(999)
    new_mem_state = new_layer.memory.init_state(B, xs.device, xs.dtype)
    new_optim_state = new_layer.algorithm.init_optim_state(
        new_layer.memory, B, xs.device, xs.dtype
    )

    # Run per-timestep: old model
    old_preds = []
    state = old_state
    for i in range(n):
        q = xs[:, i]
        out = old_model.memory.read(state, q)
        old_preds.append(out)
        if i < n - 1:
            k = xs[:, i]
            v = ys[:, i]
            state = old_model.memory.write(state, k, v, old_model.eta, old_model.alpha)
    old_out = torch.stack(old_preds, dim=1)

    # Run per-timestep: new model
    new_preds = []
    new_state = (new_mem_state, new_optim_state)
    for i in range(n):
        q = xs[:, i]
        out = new_layer.read(new_state, q)
        new_preds.append(out)
        if i < n - 1:
            k = xs[:, i]
            v = ys[:, i]
            new_state = new_layer.write(new_state, k, v)
    new_out = torch.stack(new_preds, dim=1)

    return old_out, new_out


def main():
    d_in, d_out = 10, 1
    d_model = 32
    all_pass = True

    configs = [
        # (name, update_rule, memory_type, use_proj, bias_cls, retention_cls, eta_override)
        ("Hebbian + Matrix + NoRetention",
         "hebbian", "matrix", False, DotProductBias, NoRetention, None),
        ("Delta + Matrix + NoRetention",
         "delta", "matrix", False, L2Bias, NoRetention, None),
        ("Hebbian + Matrix + ScalarL2",
         "hebbian", "matrix", False, DotProductBias, ScalarL2Retention, None),
        ("Delta + MLP + NoRetention",
         "delta", "mlp", False, L2Bias, NoRetention, 0.01),  # small eta to avoid MLP divergence
        ("Delta + Matrix + Projections",
         "delta", "matrix", True, L2Bias, NoRetention, None),
    ]

    print("MIRAS Equivalence Verification")
    print("=" * 60)

    for name, rule, mem_type, use_proj, bias_cls, ret_cls, eta_override in configs:
        d_k = d_model if use_proj else d_in
        d_v = d_model if use_proj else d_out
        is_mlp = (mem_type == "mlp")

        if mem_type == "matrix":
            old_mem = OldMatrixMemory(d_k, d_v, update_rule=rule)
            new_mem = MatrixMemory(d_k, d_v)
        else:
            d_hidden = 16
            old_mem = OldMLPMemory(d_k, d_v, d_hidden, update_rule=rule)
            new_mem = MLPMemory(d_k, d_v, d_hidden)

        old_model = AssociativeRNN(
            d_in=d_in, d_out=d_out, memory=old_mem,
            use_projections=use_proj, d_model=d_model,
        )
        if eta_override is not None:
            with torch.no_grad():
                old_model.eta.fill_(eta_override)

        layer = MIRASLayer(new_mem, bias_cls(), ret_cls(), GD())
        new_model = MIRASModel(
            d_in=d_in, d_out=d_out, layers=[layer],
            use_projections=use_proj, d_model=d_model,
        )

        if not verify_config(name, d_in, d_out, old_model, new_model, layer, is_mlp=is_mlp):
            all_pass = False

    print("=" * 60)
    print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
