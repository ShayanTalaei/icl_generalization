import torch
from src.models.miras.memory import MatrixMemory, MLPMemory


# --- MatrixMemory ---

def test_matrix_init_state_zeros():
    mem = MatrixMemory(d_k=4, d_v=3)
    state = mem.init_state(B=2, device="cpu", dtype=torch.float32)
    assert state.shape == (2, 3, 4)
    assert torch.all(state == 0)


def test_matrix_read_zero_state():
    mem = MatrixMemory(d_k=4, d_v=3)
    state = torch.zeros(2, 3, 4)
    query = torch.randn(2, 4)
    out = mem.read(state, query)
    assert out.shape == (2, 3)
    assert torch.allclose(out, torch.zeros(2, 3))


def test_matrix_read_identity():
    """M = I should return query (when d_k == d_v)."""
    mem = MatrixMemory(d_k=4, d_v=4)
    state = torch.eye(4).unsqueeze(0).expand(2, -1, -1)
    query = torch.randn(2, 4)
    out = mem.read(state, query)
    assert torch.allclose(out, query)


def test_matrix_backward_outer_product():
    """backward should return d_out outer k."""
    mem = MatrixMemory(d_k=3, d_v=2)
    state = torch.zeros(1, 2, 3)
    k = torch.tensor([[1.0, 0.0, 0.0]])
    d_out = torch.tensor([[2.0, 3.0]])
    grads = mem.backward(state, k, d_out)
    expected = torch.tensor([[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    assert torch.allclose(grads, expected)


def test_matrix_backward_shape():
    mem = MatrixMemory(d_k=5, d_v=3)
    state = torch.zeros(4, 3, 5)
    k = torch.randn(4, 5)
    d_out = torch.randn(4, 3)
    grads = mem.backward(state, k, d_out)
    assert grads.shape == (4, 3, 5)


# --- MLPMemory ---

def test_mlp_init_state_shapes():
    mem = MLPMemory(d_k=4, d_v=3, d_hidden=8)
    W1, W2 = mem.init_state(B=2, device="cpu", dtype=torch.float32)
    assert W1.shape == (2, 3, 8)
    assert W2.shape == (2, 8, 4)


def test_mlp_init_state_w1_zeros():
    """W1 should be zero-initialized."""
    mem = MLPMemory(d_k=4, d_v=3, d_hidden=8)
    W1, W2 = mem.init_state(B=2, device="cpu", dtype=torch.float32)
    assert torch.all(W1 == 0)


def test_mlp_read_residual_when_square():
    """When d_k == d_v with W1=0, output should be x (residual)."""
    mem = MLPMemory(d_k=4, d_v=4, d_hidden=8)
    state = mem.init_state(B=2, device="cpu", dtype=torch.float32)
    query = torch.randn(2, 4)
    out = mem.read(state, query)
    assert torch.allclose(out, query)


def test_mlp_read_no_residual_when_nonsquare():
    """When d_k != d_v with W1=0, output should be 0."""
    mem = MLPMemory(d_k=4, d_v=3, d_hidden=8)
    state = mem.init_state(B=2, device="cpu", dtype=torch.float32)
    query = torch.randn(2, 4)
    out = mem.read(state, query)
    assert torch.allclose(out, torch.zeros(2, 3))


def test_mlp_backward_shapes():
    mem = MLPMemory(d_k=4, d_v=3, d_hidden=8)
    state = mem.init_state(B=2, device="cpu", dtype=torch.float32)
    k = torch.randn(2, 4)
    d_out = torch.randn(2, 3)
    dW1, dW2 = mem.backward(state, k, d_out)
    assert dW1.shape == (2, 3, 8)
    assert dW2.shape == (2, 8, 4)


def test_mlp_backward_gradient_check():
    """Verify manual backward matches autograd for a single step."""
    torch.manual_seed(42)
    d_k, d_v, d_hidden, B = 3, 2, 4, 1
    mem = MLPMemory(d_k=d_k, d_v=d_v, d_hidden=d_hidden)

    W1 = torch.randn(B, d_v, d_hidden, requires_grad=True)
    W2 = torch.randn(B, d_hidden, d_k, requires_grad=True)
    k = torch.randn(B, d_k)
    target = torch.randn(B, d_v)

    # Autograd
    x = k.unsqueeze(-1)
    h = torch.nn.functional.silu(torch.bmm(W2, x))
    out = torch.bmm(W1, h).squeeze(-1)
    loss = ((out - target) ** 2).sum()
    loss.backward()

    # Manual: d_out = target - pred (L2Bias convention)
    # manual grads should equal -autograd_grads / 2
    d_out = target - out.detach()
    dW1_manual, dW2_manual = mem.backward((W1.detach(), W2.detach()), k, d_out)

    assert torch.allclose(dW1_manual, -W1.grad / 2, atol=1e-5)
    assert torch.allclose(dW2_manual, -W2.grad / 2, atol=1e-5)


def test_memory_no_learnable_params():
    """Memory modules should have no nn.Parameters (state is per-batch)."""
    for mem in [MatrixMemory(4, 3), MLPMemory(4, 3, 8)]:
        assert len(list(mem.parameters())) == 0
