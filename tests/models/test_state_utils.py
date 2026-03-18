import torch
from src.models.miras.state_utils import state_map


def test_state_map_tensor():
    t = torch.tensor([1.0, 2.0, 3.0])
    result = state_map(lambda x: x * 2, t)
    assert torch.allclose(result, torch.tensor([2.0, 4.0, 6.0]))


def test_state_map_tuple():
    t1 = torch.tensor([1.0, 2.0])
    t2 = torch.tensor([3.0, 4.0])
    result = state_map(lambda x: x * 2, (t1, t2))
    assert isinstance(result, tuple)
    assert torch.allclose(result[0], torch.tensor([2.0, 4.0]))
    assert torch.allclose(result[1], torch.tensor([6.0, 8.0]))


def test_state_map_two_args():
    s = torch.tensor([1.0, 2.0])
    g = torch.tensor([0.5, 0.5])
    result = state_map(lambda a, b: a + b, s, g)
    assert torch.allclose(result, torch.tensor([1.5, 2.5]))


def test_state_map_two_tuple_args():
    s = (torch.tensor([1.0]), torch.tensor([2.0]))
    g = (torch.tensor([0.1]), torch.tensor([0.2]))
    result = state_map(lambda a, b: a + b, s, g)
    assert torch.allclose(result[0], torch.tensor([1.1]))
    assert torch.allclose(result[1], torch.tensor([2.2]))
