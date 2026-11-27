import numpy as np
import pytest
import torch

from scratchgpt import nn
from scratchgpt.tensor import Tensor


def test_linear_matches_torch_with_transposed_weight():
    layer = nn.Linear(3, 2, rng=np.random.default_rng(0))
    ref = torch.nn.Linear(3, 2).double()
    with torch.no_grad():
        ref.weight.copy_(torch.tensor(layer.weight.data.T))  # ours is (in, out), torch's is (out, in)
        ref.bias.copy_(torch.tensor(layer.bias.data))
    x = np.random.default_rng(1).standard_normal((4, 3))
    np.testing.assert_allclose(layer(Tensor(x, dtype=np.float64)).data,
                               ref(torch.tensor(x)).detach().numpy(), rtol=1e-6)


def test_linear_without_bias_has_one_parameter():
    assert len(nn.Linear(3, 2, bias=False).parameters()) == 1


def test_embedding_looks_up_rows():
    emb = nn.Embedding(10, 4)
    idx = np.array([[1, 3], [3, 0]])
    out = emb(idx)
    assert out.shape == (2, 2, 4)
    np.testing.assert_array_equal(out.data[0, 1], emb.weight.data[3])


def test_layer_norm_starts_as_identity_scale():
    ln = nn.LayerNorm(4)
    out = ln(Tensor(np.random.default_rng(0).standard_normal((3, 4)))).data
    np.testing.assert_allclose(out.mean(axis=-1), 0, atol=1e-6)
    np.testing.assert_allclose(out.std(axis=-1), 1, atol=1e-3)


def test_parameters_and_names_are_found_through_nested_modules_and_lists():
    mlp = nn.MLP([2, 8, 8, 3])
    names = [n for n, _ in mlp.named_parameters()]
    assert names == ["layers.0.weight", "layers.0.bias", "layers.1.weight", "layers.1.bias",
                     "layers.2.weight", "layers.2.bias"]
    assert sum(p.size for p in mlp.parameters()) == 2 * 8 + 8 + 8 * 8 + 8 + 8 * 3 + 3


def test_zero_grad_clears_every_gradient():
    mlp = nn.MLP([2, 4, 1])
    mlp(Tensor(np.ones((3, 2)))).sum().backward()
    assert all(p.grad is not None for p in mlp.parameters())
    mlp.zero_grad()
    assert all(p.grad is None for p in mlp.parameters())


def test_state_dict_round_trip_restores_every_weight():
    a, b = nn.MLP([2, 4, 1], rng=np.random.default_rng(0)), nn.MLP([2, 4, 1], rng=np.random.default_rng(1))
    b.load_state_dict(a.state_dict())
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        np.testing.assert_array_equal(pa.data, pb.data)
    snapshot = a.state_dict()
    a.layers[0].weight.data += 1  # optimizers update weights in place; a snapshot must not follow
    assert not np.array_equal(a.layers[0].weight.data, snapshot["layers.0.weight"])


def test_load_state_dict_rejects_missing_names_and_wrong_shapes():
    model = nn.MLP([2, 4, 1])
    state = model.state_dict()
    with pytest.raises(ValueError, match="layers.1.bias"):
        model.load_state_dict({k: v for k, v in state.items() if k != "layers.1.bias"})
    with pytest.raises(ValueError, match="shape"):
        model.load_state_dict({**state, "layers.0.weight": np.zeros((3, 3))})


def test_load_state_dict_rejects_names_the_model_does_not_have():
    model = nn.MLP([2, 4, 1])
    with pytest.raises(ValueError, match="layers.9.weight"):
        model.load_state_dict({**model.state_dict(), "layers.9.weight": np.zeros((1, 1))})
