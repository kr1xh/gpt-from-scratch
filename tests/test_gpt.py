"""The GPT must compute exactly what a PyTorch model computes from the same weights, for the GPT-2
block, the modern block and every one-part swap between them."""
import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from experiments.torch_gpt import forward as torch_gpt
from scratchgpt import optim
from dataclasses import replace

from scratchgpt.nn import GPT, GPTConfig

SMALL = GPTConfig(vocab_size=11, block_size=6, n_layer=2, n_head=2, d_model=8)
# 4 query heads on 2 key/value heads: with a single kv head every grouping would look the same
MODERN = dict(norm="rms", mlp="swiglu", pos="rope", n_head=4, n_kv_head=2)
VARIANTS = {"gpt2": {}, "rms": dict(norm="rms"), "swiglu": dict(mlp="swiglu"), "rope": dict(pos="rope"),
            "gqa": dict(n_head=4, n_kv_head=2), "mqa": dict(n_kv_head=1), "modern": MODERN}


def as_float64(model):
    for p in model.parameters():
        p.data = p.data.astype(np.float64)
    return model


@pytest.mark.parametrize("variant", VARIANTS)
def test_logits_loss_and_every_gradient_match_a_torch_reference(variant):
    cfg = replace(SMALL, **VARIANTS[variant])
    model = as_float64(GPT(cfg, rng=np.random.default_rng(0)))
    for p in model.parameters():  # GPT-2 starts biases at zero; random ones test more of the graph
        p.data += np.random.default_rng(1).normal(0, 0.1, p.shape)
    rng = np.random.default_rng(2)
    idx = rng.integers(0, SMALL.vocab_size, (3, 5))  # shorter than block_size
    targets = rng.integers(0, SMALL.vocab_size, (3, 5))

    logits, loss = model(idx, targets)
    loss.backward()

    P = {n: torch.tensor(p.data, requires_grad=True) for n, p in model.named_parameters()}
    t_logits, t_loss = torch_gpt(P, cfg, torch.tensor(idx), torch.tensor(targets))
    t_loss.backward()

    np.testing.assert_allclose(logits.data, t_logits.detach().numpy(), rtol=1e-6, atol=1e-8)
    assert loss.item() == pytest.approx(t_loss.item(), rel=1e-6)
    for name, p in model.named_parameters():
        np.testing.assert_allclose(p.grad, P[name].grad.numpy(), rtol=1e-5, atol=1e-8, err_msg=name)


@pytest.mark.parametrize("variant", ["gpt2", "modern"])
@settings(max_examples=25, deadline=None)
@given(data=st.data())
def test_a_position_never_sees_later_tokens(variant, data):
    model = GPT(replace(SMALL, **VARIANTS[variant]), rng=np.random.default_rng(0))
    T = data.draw(st.integers(2, SMALL.block_size))
    t = data.draw(st.integers(0, T - 2))
    tokens = st.lists(st.integers(0, SMALL.vocab_size - 1), min_size=T, max_size=T)
    a = np.array(data.draw(tokens))
    b = a.copy()
    b[t + 1:] = data.draw(st.lists(st.integers(0, SMALL.vocab_size - 1), min_size=T - t - 1,
                                   max_size=T - t - 1))
    la, _ = model(a[None])
    lb, _ = model(b[None])
    np.testing.assert_allclose(la.data[0, :t + 1], lb.data[0, :t + 1], rtol=1e-5, atol=1e-6)


def test_later_positions_do_see_earlier_tokens():
    model = GPT(SMALL, rng=np.random.default_rng(0))
    la, _ = model(np.array([[1, 2, 3]]))
    lb, _ = model(np.array([[4, 2, 3]]))
    assert not np.allclose(la.data[0, 2], lb.data[0, 2])


def test_loss_at_initialisation_is_that_of_a_uniform_guess():
    cfg = GPTConfig(vocab_size=65, block_size=16, n_layer=2, n_head=2, d_model=32)
    model = GPT(cfg, rng=np.random.default_rng(0))
    rng = np.random.default_rng(1)
    # targets independent of the inputs: the tied head scores a token's own row higher, so
    # "predict the input token" would come out easier than a uniform guess
    idx, targets = rng.integers(0, 65, (4, 16)), rng.integers(0, 65, (4, 16))
    _, loss = model(idx, targets)
    assert loss.item() == pytest.approx(math.log(65), abs=0.1)


def test_parameter_count_counts_the_tied_head_once():
    V, Tmax, L, d = SMALL.vocab_size, SMALL.block_size, SMALL.n_layer, SMALL.d_model
    expected = V * d + Tmax * d + L * (12 * d * d + 13 * d) + 2 * d
    assert sum(p.size for p in GPT(SMALL).parameters()) == expected


def test_gpt2_initialisation():
    cfg = GPTConfig(vocab_size=50, block_size=8, n_layer=8, n_head=4, d_model=64)
    named = dict(GPT(cfg, rng=np.random.default_rng(0)).named_parameters())
    assert named["blocks.0.attn.qkv.weight"].data.std() == pytest.approx(0.02, rel=0.05)
    # projections into the residual stream shrink with depth so the sum of L blocks stays small
    assert named["blocks.0.attn.proj.weight"].data.std() == pytest.approx(0.02 / 4, rel=0.1)
    assert named["blocks.0.mlp.layers.1.weight"].data.std() == pytest.approx(0.02 / 4, rel=0.1)
    assert not named["blocks.0.attn.qkv.bias"].data.any()
    assert all(p.data.dtype == np.float32 for p in named.values())


def test_rejects_sequences_longer_than_the_context():
    with pytest.raises(ValueError, match="block_size"):
        GPT(SMALL)(np.zeros((1, SMALL.block_size + 1), dtype=int))


def test_heads_must_divide_the_width():
    with pytest.raises(ValueError):
        GPT(GPTConfig(vocab_size=5, block_size=4, n_layer=1, n_head=3, d_model=8))


def test_memorises_one_batch():
    cfg = GPTConfig(vocab_size=8, block_size=8, n_layer=1, n_head=2, d_model=16)
    model = GPT(cfg, rng=np.random.default_rng(0))
    idx = np.random.default_rng(1).integers(0, 8, (4, 9))
    x, y = idx[:, :-1], idx[:, 1:]
    opt = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    for _ in range(150):
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_rotary_models_have_no_learned_position_embedding():
    assert "wpe.weight" not in dict(GPT(replace(SMALL, pos="rope")).named_parameters())


def test_swiglu_hidden_size_keeps_the_mlp_the_size_of_the_gelu_one():
    def mlp_params(mlp):
        cfg = GPTConfig(vocab_size=65, n_layer=1, d_model=128, mlp=mlp)
        return sum(p.size for n, p in GPT(cfg).named_parameters() if ".mlp." in n)
    assert mlp_params("swiglu") == pytest.approx(mlp_params("gelu"), rel=0.01)


def test_fewer_kv_heads_shrink_the_key_and_value_projections():
    full = dict(GPT(SMALL).named_parameters())["blocks.0.attn.qkv.weight"]
    grouped = dict(GPT(replace(SMALL, n_kv_head=1)).named_parameters())["blocks.0.attn.qkv.weight"]
    head = SMALL.d_model // SMALL.n_head
    assert full.shape == (8, 3 * 8) and grouped.shape == (8, 8 + 2 * head)


@pytest.mark.parametrize("bad", [dict(norm="batch"), dict(mlp="relu"), dict(pos="alibi"), dict(n_kv_head=3),
                                 dict(n_kv_head=0)])
def test_unknown_options_and_kv_heads_that_do_not_divide_the_heads_are_rejected(bad):
    with pytest.raises(ValueError):
        GPT(replace(SMALL, **bad))


@pytest.mark.parametrize("mlp, name", [("gelu", "blocks.0.mlp.layers.0.weight"), ("swiglu", "blocks.0.mlp.up.weight")])
def test_mlp_hidden_overrides_the_default_width(mlp, name):
    assert dict(GPT(replace(SMALL, mlp=mlp, mlp_hidden=5)).named_parameters())[name].shape == (8, 5)
