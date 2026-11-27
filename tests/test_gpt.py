"""The GPT must compute exactly what a PyTorch GPT-2 block stack computes from the same weights."""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as TF
from hypothesis import given, settings
from hypothesis import strategies as st

from scratchgpt import optim
from scratchgpt.nn import GPT, GPTConfig

SMALL = GPTConfig(vocab_size=11, block_size=6, n_layer=2, n_head=2, d_model=8)


def as_float64(model):
    for p in model.parameters():
        p.data = p.data.astype(np.float64)
    return model


def torch_gpt(P, cfg, idx, targets):
    """GPT-2 written with torch's own layer norm, attention, GELU and cross-entropy."""
    B, T = idx.shape
    C, H = cfg.d_model, cfg.n_head
    x = P["wte.weight"][idx] + P["wpe.weight"][:T]
    for i in range(cfg.n_layer):
        p = lambda name: P[f"blocks.{i}.{name}"]  # noqa: E731
        h = TF.layer_norm(x, (C,), p("ln1.weight"), p("ln1.bias"))
        q, k, v = (h @ p("attn.qkv.weight") + p("attn.qkv.bias")).split(C, dim=-1)
        q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in (q, k, v))
        a = TF.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + a.transpose(1, 2).reshape(B, T, C) @ p("attn.proj.weight") + p("attn.proj.bias")
        h = TF.layer_norm(x, (C,), p("ln2.weight"), p("ln2.bias"))
        h = TF.gelu(h @ p("mlp.layers.0.weight") + p("mlp.layers.0.bias"), approximate="tanh")
        x = x + h @ p("mlp.layers.1.weight") + p("mlp.layers.1.bias")
    x = TF.layer_norm(x, (C,), P["ln_f.weight"], P["ln_f.bias"])
    logits = x @ P["wte.weight"].T  # the output head is tied to the token embedding
    return logits, TF.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def test_logits_loss_and_every_gradient_match_a_torch_reference():
    model = as_float64(GPT(SMALL, rng=np.random.default_rng(0)))
    for p in model.parameters():  # GPT-2 starts biases at zero; random ones test more of the graph
        p.data += np.random.default_rng(1).normal(0, 0.1, p.shape)
    rng = np.random.default_rng(2)
    idx = rng.integers(0, SMALL.vocab_size, (3, 5))  # shorter than block_size
    targets = rng.integers(0, SMALL.vocab_size, (3, 5))

    logits, loss = model(idx, targets)
    loss.backward()

    P = {n: torch.tensor(p.data, requires_grad=True) for n, p in model.named_parameters()}
    t_logits, t_loss = torch_gpt(P, SMALL, torch.tensor(idx), torch.tensor(targets))
    t_loss.backward()

    np.testing.assert_allclose(logits.data, t_logits.detach().numpy(), rtol=1e-6, atol=1e-8)
    assert loss.item() == pytest.approx(t_loss.item(), rel=1e-6)
    for name, p in model.named_parameters():
        np.testing.assert_allclose(p.grad, P[name].grad.numpy(), rtol=1e-5, atol=1e-8, err_msg=name)


@settings(max_examples=25, deadline=None)
@given(st.data())
def test_a_position_never_sees_later_tokens(data):
    model = GPT(SMALL, rng=np.random.default_rng(0))
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
