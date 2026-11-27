"""A tiny mixture of experts: each token goes to its top-k experts and gets their gated sum."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.torch_gpt import forward as torch_gpt
from scratchgpt.nn import GPT, GPTConfig, MoE
from scratchgpt.tensor import Tensor, no_grad

CFG = GPTConfig(vocab_size=11, block_size=6, n_layer=2, n_head=2, d_model=8, mlp="swiglu", n_expert=4, top_k=2)


def test_each_token_gets_the_gated_sum_of_its_top_k_experts():
    moe = MoE(CFG, rng=np.random.default_rng(0))
    x = np.random.default_rng(1).standard_normal((2, 5, 8))
    with no_grad():
        out = moe(Tensor(x)).data.reshape(-1, 8)
        for n, token in enumerate(x.reshape(-1, 8)):
            scores = (Tensor(token[None]) @ moe.router.weight).data[0]
            chosen = np.argsort(-scores)[:2]
            gates = np.exp(scores[chosen]) / np.exp(scores[chosen]).sum()
            expected = sum(g * moe.experts[e](Tensor(token[None])).data[0] for g, e in zip(gates, chosen))
            np.testing.assert_allclose(out[n], expected, rtol=1e-5, atol=1e-6)


def test_balanced_routing_costs_one_and_collapsed_routing_costs_about_n_expert():
    cfg = replace(CFG, d_model=4, top_k=1)
    moe = MoE(cfg, rng=np.random.default_rng(0))
    moe.router.weight.data = 20 * np.eye(4, dtype=np.float32)  # token i strongly prefers expert i
    with no_grad():
        moe(Tensor(np.eye(4)[None]))  # one token for each expert
        assert moe.stats["aux_loss"].item() == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_allclose(moe.stats["load"], [0.25] * 4)
        moe(Tensor(np.tile(np.eye(4)[:1], (4, 1))[None]))  # every token wants expert 0
        assert moe.stats["aux_loss"].item() == pytest.approx(4.0, rel=1e-6)
        np.testing.assert_allclose(moe.stats["load"], [1, 0, 0, 0])


@pytest.mark.parametrize("mlp", ["gelu", "swiglu"])
def test_logits_losses_and_every_gradient_match_a_torch_reference(mlp):
    cfg = replace(CFG, mlp=mlp)
    model = GPT(cfg, rng=np.random.default_rng(0))
    for p in model.parameters():
        p.data = p.data.astype(np.float64) + np.random.default_rng(1).normal(0, 0.1, p.shape)
    rng = np.random.default_rng(2)
    idx, targets = rng.integers(0, 11, (3, 5)), rng.integers(0, 11, (3, 5))

    logits, loss = model(idx, targets)
    (loss + 0.01 * model.aux_loss()).backward()
    P = {n: torch.tensor(p.data, requires_grad=True) for n, p in model.named_parameters()}
    t_logits, t_loss, t_aux = torch_gpt(P, cfg, torch.tensor(idx), torch.tensor(targets), with_aux=True)
    (t_loss + 0.01 * t_aux).backward()

    np.testing.assert_allclose(logits.data, t_logits.detach().numpy(), rtol=1e-6, atol=1e-8)
    assert loss.item() == pytest.approx(t_loss.item(), rel=1e-6)
    assert model.aux_loss().item() == pytest.approx(t_aux.item(), rel=1e-6)
    names = dict(model.named_parameters())
    last = {"gelu": "layers.1", "swiglu": "down"}[mlp]
    assert "blocks.0.mlp.router.weight" in names and f"blocks.1.mlp.experts.3.{last}.weight" in names
    for name, p in names.items():
        np.testing.assert_allclose(p.grad, P[name].grad.numpy(), rtol=1e-5, atol=1e-8, err_msg=name)


def test_the_load_of_every_layer_is_reported_after_a_forward():
    model = GPT(CFG, rng=np.random.default_rng(0))
    model(np.zeros((2, 6), dtype=int))
    loads = model.expert_load()
    assert len(loads) == CFG.n_layer
    for load in loads:
        assert load.shape == (4,) and load.sum() == pytest.approx(1.0)


def test_dense_models_have_no_aux_loss_or_load():
    model = GPT(replace(CFG, n_expert=0))
    model(np.zeros((1, 3), dtype=int))
    assert model.aux_loss() is None and model.expert_load() == []


def test_the_router_and_experts_get_gpt2_initialisation():
    named = dict(GPT(replace(CFG, n_layer=8, d_model=64, n_head=4), rng=np.random.default_rng(0)).named_parameters())
    assert named["blocks.0.mlp.router.weight"].data.std() == pytest.approx(0.02, rel=0.1)
    assert named["blocks.0.mlp.experts.2.gate.weight"].data.std() == pytest.approx(0.02, rel=0.1)
    assert named["blocks.0.mlp.experts.2.down.weight"].data.std() == pytest.approx(0.02 / 4, rel=0.1)
    assert not named["blocks.0.mlp.experts.2.down.bias"].data.any()


@pytest.mark.parametrize("bad", [dict(n_expert=1), dict(top_k=5), dict(top_k=0)])
def test_impossible_expert_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        replace(CFG, **bad)

