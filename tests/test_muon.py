"""Muon: momentum, orthogonalised by Newton-Schulz, on the hidden weight matrices."""
import numpy as np
import pytest
import torch
import torch.optim._muon as torch_muon
from hypothesis import given, settings
from hypothesis import strategies as st

from scratchgpt import optim
from scratchgpt.nn import GPT, GPTConfig
from scratchgpt.tensor import Tensor
from scratchgpt.train import make_optimizers


@settings(max_examples=30, deadline=None)
@given(rows=st.integers(2, 40), cols=st.integers(2, 40), seed=st.integers(0, 999))
def test_newton_schulz_keeps_the_singular_vectors_and_squeezes_the_singular_values(rows, cols, seed):
    g = np.random.default_rng(seed).standard_normal((rows, cols))
    u, s, vt = np.linalg.svd(g, full_matrices=False)
    core = u.T @ optim.newton_schulz(g) @ vt.T  # diagonal if the singular vectors are kept
    np.testing.assert_allclose(core - np.diag(np.diag(core)), 0, atol=1e-8)
    # the quintic is tuned for speed, not to reach exactly 1: values land around 0.7 to 1.2
    kept = np.diag(core)[s > 1e-3 * s.max()]
    assert kept.min() > 0.5 and kept.max() < 1.5


def test_newton_schulz_handles_tall_and_wide_matrices_alike():
    g = np.random.default_rng(0).standard_normal((7, 3))
    np.testing.assert_allclose(optim.newton_schulz(g), optim.newton_schulz(g.T).T, atol=1e-12)


def float64_newton_schulz(grad, ns_coefficients, ns_steps, eps):
    """torch's own iteration, minus its cast to bfloat16, so the rest of torch's Muon can be
    compared exactly."""
    a, b, c = ns_coefficients
    x = grad.clone()
    if grad.size(0) > grad.size(1):
        x = x.T
    x = x / x.norm().clamp(min=eps)
    for _ in range(ns_steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if grad.size(0) > grad.size(1) else x


def test_five_steps_match_torch_muon(monkeypatch):
    monkeypatch.setattr(torch_muon, "_zeropower_via_newtonschulz", float64_newton_schulz)
    rng = np.random.default_rng(0)
    x, y, w0 = rng.standard_normal((16, 5)), rng.standard_normal((16, 3)), rng.standard_normal((5, 3))

    w = Tensor(w0.copy(), requires_grad=True)
    opt = optim.Muon([w], lr=0.02, momentum=0.9, weight_decay=0.1)
    # torch stores a layer's weight as (out, in); ours is (in, out), so torch gets the transpose
    tw = torch.tensor(w0.T.copy(), requires_grad=True)
    topt = torch.optim.Muon([tw], lr=0.02, momentum=0.9, weight_decay=0.1, adjust_lr_fn="match_rms_adamw")
    for _ in range(5):
        opt.zero_grad()
        (((Tensor(x) @ w) - y) ** 2).mean().backward()
        opt.step()
        topt.zero_grad()
        ((torch.tensor(x) @ tw.T - torch.tensor(y)) ** 2).mean().backward()
        topt.step()
    np.testing.assert_allclose(w.data, tw.detach().numpy().T, rtol=1e-10, atol=1e-12)


def test_muon_only_takes_matrices():
    with pytest.raises(ValueError, match="2-D"):
        optim.Muon([Tensor(np.zeros(3), requires_grad=True)], lr=0.02)


def test_muon_gets_the_hidden_matrices_and_adamw_the_rest():
    model = GPT(GPTConfig(vocab_size=8, block_size=8, n_layer=2, n_head=2, d_model=16))
    opts = make_optimizers(model, lr=1e-2, weight_decay=0.1, optimizer="muon")
    names = {id(p): n for n, p in model.named_parameters()}
    muon = [names[id(p)] for o in opts if isinstance(o, optim.Muon) for p in o.params]
    assert sorted(muon) == sorted(n for n, p in model.named_parameters()
                                  if p.ndim == 2 and n not in ("wte.weight", "wpe.weight"))
    covered = [id(p) for o in opts for p in o.params]
    assert sorted(covered) == sorted(names) and len(covered) == len(set(covered))  # each exactly once


def test_a_gpt_trained_with_muon_memorises_one_batch():
    cfg = GPTConfig(vocab_size=8, block_size=8, n_layer=1, n_head=2, d_model=16)
    model = GPT(cfg, rng=np.random.default_rng(0))
    idx = np.random.default_rng(1).integers(0, 8, (4, 9))
    opts = make_optimizers(model, lr=1e-2, weight_decay=0.0, optimizer="muon")
    for _ in range(150):
        for o in opts:
            o.zero_grad()
        _, loss = model(idx[:, :-1], idx[:, 1:])
        loss.backward()
        for o in opts:
            o.step()
    assert loss.item() < 0.1
