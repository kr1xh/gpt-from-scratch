"""Each optimizer must take exactly the steps torch.optim takes from the same weights."""
import math

import numpy as np
import pytest
import torch

from scratchgpt import optim
from scratchgpt.tensor import Tensor

RNG = np.random.default_rng(0)
X = RNG.standard_normal((16, 3))
Y = RNG.standard_normal((16, 2))
W0 = RNG.standard_normal((3, 2))
B0 = RNG.standard_normal(2)


def run_ours(make, steps=10):
    w, b = Tensor(W0.copy(), requires_grad=True), Tensor(B0.copy(), requires_grad=True)
    opt = make([w, b])
    for _ in range(steps):
        opt.zero_grad()
        (((Tensor(X) @ w + b) - Y) ** 2).mean().backward()
        opt.step()
    return w.data, b.data


def run_torch(make, steps=10):
    w = torch.tensor(W0.copy(), requires_grad=True)
    b = torch.tensor(B0.copy(), requires_grad=True)
    opt = make([w, b])
    for _ in range(steps):
        opt.zero_grad()
        (((torch.tensor(X) @ w + b) - torch.tensor(Y)) ** 2).mean().backward()
        opt.step()
    return w.detach().numpy(), b.detach().numpy()


CASES = [
    ("sgd", lambda p: optim.SGD(p, lr=0.1), lambda p: torch.optim.SGD(p, lr=0.1)),
    ("sgd weight decay", lambda p: optim.SGD(p, lr=0.1, weight_decay=0.01),
     lambda p: torch.optim.SGD(p, lr=0.1, weight_decay=0.01)),
    ("momentum", lambda p: optim.SGD(p, lr=0.05, momentum=0.9),
     lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9)),
    ("nesterov", lambda p: optim.SGD(p, lr=0.05, momentum=0.9, nesterov=True),
     lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, nesterov=True)),
    ("adam", lambda p: optim.Adam(p, lr=0.01), lambda p: torch.optim.Adam(p, lr=0.01)),
    ("adam l2", lambda p: optim.Adam(p, lr=0.01, weight_decay=0.1),
     lambda p: torch.optim.Adam(p, lr=0.01, weight_decay=0.1)),
    ("adamw", lambda p: optim.AdamW(p, lr=0.01, betas=(0.9, 0.95), weight_decay=0.1),
     lambda p: torch.optim.AdamW(p, lr=0.01, betas=(0.9, 0.95), weight_decay=0.1)),
]


@pytest.mark.parametrize("name,ours,theirs", CASES, ids=[c[0] for c in CASES])
def test_ten_steps_match_torch(name, ours, theirs):
    for a, b in zip(run_ours(ours), run_torch(theirs)):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9)


def test_parameters_without_a_gradient_are_left_alone():
    used, unused = Tensor(np.ones(2), requires_grad=True), Tensor(np.ones(2), requires_grad=True)
    opt = optim.Adam([used, unused], lr=0.1)
    (used * 3).sum().backward()
    opt.step()
    np.testing.assert_array_equal(unused.data, [1.0, 1.0])
    assert not np.allclose(used.data, [1.0, 1.0])


def test_lr_can_be_changed_between_steps():
    w = Tensor(np.array([1.0]), requires_grad=True)
    opt = optim.SGD([w], lr=0.1)
    w.grad = np.array([1.0])
    opt.lr = 0.5
    opt.step()
    np.testing.assert_allclose(w.data, [0.5])


def test_clip_grad_norm_matches_torch():
    grads = [RNG.standard_normal((3, 2)) * 5, RNG.standard_normal(2) * 5]
    ours = [Tensor(np.zeros_like(g), requires_grad=True) for g in grads]
    for t, g in zip(ours, grads):
        t.grad = g.copy()
    theirs = [torch.zeros(g.shape, dtype=torch.float64, requires_grad=True) for g in grads]
    for t, g in zip(theirs, grads):
        t.grad = torch.tensor(g.copy())
    norm = optim.clip_grad_norm(ours, 1.0)
    t_norm = torch.nn.utils.clip_grad_norm_(theirs, 1.0)
    assert norm == pytest.approx(float(t_norm))
    for a, b in zip(ours, theirs):
        np.testing.assert_allclose(a.grad, b.grad.numpy(), rtol=1e-10)


def test_clip_grad_norm_leaves_small_gradients_unchanged():
    t = Tensor(np.zeros(2), requires_grad=True)
    t.grad = np.array([0.3, 0.4])
    assert optim.clip_grad_norm([t], 1.0) == pytest.approx(0.5)
    np.testing.assert_allclose(t.grad, [0.3, 0.4])


def test_cosine_schedule_warms_up_then_decays_to_the_floor():
    lr = lambda s: optim.cosine_lr(s, max_lr=1.0, warmup=10, total=110, min_lr=0.1)  # noqa: E731
    assert lr(0) == pytest.approx(0.1)  # first step is 1/warmup of the peak
    assert lr(9) == pytest.approx(1.0)
    assert lr(60) == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 50 / 100)))
    assert lr(110) == pytest.approx(0.1) and lr(500) == pytest.approx(0.1)
    values = [lr(s) for s in range(9, 111)]
    assert all(a >= b for a, b in zip(values, values[1:]))
