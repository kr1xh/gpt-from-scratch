"""Shared gradient checks: compare against torch.autograd and central finite differences."""
import numpy as np
import torch

from scratchgpt.tensor import Tensor

RNG = np.random.default_rng(0)


def rand(*shape, low=-1.0, high=1.0):
    return RNG.uniform(low, high, size=shape)


def ours(fn, arrays):
    ts = [Tensor(a, requires_grad=True) for a in arrays]
    out = fn(*ts)
    weights = RNG.standard_normal(out.data.shape)  # a random direction tests the whole Jacobian
    (out * weights).sum().backward()
    return out.data, [t.grad for t in ts], weights


def theirs(fn, arrays, weights):
    ts = [torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in arrays]
    out = fn(*ts)
    (out * torch.tensor(weights)).sum().backward()
    return out.detach().numpy(), [t.grad.numpy() for t in ts]


def finite_diff(fn, arrays, weights, eps=1e-6):
    grads = []
    for i, a in enumerate(arrays):
        g = np.zeros_like(a)
        for idx in np.ndindex(a.shape):
            def f(delta):
                moved = [x.copy() for x in arrays]
                moved[i][idx] += delta
                return float((fn(*[Tensor(x) for x in moved]).data * weights).sum())
            g[idx] = (f(eps) - f(-eps)) / (2 * eps)
        grads.append(g)
    return grads


def check(fn, torch_fn, *arrays):
    """Forward values and gradients of `fn` must match torch and finite differences (float64)."""
    out, grads, weights = ours(fn, arrays)
    t_out, t_grads = theirs(torch_fn, arrays, weights)
    np.testing.assert_allclose(out, t_out, rtol=1e-10, atol=1e-12)
    for g, tg, fd in zip(grads, t_grads, finite_diff(fn, arrays, weights)):
        np.testing.assert_allclose(g, tg, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-6)
