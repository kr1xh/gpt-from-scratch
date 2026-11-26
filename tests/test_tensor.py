"""Every op's gradient is checked against finite differences and against torch.autograd."""
import numpy as np
import pytest
import torch

from scratchgpt.tensor import Tensor, concat, no_grad, where

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
    out, grads, weights = ours(fn, arrays)
    t_out, t_grads = theirs(torch_fn, arrays, weights)
    np.testing.assert_allclose(out, t_out, rtol=1e-10, atol=1e-12)
    for g, tg, fd in zip(grads, t_grads, finite_diff(fn, arrays, weights)):
        np.testing.assert_allclose(g, tg, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-6)


BINARY = [
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
    ("div", lambda a, b: a / b),
]


@pytest.mark.parametrize("name,fn", BINARY)
@pytest.mark.parametrize("shapes", [((2, 3), (2, 3)), ((2, 3), (3,)), ((4, 1, 3), (2, 1)), ((2, 3), ())])
def test_binary_ops_with_broadcasting(name, fn, shapes):
    a = rand(*shapes[0])
    b = rand(*shapes[1], low=0.5, high=2.0) if name == "div" else rand(*shapes[1])
    check(fn, fn, a, b)


def test_scalars_on_either_side():
    check(lambda a: 2.0 * a - 1.0 + a / 4.0, lambda a: 2.0 * a - 1.0 + a / 4.0, rand(2, 3))
    check(lambda a: 1.0 - a, lambda a: 1.0 - a, rand(3))
    check(lambda a: 3.0 / a, lambda a: 3.0 / a, rand(3, low=0.5, high=2.0))


@pytest.mark.parametrize("p", [2, 3, 0.5, -1.5])
def test_pow(p):
    check(lambda a: a ** p, lambda a: a ** p, rand(2, 3, low=0.5, high=2.0))


def test_neg():
    check(lambda a: -a, lambda a: -a, rand(2, 3))


@pytest.mark.parametrize("shapes", [((3, 4), (4, 5)), ((2, 3, 4), (2, 4, 5)), ((3, 4), (2, 4, 5)),
                                    ((2, 1, 3, 4), (5, 4, 2))])
def test_matmul_including_batch_broadcasting(shapes):
    check(lambda a, b: a @ b, lambda a, b: a @ b, rand(*shapes[0]), rand(*shapes[1]))


UNARY = [
    ("exp", lambda a: a.exp(), torch.exp, (-1, 1)),
    ("log", lambda a: a.log(), torch.log, (0.5, 2)),
    ("sqrt", lambda a: a.sqrt(), torch.sqrt, (0.5, 2)),
    ("tanh", lambda a: a.tanh(), torch.tanh, (-2, 2)),
    ("sigmoid", lambda a: a.sigmoid(), torch.sigmoid, (-3, 3)),
    ("relu", lambda a: a.relu(), torch.relu, (-1, 1)),
    ("gelu", lambda a: a.gelu(), lambda a: torch.nn.functional.gelu(a, approximate="tanh"), (-3, 3)),
    ("silu", lambda a: a.silu(), torch.nn.functional.silu, (-3, 3)),
]


@pytest.mark.parametrize("name,fn,torch_fn,domain", UNARY)
def test_unary_ops(name, fn, torch_fn, domain):
    a = rand(3, 4, low=domain[0], high=domain[1])
    if name == "relu":
        a[np.abs(a) < 1e-3] = 0.5  # keep finite differences away from the kink
    check(fn, torch_fn, a)


@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (1, True), ((0, 2), False), (-1, True)])
def test_sum_and_mean(axis, keepdims):
    a = rand(2, 3, 4)
    dims = axis if axis is not None else tuple(range(3))
    check(lambda t: t.sum(axis=axis, keepdims=keepdims),
          lambda t: t.sum(dim=dims, keepdim=keepdims), a)
    check(lambda t: t.mean(axis=axis, keepdims=keepdims),
          lambda t: t.mean(dim=dims, keepdim=keepdims), a)


@pytest.mark.parametrize("axis,keepdims", [(-1, True), (0, False), (None, False)])
def test_max(axis, keepdims):
    a = rand(3, 5)
    if axis is None:
        check(lambda t: t.max(), lambda t: t.max(), a)
    else:
        check(lambda t: t.max(axis=axis, keepdims=keepdims),
              lambda t: t.max(dim=axis, keepdim=keepdims).values, a)


def test_max_splits_the_gradient_between_ties():
    t = Tensor(np.array([[1.0, 3.0, 3.0]]), requires_grad=True)
    t.max(axis=-1).sum().backward()
    np.testing.assert_allclose(t.grad, [[0.0, 0.5, 0.5]])


def test_reshape_and_transpose():
    a = rand(2, 3, 4)
    check(lambda t: t.reshape(4, 6), lambda t: t.reshape(4, 6), a)
    check(lambda t: t.reshape(-1, 2), lambda t: t.reshape(-1, 2), a)
    check(lambda t: t.transpose(2, 0, 1), lambda t: t.permute(2, 0, 1), a)
    check(lambda t: t.swapaxes(-1, -2), lambda t: t.transpose(-1, -2), a)
    check(lambda t: t.T, lambda t: t.transpose(-1, -2), a)


def test_slicing():
    a = rand(4, 5)
    check(lambda t: t[1:3, ::2], lambda t: t[1:3, ::2], a)
    check(lambda t: t[:, -1], lambda t: t[:, -1], a)


def test_integer_array_indexing_adds_up_repeated_rows():
    table = rand(5, 3)
    idx = np.array([[0, 2], [2, 4], [0, 0]])  # rows 0 and 2 are used more than once
    check(lambda t: t[idx], lambda t: t[torch.tensor(idx)], table)


def test_concat():
    a, b = rand(2, 3), rand(2, 1)
    check(lambda x, y: concat([x, y], axis=1), lambda x, y: torch.cat([x, y], dim=1), a, b)


def test_repeat_like_repeat_interleave():
    a = rand(2, 3, 4)
    check(lambda t: t.repeat(2, axis=1), lambda t: t.repeat_interleave(2, dim=1), a)


def test_where_with_broadcasting():
    cond = RNG.random((2, 3)) > 0.5
    a, b = rand(2, 3), rand(3)
    check(lambda x, y: where(cond, x, y), lambda x, y: torch.where(torch.tensor(cond), x, y), a, b)


def test_masked_fill_blocks_the_gradient_where_masked():
    mask = np.array([[False, True], [True, False]])
    t = Tensor(rand(2, 2), requires_grad=True)
    out = t.masked_fill(mask, -1e9)
    assert out.data[0, 1] == -1e9
    out.sum().backward()
    np.testing.assert_allclose(t.grad, (~mask).astype(float))


def test_a_node_used_twice_gets_both_gradients():
    x = Tensor(np.array([2.0, -3.0]), requires_grad=True)
    (x * x + x).sum().backward()
    np.testing.assert_allclose(x.grad, 2 * x.data + 1)


def test_deep_chains_do_not_hit_the_recursion_limit():
    x = Tensor(np.array(1.0), requires_grad=True)
    y = x
    for _ in range(5000):
        y = y + 0.0
    y.backward()
    assert x.grad == 1.0


def test_backward_on_a_non_scalar_needs_a_gradient():
    t = Tensor(rand(2), requires_grad=True)
    with pytest.raises(ValueError):
        (t * 2).backward()
    (t * 2).backward(np.ones(2))
    np.testing.assert_allclose(t.grad, [2.0, 2.0])


def test_constants_get_no_gradient():
    a = Tensor(rand(2), requires_grad=True)
    c = Tensor(rand(2))
    (a * c).sum().backward()
    assert c.grad is None and a.grad is not None


def test_no_grad_records_no_graph():
    a = Tensor(rand(2), requires_grad=True)
    with no_grad():
        out = a * 2
    assert not out.requires_grad


def test_default_dtype_is_float32_but_float64_is_kept():
    assert Tensor([1, 2]).data.dtype == np.float32
    assert Tensor(np.zeros(2)).data.dtype == np.float64
    f32 = Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)
    (f32 * 2.0).sum().backward()
    assert (f32 * 2.0).data.dtype == np.float32 and f32.grad.dtype == np.float32
