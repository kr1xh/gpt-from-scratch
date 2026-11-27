"""A NumPy array that remembers how it was made, so gradients can flow back through it.

Each op returns a new Tensor holding its inputs (`_parents`) and a function that turns the
gradient of the output into gradients of the inputs (`_backward`). `backward()` walks the graph
from the output to the leaves in reverse topological order and adds those gradients up.
"""
from contextlib import contextmanager

import numpy as np

_GRAD_ENABLED = True


@contextmanager
def no_grad():
    """Run ops without recording a graph (evaluation, sampling, optimizer updates)."""
    global _GRAD_ENABLED
    previous, _GRAD_ENABLED = _GRAD_ENABLED, False
    try:
        yield
    finally:
        _GRAD_ENABLED = previous


def _as_array(data, dtype=None) -> np.ndarray:
    if dtype is not None:
        return np.asarray(data, dtype=dtype)
    arr = np.asarray(data)
    return arr if arr.dtype in (np.float32, np.float64) else arr.astype(np.float32)


def unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum a gradient back down to `shape`, undoing NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, size in enumerate(shape):
        if size == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def _axes(axis, ndim: int) -> tuple:
    if axis is None:
        return tuple(range(ndim))
    return tuple(a % ndim for a in (axis if isinstance(axis, tuple) else (axis,)))


class Tensor:
    __array_priority__ = 100  # so `ndarray * Tensor` calls Tensor.__rmul__

    def __init__(self, data, requires_grad: bool = False, dtype=None):
        self.data = _as_array(data, dtype)
        self.requires_grad = requires_grad
        self.grad = None
        self._parents = ()
        self._backward = None

    # ---- graph plumbing -------------------------------------------------
    def _wrap(self, other) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other, dtype=self.data.dtype)

    @staticmethod
    def _make(data, parents, backward) -> "Tensor":
        out = Tensor(data, dtype=data.dtype if isinstance(data, np.ndarray) else None)
        if _GRAD_ENABLED and any(p.requires_grad for p in parents):
            out.requires_grad = True
            out._parents = parents
            out._backward = backward
        return out

    def backward(self, grad=None) -> None:
        if grad is None:
            if self.data.size != 1:
                raise ValueError("backward() on a non-scalar needs an explicit gradient")
            grad = np.ones_like(self.data)
        order, seen, stack = [], set(), [(self, False)]
        while stack:  # iterative post-order DFS: deep graphs must not hit the recursion limit
            node, finished = stack.pop()
            if finished:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            stack.extend((p, False) for p in node._parents if p.requires_grad and id(p) not in seen)

        grads = {id(self): _as_array(grad, self.data.dtype)}
        for node in reversed(order):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if not node._parents:  # a leaf: keep its gradient (added to, until zeroed)
                # its own writeable copy: ops pass one array to several inputs, and sum's is read-only
                node.grad = np.array(g) if node.grad is None else node.grad + g
                continue
            for parent, pg in zip(node._parents, node._backward(g)):
                if parent.requires_grad and pg is not None:
                    pg = pg.astype(parent.data.dtype, copy=False)
                    grads[id(parent)] = grads[id(parent)] + pg if id(parent) in grads else pg

    # ---- conveniences ---------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def item(self) -> float:
        return self.data.item()

    def numpy(self) -> np.ndarray:
        return self.data

    def detach(self) -> "Tensor":
        return Tensor(self.data)

    def __repr__(self):
        return f"Tensor({self.data!r}, requires_grad={self.requires_grad})"

    # ---- arithmetic -----------------------------------------------------
    def __add__(self, other):
        other = self._wrap(other)
        a, b = self, other
        return Tensor._make(a.data + b.data, (a, b),
                            lambda g: (unbroadcast(g, a.shape), unbroadcast(g, b.shape)))

    def __mul__(self, other):
        other = self._wrap(other)
        a, b = self, other
        return Tensor._make(a.data * b.data, (a, b),
                            lambda g: (unbroadcast(g * b.data, a.shape), unbroadcast(g * a.data, b.shape)))

    def __truediv__(self, other):
        other = self._wrap(other)
        a, b = self, other
        return Tensor._make(a.data / b.data, (a, b),
                            lambda g: (unbroadcast(g / b.data, a.shape),
                                       unbroadcast(-g * a.data / (b.data * b.data), b.shape)))

    def __neg__(self):
        return Tensor._make(-self.data, (self,), lambda g: (-g,))

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return self._wrap(other) - self

    def __rtruediv__(self, other):
        return self._wrap(other) / self

    def __pow__(self, p: float):
        x = self
        return Tensor._make(x.data ** p, (x,), lambda g: (g * p * x.data ** (p - 1),))

    def __matmul__(self, other):
        other = self._wrap(other)
        a, b = self, other
        if a.ndim > 2 and b.ndim == 2:
            # activations (..., K) times a weight (K, N): one large GEMM on (rows, K). NumPy runs
            # the stacked form as many small products, measured 20x slower for (32, 128, 128) @ (128, 512)
            a2 = a.data.reshape(-1, a.shape[-1])

            def backward_2d(g):
                g2 = g.reshape(-1, g.shape[-1])
                return (g2 @ b.data.T).reshape(a.shape), a2.T @ g2
            return Tensor._make((a2 @ b.data).reshape(*a.shape[:-1], b.shape[1]), (a, b), backward_2d)

        def backward(g):
            return (unbroadcast(g @ np.swapaxes(b.data, -1, -2), a.shape),
                    unbroadcast(np.swapaxes(a.data, -1, -2) @ g, b.shape))
        return Tensor._make(a.data @ b.data, (a, b), backward)

    # ---- elementwise functions -----------------------------------------
    def exp(self):
        out = np.exp(self.data)
        return Tensor._make(out, (self,), lambda g: (g * out,))

    def log(self):
        x = self
        return Tensor._make(np.log(x.data), (x,), lambda g: (g / x.data,))

    def sqrt(self):
        out = np.sqrt(self.data)
        return Tensor._make(out, (self,), lambda g: (g * 0.5 / out,))

    def tanh(self):
        out = np.tanh(self.data)
        return Tensor._make(out, (self,), lambda g: (g * (1 - out * out),))

    def sigmoid(self):
        out = 1 / (1 + np.exp(-self.data))
        return Tensor._make(out, (self,), lambda g: (g * out * (1 - out),))

    def relu(self):
        x = self
        return Tensor._make(np.maximum(x.data, 0), (x,), lambda g: (g * (x.data > 0),))

    def gelu(self):
        """GELU, tanh approximation (as in GPT-2)."""
        x = self.data
        k = np.sqrt(2 / np.pi).astype(x.dtype)
        x2 = x * x  # x * x * x, not x ** 3: NumPy's general pow was 5x slower here
        t = np.tanh(k * x * (1 + 0.044715 * x2))
        out = 0.5 * x * (1 + t)

        def backward(g):
            return (g * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * k * (1 + 3 * 0.044715 * x2)),)
        return Tensor._make(out, (self,), backward)

    def silu(self):
        x = self.data
        s = 1 / (1 + np.exp(-x))
        return Tensor._make(x * s, (self,), lambda g: (g * s * (1 + x * (1 - s)),))

    # ---- reductions -----------------------------------------------------
    def sum(self, axis=None, keepdims: bool = False):
        shape, axes = self.shape, _axes(axis, self.ndim)

        def backward(g):
            if not keepdims:
                g = np.expand_dims(g, axes)
            return (np.broadcast_to(g, shape),)
        return Tensor._make(self.data.sum(axis=axes, keepdims=keepdims), (self,), backward)

    def mean(self, axis=None, keepdims: bool = False):
        axes = _axes(axis, self.ndim)
        n = int(np.prod([self.shape[a] for a in axes]))
        return self.sum(axis=axes, keepdims=keepdims) / n

    def max(self, axis=None, keepdims: bool = False):
        axes = _axes(axis, self.ndim)
        top = self.data.max(axis=axes, keepdims=True)
        mask = self.data == top
        share = mask / mask.sum(axis=axes, keepdims=True)  # ties split the gradient equally

        def backward(g):
            if not keepdims:
                g = np.expand_dims(g, axes)
            return (g * share,)
        return Tensor._make(top if keepdims else np.squeeze(top, axis=axes), (self,), backward)

    # ---- shape ----------------------------------------------------------
    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        old = self.shape
        return Tensor._make(self.data.reshape(shape), (self,), lambda g: (g.reshape(old),))

    def transpose(self, *axes):
        axes = axes or tuple(reversed(range(self.ndim)))
        inverse = tuple(np.argsort(axes))
        return Tensor._make(np.transpose(self.data, axes), (self,), lambda g: (np.transpose(g, inverse),))

    def swapaxes(self, a: int, b: int):
        return Tensor._make(np.swapaxes(self.data, a, b), (self,), lambda g: (np.swapaxes(g, a, b),))

    @property
    def T(self):
        """Swap the last two axes (matrix transpose, batched)."""
        return self.swapaxes(-1, -2)

    def __getitem__(self, idx):
        idx = idx.data.astype(np.int64) if isinstance(idx, Tensor) else idx
        shape, dtype = self.shape, self.data.dtype
        parts = idx if isinstance(idx, tuple) else (idx,)
        basic = all(isinstance(i, (int, np.integer, slice)) or i is None or i is Ellipsis for i in parts)

        def backward(g):
            full = np.zeros(shape, dtype=dtype)
            if basic:  # ints and slices pick each element at most once: plain assignment is enough
                full[idx] = g
            else:
                np.add.at(full, idx, g)  # repeated indices (e.g. the same token twice) add up
            return (full,)
        return Tensor._make(self.data[idx], (self,), backward)

    def repeat(self, repeats: int, axis: int):
        """Repeat each element `repeats` times along `axis` (like torch.repeat_interleave)."""
        axis = axis % self.ndim
        shape = self.shape

        def backward(g):
            g = g.reshape(shape[:axis] + (shape[axis], repeats) + shape[axis + 1:])
            return (g.sum(axis=axis + 1),)
        return Tensor._make(np.repeat(self.data, repeats, axis=axis), (self,), backward)

    def masked_fill(self, mask, value: float):
        """Replace entries where `mask` is True with a constant (no gradient flows there)."""
        return where(mask, value, self)


def concat(tensors: list[Tensor], axis: int = 0) -> Tensor:
    tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
    cuts = np.cumsum([t.shape[axis] for t in tensors])[:-1]
    return Tensor._make(np.concatenate([t.data for t in tensors], axis=axis), tuple(tensors),
                        lambda g: tuple(np.split(g, cuts, axis=axis)))


def where(cond, a, b) -> Tensor:
    """Pick from `a` where `cond` is True, else from `b`; `a` and `b` may be constants."""
    cond = np.asarray(cond.data if isinstance(cond, Tensor) else cond, dtype=bool)
    like = a if isinstance(a, Tensor) else b
    a, b = like._wrap(a), like._wrap(b)

    def backward(g):
        return (unbroadcast(np.where(cond, g, 0), a.shape), unbroadcast(np.where(cond, 0, g), b.shape))
    return Tensor._make(np.where(cond, a.data, b.data), (a, b), backward)
