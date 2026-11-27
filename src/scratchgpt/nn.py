"""Layers. A Module finds its parameters by looking through its attributes, lists included."""
import numpy as np

from . import functional as F
from .tensor import Tensor


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def named_parameters(self, prefix: str = ""):
        seen = set()
        for name, value in self._walk(prefix):
            if id(value) not in seen:  # a tied weight is listed once
                seen.add(id(value))
                yield name, value

    def _walk(self, prefix):
        for name, value in vars(self).items():
            full = f"{prefix}{name}"
            items = list(enumerate(value)) if isinstance(value, (list, tuple)) else [(None, value)]
            for i, item in items:
                key = full if i is None else f"{full}.{i}"
                if isinstance(item, Tensor) and item.requires_grad:
                    yield key, item
                elif isinstance(item, Module):
                    yield from item._walk(f"{key}.")

    def parameters(self) -> list[Tensor]:
        return [p for _, p in self.named_parameters()]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None


class Linear(Module):
    """y = x @ W + b, with W stored as (in, out). Initialised like torch: U(-1/sqrt(in), 1/sqrt(in))."""

    def __init__(self, n_in: int, n_out: int, bias: bool = True, rng=None):
        rng = rng or np.random.default_rng()
        bound = 1 / np.sqrt(n_in)
        self.weight = Tensor(rng.uniform(-bound, bound, (n_in, n_out)), requires_grad=True, dtype=np.float32)
        self.bias = Tensor(rng.uniform(-bound, bound, n_out), requires_grad=True, dtype=np.float32) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weight
        return y + self.bias if self.bias is not None else y


class Embedding(Module):
    def __init__(self, num: int, dim: int, std: float = 0.02, rng=None):
        rng = rng or np.random.default_rng()
        self.weight = Tensor(rng.normal(0, std, (num, dim)), requires_grad=True, dtype=np.float32)

    def forward(self, idx) -> Tensor:
        return self.weight[np.asarray(idx)]


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = Tensor(np.ones(dim), requires_grad=True, dtype=np.float32)
        self.bias = Tensor(np.zeros(dim), requires_grad=True, dtype=np.float32)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.weight, self.bias, self.eps)


class MLP(Module):
    """Linear layers of the given sizes with an activation between them (none after the last)."""

    def __init__(self, sizes: list[int], activation: str = "relu", rng=None):
        rng = rng or np.random.default_rng()
        self.layers = [Linear(a, b, rng=rng) for a, b in zip(sizes, sizes[1:])]
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = getattr(x, self.activation)()
        return x
