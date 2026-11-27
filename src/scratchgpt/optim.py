"""Optimizers, a learning-rate schedule and gradient clipping. Updates are in place on .data."""
import math

import numpy as np

from .tensor import Tensor


class Optimizer:
    def __init__(self, params, lr: float):
        self.params: list[Tensor] = list(params)
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is not None:  # a parameter not used this step is left alone
                self._update(i, p, p.grad)

    def _update(self, i: int, p: Tensor, g: np.ndarray) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Plain SGD, optionally with (Nesterov) momentum and L2 weight decay, as torch.optim.SGD."""

    def __init__(self, params, lr: float, momentum: float = 0.0, nesterov: bool = False,
                 weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.momentum, self.nesterov, self.weight_decay = momentum, nesterov, weight_decay
        self.buf = [None] * len(self.params)

    def _update(self, i, p, g):
        if self.weight_decay:
            g = g + self.weight_decay * p.data
        if self.momentum:
            self.buf[i] = g.copy() if self.buf[i] is None else self.momentum * self.buf[i] + g
            g = g + self.momentum * self.buf[i] if self.nesterov else self.buf[i]
        p.data -= self.lr * g


class Adam(Optimizer):
    """Adam (Kingma and Ba). `weight_decay` here is L2 added to the gradient, as torch.optim.Adam."""

    decoupled = False

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps, self.weight_decay = eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = [0] * len(self.params)

    def _update(self, i, p, g):
        if self.weight_decay:
            if self.decoupled:
                p.data *= 1 - self.lr * self.weight_decay
            else:
                g = g + self.weight_decay * p.data
        self.t[i] += 1
        self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
        self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
        bias1 = 1 - self.b1 ** self.t[i]
        bias2 = 1 - self.b2 ** self.t[i]
        p.data -= (self.lr / bias1) * self.m[i] / (np.sqrt(self.v[i]) / math.sqrt(bias2) + self.eps)


class AdamW(Adam):
    """Adam with decoupled weight decay (Loshchilov and Hutter): the weights shrink directly."""

    decoupled = True

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        super().__init__(params, lr, betas, eps, weight_decay)


def cosine_lr(step: int, max_lr: float, warmup: int, total: int, min_lr: float = 0.0) -> float:
    """Linear warmup to `max_lr` over `warmup` steps, then cosine decay to `min_lr` at `total`."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def clip_grad_norm(params, max_norm: float) -> float:
    """Scale all gradients together so their global L2 norm is at most `max_norm`; return the norm."""
    grads = [p.grad for p in params if p.grad is not None]
    total = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads))
    scale = max_norm / (total + 1e-6)
    if scale < 1:
        for g in grads:
            g *= scale
    return total
