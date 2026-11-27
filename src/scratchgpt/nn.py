"""Layers, up to a GPT-2-style GPT. A Module finds its parameters through its attributes and lists."""
from dataclasses import dataclass

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

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state) -> None:
        params = dict(self.named_parameters())
        missing, unknown = sorted(set(params) - set(state)), sorted(set(state) - set(params))
        if missing:
            raise ValueError(f"state has no value for {', '.join(missing)}")
        if unknown:  # e.g. a deeper model's checkpoint: refuse rather than drop layers silently
            raise ValueError(f"model has no parameter {', '.join(unknown)}")
        for name, p in params.items():
            if state[name].shape != p.shape:
                raise ValueError(f"{name}: shape {state[name].shape} does not match {p.shape}")
            p.data = np.array(state[name], dtype=p.data.dtype)


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


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128  # longest context the model can read
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128


class CausalSelfAttention(Module):
    """Multi-head attention where position t can only look at positions <= t."""

    def __init__(self, d_model: int, n_head: int, rng=None):
        if d_model % n_head:
            raise ValueError(f"d_model={d_model} is not divisible by n_head={n_head}")
        self.n_head = n_head
        self.qkv = Linear(d_model, 3 * d_model, rng=rng)
        self.proj = Linear(d_model, d_model, rng=rng)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        H = self.n_head
        # one matmul makes q, k and v; split them into heads: (3, B, H, T, C // H)
        qkv = self.qkv(x).reshape(B, T, 3, H, C // H).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.T) * (1.0 / np.sqrt(C // H))
        future = np.triu(np.ones((T, T), dtype=bool), k=1)
        weights = F.softmax(scores.masked_fill(future, -1e9), axis=-1)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(out)


class Block(Module):
    """Pre-norm transformer block (GPT-2): x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, cfg: GPTConfig, rng=None):
        self.ln1 = LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, rng=rng)
        self.ln2 = LayerNorm(cfg.d_model)
        self.mlp = MLP([cfg.d_model, 4 * cfg.d_model, cfg.d_model], activation="gelu", rng=rng)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(Module):
    """GPT-2: token + position embeddings, n_layer blocks, a final norm and a head tied to the
    token embedding. Called with token ids (B, T); returns (logits, loss or None)."""

    def __init__(self, cfg: GPTConfig, rng=None):
        rng = rng or np.random.default_rng()
        self.config = cfg
        self.wte = Embedding(cfg.vocab_size, cfg.d_model, rng=rng)
        self.wpe = Embedding(cfg.block_size, cfg.d_model, rng=rng)
        self.blocks = [Block(cfg, rng=rng) for _ in range(cfg.n_layer)]
        self.ln_f = LayerNorm(cfg.d_model)
        # GPT-2 init: N(0, 0.02) weights and zero biases; the projections that add into the
        # residual stream get 0.02 / sqrt(2 * n_layer), so the stream does not grow with depth
        residual_std = 0.02 / np.sqrt(2 * cfg.n_layer)
        for block in self.blocks:
            for layer in (block.attn.qkv, block.attn.proj, *block.mlp.layers):
                std = residual_std if layer in (block.attn.proj, block.mlp.layers[-1]) else 0.02
                layer.weight.data = rng.normal(0, std, layer.weight.shape).astype(np.float32)
                layer.bias.data = np.zeros_like(layer.bias.data)

    def forward(self, idx, targets=None):
        idx = np.asarray(idx)
        T = idx.shape[1]
        if T > self.config.block_size:
            raise ValueError(f"sequence of {T} tokens is longer than block_size={self.config.block_size}")
        x = self.wte(idx) + self.wpe.weight[:T]
        for block in self.blocks:
            x = block(x)
        logits = self.ln_f(x) @ self.wte.weight.T
        loss = F.cross_entropy(logits, targets) if targets is not None else None
        return logits, loss
