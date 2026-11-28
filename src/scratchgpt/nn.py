"""Layers, up to a GPT that is GPT-2 by default and Llama-style with the modern options.
A Module finds its parameters through its attributes and lists."""
from dataclasses import dataclass

import numpy as np

from . import functional as F
from .tensor import Tensor, concat, scatter_add


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


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = Tensor(np.ones(dim), requires_grad=True, dtype=np.float32)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, self.weight, self.eps)


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


class SwiGLU(Module):
    """down(silu(gate(x)) * up(x)) (Shazeer 2020): a gated MLP with three weight matrices."""

    def __init__(self, dim: int, hidden: int, rng=None):
        self.gate = Linear(dim, hidden, rng=rng)
        self.up = Linear(dim, hidden, rng=rng)
        self.down = Linear(hidden, dim, rng=rng)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.gate(x).silu() * self.up(x))


OPTIONS = {"norm": ("layer", "rms"), "mlp": ("gelu", "swiglu"), "pos": ("learned", "rope")}


@dataclass
class GPTConfig:
    """The defaults are GPT-2. The modern block (as in Llama) is norm="rms", mlp="swiglu",
    pos="rope" and n_kv_head < n_head; each can also be switched on alone."""
    vocab_size: int
    block_size: int = 128  # longest context the model can read
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    norm: str = "layer"  # "layer": LayerNorm; "rms": RMSNorm
    mlp: str = "gelu"  # "gelu": a 4d GELU MLP; "swiglu": SwiGLU of hidden size 8d/3, about as big
    pos: str = "learned"  # "learned": a position embedding table; "rope": rotary embeddings in attention
    n_kv_head: int | None = None  # fewer key/value heads than query heads: grouped-query attention
    mlp_hidden: int | None = None  # the MLP's inner width; default 4d (gelu) or 8d/3 (swiglu)
    n_expert: int = 0  # 0: one dense MLP per block; otherwise a mixture of n_expert MLPs of the kind above
    top_k: int = 2  # experts each token is sent to

    def __post_init__(self):
        for name, allowed in OPTIONS.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name}={getattr(self, name)!r}: use one of {', '.join(allowed)}")
        if self.n_expert and not (self.n_expert >= 2 and 1 <= self.top_k <= self.n_expert):
            raise ValueError(f"n_expert={self.n_expert}, top_k={self.top_k}: need 2+ experts and 1 <= top_k <= n_expert")
        kv = self.n_head if self.n_kv_head is None else self.n_kv_head
        if kv < 1 or self.n_head % kv:
            raise ValueError(f"n_kv_head={self.n_kv_head} must divide n_head={self.n_head}")


class KVCache:
    """The keys and values of every token seen so far, per layer, so that sampling can feed the
    model only the newest token. Pass the same cache to every call: model(idx, cache=cache)."""

    def __init__(self):
        self.layers = []  # (keys, values) per layer, each (B, n_kv_head, length, head size)
        self.length = 0

    def extend(self, layer: int, k: Tensor, v: Tensor):
        if layer == len(self.layers):
            self.layers.append((k, v))
        else:
            old_k, old_v = self.layers[layer]
            self.layers[layer] = (concat([old_k, k], axis=2), concat([old_v, v], axis=2))
        return self.layers[layer]


class CausalSelfAttention(Module):
    """Multi-head attention where position t can only look at positions <= t. With n_kv_head < n_head,
    groups of query heads share one key/value head (grouped-query attention, Ainslie et al. 2023)."""

    def __init__(self, d_model: int, n_head: int, n_kv_head: int | None = None, rope: bool = False, rng=None):
        if d_model % n_head:
            raise ValueError(f"d_model={d_model} is not divisible by n_head={n_head}")
        self.n_head, self.n_kv_head, self.rope = n_head, n_kv_head or n_head, rope
        # inspection hooks (scratchgpt.interp): keep the last attention weights, zero chosen heads
        self.record, self.last_weights, self.ablated_heads = False, None, ()
        self.qkv = Linear(d_model, d_model + 2 * self.n_kv_head * (d_model // n_head), rng=rng)
        self.proj = Linear(d_model, d_model, rng=rng)

    def forward(self, x: Tensor, cache: KVCache | None = None, layer: int = 0) -> Tensor:
        B, T, C = x.shape
        H, KV = self.n_head, self.n_kv_head
        # one matmul makes q, k and v; split into heads: (B, H + 2 KV, T, head size)
        qkv = self.qkv(x).reshape(B, T, H + 2 * KV, C // H).transpose(0, 2, 1, 3)
        q, k, v = qkv[:, :H], qkv[:, H:H + KV], qkv[:, H + KV:]
        if self.rope:
            start = cache.length if cache is not None else 0  # position of the first new token
            q, k = F.rope(q, start), F.rope(k, start)
        if cache is not None:
            k, v = cache.extend(layer, k, v)  # the cache keeps KV heads, not H: GQA's memory saving
        if KV < H:
            k, v = k.repeat(H // KV, axis=1), v.repeat(H // KV, axis=1)
        scores = (q @ k.T) * (1.0 / np.sqrt(C // H))
        Tk = k.shape[2]  # the new queries are the last T of the Tk positions
        future = np.arange(Tk)[None, :] > np.arange(Tk - T, Tk)[:, None]
        weights = F.softmax(scores.masked_fill(future, -1e9), axis=-1)
        if self.record:
            self.last_weights = weights.data
        heads = weights @ v  # (B, H, T, head size)
        if self.ablated_heads:
            keep = np.ones((1, H, 1, 1), dtype=heads.data.dtype)
            keep[:, list(self.ablated_heads)] = 0
            heads = heads * keep
        out = heads.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(out)


def make_mlp(cfg: GPTConfig, rng=None) -> Module:
    d = cfg.d_model
    if cfg.mlp == "swiglu":  # 3 matrices of d x 8d/3 hold as much as the 2 of d x 4d in the GELU MLP
        return SwiGLU(d, cfg.mlp_hidden or 8 * d // 3, rng=rng)
    return MLP([d, cfg.mlp_hidden or 4 * d, d], activation="gelu", rng=rng)


class MoE(Module):
    """A router sends each token to its top_k of n_expert MLPs, and the token gets their outputs
    weighted by a softmax over the chosen router scores (Shazeer et al. 2017; Mixtral). Each expert
    runs only on the tokens sent to it. After a forward, stats holds the Switch Transformer
    load-balancing loss (Fedus et al. 2021) and each expert's share of the assignments."""

    def __init__(self, cfg: GPTConfig, rng=None):
        self.router = Linear(cfg.d_model, cfg.n_expert, bias=False, rng=rng)
        self.experts = [make_mlp(cfg, rng) for _ in range(cfg.n_expert)]
        self.top_k = cfg.top_k
        self.stats = {}  # a dict, so the aux loss in it is not mistaken for a parameter

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        N, E = B * T, len(self.experts)
        flat = x.reshape(N, C)
        scores = self.router(flat)  # (N, E)
        chosen = np.argsort(-scores.data, axis=-1, kind="stable")[:, :self.top_k]  # (N, k), best first
        gates = F.softmax(scores[np.arange(N)[:, None], chosen], axis=-1)  # (N, k)
        outputs, rows = [], []
        for e, expert in enumerate(self.experts):
            tok, slot = np.nonzero(chosen == e)
            if tok.size:
                outputs.append(expert(flat[tok]) * gates[tok, slot].reshape(-1, 1))
                rows.append(tok)
        out = scatter_add(np.concatenate(rows), concat(outputs, axis=0), N)
        # E * sum_e share_e * mean router probability_e: 1 when both are uniform, E when one expert takes
        # everything. The share is a count (no gradient); the probabilities carry it to the router.
        share = np.bincount(chosen.ravel(), minlength=E) / chosen.size
        self.stats = {"aux_loss": (F.softmax(scores, axis=-1).mean(axis=0) * share).sum() * E, "load": share}
        return out.reshape(B, T, C)


def norm(cfg: GPTConfig) -> Module:
    return RMSNorm(cfg.d_model) if cfg.norm == "rms" else LayerNorm(cfg.d_model)


class Block(Module):
    """Pre-norm transformer block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, cfg: GPTConfig, rng=None):
        d = cfg.d_model
        self.ln1 = norm(cfg)
        self.attn = CausalSelfAttention(d, cfg.n_head, cfg.n_kv_head, rope=cfg.pos == "rope", rng=rng)
        self.ln2 = norm(cfg)
        self.mlp = MoE(cfg, rng=rng) if cfg.n_expert else make_mlp(cfg, rng=rng)

    def forward(self, x: Tensor, cache: KVCache | None = None, layer: int = 0) -> Tensor:
        x = x + self.attn(self.ln1(x), cache, layer)
        return x + self.mlp(self.ln2(x))


class GPT(Module):
    """Token embeddings (plus learned position embeddings unless pos="rope"), n_layer blocks, a final
    norm and a head tied to the token embedding. Called with token ids (B, T); returns (logits, loss
    or None). With a KVCache, idx holds only the tokens that follow the cached ones."""

    def __init__(self, cfg: GPTConfig, rng=None):
        rng = rng or np.random.default_rng()
        self.config = cfg
        self.wte = Embedding(cfg.vocab_size, cfg.d_model, rng=rng)
        self.wpe = Embedding(cfg.block_size, cfg.d_model, rng=rng) if cfg.pos == "learned" else None
        self.blocks = [Block(cfg, rng=rng) for _ in range(cfg.n_layer)]
        self.ln_f = norm(cfg)
        # GPT-2 init: N(0, 0.02) weights and zero biases; the projections that add into the
        # residual stream get 0.02 / sqrt(2 * n_layer), so the stream does not grow with depth
        residual_std = 0.02 / np.sqrt(2 * cfg.n_layer)
        for block in self.blocks:
            mlps = block.mlp.experts if isinstance(block.mlp, MoE) else [block.mlp]
            layers, into_residual = [block.attn.qkv, block.attn.proj], [block.attn.proj]
            for mlp in mlps:
                own = mlp.layers if isinstance(mlp, MLP) else [mlp.gate, mlp.up, mlp.down]
                layers += own
                into_residual.append(own[-1])
            if isinstance(block.mlp, MoE):
                layers.append(block.mlp.router)
            for layer in layers:
                std = residual_std if any(layer is r for r in into_residual) else 0.02
                layer.weight.data = rng.normal(0, std, layer.weight.shape).astype(np.float32)
                if layer.bias is not None:
                    layer.bias.data = np.zeros_like(layer.bias.data)

    def forward(self, idx, targets=None, cache: KVCache | None = None):
        idx = np.asarray(idx)
        start = cache.length if cache is not None else 0
        end = start + idx.shape[1]
        if end > self.config.block_size:
            raise ValueError(f"sequence of {end} tokens is longer than block_size={self.config.block_size}")
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe.weight[start:end]
        for i, block in enumerate(self.blocks):
            x = block(x, cache, i)
        if cache is not None:
            cache.length = end
        logits = self.ln_f(x) @ self.wte.weight.T
        loss = F.cross_entropy(logits, targets) if targets is not None else None
        return logits, loss

    def _moe_layers(self):
        return [b.mlp for b in self.blocks if isinstance(b.mlp, MoE)]

    def aux_loss(self):
        """The load-balancing loss of the last forward, summed over MoE layers; None for a dense
        model. Training adds it, scaled, to the cross-entropy that forward returns."""
        layers = self._moe_layers()
        if not layers:
            return None
        total = layers[0].stats["aux_loss"]
        for layer in layers[1:]:
            total = total + layer.stats["aux_loss"]
        return total

    def expert_load(self) -> list:
        """Each MoE layer's share of the last forward's assignments, per expert."""
        return [layer.stats["load"] for layer in self._moe_layers()]
