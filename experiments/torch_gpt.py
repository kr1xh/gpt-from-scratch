"""The same GPT written with PyTorch's own layer and RMS norm, attention (with its grouped-query
mode), GELU, SiLU and cross-entropy. Rotary embeddings are written with complex numbers.

It reads parameters by this repository's names, so a scratchgpt state_dict drives it directly.
It is the reference the tests compare against and the opponent in the engine-vs-PyTorch study.
"""
import torch
import torch.nn.functional as F


def rope(x, offset=0, base=10000.0):
    """Rotate the pair (x[i], x[i + D/2]) at position m by multiplying it, as a complex number,
    by e^(i * m * base^(-2i/D))."""
    T, D = x.shape[-2:]
    freqs = base ** (-torch.arange(0, D, 2, dtype=torch.float64) / D)
    angles = torch.outer(torch.arange(offset, offset + T, dtype=torch.float64), freqs)
    z = torch.complex(x[..., :D // 2], x[..., D // 2:]) * torch.polar(torch.ones_like(angles), angles).to(
        torch.complex64 if x.dtype == torch.float32 else torch.complex128)
    return torch.cat([z.real, z.imag], dim=-1)


def mlp(P, cfg, h, name):
    if cfg.mlp == "swiglu":
        g = F.silu(h @ P[f"{name}.gate.weight"] + P[f"{name}.gate.bias"]) * (h @ P[f"{name}.up.weight"] + P[f"{name}.up.bias"])
        return g @ P[f"{name}.down.weight"] + P[f"{name}.down.bias"]
    g = F.gelu(h @ P[f"{name}.layers.0.weight"] + P[f"{name}.layers.0.bias"], approximate="tanh")
    return g @ P[f"{name}.layers.1.weight"] + P[f"{name}.layers.1.bias"]


def moe(P, cfg, h, name):
    """Top-k routing with torch.topk, gates by softmax over the chosen scores, experts gathered
    and added back with index_add; returns the output and the Switch load-balancing loss."""
    flat = h.reshape(-1, h.shape[-1])
    scores = flat @ P[f"{name}.router.weight"]
    top_scores, chosen = scores.topk(cfg.top_k, dim=-1)
    gates = F.softmax(top_scores, dim=-1)
    out = torch.zeros_like(flat)
    for e in range(cfg.n_expert):
        tok, slot = torch.nonzero(chosen == e, as_tuple=True)
        out = out.index_add(0, tok, mlp(P, cfg, flat[tok], f"{name}.experts.{e}") * gates[tok, slot, None])
    share = torch.bincount(chosen.flatten(), minlength=cfg.n_expert).to(scores.dtype) / chosen.numel()
    aux = cfg.n_expert * (F.softmax(scores, dim=-1).mean(0) * share).sum()
    return out.reshape(h.shape), aux


def forward(P: dict, cfg, idx, targets=None, with_aux=False):
    """Returns (logits, loss), or (logits, loss, summed load-balancing loss) with with_aux."""
    B, T = idx.shape
    C, H = cfg.d_model, cfg.n_head
    KV = cfg.n_kv_head or H
    D = C // H

    def norm(x, name):
        if cfg.norm == "rms":
            return F.rms_norm(x, (C,), P[f"{name}.weight"], eps=1e-5)
        return F.layer_norm(x, (C,), P[f"{name}.weight"], P[f"{name}.bias"])

    x = P["wte.weight"][idx]
    aux = None
    if cfg.pos == "learned":
        x = x + P["wpe.weight"][:T]
    for i in range(cfg.n_layer):
        def p(name):
            return P[f"blocks.{i}.{name}"]
        h = norm(x, f"blocks.{i}.ln1")
        q, k, v = (h @ p("attn.qkv.weight") + p("attn.qkv.bias")).split([C, KV * D, KV * D], dim=-1)
        q = q.view(B, T, H, D).transpose(1, 2)
        k, v = (t.view(B, T, KV, D).transpose(1, 2) for t in (k, v))
        if cfg.pos == "rope":
            q, k = rope(q), rope(k)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=KV < H)
        x = x + a.transpose(1, 2).reshape(B, T, C) @ p("attn.proj.weight") + p("attn.proj.bias")
        h = norm(x, f"blocks.{i}.ln2")
        if cfg.n_expert:
            h, layer_aux = moe(P, cfg, h, f"blocks.{i}.mlp")
            aux = layer_aux if aux is None else aux + layer_aux
            x = x + h
        else:
            x = x + mlp(P, cfg, h, f"blocks.{i}.mlp")
    x = norm(x, "ln_f")
    logits = x @ P["wte.weight"].T  # the output head is tied to the token embedding
    loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return (logits, loss, aux) if with_aux else (logits, loss)


def parameters(state: dict, dtype=torch.float32) -> dict:
    """Leaf tensors (requiring grad) copied from a scratchgpt state_dict."""
    return {k: torch.tensor(v, dtype=dtype).requires_grad_() for k, v in state.items()}
