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


def forward(P: dict, cfg, idx, targets=None):
    B, T = idx.shape
    C, H = cfg.d_model, cfg.n_head
    KV = cfg.n_kv_head or H
    D = C // H

    def norm(x, name):
        if cfg.norm == "rms":
            return F.rms_norm(x, (C,), P[f"{name}.weight"], eps=1e-5)
        return F.layer_norm(x, (C,), P[f"{name}.weight"], P[f"{name}.bias"])

    x = P["wte.weight"][idx]
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
        if cfg.mlp == "swiglu":
            h = F.silu(h @ p("mlp.gate.weight") + p("mlp.gate.bias")) * (h @ p("mlp.up.weight") + p("mlp.up.bias"))
            x = x + h @ p("mlp.down.weight") + p("mlp.down.bias")
        else:
            h = F.gelu(h @ p("mlp.layers.0.weight") + p("mlp.layers.0.bias"), approximate="tanh")
            x = x + h @ p("mlp.layers.1.weight") + p("mlp.layers.1.bias")
    x = norm(x, "ln_f")
    logits = x @ P["wte.weight"].T  # the output head is tied to the token embedding
    if targets is None:
        return logits, None
    return logits, F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def parameters(state: dict, dtype=torch.float32) -> dict:
    """Leaf tensors (requiring grad) copied from a scratchgpt state_dict."""
    return {k: torch.tensor(v, dtype=dtype).requires_grad_() for k, v in state.items()}
