"""The same GPT-2 written with PyTorch's own layer norm, attention, GELU and cross-entropy.

It reads parameters by this repository's names, so a scratchgpt state_dict drives it directly.
It is the reference the tests compare against and the opponent in the engine-vs-PyTorch study.
"""
import torch
import torch.nn.functional as F


def forward(P: dict, cfg, idx, targets=None):
    B, T = idx.shape
    C, H = cfg.d_model, cfg.n_head
    x = P["wte.weight"][idx] + P["wpe.weight"][:T]
    for i in range(cfg.n_layer):
        def p(name):
            return P[f"blocks.{i}.{name}"]
        h = F.layer_norm(x, (C,), p("ln1.weight"), p("ln1.bias"))
        q, k, v = (h @ p("attn.qkv.weight") + p("attn.qkv.bias")).split(C, dim=-1)
        q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + a.transpose(1, 2).reshape(B, T, C) @ p("attn.proj.weight") + p("attn.proj.bias")
        h = F.layer_norm(x, (C,), p("ln2.weight"), p("ln2.bias"))
        h = F.gelu(h @ p("mlp.layers.0.weight") + p("mlp.layers.0.bias"), approximate="tanh")
        x = x + h @ p("mlp.layers.1.weight") + p("mlp.layers.1.bias")
    x = F.layer_norm(x, (C,), P["ln_f.weight"], P["ln_f.bias"])
    logits = x @ P["wte.weight"].T  # the output head is tied to the token embedding
    if targets is None:
        return logits, None
    return logits, F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def parameters(state: dict, dtype=torch.float32) -> dict:
    """Leaf tensors (requiring grad) copied from a scratchgpt state_dict."""
    return {k: torch.tensor(v, dtype=dtype).requires_grad_() for k, v in state.items()}
