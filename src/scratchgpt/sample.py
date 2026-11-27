"""Generate text from a trained checkpoint: python -m scratchgpt.sample --checkpoint models/baseline.npz"""
import argparse
import json
from pathlib import Path

import numpy as np

from .nn import GPT, GPTConfig
from .tensor import no_grad
from .tokenizers import CharTokenizer


def generate(model: GPT, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None,
             rng=None) -> np.ndarray:
    """Append `max_new_tokens` sampled tokens to each row of `idx` (B, T)."""
    if temperature <= 0:
        raise ValueError("temperature must be positive (use top_k=1 for greedy decoding)")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    rng = rng or np.random.default_rng()
    idx = np.asarray(idx)
    if idx.shape[1] == 0:
        raise ValueError("the prompt needs at least one token")
    # ponytail: re-reads the whole context for every new token. Upgrade path: the KV cache (#8).
    with no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(idx[:, -model.config.block_size:])
            logits = logits.data[:, -1].astype(np.float64) / temperature
            if top_k is not None and top_k < logits.shape[-1]:
                kth = np.sort(logits, axis=-1)[:, -top_k][:, None]
                logits = np.where(logits < kth, -np.inf, logits)
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            nxt = np.array([rng.choice(len(p), p=p) for p in probs])
            idx = np.concatenate([idx, nxt[:, None]], axis=1)
    return idx


def load(checkpoint):
    """A checkpoint is models/<name>.npz (weights + config) next to <name>.tokenizer.json."""
    checkpoint = Path(checkpoint)
    with np.load(checkpoint) as f:
        state = {k: f[k] for k in f.files if k != "__config__"}
        cfg = GPTConfig(**json.loads(str(f["__config__"])))
    model = GPT(cfg)
    model.load_state_dict(state)
    return model, CharTokenizer.load(checkpoint.with_suffix(".tokenizer.json"))


def main(argv=None) -> str:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="models/baseline.npz")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    model, tok = load(args.checkpoint)
    out = generate(model, np.array([tok.encode(args.prompt)]), args.tokens, args.temperature, args.top_k,
                   np.random.default_rng(args.seed))
    text = tok.decode(out[0].tolist())
    print(text)
    return text


if __name__ == "__main__":
    main()
