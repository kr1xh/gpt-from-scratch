"""Loss per byte of text, so models with different tokenizers can be compared.

Loss per token is not comparable: a token stands for one character in one tokenizer and for a
whole word in another. The total loss divided by the bytes of text predicted is. For ASCII text
such as TinyShakespeare, bits per byte is bits per character.
"""
import math

import numpy as np

from .tensor import no_grad


def bits_per_byte(model, tok, ids, batch_size: int = 16) -> dict:
    """Predict every token of the encoded text `ids` once, in windows of the model's context, and return the
    total loss in bits divided by the bytes of the predicted tokens. The first token is context
    only. Each window starts afresh, so its first predictions see little context."""
    ids = np.asarray(ids)
    block = model.config.block_size
    starts = list(range(0, len(ids) - 1, block))
    full = [s for s in starts if s + block + 1 <= len(ids)]
    ragged = [s for s in starts if s + block + 1 > len(ids)]  # at most one: the shorter last window
    nats, targets = 0.0, []
    with no_grad():
        for group, length in ((full, block), (ragged, len(ids) - 1 - (ragged[0] if ragged else 0))):
            for i in range(0, len(group), batch_size):
                rows = np.stack([ids[s:s + length + 1] for s in group[i:i + batch_size]])
                logits, _ = model(rows[:, :-1])
                z = logits.data.astype(np.float64)
                z = z - z.max(axis=-1, keepdims=True)
                log_probs = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
                y = rows[:, 1:]
                nats -= float(np.take_along_axis(log_probs, y[..., None], axis=-1).sum())
                targets.append(y.ravel())
    targets = np.concatenate(targets)
    n_bytes = int(tok.byte_lengths()[targets].sum())
    return {"bits_per_byte": nats / math.log(2) / n_bytes, "tokens": int(len(targets)), "bytes": n_bytes}
