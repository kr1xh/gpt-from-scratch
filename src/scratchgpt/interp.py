"""Looking inside a trained GPT: attention maps, the logit lens, induction-head scores, ablation.

Induction heads (Olsson et al. 2022) complete patterns of the form [A][B] ... [A] -> [B]: from
the current token they attend to the token that followed its previous occurrence. On a random
sequence repeated twice, the second half is predictable only that way, so a head's average
attention to "the token after the earlier copy" (its prefix-matching score) finds them, and
ablating them should hurt the second half of such sequences far more than ordinary text.
"""
from contextlib import contextmanager

import numpy as np

from . import functional as F
from .tensor import no_grad


def attention_maps(model, idx) -> list:
    """The attention weights of every layer, each (B, n_head, T, T)."""
    attns = [block.attn for block in model.blocks]
    for attn in attns:
        attn.record = True
    try:
        with no_grad():
            model(np.asarray(idx))
        return [attn.last_weights for attn in attns]
    finally:
        for attn in attns:
            attn.record, attn.last_weights = False, None


@contextmanager
def ablated(model, heads):
    """Within the block, the given (layer, head) pairs output zeros instead of their mix of values."""
    for layer, head in heads:
        attn = model.blocks[layer].attn
        attn.ablated_heads = tuple(attn.ablated_heads) + (head,)
    try:
        yield model
    finally:
        for block in model.blocks:
            block.attn.ablated_heads = ()


def logit_lens(model, idx) -> list:
    """Log-probabilities of the next token read straight off the residual stream (through the final
    norm and the tied head) after the embedding and after each block (nostalgebraist 2020)."""
    idx = np.asarray(idx)
    lens = []
    with no_grad():
        x = model.wte(idx)
        if model.wpe is not None:
            x = x + model.wpe.weight[:idx.shape[1]]
        for i in range(len(model.blocks) + 1):
            if i:
                x = model.blocks[i - 1](x)
            lens.append(F.log_softmax(model.ln_f(x) @ model.wte.weight.T).data.astype(np.float64))
    return lens


def repeated_random_tokens(vocab_size: int, n: int, batch: int, seed: int = 0) -> np.ndarray:
    """`batch` rows of n uniformly random tokens followed by the same n tokens again."""
    first = np.random.default_rng(seed).integers(0, vocab_size, (batch, n))
    return np.concatenate([first, first], axis=1)


def prefix_matching_scores(maps: list, n: int) -> np.ndarray:
    """(layers, heads): mean attention, over the second half of sequences repeated after n tokens,
    from position t to position t - n + 1, the token that followed the earlier copy of token t."""
    t = np.arange(n, 2 * n)
    return np.stack([m[:, :, t, t - n + 1].mean(axis=(0, 2)) for m in maps])


def mean_loss(model, idx, start: int = 0) -> float:
    """Mean next-token cross-entropy of the predictions made at positions start, start + 1, ..."""
    idx = np.asarray(idx)
    with no_grad():
        logits, _ = model(idx[:, :-1])
        return F.cross_entropy(logits[:, start:], idx[:, start + 1:]).item()
