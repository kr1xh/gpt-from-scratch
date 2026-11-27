"""Functions built from Tensor ops: softmax, cross-entropy, layer and RMS norm, rotary embeddings."""
import numpy as np

from .tensor import Tensor, concat


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    # Subtracting the max keeps exp() from overflowing. It is a constant here: log_softmax does not
    # change when every input shifts by the same amount, so its gradient through the max is zero.
    shifted = x - x.data.max(axis=axis, keepdims=True)
    return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return log_softmax(x, axis=axis).exp()


def cross_entropy(logits: Tensor, targets) -> Tensor:
    """Mean negative log-likelihood of integer `targets`; logits are (..., classes)."""
    classes = logits.shape[-1]
    flat = logits.reshape(-1, classes)
    targets = np.asarray(targets).reshape(-1)
    picked = log_softmax(flat, axis=-1)[np.arange(len(targets)), targets]
    return -picked.mean()


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    centred = x - x.mean(axis=-1, keepdims=True)
    var = (centred * centred).mean(axis=-1, keepdims=True)
    return centred / (var + eps).sqrt() * weight + bias


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """Layer norm without the centring and the bias: scale x to unit root-mean-square (Zhang & Sennrich 2019)."""
    return x / ((x * x).mean(axis=-1, keepdims=True) + eps).sqrt() * weight


def rope(x: Tensor, offset: int = 0, base: float = 10000.0) -> Tensor:
    """Rotary position embedding (Su et al. 2021) of x (..., T, D), whose rows sit at positions
    offset, offset + 1, ... The pair (x[i], x[i + D/2]) is rotated by the angle position * base^(-2i/D),
    so the dot product of two rotated vectors depends on how far apart they are, not where they are."""
    T, D = x.shape[-2:]
    if D % 2:
        raise ValueError(f"rotary embeddings need an even head size, got {D}")
    angles = np.outer(np.arange(offset, offset + T), base ** (-np.arange(0, D, 2) / D)).astype(x.data.dtype)
    cos, sin = np.cos(angles), np.sin(angles)
    x1, x2 = x[..., :D // 2], x[..., D // 2:]
    return concat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
