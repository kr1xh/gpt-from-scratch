"""Functions built from Tensor ops: softmax, cross-entropy and layer norm."""
import numpy as np

from .tensor import Tensor


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
