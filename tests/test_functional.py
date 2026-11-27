import numpy as np
import torch
import torch.nn.functional as F

from gradcheck import RNG, check, rand
from scratchgpt import functional as fn
from scratchgpt.tensor import Tensor


def test_log_softmax_and_softmax():
    x = rand(3, 5, low=-3, high=3)
    check(lambda t: fn.log_softmax(t, axis=-1), lambda t: F.log_softmax(t, dim=-1), x)
    check(lambda t: fn.softmax(t, axis=-1), lambda t: F.softmax(t, dim=-1), x)
    check(lambda t: fn.softmax(t, axis=0), lambda t: F.softmax(t, dim=0), x)


def test_softmax_is_stable_for_huge_logits():
    out = fn.softmax(Tensor(np.array([[1000.0, 1000.0, -1000.0]])), axis=-1).data
    np.testing.assert_allclose(out, [[0.5, 0.5, 0.0]])


def test_cross_entropy_on_2d_and_3d_logits():
    targets = RNG.integers(0, 5, size=4)
    check(lambda t: fn.cross_entropy(t, targets),
          lambda t: F.cross_entropy(t, torch.tensor(targets)), rand(4, 5, low=-2, high=2))
    targets3 = RNG.integers(0, 6, size=(2, 3))  # (batch, time) like a language model
    check(lambda t: fn.cross_entropy(t, targets3),
          lambda t: F.cross_entropy(t.reshape(-1, 6), torch.tensor(targets3).reshape(-1)),
          rand(2, 3, 6, low=-2, high=2))


def test_layer_norm():
    x, w, b = rand(2, 3, 4), rand(4), rand(4)
    check(lambda t, w_, b_: fn.layer_norm(t, w_, b_),
          lambda t, w_, b_: F.layer_norm(t, (4,), w_, b_, eps=1e-5), x, w, b)
