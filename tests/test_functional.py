import numpy as np
import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from gradcheck import RNG, check, rand
from experiments.torch_gpt import rope as torch_rope
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


def test_rms_norm():
    x, w = rand(2, 3, 4), rand(4)
    check(lambda t, w_: fn.rms_norm(t, w_), lambda t, w_: F.rms_norm(t, (4,), w_, eps=1e-5), x, w)


def test_rope_matches_a_complex_number_rotation():
    check(lambda t: fn.rope(t), lambda t: torch_rope(t), rand(2, 2, 5, 6))
    check(lambda t: fn.rope(t, offset=7), lambda t: torch_rope(t, offset=7), rand(1, 3, 2, 4))


@settings(max_examples=40, deadline=None)
@given(m=st.integers(0, 60), n=st.integers(0, 60), shift=st.integers(0, 60), seed=st.integers(0, 99))
def test_rotated_scores_depend_only_on_how_far_apart_two_positions_are(m, n, shift, seed):
    q, k = np.random.default_rng(seed).standard_normal((2, 1, 1, 1, 8))

    def score(a, b):
        return float((fn.rope(Tensor(q), offset=a).data * fn.rope(Tensor(k), offset=b).data).sum())
    assert score(m, n) == pytest.approx(score(m + shift, n + shift), rel=1e-9, abs=1e-12)


def test_rope_keeps_each_vector_s_length():
    x = rand(2, 2, 9, 6)
    np.testing.assert_allclose(np.linalg.norm(fn.rope(Tensor(x)).data, axis=-1), np.linalg.norm(x, axis=-1))
