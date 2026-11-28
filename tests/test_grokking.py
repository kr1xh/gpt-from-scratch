"""Modular addition: the data, a full-batch trainer, and the steps at which accuracy jumps."""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scratchgpt import grokking


@settings(max_examples=20, deadline=None)
@given(p=st.integers(2, 23), frac=st.floats(0.1, 0.9), seed=st.integers(0, 99))
def test_every_pair_lands_in_exactly_one_split_with_its_sum_mod_p(p, frac, seed):
    data = grokking.modular_addition(p, frac, seed)
    x = np.concatenate([data["train_x"], data["test_x"]])
    y = np.concatenate([data["train_y"], data["test_y"]])
    assert sorted(map(tuple, x[:, :2])) == [(a, b) for a in range(p) for b in range(p)]
    assert (x[:, 2] == p).all()  # the "=" token
    np.testing.assert_array_equal(y, (x[:, 0] + x[:, 1]) % p)
    assert len(data["train_x"]) == round(frac * p * p)


def test_the_split_depends_only_on_the_seed():
    a, b = grokking.modular_addition(11, 0.5, seed=3), grokking.modular_addition(11, 0.5, seed=3)
    np.testing.assert_array_equal(a["train_x"], b["train_x"])
    assert not np.array_equal(a["train_x"], grokking.modular_addition(11, 0.5, seed=4)["train_x"])


def test_first_step_finds_where_a_curve_reaches_a_threshold():
    history = [{"step": s, "test_acc": acc} for s, acc in ((0, 0.0), (10, 0.5), (20, 0.995), (30, 1.0))]
    assert grokking.first_step(history, "test_acc", 0.99) == 20
    assert grokking.first_step(history, "test_acc", 0.5) == 10  # reaching the threshold is enough
    assert grokking.first_step(history, "test_acc", 1.01) is None


def test_full_batch_training_memorises_the_training_pairs():
    # lr 3e-3: at 1e-2 the loss spikes and settles with a few pairs still wrong
    history = grokking.train(p=7, frac=0.8, steps=300, lr=3e-3, weight_decay=0.0, d_model=32, n_head=2,
                             eval_every=50, seed=0)
    assert [h["step"] for h in history] == [0, 50, 100, 150, 200, 250, 299]
    assert history[0]["train_acc"] < 0.5 and history[-1]["train_acc"] == 1.0
    for h in history:
        assert 0 <= h["test_acc"] <= 1 and h["train_loss"] > 0 and np.isfinite(h["test_loss"])


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_the_same_seed_gives_the_same_run(optimizer):
    run = dict(p=5, frac=0.6, steps=5, lr=1e-2, weight_decay=1.0, d_model=16, n_head=2, eval_every=2,
               seed=1, optimizer=optimizer)
    assert grokking.train(**run) == grokking.train(**run)
