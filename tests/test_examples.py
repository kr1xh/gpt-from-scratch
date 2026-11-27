"""The math-path warm-ups must actually learn their toy problems."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import logreg  # noqa: E402
import mlp  # noqa: E402


def test_logistic_regression_separates_two_blobs():
    assert logreg.train(seed=0) > 0.95


def test_mlp_learns_two_moons():
    assert mlp.train(seed=0) > 0.95


def test_logistic_regression_stays_finite_when_trained_long():
    # a confident model drives sigmoid to exactly 1.0 in float32; log(1 - p) must not become -inf
    assert logreg.train(seed=0, steps=3000) > 0.95
