import numpy as np
import pytest

from scratchgpt.nn import GPT, GPTConfig
from scratchgpt.sample import generate

CFG = GPTConfig(vocab_size=13, block_size=5, n_layer=1, n_head=2, d_model=8)
MODEL = GPT(CFG, rng=np.random.default_rng(0))


def last_logits(prefix):
    logits, _ = MODEL(np.asarray(prefix)[None, -CFG.block_size:])
    return logits.data[0, -1]


def test_top_k_of_one_is_greedy_decoding():
    out = generate(MODEL, np.array([[3, 1]]), 8, top_k=1, rng=np.random.default_rng(0))
    seq = [3, 1]
    for _ in range(8):
        seq.append(int(np.argmax(last_logits(seq))))
    assert out[0].tolist() == seq


def test_top_k_only_samples_from_the_k_most_likely_tokens():
    out = generate(MODEL, np.array([[2]]), 30, temperature=2.0, top_k=3, rng=np.random.default_rng(1))
    seq = out[0].tolist()
    for i in range(1, len(seq)):
        allowed = np.argsort(last_logits(seq[:i]))[-3:]
        assert seq[i] in allowed


def test_prompts_longer_than_the_context_are_cropped_not_rejected():
    out = generate(MODEL, np.zeros((2, 9), dtype=int), 4, rng=np.random.default_rng(0))
    assert out.shape == (2, 13)


def test_the_same_seed_gives_the_same_text():
    a = generate(MODEL, np.array([[0]]), 20, rng=np.random.default_rng(5))
    b = generate(MODEL, np.array([[0]]), 20, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)


def test_temperature_near_zero_becomes_greedy():
    # an untrained model's logits differ by about 1e-3, so the temperature has to be tiny
    greedy = generate(MODEL, np.zeros((4, 1), dtype=int), 6, top_k=1, rng=np.random.default_rng(0))
    cold = generate(MODEL, np.zeros((4, 1), dtype=int), 6, temperature=1e-6, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(cold, greedy)
    hot = generate(MODEL, np.zeros((50, 1), dtype=int), 1, temperature=1.0, rng=np.random.default_rng(0))
    assert len(set(hot[:, 1].tolist())) > 5  # and at 1.0 a near-uniform model spreads out


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        generate(MODEL, np.array([[0]]), 1, temperature=0.0)


def test_top_k_larger_than_the_vocabulary_samples_from_everything():
    out = generate(MODEL, np.array([[0]]), 3, top_k=10 * CFG.vocab_size, rng=np.random.default_rng(0))
    assert out.shape == (1, 4)


def test_bad_top_k_and_empty_prompts_are_rejected_with_a_reason():
    with pytest.raises(ValueError, match="top_k"):
        generate(MODEL, np.array([[0]]), 1, top_k=0)
    with pytest.raises(ValueError, match="token"):
        generate(MODEL, np.zeros((1, 0), dtype=int), 1)
