"""Tools to look inside a trained GPT: attention maps, the logit lens, induction scores, ablation."""
import numpy as np
import pytest

from scratchgpt import interp
from scratchgpt.nn import GPT, GPTConfig
from scratchgpt.tensor import no_grad

CFG = GPTConfig(vocab_size=13, block_size=12, n_layer=2, n_head=4, d_model=16, norm="rms", mlp="swiglu",
                pos="rope", n_kv_head=2)


def model():
    m = GPT(CFG, rng=np.random.default_rng(0))
    for p in m.parameters():  # random biases and gains: a model whose heads actually differ
        p.data = p.data + np.random.default_rng(1).normal(0, 0.3, p.shape).astype(np.float32)
    return m


def test_attention_maps_are_causal_distributions_per_head():
    idx = np.random.default_rng(2).integers(0, 13, (3, 9))
    maps = interp.attention_maps(model(), idx)
    assert len(maps) == CFG.n_layer
    for m in maps:
        assert m.shape == (3, 4, 9, 9)
        np.testing.assert_allclose(m.sum(axis=-1), 1, rtol=1e-5)
        assert np.abs(np.triu(m, k=1)).max() < 1e-6  # nothing from the future


def test_ablating_a_head_is_the_same_as_cutting_its_rows_out_of_the_output_projection():
    m = model()
    idx = np.random.default_rng(3).integers(0, 13, (2, 8))
    with interp.ablated(m, [(1, 2)]), no_grad():
        ablated = m(idx)[0].data
    head = CFG.d_model // CFG.n_head
    m.blocks[1].attn.proj.weight.data[2 * head:3 * head] = 0  # the rows that read head 2's output
    with no_grad():
        np.testing.assert_allclose(ablated, m(idx)[0].data, rtol=1e-5, atol=1e-6)


def test_ablation_is_undone_when_the_block_ends():
    m = model()
    idx = np.zeros((1, 4), dtype=int)
    with no_grad():
        before = m(idx)[0].data
        with interp.ablated(m, [(0, 0), (1, 3)]):
            assert not np.allclose(m(idx)[0].data, before)
        np.testing.assert_array_equal(m(idx)[0].data, before)


def test_the_logit_lens_of_the_last_layer_is_the_model_itself():
    m = model()
    idx = np.random.default_rng(4).integers(0, 13, (2, 7))
    lens = interp.logit_lens(m, idx)
    assert len(lens) == CFG.n_layer + 1  # after the embedding and after every block
    with no_grad():
        logits = m(idx)[0].data.astype(np.float64)
    expected = logits - logits.max(-1, keepdims=True)
    expected -= np.log(np.exp(expected).sum(-1, keepdims=True))
    np.testing.assert_allclose(lens[-1], expected, rtol=1e-5, atol=1e-5)
    for layer in lens:
        np.testing.assert_allclose(np.exp(layer).sum(-1), 1, rtol=1e-5)


def test_repeated_random_tokens_repeat_their_first_half():
    idx = interp.repeated_random_tokens(13, n=6, batch=4, seed=0)
    assert idx.shape == (4, 12)
    np.testing.assert_array_equal(idx[:, :6], idx[:, 6:])


def test_a_perfect_induction_head_scores_one_and_a_uniform_head_scores_little():
    n = 5
    T = 2 * n
    perfect = np.zeros((1, 2, T, T))
    for t in range(T):
        perfect[0, 0, t, 0] = 1  # head 0 just looks at the first token...
        if t >= n:
            perfect[0, 1, t] = 0
            perfect[0, 1, t, t - n + 1] = 1  # ...head 1 looks at the token after the earlier copy
        else:
            perfect[0, 1, t, 0] = 1
    uniform = np.tril(np.ones((T, T))) / np.arange(1, T + 1)[:, None]
    scores = interp.prefix_matching_scores([perfect, np.broadcast_to(uniform, (1, 2, T, T))], n)
    assert scores.shape == (2, 2)
    assert scores[0, 1] == pytest.approx(1.0) and scores[0, 0] < 0.2
    assert scores[1].max() < 0.2


def test_mean_loss_counts_only_the_predictions_from_a_given_position_on():
    m = model()
    idx = np.random.default_rng(5).integers(0, 13, (2, 10))
    with no_grad():
        logits = m(idx)[0].data.astype(np.float64)
    log_probs = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    nll = -np.take_along_axis(log_probs[:, :-1], idx[:, 1:, None], -1)[..., 0]  # nll[:, t] predicts token t+1
    assert interp.mean_loss(m, idx, start=0) == pytest.approx(nll.mean(), rel=1e-5)
    assert interp.mean_loss(m, idx, start=5) == pytest.approx(nll[:, 5:].mean(), rel=1e-5)
