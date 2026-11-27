"""A KV cache must change how fast sampling is, never what it samples."""
import numpy as np
import pytest

from scratchgpt.nn import GPT, GPTConfig, KVCache
from scratchgpt.sample import generate

CONFIGS = {
    "gpt2": GPTConfig(vocab_size=17, block_size=12, n_layer=2, n_head=4, d_model=16),
    "modern": GPTConfig(vocab_size=17, block_size=12, n_layer=2, n_head=4, d_model=16,
                        norm="rms", mlp="swiglu", pos="rope", n_kv_head=2),
    "moe": GPTConfig(vocab_size=17, block_size=12, n_layer=2, n_head=4, d_model=16,
                     norm="rms", mlp="swiglu", pos="rope", n_kv_head=2, n_expert=4, top_k=2),
}


def model(name):
    return GPT(CONFIGS[name], rng=np.random.default_rng(0))


@pytest.mark.parametrize("name", CONFIGS)
def test_feeding_tokens_through_the_cache_gives_the_logits_of_one_full_pass(name):
    m = model(name)
    idx = np.random.default_rng(1).integers(0, 17, (2, 10))
    full, _ = m(idx)
    cache = KVCache()
    parts = [m(idx[:, :4], cache=cache)[0].data]  # a prompt, then one token at a time
    parts += [m(idx[:, t:t + 1], cache=cache)[0].data for t in range(4, 10)]
    np.testing.assert_allclose(np.concatenate(parts, axis=1), full.data, rtol=1e-4, atol=1e-5)
    assert cache.length == 10


@pytest.mark.parametrize("name", CONFIGS)
def test_the_cache_holds_one_key_and_value_per_kv_head(name):
    cfg = CONFIGS[name]
    cache = KVCache()
    model(name)(np.zeros((3, 5), dtype=int), cache=cache)
    kv_heads = cfg.n_kv_head or cfg.n_head
    assert len(cache.layers) == cfg.n_layer
    for k, v in cache.layers:
        assert k.shape == v.shape == (3, kv_heads, 5, cfg.d_model // cfg.n_head)


def test_the_cache_cannot_grow_past_the_context():
    m, cache = model("gpt2"), KVCache()
    m(np.zeros((1, 12), dtype=int), cache=cache)
    with pytest.raises(ValueError, match="block_size"):
        m(np.zeros((1, 1), dtype=int), cache=cache)


@pytest.mark.parametrize("name", CONFIGS)
def test_cached_greedy_generation_equals_uncached_token_for_token(name):
    m = model(name)
    prompt = np.random.default_rng(2).integers(0, 17, (2, 3))
    # 30 new tokens run well past block_size = 12, where the context starts to slide
    cached = generate(m, prompt, 30, top_k=1, rng=np.random.default_rng(0))
    uncached = generate(m, prompt, 30, top_k=1, rng=np.random.default_rng(0), use_cache=False)
    np.testing.assert_array_equal(cached, uncached)
