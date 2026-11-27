"""Bits per byte must count every validation target exactly once, whatever the tokenizer."""
import math

import numpy as np
import pytest

from scratchgpt import functional as F
from scratchgpt.evaluate import bits_per_byte
from scratchgpt.nn import GPT, GPTConfig
from scratchgpt.tokenizers import BPETokenizer, CharTokenizer

TEXT = "Now is the winter of our discontent\nMade glorious summer by this sun of York;\n" * 6


def uniform_model(vocab_size, block_size=16):
    model = GPT(GPTConfig(vocab_size, block_size, n_layer=1, n_head=2, d_model=8), rng=np.random.default_rng(0))
    model.wte.weight.data[:] = 0  # the tied head then gives every token logit 0: a uniform guess
    return model


def test_a_uniform_guess_over_characters_costs_log2_of_the_vocabulary():
    tok = CharTokenizer.train(TEXT)
    result = bits_per_byte(uniform_model(tok.vocab_size), tok, tok.encode(TEXT))
    assert result["bits_per_byte"] == pytest.approx(math.log2(tok.vocab_size), rel=1e-5)


def test_every_target_is_counted_once_including_a_ragged_last_window():
    tok = BPETokenizer.train(TEXT, 300)
    ids = tok.encode(TEXT)
    assert (len(ids) - 1) % 16 != 0  # the last window is shorter than the block
    result = bits_per_byte(uniform_model(tok.vocab_size), tok, ids, batch_size=3)
    assert result["tokens"] == len(ids) - 1
    assert result["bytes"] == int(tok.byte_lengths()[ids[1:]].sum())
    expected = (len(ids) - 1) * math.log2(tok.vocab_size) / result["bytes"]
    assert result["bits_per_byte"] == pytest.approx(expected, rel=1e-5)


def test_matches_the_cross_entropy_of_a_single_window():
    tok = CharTokenizer.train(TEXT)
    model = GPT(GPTConfig(tok.vocab_size, 32, n_layer=1, n_head=2, d_model=16), rng=np.random.default_rng(3))
    text = TEXT[:33]  # exactly one window: 32 inputs, 32 targets
    ids = np.array([tok.encode(text)])
    logits, _ = model(ids[:, :-1])
    nats = F.cross_entropy(logits, ids[:, 1:]).item() * 32
    assert bits_per_byte(model, tok, tok.encode(text))["bits_per_byte"] == pytest.approx(nats / math.log(2) / 32, rel=1e-5)
