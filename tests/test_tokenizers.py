import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from scratchgpt import tokenizers
from scratchgpt.tokenizers import BPETokenizer, CharTokenizer

CORPUS = "To be, or not to be, that is the question. नमस्ते दुनिया 🙂🙂 café " * 20
BPE = BPETokenizer.train(CORPUS, vocab_size=300)


# ---- char -----------------------------------------------------------------
@given(st.text(alphabet=sorted(set(CORPUS))))
def test_char_round_trip_over_the_training_alphabet(text):
    tok = CharTokenizer.train(CORPUS)
    assert tok.decode(tok.encode(text)) == text


def test_char_vocab_is_sorted_unique_characters():
    tok = CharTokenizer.train("banana")
    assert tok.chars == ["a", "b", "n"] and tok.vocab_size == 3
    assert tok.encode("nab") == [2, 0, 1]


def test_char_rejects_unseen_characters_by_name():
    with pytest.raises(ValueError, match="'z'"):
        CharTokenizer.train("abc").encode("az")


def test_char_save_and_load(tmp_path):
    tok = CharTokenizer.train(CORPUS)
    tok.save(tmp_path / "char.json")
    assert CharTokenizer.load(tmp_path / "char.json").encode(CORPUS) == tok.encode(CORPUS)


# ---- BPE --------------------------------------------------------------------
@settings(max_examples=300, deadline=None)
@given(st.text())  # any Unicode, including text the tokenizer never saw: bytes cover everything
@example("")
@example("🙂")
@example("\x00\n\t")
@example("नमस्ते")
def test_bpe_round_trip_for_any_text(text):
    assert BPE.decode(BPE.encode(text)) == text


def naive_encode(tok, text):
    """Reference: apply every learned merge, in training order, across the whole sequence."""
    ids = list(text.encode("utf-8"))
    for (a, b), new in tok.merges.items():
        out, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                out.append(new)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out
    return ids


@settings(max_examples=200, deadline=None)
@given(st.text(alphabet=sorted(set(CORPUS)) + ["x", "é", "\n"]))
def test_bpe_encode_matches_the_naive_reference(text):
    assert BPE.encode(text) == naive_encode(BPE, text)


@settings(deadline=None)
@given(st.text())
def test_bpe_ids_are_in_vocab_and_never_longer_than_the_bytes(text):
    ids = BPE.encode(text)
    assert all(0 <= i < BPE.vocab_size for i in ids)
    assert len(ids) <= len(text.encode("utf-8"))


def test_bpe_learns_the_textbook_merges():
    # Wikipedia's example: aa -> Z, then ab (ties with Za; the smaller pair wins) -> Y, then ZY -> X
    tok = BPETokenizer.train("aaabdaaabac", vocab_size=259)
    assert list(tok.merges) == [(97, 97), (97, 98), (256, 257)]
    assert tok.encode("aaabdaaabac") == [258, 100, 258, 97, 99]


def test_bpe_compresses_its_training_text():
    assert len(BPE.encode(CORPUS)) < 0.6 * len(CORPUS.encode("utf-8"))


def test_bpe_training_is_deterministic():
    assert BPETokenizer.train(CORPUS, 300).merges == BPE.merges


def test_bpe_stops_early_when_no_pair_repeats():
    tok = BPETokenizer.train("abc", vocab_size=1000)
    assert tok.merges == {} and tok.vocab_size == 256


def test_bpe_needs_room_for_the_256_bytes():
    with pytest.raises(ValueError):
        BPETokenizer.train("abc", vocab_size=100)


def test_bpe_save_and_load(tmp_path):
    BPE.save(tmp_path / "bpe.json")
    loaded = BPETokenizer.load(tmp_path / "bpe.json")
    assert loaded.merges == BPE.merges and loaded.encode(CORPUS) == BPE.encode(CORPUS)


def test_bpe_applies_merges_in_learned_order_when_they_overlap():
    # "bc" was learned before "ab", so "abc" is a + bc, not ab + c
    tok = BPETokenizer({(98, 99): 256, (97, 98): 257})
    assert tok.encode("abc") == [97, 256]


def test_load_reads_either_kind_from_its_file(tmp_path):
    CharTokenizer.train("abc").save(tmp_path / "c.json")
    BPE.save(tmp_path / "b.json")
    assert tokenizers.load(tmp_path / "c.json").encode("cab") == [2, 0, 1]
    assert tokenizers.load(tmp_path / "b.json").merges == BPE.merges


def test_byte_lengths_say_how_much_text_each_id_stands_for():
    np.testing.assert_array_equal(CharTokenizer.train("aé").byte_lengths(), [1, 2])  # é is 2 bytes
    tok = BPETokenizer({(97, 98): 256, (256, 99): 257})  # ab, then abc
    lengths = tok.byte_lengths()
    assert len(lengths) == 258 and lengths[97] == 1 and lengths[256] == 2 and lengths[257] == 3


@given(st.text())
def test_the_byte_lengths_of_an_encoding_add_up_to_the_text(text):
    assert BPE.byte_lengths()[BPE.encode(text)].sum() == len(text.encode("utf-8"))
