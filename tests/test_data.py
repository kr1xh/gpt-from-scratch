import hashlib

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from scratchgpt import data


@given(n=st.integers(2, 300), block=st.integers(1, 40), batch=st.integers(1, 8), seed=st.integers(0, 99))
def test_batches_are_windows_of_the_data_with_targets_one_step_ahead(n, block, batch, seed):
    block = min(block, n - 1)
    ids = np.arange(n) * 7 % 1000  # distinct values reveal where each window came from
    x, y = data.get_batch(ids, batch, block, np.random.default_rng(seed))
    assert x.shape == y.shape == (batch, block) and x.dtype == np.int64
    for row_x, row_y in zip(x, y):
        start = int(np.flatnonzero(ids == row_x[0])[0])
        np.testing.assert_array_equal(row_x, ids[start:start + block])
        np.testing.assert_array_equal(row_y, ids[start + 1:start + block + 1])


def test_every_window_start_can_be_drawn():
    ids = np.arange(6)
    starts = {int(data.get_batch(ids, 1, 3, np.random.default_rng(s))[0][0, 0]) for s in range(200)}
    assert starts == {0, 1, 2}


def test_batches_need_one_more_token_than_the_block():
    with pytest.raises(ValueError):
        data.get_batch(np.arange(4), 1, 4, np.random.default_rng(0))


def test_split_keeps_order_and_holds_out_the_end():
    train, val = data.split("abcdefghij", val_frac=0.2)
    assert (train, val) == ("abcdefgh", "ij")


def test_download_fetches_once_and_checks_the_hash(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"hello")
    good = hashlib.sha256(b"hello").hexdigest()
    target = tmp_path / "cache" / "out.txt"

    assert data.download(source.as_uri(), target, sha256=good).read_bytes() == b"hello"
    source.unlink()  # a second call must use the cached file, not the network
    assert data.download(source.as_uri(), target, sha256=good).read_bytes() == b"hello"


def test_download_refuses_a_file_with_the_wrong_hash(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"tampered")
    target = tmp_path / "out.txt"
    with pytest.raises(ValueError, match="sha256"):
        data.download(source.as_uri(), target, sha256="0" * 64)
    assert not target.exists()  # nothing half-checked is left behind to be trusted next time


def test_a_corrupted_cached_file_is_not_trusted(tmp_path):
    target = tmp_path / "out.txt"
    target.write_bytes(b"half a downl")  # e.g. an interrupted earlier download
    with pytest.raises(ValueError, match="sha256"):
        data.download((tmp_path / "missing.txt").as_uri(), target, sha256=hashlib.sha256(b"hello").hexdigest())
