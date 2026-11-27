import json

import pytest

from experiments import char_vs_bpe as study
from scratchgpt import train

TEXT = "First Citizen:\nBefore we proceed any further, hear me speak.\n" * 40
TINY = ["--n-layer", "1", "--n-head", "2", "--d-model", "16", "--block-size", "16", "--batch-size", "4",
        "--steps", "4", "--eval-every", "2", "--eval-iters", "1", "--warmup", "1"]


def run(tmp_path, *extra):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    return study.main(["--data", str(corpus), "--out-dir", str(tmp_path), "--variants", "char,bpe-270", *TINY, *extra])


def test_variant_names_become_train_arguments():
    assert study.variant_args("char") == ["--tokenizer", "char"]
    assert study.variant_args("bpe-512") == ["--tokenizer", "bpe", "--vocab-size", "512"]
    with pytest.raises(ValueError):
        study.variant_args("wordpiece-9")


def test_one_row_per_variant_with_the_comparable_score(tmp_path):
    summary = run(tmp_path)
    rows = {r["variant"]: r for r in summary["rows"]}
    assert set(rows) == {"char", "bpe-270"}
    assert rows["char"]["bytes_per_token"] == pytest.approx(1.0)
    assert rows["bpe-270"]["bytes_per_token"] > 1.0  # merges make tokens longer than one byte
    for r in rows.values():
        assert 0 < r["val_bits_per_byte"] < 8 and r["params"] > 0
        assert r["bytes_per_step"] == pytest.approx(4 * 16 * r["bytes_per_token"])
    assert json.loads((tmp_path / "results" / "char_vs_bpe.json").read_text()) == summary
    assert (tmp_path / "docs" / "figures" / "char_vs_bpe.png").stat().st_size > 1000


def test_finished_runs_are_not_trained_again(tmp_path, monkeypatch):
    first = run(tmp_path)
    monkeypatch.setattr(train, "main", lambda *a: pytest.fail("a finished run was retrained"))
    assert run(tmp_path) == first


def test_a_run_without_its_final_score_is_trained_again(tmp_path, monkeypatch):
    run(tmp_path)
    path = tmp_path / "results" / "char_vs_bpe-char.json"
    unfinished = json.loads(path.read_text())
    del unfinished["final"]  # as if the run was stopped before its last eval
    path.write_text(json.dumps(unfinished))
    calls = []
    real = train.main
    monkeypatch.setattr(train, "main", lambda argv: (calls.append(argv), real(argv))[1])
    run(tmp_path)
    assert len(calls) == 1 and "char_vs_bpe-char" in calls[0]


def test_a_finished_run_with_different_settings_is_trained_again(tmp_path, monkeypatch):
    run(tmp_path)
    calls = []
    real = train.main
    monkeypatch.setattr(train, "main", lambda argv: (calls.append(argv), real(argv))[1])
    summary = run(tmp_path, "--steps", "5")  # passed after TINY, so it wins
    assert len(calls) == 2
    assert all(r["variant"] in ("char", "bpe-270") for r in summary["rows"])


def test_a_run_saved_before_a_train_option_existed_counts_as_using_its_default(tmp_path, monkeypatch):
    run(tmp_path)
    for variant in ("char", "bpe-270"):
        path = tmp_path / "results" / f"char_vs_bpe-{variant}.json"
        older = json.loads(path.read_text())
        del older["args"]["norm"]  # as saved by a version of train.py without --norm
        path.write_text(json.dumps(older))
    monkeypatch.setattr(train, "main", lambda *a: pytest.fail("a finished run was retrained"))
    run(tmp_path)
