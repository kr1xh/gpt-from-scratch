import json

import pytest

from experiments import modern_block as study
from scratchgpt import train

TEXT = "First Citizen:\nBefore we proceed any further, hear me speak.\n" * 40
TINY = ["--n-layer", "1", "--n-head", "2", "--d-model", "16", "--block-size", "16", "--batch-size", "4",
        "--steps", "4", "--eval-every", "2", "--eval-iters", "1", "--warmup", "1"]


def run(tmp_path, *extra):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    return study.main(["--data", str(corpus), "--out-dir", str(tmp_path), "--variants", "gpt2,gqa,modern",
                       "--n-kv-head", "1", "--seeds", "2", *TINY, *extra])


def test_every_variant_is_matched_to_the_size_of_the_gpt2_model():
    size = ["--n-layer", "4", "--n-head", "4", "--d-model", "128", "--block-size", "128"]
    target = study.count_params(size)
    per_hidden_unit = 3 * 128 + 2  # the largest a single unit of MLP width adds, per layer (SwiGLU)
    for variant in study.VARIANTS:
        argv = [*size, *study.variant_args(variant, n_kv_head=2)]
        hidden = study.matched_hidden(argv, target)
        assert abs(study.count_params([*argv, "--mlp-hidden", str(hidden)]) - target) <= 4 * per_hidden_unit / 2
    assert study.matched_hidden(size, target) == 4 * 128  # GPT-2 keeps its own width


def test_one_row_per_run_and_repeated_seeds_for_the_two_main_blocks(tmp_path):
    summary = run(tmp_path)
    runs = {(r["variant"], r["seed"]) for r in summary["runs"]}
    assert runs == {("gpt2", 0), ("gpt2", 1), ("gqa", 0), ("modern", 0), ("modern", 1)}
    target = next(r["params"] for r in summary["runs"] if r["variant"] == "gpt2")
    for r in summary["runs"]:
        assert 0 < r["val_bits_per_byte"] < 8
        assert r["params"] == pytest.approx(target, rel=0.05)
    by_variant = {v["variant"]: v for v in summary["variants"]}
    gpt2 = [r["val_bits_per_byte"] for r in summary["runs"] if r["variant"] == "gpt2"]
    assert by_variant["gpt2"]["mean"] == pytest.approx(sum(gpt2) / 2)
    assert by_variant["gpt2"]["low"] == min(gpt2) and by_variant["gpt2"]["high"] == max(gpt2)
    assert json.loads((tmp_path / "results" / "modern_block.json").read_text()) == summary
    assert (tmp_path / "docs" / "figures" / "modern_block.png").stat().st_size > 1000


def test_finished_runs_are_not_trained_again(tmp_path, monkeypatch):
    first = run(tmp_path)
    monkeypatch.setattr(train, "main", lambda *a: pytest.fail("a finished run was retrained"))
    assert run(tmp_path) == first


def test_unknown_variants_are_rejected():
    with pytest.raises(ValueError):
        study.variant_args("alibi", n_kv_head=2)
