import json

import pytest

from experiments import moe as study
from scratchgpt import train

TEXT = "First Citizen:\nBefore we proceed any further, hear me speak.\n" * 40
TINY = ["--n-layer", "1", "--n-head", "2", "--d-model", "16", "--block-size", "16", "--batch-size", "4",
        "--steps", "6", "--eval-every", "3", "--eval-iters", "1", "--warmup", "1"]


def run(tmp_path, *extra):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    return study.main(["--data", str(corpus), "--out-dir", str(tmp_path), "--n-kv-head", "1", *TINY, *extra])


def test_moe_variants_keep_the_dense_model_s_active_mlp_size():
    size = ["--d-model", "128"]
    for variant in ("moe-4", "moe-8", "moe-4-noaux"):
        args = train.parse([*size, *study.variant_args(variant, d_model=128, n_kv_head=2)])
        assert args.mlp_hidden * args.top_k == 344  # top-2 of experts half as wide as the dense MLP
    assert train.parse(study.variant_args("moe-4-noaux", d_model=128, n_kv_head=2)).moe_aux == 0
    dense = train.parse(study.variant_args("dense", d_model=128, n_kv_head=2))
    assert dense.n_expert == 0 and dense.mlp_hidden == 344  # the width the experts are half of
    with pytest.raises(ValueError):
        study.variant_args("moe-3", d_model=128, n_kv_head=2)


def test_imbalance_is_the_busiest_expert_s_share_over_the_fair_share():
    assert study.imbalance([0.25, 0.25, 0.25, 0.25]) == pytest.approx(1.0)
    assert study.imbalance([0.7, 0.1, 0.1, 0.1]) == pytest.approx(2.8)


def test_one_row_per_variant_with_scores_sizes_and_load_balance(tmp_path):
    summary = run(tmp_path)
    rows = {r["variant"]: r for r in summary["rows"]}
    assert set(rows) == {"dense", "moe-4", "moe-8", "moe-4-noaux"}
    assert rows["moe-8"]["total_params"] > rows["moe-4"]["total_params"] > rows["dense"]["total_params"]
    assert rows["dense"]["imbalance"] is None
    for name in ("moe-4", "moe-8", "moe-4-noaux"):
        r = rows[name]
        assert 1.0 <= r["imbalance"] <= int(name.split("-")[1])
        saved = json.loads((tmp_path / "results" / f"moe-{name}.json").read_text())
        assert r["load"] == saved["evals"][-1]["expert_load"]
    for r in rows.values():
        assert 0 < r["val_bits_per_byte"] < 8
    assert json.loads((tmp_path / "results" / "moe.json").read_text()) == summary
    assert (tmp_path / "docs" / "figures" / "moe.png").stat().st_size > 1000


def test_finished_runs_are_not_trained_again(tmp_path, monkeypatch):
    first = run(tmp_path)
    monkeypatch.setattr(train, "main", lambda *a: pytest.fail("a finished run was retrained"))
    assert run(tmp_path) == first
