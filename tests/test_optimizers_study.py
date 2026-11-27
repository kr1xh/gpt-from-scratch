import json
import math

import pytest

from experiments import optimizers as study
from scratchgpt import train

TEXT = "First Citizen:\nBefore we proceed any further, hear me speak.\n" * 40
TINY = ["--n-layer", "1", "--n-head", "2", "--d-model", "16", "--block-size", "16", "--batch-size", "4",
        "--steps", "6", "--eval-every", "3", "--eval-iters", "1", "--warmup", "1"]
GRID = "adamw:0.01,0.003;muon:0.01;sgd:0.3"


def run(tmp_path, *extra):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    return study.main(["--data", str(corpus), "--out-dir", str(tmp_path), "--grid", GRID, "--smooth", "2",
                       "--target-bits", "5", *TINY, *extra])


def test_grids_are_parsed_per_optimizer():
    assert study.parse_grid("adamw:1e-3,2e-3;sgd:0.1") == {"adamw": [1e-3, 2e-3], "sgd": [0.1]}
    with pytest.raises(ValueError):
        study.parse_grid("lion:1e-4")


def test_steps_to_target_uses_a_running_mean_of_the_training_loss():
    bits = [5.0, 4.0, 3.0, 2.0, 1.0]
    nats = [b * math.log(2) for b in bits]
    # window 2: the means at steps 1 to 4 are 4.5, 3.5, 2.5 and 1.5 bits
    assert study.steps_to_target(nats, target_bits=2.5, window=2) == 3
    assert study.steps_to_target(nats, target_bits=3.5, window=1) == 2
    assert study.steps_to_target(nats, target_bits=0.5, window=1) is None  # never reached


def test_one_row_per_run_and_a_summary_per_optimizer(tmp_path):
    summary = run(tmp_path)
    assert {(r["optimizer"], r["lr"]) for r in summary["runs"]} == {
        ("adamw", 0.01), ("adamw", 0.003), ("muon", 0.01), ("sgd", 0.3)}
    for r in summary["runs"]:
        saved = json.loads((tmp_path / "results" / f"{r['name']}.json").read_text())
        assert saved["args"]["optimizer"] == r["optimizer"] and saved["args"]["lr"] == r["lr"]
        assert saved["args"]["weight_decay"] == (0.1 if r["optimizer"] in ("adamw", "muon") else 0.0)
        assert r["val_bits_per_byte"] == saved["final"]["val_bits_per_byte"]
    adamw = next(o for o in summary["optimizers"] if o["optimizer"] == "adamw")
    scores = {r["lr"]: r["val_bits_per_byte"] for r in summary["runs"] if r["optimizer"] == "adamw"}
    assert adamw["best_bits_per_byte"] == min(scores.values())
    assert adamw["best_lr"] == min(scores, key=scores.get)
    assert adamw["lr_spread"] == pytest.approx(max(scores.values()) - min(scores.values()))
    assert json.loads((tmp_path / "results" / "optimizers.json").read_text()) == summary
    assert (tmp_path / "docs" / "figures" / "optimizers.png").stat().st_size > 1000


def test_finished_runs_are_not_trained_again(tmp_path, monkeypatch):
    first = run(tmp_path)
    monkeypatch.setattr(train, "main", lambda *a: pytest.fail("a finished run was retrained"))
    assert run(tmp_path) == first
