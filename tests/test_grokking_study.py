import json

import pytest

from experiments import grokking as study
from experiments.runs import cached

TINY = ["--p", "5", "--frac", "0.6", "--steps", "20", "--d-model", "16", "--n-head", "2", "--eval-every", "5",
        "--runs", "adamw:0.1,adamw:1,muon:1"]


def test_cached_reuses_a_result_only_for_the_same_settings(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return {"value": len(calls)}
    path = tmp_path / "r.json"
    assert cached(path, {"a": 1}, compute) == {"value": 1}
    assert cached(path, {"a": 1}, compute) == {"value": 1} and len(calls) == 1
    assert cached(path, {"a": 2}, compute) == {"value": 2} and len(calls) == 2


def test_runs_are_parsed_as_optimizer_and_weight_decay():
    assert study.parse_runs("adamw:0.1,muon:1") == [("adamw", 0.1), ("muon", 1.0)]
    with pytest.raises(ValueError):
        study.parse_runs("lion:1")


def test_one_row_per_run_with_the_steps_to_memorise_and_to_generalise(tmp_path):
    summary = study.main(["--out-dir", str(tmp_path), *TINY])
    assert [(r["optimizer"], r["weight_decay"]) for r in summary["rows"]] == [("adamw", 0.1), ("adamw", 1.0),
                                                                              ("muon", 1.0)]
    for r in summary["rows"]:
        saved = json.loads((tmp_path / "results" / f"grokking-{r['optimizer']}-wd{r['weight_decay']:g}.json").read_text())
        history = saved["result"]
        assert saved["settings"]["weight_decay"] == r["weight_decay"]
        assert r["final_test_acc"] == history[-1]["test_acc"] and r["final_train_acc"] == history[-1]["train_acc"]
        for key in ("memorised_at", "generalised_at"):
            assert r[key] is None or 0 <= r[key] <= 19
    assert json.loads((tmp_path / "results" / "grokking.json").read_text()) == summary
    assert (tmp_path / "docs" / "figures" / "grokking.png").stat().st_size > 1000


def test_finished_runs_are_not_trained_again(tmp_path, monkeypatch):
    first = study.main(["--out-dir", str(tmp_path), *TINY])
    monkeypatch.setattr(study.grokking, "train", lambda **kw: pytest.fail("a finished run was retrained"))
    assert study.main(["--out-dir", str(tmp_path), *TINY]) == first
