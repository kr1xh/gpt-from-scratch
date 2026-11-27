"""The study is only fair if both sides start from the same weights, see the same batches and take
the same optimizer steps; in float64 they must then agree step for step."""
import json

import numpy as np
import pytest

from experiments import engine_vs_torch as study
from scratchgpt.nn import GPT, GPTConfig

CFG = GPTConfig(vocab_size=12, block_size=8, n_layer=2, n_head=2, d_model=16)
HP = {**study.HP, "warmup": 3}  # a short warmup so 8 steps cover warmup and decay


def setup(steps=8):
    state = GPT(CFG, rng=np.random.default_rng(0)).state_dict()
    ids = np.tile(np.arange(CFG.vocab_size), 40)  # a repeating pattern: learnable, so the loss falls
    return state, study.make_batches(ids, steps, batch_size=4, block_size=CFG.block_size, seed=2)


def test_batches_are_fixed_by_the_seed_and_shaped_for_the_model():
    ids = np.arange(100)
    a = study.make_batches(ids, 5, 3, 8, seed=0)
    b = study.make_batches(ids, 5, 3, 8, seed=0)
    assert len(a) == 5 and a[0][0].shape == (3, 8)
    for (xa, ya), (xb, yb) in zip(a, b):
        np.testing.assert_array_equal(xa, xb)
        np.testing.assert_array_equal(ya, xa + 1)  # ids are arange: targets are one step on


def test_in_float64_both_engines_take_the_same_steps():
    state, batches = setup()
    ours = study.run_ours(CFG, state, batches, HP, np.float64)
    theirs = study.run_torch(CFG, state, batches, HP, np.float64)
    np.testing.assert_allclose(ours["loss"], theirs["loss"], rtol=1e-9)
    assert ours["loss"][-1] < ours["loss"][0]  # it trained, so the steps were not trivially equal
    for name, value in ours["state"].items():
        np.testing.assert_allclose(value, theirs["state"][name], rtol=1e-7, atol=1e-9, err_msg=name)


def test_in_float32_they_stay_close_but_not_bitwise_equal():
    state, batches = setup()
    ours = study.run_ours(CFG, state, batches, HP, np.float32)
    theirs = study.run_torch(CFG, state, batches, HP, np.float32)
    np.testing.assert_allclose(ours["loss"], theirs["loss"], rtol=1e-4)
    assert len(ours["seconds"]) == len(theirs["seconds"]) == len(batches)
    assert ours["state"]["wte.weight"].dtype == np.float32


def test_runs_do_not_change_the_initial_weights_they_were_given():
    state, batches = setup(2)
    before = {k: v.copy() for k, v in state.items()}
    study.run_ours(CFG, state, batches, HP, np.float32)
    study.run_torch(CFG, state, batches, HP, np.float32)
    for k in state:
        np.testing.assert_array_equal(state[k], before[k])


def test_summary_reports_the_gap_over_time_and_the_speed_ratio():
    ours = {"loss": [4.0, 3.0, 2.0, 1.0], "seconds": [9.0, 2.0, 2.0, 4.0]}
    theirs = {"loss": [4.0, 3.1, 2.0, 0.8], "seconds": [5.0, 0.1, 0.1, 0.2]}
    s = study.summarize(ours, theirs, windows=2)
    assert s["max_abs_loss_diff"] == pytest.approx(0.2)
    assert [w["max_abs_loss_diff"] for w in s["windows"]] == pytest.approx([0.1, 0.2])
    assert [w["steps"] for w in s["windows"]] == [[0, 1], [2, 3]]
    # the first step includes one-off warm-up costs, so medians skip it
    assert s["ours_ms_per_step"] == pytest.approx(2000) and s["torch_ms_per_step"] == pytest.approx(100)
    assert s["slowdown"] == pytest.approx(20)
    assert s["exact_match_fraction"] == pytest.approx(0.5)  # steps 0 and 2 agree bit for bit


def test_summary_of_a_run_shorter_than_the_windows():
    run = {"loss": [1.0, 2.0], "seconds": [1.0, 1.0]}
    assert [w["steps"] for w in study.summarize(run, run, windows=5)["windows"]] == [[0, 0], [1, 1]]


def test_main_writes_results_and_a_figure(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("to be or not to be, that is the question\n" * 30, encoding="utf-8")
    out = study.main(["--data", str(corpus), "--out-dir", str(tmp_path), "--steps", "4", "--n-layer", "1",
                      "--d-model", "16", "--n-head", "2", "--block-size", "8", "--batch-size", "2",
                      "--warmup", "2"])
    saved = json.loads((tmp_path / "results" / "engine_vs_torch.json").read_text())
    assert saved == out
    assert set(saved["runs"]) == {"float32", "float64"}
    assert len(saved["runs"]["float32"]["ours"]["loss"]) == 4
    assert (tmp_path / "docs" / "figures" / "engine_vs_torch.png").stat().st_size > 1000


def test_a_run_stopped_during_float64_keeps_the_float32_results(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("to be or not to be, that is the question\n" * 30, encoding="utf-8")
    real = study.run_ours

    def dies_in_float64(cfg, state, batches, hp, dtype):
        if dtype == np.float64:
            raise MemoryError("killed")
        return real(cfg, state, batches, hp, dtype)
    monkeypatch.setattr(study, "run_ours", dies_in_float64)
    with pytest.raises(MemoryError):
        study.main(["--data", str(corpus), "--out-dir", str(tmp_path), "--steps", "3", "--n-layer", "1",
                    "--d-model", "16", "--n-head", "2", "--block-size", "8", "--batch-size", "2", "--warmup", "1"])
    saved = json.loads((tmp_path / "results" / "engine_vs_torch.json").read_text())
    assert set(saved["runs"]) == {"float32"}


def test_replot_redraws_from_saved_results_without_training(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("to be or not to be, that is the question\n" * 30, encoding="utf-8")
    args = ["--data", str(corpus), "--out-dir", str(tmp_path), "--steps", "3", "--n-layer", "1", "--d-model", "16",
            "--n-head", "2", "--block-size", "8", "--batch-size", "2", "--warmup", "1"]
    first = study.main(args)
    figure = tmp_path / "docs" / "figures" / "engine_vs_torch.png"
    figure.unlink()
    monkeypatch.setattr(study, "run_ours", lambda *a: pytest.fail("replot must not train"))
    again = study.main(args + ["--replot"])
    assert again["runs"]["float32"]["ours"] == first["runs"]["float32"]["ours"] and figure.exists()
