import json

import pytest

from experiments import interp as study
from scratchgpt import train

TEXT = "First Citizen:\nBefore we proceed any further, hear me speak.\n" * 40


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    out = tmp_path_factory.mktemp("trained")
    corpus = out / "corpus.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    train.main(["--data", str(corpus), "--out-dir", str(out), "--name", "tiny", "--n-layer", "2", "--n-head", "2",
                "--d-model", "16", "--block-size", "16", "--batch-size", "4", "--steps", "20", "--eval-every", "10",
                "--eval-iters", "1", "--warmup", "2", "--norm", "rms", "--mlp", "swiglu", "--pos", "rope"])
    return out / "models" / "tiny.npz", corpus


def test_scores_losses_and_ablations_are_reported_and_saved(tmp_path, checkpoint):
    path, corpus = checkpoint
    summary = study.main(["--checkpoint", str(path), "--data", str(corpus), "--out-dir", str(tmp_path),
                          "--batch", "4", "--ablate", "1"])
    assert len(summary["prefix_matching"]) == 2 and len(summary["prefix_matching"][0]) == 2
    top, bottom = summary["ablation"]["top"], summary["ablation"]["bottom"]
    assert len(top["heads"]) == len(bottom["heads"]) == 1 and top["heads"] != bottom["heads"]
    scores = summary["prefix_matching"]
    (tl, th), (bl, bh) = top["heads"][0], bottom["heads"][0]
    assert scores[tl][th] == max(max(row) for row in scores) and scores[bl][bh] == min(min(row) for row in scores)
    for key in ("repeated", "text"):
        assert summary["loss"][key] > 0 and top["loss"][key] > 0 and bottom["loss"][key] > 0
    assert summary["loss"]["random_half"] > 0  # the first, unpredictable half
    assert len(summary["logit_lens"]) == 3  # embedding + 2 blocks
    assert json.loads((tmp_path / "results" / "interp.json").read_text()) == summary
    assert (tmp_path / "docs" / "figures" / "interp.png").stat().st_size > 1000
