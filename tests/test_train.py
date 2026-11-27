"""End to end on a tiny text: train writes metrics and a checkpoint, sample reads it back."""
import json

import numpy as np
import pytest

from scratchgpt import data, sample, train
from scratchgpt.nn import GPT

TEXT = "First Citizen:\nBefore we proceed any further, hear me speak.\n" * 40
TINY = ["--n-layer", "1", "--n-head", "2", "--d-model", "16", "--block-size", "16",
        "--batch-size", "8", "--eval-every", "20", "--eval-iters", "2", "--warmup", "5", "--lr", "1e-2"]


def run(tmp_path, *extra):
    tmp_path.mkdir(parents=True, exist_ok=True)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    return train.main(["--data", str(corpus), "--out-dir", str(tmp_path), "--name", "tiny", *TINY, *extra])


def test_training_writes_metrics_and_learns(tmp_path):
    metrics = run(tmp_path, "--steps", "100")
    saved = json.loads((tmp_path / "results" / "tiny.json").read_text())
    assert saved == metrics
    assert saved["config"]["d_model"] == 16 and saved["config"]["vocab_size"] == len(set(TEXT))
    assert len(saved["train_loss"]) == 100
    assert [e["step"] for e in saved["evals"]] == [0, 20, 40, 60, 80, 99]
    first, last = saved["evals"][0], saved["evals"][-1]
    assert first["val_loss"] > 3.0 and last["val_loss"] < 1.0  # the text repeats: learnable
    assert {"train_loss", "val_loss", "lr", "seconds"} <= set(last)


def test_same_seed_same_run(tmp_path):
    a = run(tmp_path / "a", "--steps", "5")
    b = run(tmp_path / "b", "--steps", "5")
    assert a["train_loss"] == b["train_loss"]


def test_sample_continues_a_prompt_from_the_checkpoint(tmp_path):
    run(tmp_path, "--steps", "100")
    text = sample.main(["--checkpoint", str(tmp_path / "models" / "tiny.npz"), "--prompt", "First",
                        "--tokens", "30", "--top-k", "5", "--seed", "0"])
    assert text.startswith("First") and len(text) == len("First") + 30
    again = sample.main(["--checkpoint", str(tmp_path / "models" / "tiny.npz"), "--prompt", "First",
                         "--tokens", "30", "--top-k", "5", "--seed", "0"])
    assert again == text


def test_checkpoint_holds_the_trained_weights_not_the_initial_ones(tmp_path):
    run(tmp_path, "--steps", "100")
    model, tok = sample.load(tmp_path / "models" / "tiny.npz")
    ids = np.array([tok.encode(TEXT[:17])])
    _, loss = model(ids[:, :-1], ids[:, 1:])
    assert loss.item() < 1.0  # a fresh model scores about ln(27) = 3.3


def test_validation_text_may_use_characters_the_training_part_lacks(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TEXT + "Zz!" * 5, encoding="utf-8")  # 'Z', 'z' and '!' only in the last 10%
    metrics = train.main(["--data", str(corpus), "--out-dir", str(tmp_path), "--name", "t", *TINY, "--steps", "2"])
    assert metrics["config"]["vocab_size"] == len(set(TEXT + "Zz!"))


def test_a_checkpoint_is_written_at_every_eval_so_a_stopped_run_keeps_its_weights(tmp_path, monkeypatch):
    calls = []
    real_savez = np.savez
    monkeypatch.setattr(train.np, "savez", lambda *a, **k: (calls.append(a[0]), real_savez(*a, **k)))
    run(tmp_path, "--steps", "41")  # evals at 0, 20, 40
    assert len(calls) == 3


def moved_from_init(tmp_path) -> float:
    """Largest weight change between the checkpoint and the model the seed initialised."""
    model, _ = sample.load(tmp_path / "models" / "tiny.npz")
    init = GPT(model.config, rng=np.random.default_rng(0)).state_dict()
    return max(float(np.abs(v - init[k]).max()) for k, v in model.state_dict().items())


def test_the_learning_rate_schedule_is_applied(tmp_path):
    run(tmp_path, "--steps", "3", "--warmup", "1000000")  # warmup keeps the lr near 1e-8
    assert moved_from_init(tmp_path) < 1e-5


def test_gradients_are_clipped_before_the_optimizer_sees_them(tmp_path):
    # clipped to 1e-12 the gradients drown in Adam's eps (1e-8), so the steps all but vanish
    run(tmp_path, "--steps", "3", "--grad-clip", "1e-12")
    assert moved_from_init(tmp_path) < 1e-4


def test_every_eval_uses_the_same_batches(tmp_path):
    metrics = run(tmp_path, "--steps", "41", "--lr", "0")  # weights never change
    losses = {(e["train_loss"], e["val_loss"]) for e in metrics["evals"]}
    assert len(losses) == 1


def test_every_run_ends_with_an_exact_bits_per_byte_on_validation(tmp_path):
    metrics = run(tmp_path, "--steps", "3")
    final = metrics["final"]
    val_text = TEXT[int(len(TEXT) * 0.9):]
    assert final["val_bytes"] == len(val_text.encode()) - 1  # every character after the first
    assert 0 < final["val_bits_per_byte"] < 8


def test_bpe_runs_train_cache_and_sample(tmp_path, monkeypatch):
    bpe = ["--tokenizer", "bpe", "--vocab-size", "270", "--cache-dir", str(tmp_path / "cache")]
    first = run(tmp_path / "one", "--steps", "3", *bpe)
    assert 256 < first["config"]["vocab_size"] <= 270  # BPE stops early once no pair repeats
    assert json.loads((tmp_path / "one" / "models" / "tiny.tokenizer.json").read_text())["type"] == "bpe"
    text = sample.main(["--checkpoint", str(tmp_path / "one" / "models" / "tiny.npz"), "--prompt", "First",
                        "--tokens", "5", "--seed", "0"])
    assert text.startswith("First")

    # a second run with the same text and vocabulary reuses the cache: no training, no encoding
    from scratchgpt import tokenizers
    monkeypatch.setattr(tokenizers.BPETokenizer, "train", lambda *a: pytest.fail("retrained"))
    monkeypatch.setattr(tokenizers.BPETokenizer, "encode", lambda *a: pytest.fail("re-encoded"))
    again = run(tmp_path / "two", "--steps", "3", *bpe)
    assert again["train_loss"] == first["train_loss"]


def test_the_bpe_tokenizer_only_sees_the_training_text(tmp_path, monkeypatch):
    from scratchgpt import tokenizers
    seen = []
    real = tokenizers.BPETokenizer.train.__func__
    monkeypatch.setattr(tokenizers.BPETokenizer, "train", classmethod(lambda cls, t, v: (seen.append(t), real(cls, t, v))[1]))
    run(tmp_path, "--steps", "1", "--tokenizer", "bpe", "--vocab-size", "260", "--cache-dir", str(tmp_path / "c"))
    assert seen == [TEXT[:int(len(TEXT) * 0.9)]]


def test_the_cache_keeps_tokenizers_for_different_vocabularies_and_texts_apart(tmp_path):
    cache = ["--tokenizer", "bpe", "--cache-dir", str(tmp_path / "cache")]
    small = run(tmp_path / "a", "--steps", "1", *cache, "--vocab-size", "260")
    large = run(tmp_path / "b", "--steps", "1", *cache, "--vocab-size", "270")
    assert small["config"]["vocab_size"] == 260 and large["config"]["vocab_size"] > 260
    other = tmp_path / "c"
    other.mkdir()
    (other / "corpus.txt").write_text(TEXT.upper(), encoding="utf-8")
    shouted = train.main(["--data", str(other / "corpus.txt"), "--out-dir", str(other), "--name", "tiny", *TINY,
                          "--steps", "1", *cache, "--vocab-size", "260"])
    # different text, different merges: the upper-case corpus must not reuse the first tokenizer
    assert sample.load(other / "models" / "tiny.npz")[1].merges != sample.load(tmp_path / "a" / "models" / "tiny.npz")[1].merges
    assert shouted["config"]["vocab_size"] == 260


def test_a_corrupt_cache_is_rebuilt_not_trusted(tmp_path):
    cache = ["--tokenizer", "bpe", "--vocab-size", "260", "--cache-dir", str(tmp_path / "cache")]
    first = run(tmp_path / "a", "--steps", "2", *cache)
    (ids_file,) = (tmp_path / "cache").glob("*-ids.npz")
    ids_file.write_bytes(ids_file.read_bytes()[:100])  # a run killed while writing it
    again = run(tmp_path / "b", "--steps", "2", *cache)
    assert again["train_loss"] == first["train_loss"]


def test_modern_block_options_reach_the_model_and_its_checkpoint(tmp_path):
    metrics = run(tmp_path, "--steps", "2", "--norm", "rms", "--mlp", "swiglu", "--pos", "rope", "--n-kv-head", "1")
    assert {k: metrics["config"][k] for k in ("norm", "mlp", "pos", "n_kv_head")} == \
        {"norm": "rms", "mlp": "swiglu", "pos": "rope", "n_kv_head": 1}
    model, _ = sample.load(tmp_path / "models" / "tiny.npz")
    names = dict(model.named_parameters())
    assert "wpe.weight" not in names and "blocks.0.mlp.gate.weight" in names
    text = sample.main(["--checkpoint", str(tmp_path / "models" / "tiny.npz"), "--prompt", "First",
                        "--tokens", "20", "--seed", "0"])
    assert text.startswith("First") and len(text) == len("First") + 20


def test_the_batches_depend_on_the_seed_not_on_the_model(tmp_path, monkeypatch):
    seen = {}
    real = data.get_batch

    def record(ids, batch, block, rng):
        x, y = real(ids, batch, block, rng)
        seen.setdefault(width, []).append(x.copy())
        return x, y
    monkeypatch.setattr(data, "get_batch", record)
    for width in ("16", "24"):  # different models draw different amounts of randomness at init
        run(tmp_path / width, "--steps", "3", "--d-model", width, "--eval-every", "100")
    assert len(seen["16"]) == len(seen["24"]) > 3
    for a, b in zip(seen["16"], seen["24"]):
        np.testing.assert_array_equal(a, b)


def test_mlp_hidden_sets_the_width_of_the_mlp(tmp_path):
    metrics = run(tmp_path, "--steps", "1", "--mlp", "swiglu", "--mlp-hidden", "20")
    model, _ = sample.load(tmp_path / "models" / "tiny.npz")
    assert metrics["config"]["mlp_hidden"] == 20
    assert dict(model.named_parameters())["blocks.0.mlp.gate.weight"].shape == (16, 20)


@pytest.mark.parametrize("optimizer", ["adam", "sgd", "momentum", "muon"])
def test_the_optimizer_option_changes_the_updates_not_the_start(tmp_path, optimizer):
    adamw = run(tmp_path / "adamw", "--steps", "3")["train_loss"]
    other = run(tmp_path / optimizer, "--steps", "3", "--optimizer", optimizer)
    assert other["args"]["optimizer"] == optimizer
    assert other["train_loss"][0] == adamw[0]  # same weights and batch before any update
    assert other["train_loss"][1:] != adamw[1:]


def test_momentum_is_not_plain_sgd(tmp_path):
    sgd = run(tmp_path / "sgd", "--steps", "4", "--optimizer", "sgd")["train_loss"]
    momentum = run(tmp_path / "momentum", "--steps", "4", "--optimizer", "momentum")["train_loss"]
    # Nesterov momentum already looks ahead on the first update, so the runs part from step 1 on
    assert sgd[0] == momentum[0] and sgd[1:] != momentum[1:]
