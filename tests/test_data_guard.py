import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_data_guard as guard  # noqa: E402

MB = 1024 * 1024


def sizes(**by_path):
    return lambda p: by_path.get(p, 0)


def test_code_docs_and_figures_pass():
    paths = ["README.md", "src/scratchgpt/tensor.py", "configs/baseline.json", "docs/figures/curves.png"]
    assert guard.check(paths, size_of=sizes()) == []


def test_datasets_checkpoints_and_run_outputs_are_blocked():
    for path in ["data/input.txt", "models/baseline.npz", "results/metrics.json",
                 "ckpt.npz", "x.npy", "w.pt", "w.pth", "w.ckpt", "w.safetensors", "w.bin"]:
        assert guard.check([path], size_of=sizes()), path


def test_small_demo_weights_are_allowed():
    assert guard.check(["web/weights/model.bin"], size_of=sizes(**{"web/weights/model.bin": 3 * MB})) == []


def test_oversized_demo_weights_are_blocked():
    problems = guard.check(["web/weights/model.bin"], size_of=sizes(**{"web/weights/model.bin": 9 * MB}))
    assert problems and "8 MB" in problems[0]


def test_weights_outside_the_demo_folder_are_blocked_even_when_small():
    assert guard.check(["web/model.bin"], size_of=sizes(**{"web/model.bin": 1}))


def test_windows_separators_are_normalised():
    assert guard.check(["data\\input.txt"], size_of=sizes())
