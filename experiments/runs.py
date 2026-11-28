"""Shared by the studies: train a run, or reuse it if it already finished with the same settings."""
import json
from pathlib import Path

from scratchgpt import train


def cached(path: Path, settings: dict, compute):
    """compute() and save its result with its settings, unless `path` already holds a result saved
    with exactly these settings; either way return the result."""
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("settings") == settings:
            return saved["result"]
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"settings": settings, "result": result}, indent=1), encoding="utf-8")
    return result


def train_or_resume(argv: list[str]) -> dict:
    """Return the metrics of `python -m scratchgpt.train <argv>`. If results/<name>.json under
    --out-dir already holds a finished run (with its final score) and the same settings, it is
    reused instead of trained again. An option that train.py gained after that run was saved
    counts as left at its default."""
    wanted = vars(train.parse(argv))
    saved = Path(wanted["out_dir"]) / "results" / f"{wanted['name']}.json"
    metrics = json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else {}
    same_settings = "args" in metrics and {**vars(train.parse([])), **metrics["args"]} == wanted
    if "final" not in metrics or not same_settings:  # not run, stopped early, or other settings
        metrics = train.main(argv)
    return metrics
