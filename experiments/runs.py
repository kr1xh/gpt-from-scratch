"""Shared by the studies: train a run, or reuse it if it already finished with the same settings."""
import json
from pathlib import Path

from scratchgpt import train


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
