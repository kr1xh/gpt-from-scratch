"""Study: which optimizer trains the baseline GPT best, how fast, and how picky is it about lr?

SGD, Nesterov momentum, Adam, AdamW and Muon each train the baseline model at three learning
rates around their usual range, with the same warmup, cosine schedule, clipping, batches and
seed. Muon updates the hidden matrices; the embeddings and vectors in its runs use AdamW at the
same lr (its update is scaled to match AdamW's size, so one lr serves both). Weight decay 0.1 is
decoupled, so AdamW and Muon get it; SGD, momentum and Adam get none, because the same 0.1 added
to their gradients would be a different and much stronger regulariser.

    python -m experiments.optimizers     # about 5 hours on a 4-core CPU: run it overnight

For each optimizer: the best validation bits per character over its lrs (whole split), the steps
its best run took to bring a running mean of the training loss under a target, and the spread
of its final scores across the lrs. Finished runs are reused. Writes
results/optimizers-<optimizer>-lr<lr>.json per run, results/optimizers.json and
docs/figures/optimizers.png.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.runs import train_or_resume
from scratchgpt.train import OPTIMIZERS

GRID = "sgd:0.1,0.3,1;momentum:0.03,0.1,0.3;adam:1e-3,2e-3,4e-3;adamw:1e-3,2e-3,4e-3;muon:1e-3,2e-3,4e-3"
DECOUPLED = ("adamw", "muon")


def parse_grid(text: str) -> dict:
    grid = {}
    for part in text.split(";"):
        name, _, lrs = part.partition(":")
        if name not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer {name!r}: use one of {', '.join(OPTIMIZERS)}")
        grid[name] = [float(lr) for lr in lrs.split(",")]
    return grid


def steps_to_target(train_loss: list, target_bits: float, window: int):
    """The first step at which the mean of the last `window` training losses (nats) is at or
    below `target_bits`, or None if the run never got there."""
    bits = np.asarray(train_loss) / math.log(2)
    means = np.convolve(bits, np.ones(window) / window, mode="valid")  # means[i] ends at step i + window - 1
    hit = np.flatnonzero(means <= target_bits)
    return int(hit[0]) + window - 1 if hit.size else None


def summarize(runs: list) -> list:
    """Per optimizer: its best run, that run's steps to the target, and the spread of its scores
    over the lrs. A run that diverged (a NaN or infinite score) is never the best, and makes the
    spread infinite: that optimizer cannot use every lr in its grid."""
    optimizers = []
    for name in dict.fromkeys(r["optimizer"] for r in runs):
        mine = [r for r in runs if r["optimizer"] == name]
        diverged = [r["lr"] for r in mine if not math.isfinite(r["val_bits_per_byte"])]
        finite = [r for r in mine if math.isfinite(r["val_bits_per_byte"])] or mine
        best = min(finite, key=lambda r: r["val_bits_per_byte"])
        scores = [r["val_bits_per_byte"] for r in finite]
        optimizers.append({"optimizer": name, "best_name": best["name"], "best_lr": best["lr"],
                           "best_bits_per_byte": best["val_bits_per_byte"], "steps_to_target": best["steps_to_target"],
                           "lr_spread": math.inf if diverged else max(scores) - min(scores), "diverged": diverged})
    return optimizers


INK, INK_2, GRID_LINE, SURFACE ="#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"sgd": "#2a78d6", "momentum": "#eb6834", "adam": "#1baf7a", "adamw": "#eda100", "muon": "#6250d6"}


def plot(runs: list, best_curves: dict, smooth: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        ax.grid(True, color=GRID_LINE, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID_LINE)
        ax.tick_params(colors=INK_2, labelsize=9)

    # left: final score against learning rate, one line per optimizer
    for name, color in COLORS.items():
        mine = sorted((r["lr"], r["val_bits_per_byte"]) for r in runs if r["optimizer"] == name)
        if mine:
            left.plot(*zip(*mine), color=color, linewidth=2, marker="o", markersize=5, label=name)
    left.set_xscale("log")
    left.set_title("Final validation bits per character by learning rate", color=INK, fontsize=11, loc="left")
    left.set_xlabel("peak learning rate (log scale)", color=INK_2, fontsize=9)
    left.set_ylabel("bits per character (lower is better)", color=INK_2, fontsize=9)
    left.legend(frameon=False, fontsize=9, labelcolor=INK)

    # right: training loss of each optimizer's best run, as a running mean
    for name, loss in best_curves.items():
        bits = np.convolve(np.asarray(loss) / math.log(2), np.ones(smooth) / smooth, mode="valid")
        right.plot(np.arange(smooth - 1, len(loss)), bits, color=COLORS[name], linewidth=2, label=name)
    right.set_title(f"Training loss of each best run ({smooth}-step mean)", color=INK, fontsize=11, loc="left")
    right.set_xlabel("step", color=INK_2, fontsize=9)
    right.set_ylabel("bits per character", color=INK_2, fontsize=9)
    right.legend(frameon=False, fontsize=9, labelcolor=INK)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default=GRID, help="optimizer:lr,lr;optimizer:lr,...")
    ap.add_argument("--target-bits", type=float, default=2.75, help="training loss target, bits per character")
    ap.add_argument("--smooth", type=int, default=50, help="steps in the running mean of the training loss")
    ap.add_argument("--out-dir", default=".")
    args, passthrough = ap.parse_known_args(argv)  # the rest (model size, steps, --data) go to train
    out = Path(args.out_dir)

    runs, losses = [], {}
    for name, lrs in parse_grid(args.grid).items():
        for lr in lrs:
            run_name = f"optimizers-{name}-lr{lr:g}"
            metrics = train_or_resume(["--name", run_name, "--out-dir", str(out), *passthrough,
                                       "--optimizer", name, "--lr", str(lr),
                                       "--weight-decay", "0.1" if name in DECOUPLED else "0"])
            runs.append({"name": run_name, "optimizer": name, "lr": lr,
                         "val_bits_per_byte": metrics["final"]["val_bits_per_byte"],
                         "steps_to_target": steps_to_target(metrics["train_loss"], args.target_bits, args.smooth),
                         "minutes": metrics["final"]["seconds"] / 60})
            losses[run_name] = metrics["train_loss"]

    optimizers = summarize(runs)
    best_curves = {o["optimizer"]: losses[o["best_name"]] for o in optimizers}
    summary = {"runs": runs, "optimizers": optimizers, "target_bits": args.target_bits, "smooth": args.smooth,
               "passthrough": passthrough}
    (out / "results" / "optimizers.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    plot(runs, best_curves, args.smooth, out / "docs" / "figures" / "optimizers.png")
    for o in optimizers:
        print(f"{o['optimizer']:>8}: best {o['best_bits_per_byte']:.4f} bits/char at lr {o['best_lr']:g}, "
              f"target after {o['steps_to_target']} steps, spread over lrs {o['lr_spread']:.3f}")
    return summary


if __name__ == "__main__":
    main()
