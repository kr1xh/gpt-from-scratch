"""Study: when does a transformer trained on part of a + b mod p generalise to the rest, and how
do weight decay and the optimizer change that?

Each run trains a 1-layer transformer full batch on a fixed random part of the addition table
(Power et al. 2022) with a constant learning rate, logging train and test accuracy. The runs
differ only in the optimizer (AdamW or Muon) and the weight decay. A run has memorised when its
training accuracy reaches 99%, and generalised when its test accuracy does.

    python -m experiments.grokking

Finished runs are reused. Writes results/grokking-<optimizer>-wd<decay>.json per run,
results/grokking.json and docs/figures/grokking.png.
"""
import argparse
import json
from pathlib import Path

from experiments.runs import cached
from scratchgpt import grokking
from scratchgpt.train import OPTIMIZERS

RUNS = "adamw:0,adamw:0.1,adamw:0.3,adamw:1,muon:1"


def parse_runs(text: str) -> list:
    runs = []
    for part in text.split(","):
        name, _, decay = part.partition(":")
        if name not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer {name!r}: use one of {', '.join(OPTIMIZERS)}")
        runs.append((name, float(decay)))
    return runs


INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]  # validated slots 1-5, in run order


def plot(histories: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.set_xscale("symlog", linthresh=100)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("step (linear to 100, then log)", color=INK_2, fontsize=9)

    # left: the classic picture for the strongest-decay AdamW run: train accuracy jumps first
    label = max((k for k in histories if k.startswith("adamw")), key=lambda k: float(k.split("wd")[1]), default=None)
    if label:
        h = histories[label]
        steps = [r["step"] for r in h]
        left.plot(steps, [r["train_acc"] for r in h], color=COLORS[0], linewidth=2, label="train")
        left.plot(steps, [r["test_acc"] for r in h], color=COLORS[1], linewidth=2, label="test")
        left.set_title(f"Accuracy, {label.replace('-', ', ')}", color=INK, fontsize=11, loc="left")
        left.legend(frameon=False, fontsize=9, labelcolor=INK)
    left.set_ylabel("accuracy", color=INK_2, fontsize=9)

    # right: test accuracy of every run
    for (name, h), color in zip(histories.items(), COLORS):
        right.plot([r["step"] for r in h], [r["test_acc"] for r in h], color=color, linewidth=2,
                   label=name.replace("-", ", "))
    right.set_title("Test accuracy of every run", color=INK, fontsize=11, loc="left")
    right.set_ylabel("test accuracy", color=INK_2, fontsize=9)
    right.legend(frameon=False, fontsize=9, labelcolor=INK)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=RUNS, help="optimizer:weight_decay,...")
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--frac", type=float, default=0.4, help="share of the table used for training")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)

    rows, histories = [], {}
    for optimizer, decay in parse_runs(args.runs):
        settings = {"p": args.p, "frac": args.frac, "steps": args.steps, "lr": args.lr, "weight_decay": decay,
                    "optimizer": optimizer, "d_model": args.d_model, "n_head": args.n_head,
                    "eval_every": args.eval_every, "seed": args.seed}
        name = f"grokking-{optimizer}-wd{decay:g}"

        def log(row, name=name):
            print(f"{name}  step {row['step']:6d}  train {row['train_acc']:.3f}  test {row['test_acc']:.3f}", flush=True)
        history = cached(out / "results" / f"{name}.json", settings, lambda: grokking.train(**settings, log=log))
        histories[f"{optimizer}-wd{decay:g}"] = history
        rows.append({"optimizer": optimizer, "weight_decay": decay,
                     "memorised_at": grokking.first_step(history, "train_acc", 0.99),
                     "generalised_at": grokking.first_step(history, "test_acc", 0.99),
                     "final_train_acc": history[-1]["train_acc"], "final_test_acc": history[-1]["test_acc"]})
    summary = {"rows": rows, "settings": {k: v for k, v in vars(args).items() if k not in ("runs", "out_dir")}}
    (out / "results" / "grokking.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    plot(histories, out / "docs" / "figures" / "grokking.png")
    for r in rows:
        print(f"{r['optimizer']:>6} wd {r['weight_decay']:<4g}: memorised at {r['memorised_at']}, "
              f"generalised at {r['generalised_at']}, final test accuracy {r['final_test_acc']:.3f}")
    return summary


if __name__ == "__main__":
    main()
