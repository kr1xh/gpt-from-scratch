"""Study: at equal compute, which tokenizer gives the best validation bits per character?

Every variant trains the same model for the same steps on the same number of tokens per step;
only the tokenizer changes (characters, or byte-level BPE with 256, 512 or 1024 ids). Loss per
token cannot be compared across tokenizers, so each run is scored in bits per byte of the whole
validation split (for ASCII TinyShakespeare, bits per character); see scratchgpt.evaluate.

    python -m experiments.char_vs_bpe     # about 1.5 hours on a 4-core CPU

Runs are resumable: a variant whose results already hold a final score is not trained again.
Writes results/char_vs_bpe-<variant>.json per run, results/char_vs_bpe.json and
docs/figures/char_vs_bpe.png.
"""
import argparse
import json
import math
from pathlib import Path

from scratchgpt import train

VARIANTS = "char,bpe-256,bpe-512,bpe-1024"


def variant_args(name: str) -> list[str]:
    if name == "char":
        return ["--tokenizer", "char"]
    kind, _, size = name.partition("-")
    if kind != "bpe" or not size.isdigit():
        raise ValueError(f"unknown variant {name!r}: use 'char' or 'bpe-<vocab size>'")
    return ["--tokenizer", "bpe", "--vocab-size", size]


def row(variant: str, metrics: dict) -> dict:
    cfg, final = metrics["config"], metrics["final"]
    bytes_per_token = final["val_bytes"] / final["val_tokens"]
    return {"variant": variant, "vocab_size": cfg["vocab_size"], "params": cfg["params"],
            "val_bits_per_byte": final["val_bits_per_byte"], "bytes_per_token": bytes_per_token,
            "bytes_per_step": cfg["batch_size"] * cfg["block_size"] * bytes_per_token,
            "minutes": final["seconds"] / 60}


INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4, in order


def plot(runs: dict, rows: list, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE,
                                      gridspec_kw={"width_ratios": [1.5, 1]})
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)

    # during training: eval loss (nats per token, sampled batches) turned into bits per byte with
    # each tokenizer's measured bytes per token; the exact whole-split score is on the right
    left.grid(True, color=GRID, linewidth=0.8)
    ends = []
    for (variant, metrics), r, color in zip(runs.items(), rows, COLORS):
        steps = [e["step"] for e in metrics["evals"]]
        bpb = [e["val_loss"] / math.log(2) / r["bytes_per_token"] for e in metrics["evals"]]
        left.plot(steps, bpb, color=color, linewidth=2, marker="o", markersize=4, label=variant)
        ends.append((bpb[-1], variant, steps[-1]))
    lo, hi = left.get_ylim()
    gap, previous = 0.05 * (hi - lo), -math.inf
    for y, variant, x in sorted(ends):  # curves end close together: spread their labels apart
        previous = max(y, previous + gap)
        left.annotate(variant, (x, y), xytext=(x + 40, previous), textcoords="data",
                      color=INK, fontsize=9, va="center", annotation_clip=False)
    left.set_title("Validation bits per character during training", color=INK, fontsize=11, loc="left")
    left.set_xlabel("step (same tokens per step for every variant)", color=INK_2, fontsize=9)
    left.set_ylabel("bits per character (estimated)", color=INK_2, fontsize=9)
    left.legend(frameon=False, fontsize=9, labelcolor=INK)

    names = [r["variant"] for r in rows]
    scores = [r["val_bits_per_byte"] for r in rows]
    bars = right.barh(names, scores, color=COLORS[:len(rows)], height=0.6)
    right.invert_yaxis()
    right.grid(True, axis="x", color=GRID, linewidth=0.8)
    for bar, value in zip(bars, scores):
        right.annotate(f"{value:.3f}", (value, bar.get_y() + bar.get_height() / 2), xytext=(4, 0),
                       textcoords="offset points", color=INK, fontsize=9, va="center")
    right.set_xlim(0, max(scores) * 1.18)
    right.set_title("Final, whole validation split", color=INK, fontsize=11, loc="left")
    right.set_xlabel("bits per character (lower is better)", color=INK_2, fontsize=9)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default=VARIANTS)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--cache-dir", default=None, help="default: <out-dir>/data/cache")
    args, passthrough = ap.parse_known_args(argv)  # the rest (model size, steps, --data) go to train
    out = Path(args.out_dir)
    cache = args.cache_dir or str(out / "data" / "cache")

    runs = {}
    for variant in args.variants.split(","):
        name = f"char_vs_bpe-{variant}"
        saved = out / "results" / f"{name}.json"
        argv_for_run = ["--name", name, "--out-dir", str(out), "--cache-dir", cache,
                        *variant_args(variant), *passthrough]
        metrics = json.loads(saved.read_text()) if saved.exists() else {}
        wanted = vars(train.parse(argv_for_run))
        # an option that train.py gained after the run was saved counts as left at its default
        same_settings = "args" in metrics and {**vars(train.parse([])), **metrics["args"]} == wanted
        if "final" not in metrics or not same_settings:  # not run, stopped early, or other settings
            metrics = train.main(argv_for_run)
        runs[variant] = metrics

    rows = [row(v, m) for v, m in runs.items()]
    summary = {"rows": rows, "passthrough": passthrough}
    (out / "results" / "char_vs_bpe.json").write_text(json.dumps(summary, indent=1))
    plot(runs, rows, out / "docs" / "figures" / "char_vs_bpe.png")
    for r in rows:
        print(f"{r['variant']:>9}: {r['val_bits_per_byte']:.4f} bits/char, {r['bytes_per_token']:.2f} chars/token, "
              f"{r['params']:,} params, {r['minutes']:.0f} min")
    return summary


if __name__ == "__main__":
    main()
