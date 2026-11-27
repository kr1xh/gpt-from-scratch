"""Study: at equal size and steps, does the modern block beat the GPT-2 block, and which part helps?

Six variants of the baseline model: the GPT-2 block, each modern part alone (RMSNorm, SwiGLU,
rotary embeddings, grouped-query attention) and all four together. Each variant gets the MLP
width that brings it to the GPT-2 model's parameter count, so parts that save parameters (RoPE
drops the position table, GQA shrinks the key/value projections) hand them back to the MLP.
GPT-2 and the full modern block are repeated over several seeds to measure run-to-run noise.

    python -m experiments.modern_block     # about 4 hours on a 4-core CPU: run it overnight

Each run is scored in bits per character on the whole validation split (char tokenizer).
Finished runs are reused. Writes results/modern_block-<variant>-s<seed>.json per run,
results/modern_block.json and docs/figures/modern_block.png.
"""
import argparse
import json
import math
from pathlib import Path

from experiments.runs import train_or_resume
from scratchgpt import train
from scratchgpt.nn import GPT, GPTConfig

VARIANTS = {
    "gpt2": [],
    "rmsnorm": ["--norm", "rms"],
    "swiglu": ["--mlp", "swiglu"],
    "rope": ["--pos", "rope"],
    "gqa": [],  # filled in with --n-kv-head
    "modern": ["--norm", "rms", "--mlp", "swiglu", "--pos", "rope"],
}
REPEATED = ("gpt2", "modern")  # the two that get several seeds


def variant_args(variant: str, n_kv_head: int) -> list[str]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}: use one of {', '.join(VARIANTS)}")
    gqa = ["--n-kv-head", str(n_kv_head)] if variant in ("gqa", "modern") else []
    return VARIANTS[variant] + gqa


def count_params(argv: list[str]) -> int:
    """Parameters of the model `train` would build from argv, without the token embedding (the
    same for every variant, and dependent on the data)."""
    a = train.parse(argv)
    cfg = GPTConfig(1, a.block_size, a.n_layer, a.n_head, a.d_model, norm=a.norm, mlp=a.mlp, pos=a.pos,
                    n_kv_head=a.n_kv_head, mlp_hidden=a.mlp_hidden)
    return sum(p.size for name, p in GPT(cfg).named_parameters() if name != "wte.weight")


def matched_hidden(argv: list[str], target: int) -> int:
    """The MLP width that brings the model built from argv closest to `target` parameters. The
    count grows linearly with the width, so two counts give the line."""
    one, two = (count_params([*argv, "--mlp-hidden", str(h)]) for h in (1, 2))
    return max(1, round(1 + (target - one) / (two - one)))


def summarize(runs: list) -> list:
    """Per variant, in order of first appearance: the number of seeds, the mean score and its range."""
    variants = []
    for variant in dict.fromkeys(r["variant"] for r in runs):
        scores = [r["val_bits_per_byte"] for r in runs if r["variant"] == variant]
        variants.append({"variant": variant, "n": len(scores), "mean": sum(scores) / len(scores),
                         "low": min(scores), "high": max(scores)})
    return variants


INK, INK_2, GRID, SURFACE ="#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, BAND = "#2a78d6", "#eb6834", "#e9eef6"


def plot(curves: dict, runs: list, variants: list, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE,
                                      gridspec_kw={"width_ratios": [1.2, 1]})
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)

    # left: the sampled validation loss during training, seed 0 of GPT-2 and of the modern block;
    # step 0 (a uniform guess) is left out so the part where they differ is readable
    left.grid(True, color=GRID, linewidth=0.8)
    for variant, color in (("gpt2", BLUE), ("modern", ORANGE)):
        if variant in curves:
            evals = [e for e in curves[variant]["evals"] if e["step"] > 0]
            left.plot([e["step"] for e in evals], [e["val_loss"] / math.log(2) for e in evals], color=color,
                      linewidth=2, marker="o", markersize=4, label=variant)
    left.set_title("Validation bits per character during training (seed 0)", color=INK, fontsize=11, loc="left")
    left.set_xlabel("step", color=INK_2, fontsize=9)
    left.set_ylabel("bits per character (sampled batches)", color=INK_2, fontsize=9)
    left.legend(frameon=False, fontsize=9, labelcolor=INK)

    # right: every run's final score; the band is the spread of the GPT-2 seeds
    names = [v["variant"] for v in variants]
    rows = {name: i for i, name in enumerate(names)}
    base = next((v for v in variants if v["variant"] == "gpt2"), None)
    if base is not None:
        right.axvspan(base["low"], base["high"], color=BAND, zorder=0)
        right.axvline(base["mean"], color=BLUE, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    right.grid(True, axis="x", color=GRID, linewidth=0.8)
    for r in runs:
        right.scatter(r["val_bits_per_byte"], rows[r["variant"]], s=40, color=BLUE, edgecolor=SURFACE,
                      linewidth=1.5, zorder=3)
    for v in variants:
        right.annotate(f"{v['mean']:.3f}", (v["high"], rows[v["variant"]]), xytext=(8, 0),
                       textcoords="offset points", color=INK, fontsize=9, va="center")
    right.set_yticks(range(len(names)), names)
    right.set_ylim(len(names) - 0.5, -0.5)
    lo = min(r["val_bits_per_byte"] for r in runs)
    hi = max(r["val_bits_per_byte"] for r in runs)
    pad = max(hi - lo, 0.01)
    right.set_xlim(lo - 0.15 * pad, hi + 0.45 * pad)
    right.set_title("Final, whole validation split", color=INK, fontsize=11, loc="left")
    right.set_xlabel("bits per character (lower is better; band: GPT-2 seeds)", color=INK_2, fontsize=9)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", type=int, default=3, help="seeds for gpt2 and modern; the others run seed 0")
    ap.add_argument("--n-kv-head", type=int, default=2, help="key/value heads in the gqa and modern variants")
    ap.add_argument("--out-dir", default=".")
    args, passthrough = ap.parse_known_args(argv)  # the rest (model size, steps, --data) go to train
    out = Path(args.out_dir)
    target = count_params(passthrough)

    runs, curves = [], {}
    for variant in args.variants.split(","):
        model_args = [*passthrough, *variant_args(variant, args.n_kv_head)]
        hidden = matched_hidden(model_args, target)
        for seed in range(args.seeds if variant in REPEATED else 1):
            name = f"modern_block-{variant}-s{seed}"
            metrics = train_or_resume(["--name", name, "--out-dir", str(out), *model_args,
                                       "--mlp-hidden", str(hidden), "--seed", str(seed)])
            final = metrics["final"]
            runs.append({"variant": variant, "seed": seed, "params": metrics["config"]["params"],
                         "mlp_hidden": hidden, "val_bits_per_byte": final["val_bits_per_byte"],
                         "minutes": final["seconds"] / 60})
            if seed == 0:
                curves[variant] = metrics

    variants = summarize(runs)
    summary = {"runs": runs, "variants": variants, "passthrough": passthrough}
    (out / "results" / "modern_block.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    plot(curves, runs, variants, out / "docs" / "figures" / "modern_block.png")
    for v in variants:
        print(f"{v['variant']:>8}: {v['mean']:.4f} bits/char (n={v['n']}, {v['low']:.4f} to {v['high']:.4f})")
    return summary


if __name__ == "__main__":
    main()
