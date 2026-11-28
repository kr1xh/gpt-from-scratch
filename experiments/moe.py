"""Study: at equal active parameters per token, does a mixture of experts beat a dense MLP, and
does it keep its experts busy?

All runs use the modern block (RMSNorm, SwiGLU, rotary embeddings, grouped-query attention).
The dense model's SwiGLU MLP has width h (8d/3 rounded to a multiple of 8). The mixtures send
each token to its top 2 of 4 or 8 SwiGLU experts of width h/2, so every token passes through
the same number of MLP weights; the mixtures only hold more of them in total. A fourth run drops
the load-balancing loss, to see what it is for.

    python -m experiments.moe     # about 2 hours on a 4-core CPU

Load balance is the busiest expert's share of the assignments divided by the fair share 1/E
(1 is perfect, E means one expert takes everything), taken at each eval on a validation batch.
Writes results/moe-<variant>.json per run, results/moe.json and docs/figures/moe.png.
"""
import argparse
import json
from pathlib import Path

from experiments.runs import train_or_resume
from scratchgpt import train

VARIANTS = ("dense", "moe-4", "moe-8", "moe-4-noaux")


def dense_hidden(d_model: int) -> int:
    return 8 * round(8 * d_model / 3 / 8)  # SwiGLU's 8d/3, made even for two experts of half the width


def variant_args(variant: str, d_model: int, n_kv_head: int) -> list[str]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}: use one of {', '.join(VARIANTS)}")
    modern = ["--norm", "rms", "--mlp", "swiglu", "--pos", "rope", "--n-kv-head", str(n_kv_head)]
    h = dense_hidden(d_model)
    if variant == "dense":
        return [*modern, "--mlp-hidden", str(h)]
    experts = variant.split("-")[1]
    aux = ["--moe-aux", "0"] if variant.endswith("noaux") else []
    return [*modern, "--n-expert", experts, "--top-k", "2", "--mlp-hidden", str(h // 2), *aux]


def imbalance(load) -> float:
    return max(load) * len(load)


INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"dense": "#2a78d6", "moe-4": "#eb6834", "moe-8": "#1baf7a", "moe-4-noaux": "#eda100"}


def plot(runs: dict, rows: list, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE,
                                      gridspec_kw={"width_ratios": [1, 1.3]})
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)

    names = [r["variant"] for r in rows]
    scores = [r["val_bits_per_byte"] for r in rows]
    bars = left.barh(names, scores, color=[COLORS[n] for n in names], height=0.6)
    left.invert_yaxis()
    left.grid(True, axis="x", color=GRID, linewidth=0.8)
    for bar, value in zip(bars, scores):
        left.annotate(f"{value:.3f}", (value, bar.get_y() + bar.get_height() / 2), xytext=(4, 0),
                      textcoords="offset points", color=INK, fontsize=9, va="center")
    left.set_xlim(0, max(scores) * 1.2)
    left.set_title("Final validation bits per character", color=INK, fontsize=11, loc="left")
    left.set_xlabel("bits per character (lower is better)", color=INK_2, fontsize=9)

    right.grid(True, color=GRID, linewidth=0.8)
    for variant, metrics in runs.items():
        evals = [e for e in metrics["evals"] if "expert_load" in e]
        if evals:
            worst = [max(imbalance(load) for load in e["expert_load"]) for e in evals]
            right.plot([e["step"] for e in evals], worst, color=COLORS[variant], linewidth=2, marker="o",
                       markersize=4, label=variant)
    right.axhline(1, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
    right.set_title("Load imbalance of the worst layer (1 = perfectly even)", color=INK, fontsize=11, loc="left")
    right.set_xlabel("step", color=INK_2, fontsize=9)
    right.set_ylabel("busiest expert's share / fair share", color=INK_2, fontsize=9)
    right.legend(frameon=False, fontsize=9, labelcolor=INK)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--n-kv-head", type=int, default=2)
    ap.add_argument("--out-dir", default=".")
    args, passthrough = ap.parse_known_args(argv)  # the rest (model size, steps, --data) go to train
    out = Path(args.out_dir)
    d_model = train.parse(passthrough).d_model

    runs, rows = {}, []
    for variant in args.variants.split(","):
        metrics = train_or_resume(["--name", f"moe-{variant}", "--out-dir", str(out), *passthrough,
                                   *variant_args(variant, d_model, args.n_kv_head)])
        runs[variant] = metrics
        last = metrics["evals"][-1]
        load = last.get("expert_load")
        rows.append({"variant": variant, "total_params": metrics["config"]["params"],
                     "val_bits_per_byte": metrics["final"]["val_bits_per_byte"],
                     "minutes": metrics["final"]["seconds"] / 60, "load": load,
                     "imbalance": max(imbalance(layer) for layer in load) if load else None})
    summary = {"rows": rows, "passthrough": passthrough}
    (out / "results" / "moe.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    plot(runs, rows, out / "docs" / "figures" / "moe.png")
    for r in rows:
        balance = f", imbalance {r['imbalance']:.2f}" if r["imbalance"] is not None else ""
        print(f"{r['variant']:>12}: {r['val_bits_per_byte']:.4f} bits/char, {r['total_params']:,} params, "
              f"{r['minutes']:.0f} min{balance}")
    return summary


if __name__ == "__main__":
    main()
