"""Study: does the trained model have induction heads, and does it rely on them?

On random token sequences repeated twice, the second half can only be predicted by copying from
the first. The study scores every head by its prefix-matching attention on such sequences, then
ablates the top-scoring heads, and as a control the same number of lowest-scoring heads, and
measures the loss on the repeated halves and on ordinary validation text. It also reads the
next-token loss off the residual stream after each layer (the logit lens).

    python -m experiments.interp --checkpoint models/modern_block-modern-s0.npz

Writes results/interp.json and docs/figures/interp.png.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scratchgpt import data, interp, sample
from scratchgpt.tensor import no_grad


def text_batch(tok, text: str, block: int, batch: int, seed: int) -> np.ndarray:
    _, val = data.split(np.array(tok.encode(text)))
    starts = np.random.default_rng(seed).integers(0, len(val) - block, batch)
    return np.stack([val[s:s + block] for s in starts])


def losses(model, repeated, text, n) -> dict:
    return {"repeated": interp.mean_loss(model, repeated, start=n), "text": interp.mean_loss(model, text)}


INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8d8b85"


def plot(summary: dict, top_map: np.ndarray, n: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)
    (heat, attn), (bars, lens) = axes

    scores = np.array(summary["prefix_matching"])
    im = heat.imshow(scores, cmap="Blues", vmin=0, vmax=max(1e-6, scores.max()), aspect="auto")
    for (layer, head), value in np.ndenumerate(scores):
        heat.annotate(f"{value:.2f}", (head, layer), ha="center", va="center", fontsize=8,
                      color="white" if value > 0.6 * scores.max() else INK)
    heat.set_xticks(range(scores.shape[1]))
    heat.set_yticks(range(scores.shape[0]))
    heat.set_xlabel("head", color=INK_2, fontsize=9)
    heat.set_ylabel("layer", color=INK_2, fontsize=9)
    heat.set_title("Prefix-matching score per head", color=INK, fontsize=11, loc="left")
    fig.colorbar(im, ax=heat, fraction=0.04)

    attn.imshow(top_map, cmap="Blues", vmin=0, vmax=top_map.max(), aspect="auto")
    attn.axhline(n - 0.5, color=ORANGE, linewidth=1)
    attn.axvline(n - 0.5, color=ORANGE, linewidth=1)
    layer, head = summary["ablation"]["top"]["heads"][0]
    attn.set_title(f"Layer {layer}, head {head} on a repeated sequence", color=INK, fontsize=11, loc="left")
    attn.set_xlabel("attends to position", color=INK_2, fontsize=9)
    attn.set_ylabel("from position (copy starts at the line)", color=INK_2, fontsize=9)

    base = summary["loss"]
    groups = [("repeated second half", "repeated"), ("validation text", "text")]
    x = np.arange(len(groups))
    for offset, (label, key, color) in zip((-0.2, 0.2), (("top heads ablated", "top", ORANGE),
                                                        ("lowest heads ablated", "bottom", GRAY))):
        rises = [summary["ablation"][key]["loss"][k] - base[k] for _, k in groups]
        drawn = bars.bar(x + offset, rises, width=0.38, color=color, label=label)
        for rect, value in zip(drawn, rises):
            bars.annotate(f"{value:+.2f}", (rect.get_x() + rect.get_width() / 2, value), xytext=(0, 3),
                          textcoords="offset points", ha="center", fontsize=8, color=INK)
    bars.axhline(0, color=GRID, linewidth=1)
    bars.set_xticks(x, [g for g, _ in groups])
    bars.set_ylabel("loss increase (nats per token)", color=INK_2, fontsize=9)
    bars.set_title(f"Ablating {len(summary['ablation']['top']['heads'])} heads", color=INK, fontsize=11, loc="left")
    bars.legend(frameon=False, fontsize=9, labelcolor=INK)
    bars.grid(True, axis="y", color=GRID, linewidth=0.8)
    bars.set_axisbelow(True)

    lens.plot(range(len(summary["logit_lens"])), summary["logit_lens"], color=BLUE, linewidth=2, marker="o")
    lens.set_xticks(range(len(summary["logit_lens"])),
                    ["embedding"] + [f"block {i}" for i in range(len(summary["logit_lens"]) - 1)])
    lens.set_ylabel("next-token loss on validation text (nats)", color=INK_2, fontsize=9)
    lens.set_title("Logit lens: loss read off the residual stream", color=INK, fontsize=11, loc="left")
    lens.grid(True, color=GRID, linewidth=0.8)
    lens.set_axisbelow(True)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="models/modern_block-modern-s0.npz")
    ap.add_argument("--data", help="the text the model was trained on; default: TinyShakespeare")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ablate", type=int, default=2, help="heads to ablate in each group")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)

    model, tok = sample.load(args.checkpoint)
    block = model.config.block_size
    n = block // 2
    text = Path(args.data).read_text(encoding="utf-8") if args.data else data.load_tinyshakespeare()
    repeated = interp.repeated_random_tokens(tok.vocab_size, n, args.batch, args.seed)
    normal = text_batch(tok, text, block, args.batch, args.seed)

    maps = interp.attention_maps(model, repeated)
    scores = interp.prefix_matching_scores(maps, n)
    order = [divmod(int(i), scores.shape[1]) for i in np.argsort(-scores, axis=None)]  # best first
    groups = {"top": order[:args.ablate], "bottom": order[::-1][:args.ablate]}

    summary = {"checkpoint": str(args.checkpoint), "prefix_matching": scores.tolist(),
               "loss": {**losses(model, repeated, normal, n), "random_half": interp.mean_loss(model, repeated[:, :n])},
               "ablation": {}, "logit_lens": []}
    for name, heads in groups.items():
        with interp.ablated(model, heads):
            summary["ablation"][name] = {"heads": [list(h) for h in heads], "loss": losses(model, repeated, normal, n)}
    for log_probs in interp.logit_lens(model, normal[:, :-1]):
        picked = np.take_along_axis(log_probs, normal[:, 1:, None], axis=-1)
        summary["logit_lens"].append(float(-picked.mean()))

    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results" / "interp.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    layer, head = groups["top"][0]
    plot(summary, maps[layer][0, head], n, out / "docs" / "figures" / "interp.png")
    print(f"top heads {groups['top']} (scores {[round(float(scores[l, h]), 3) for l, h in groups['top']]})")
    for name in ("top", "bottom"):
        ab = summary["ablation"][name]["loss"]
        print(f"ablate {name:>6}: repeated {summary['loss']['repeated']:.3f} -> {ab['repeated']:.3f}, "
              f"text {summary['loss']['text']:.3f} -> {ab['text']:.3f}")
    return summary


if __name__ == "__main__":
    main()
