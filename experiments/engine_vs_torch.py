"""Study: does training on this engine follow PyTorch, and how much slower is it?

Both sides start from the same weights, see the same batches in the same order and use the same
recipe as scratchgpt.train (AdamW with decay on matrices only, warmup + cosine, clipping at 1.0).
The run is repeated in float64 as a control: there the two must agree to many digits, so any gap
in float32 is rounding, not a difference in the maths.

    python -m experiments.engine_vs_torch            # needs the [dev] extra (torch)
    python -m experiments.engine_vs_torch --replot   # redraw from the saved results

Writes results/engine_vs_torch.json and docs/figures/engine_vs_torch.png.
"""
import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

# the same thread budget on both sides: one per physical core (see scratchgpt.train)
THREADS = max(1, (os.cpu_count() or 2) // 2)
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(THREADS))

import numpy as np
import torch

from experiments import torch_gpt
from scratchgpt import data, optim
from scratchgpt.nn import GPT, GPTConfig
from scratchgpt.tokenizers import CharTokenizer
from scratchgpt.train import BETAS, make_optimizers

HP = {"lr": 2e-3, "min_lr": 2e-4, "warmup": 100, "weight_decay": 0.1, "grad_clip": 1.0}


def make_batches(ids, steps: int, batch_size: int, block_size: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    return [data.get_batch(ids, batch_size, block_size, rng) for _ in range(steps)]


def lr_at(step: int, steps: int, hp: dict) -> float:
    return optim.cosine_lr(step, hp["lr"], hp["warmup"], steps, hp["min_lr"])


def run_ours(cfg, state, batches, hp, dtype) -> dict:
    model = GPT(cfg)
    model.load_state_dict(state)  # copies, so the caller's initial weights stay untouched
    params = model.parameters()
    for p in params:
        p.data = p.data.astype(dtype)
    opts = make_optimizers(model, hp["lr"], hp["weight_decay"])
    losses, seconds = [], []
    for step, (x, y) in enumerate(batches):
        start = time.perf_counter()
        for opt in opts:
            opt.lr = lr_at(step, len(batches), hp)
        model.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optim.clip_grad_norm(params, hp["grad_clip"])
        for opt in opts:
            opt.step()
        seconds.append(time.perf_counter() - start)
        losses.append(loss.item())
    return {"loss": losses, "seconds": seconds, "state": model.state_dict()}


def run_torch(cfg, state, batches, hp, dtype) -> dict:
    P = torch_gpt.parameters(state, torch.float64 if dtype == np.float64 else torch.float32)
    matrices = [p for p in P.values() if p.ndim >= 2]
    others = [p for p in P.values() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": matrices, "weight_decay": hp["weight_decay"]},
                             {"params": others, "weight_decay": 0.0}], lr=hp["lr"], betas=BETAS, eps=1e-8)
    losses, seconds = [], []
    for step, (x, y) in enumerate(batches):
        start = time.perf_counter()
        for group in opt.param_groups:
            group["lr"] = lr_at(step, len(batches), hp)
        opt.zero_grad(set_to_none=True)
        _, loss = torch_gpt.forward(P, cfg, torch.from_numpy(x), torch.from_numpy(y))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(P.values()), hp["grad_clip"])
        opt.step()
        seconds.append(time.perf_counter() - start)
        losses.append(loss.item())
    return {"loss": losses, "seconds": seconds, "state": {k: v.detach().numpy().copy() for k, v in P.items()}}


def summarize(ours: dict, theirs: dict, windows: int = 5) -> dict:
    diff = np.abs(np.array(ours["loss"]) - np.array(theirs["loss"]))
    # the first step pays one-off costs (allocations, thread start-up), so medians skip it
    ours_ms = float(np.median(ours["seconds"][1:]) * 1000)
    torch_ms = float(np.median(theirs["seconds"][1:]) * 1000)
    return {
        "max_abs_loss_diff": float(diff.max()),
        "final_abs_loss_diff": float(diff[-1]),
        "exact_match_fraction": float(np.mean(diff == 0)),
        "windows": [{"steps": [int(w[0]), int(w[-1])], "max_abs_loss_diff": float(diff[w].max()),
                     "mean_abs_loss_diff": float(diff[w].mean())}
                    for w in np.array_split(np.arange(len(diff)), min(windows, len(diff)))],
        "ours_ms_per_step": ours_ms,
        "torch_ms_per_step": torch_ms,
        "slowdown": ours_ms / torch_ms,
    }


INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
OURS, TORCH, F32, F64 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"  # validated categorical slots


def plot(runs: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.set_xlabel("step", color=INK_2, fontsize=9)

    f32 = runs["float32"]
    steps = np.arange(len(f32["ours"]["loss"]))
    left.plot(steps, f32["ours"]["loss"], color=OURS, linewidth=2, label="this engine (NumPy)")
    left.plot(steps, f32["torch"]["loss"], color=TORCH, linewidth=2, linestyle=(0, (4, 3)), label="PyTorch")
    left.set_title("Training loss, float32: the two curves overlap", color=INK, fontsize=11, loc="left")
    left.set_ylabel("loss (nats per character)", color=INK_2, fontsize=9)
    left.legend(frameon=False, fontsize=9, labelcolor=INK)

    # many steps agree bit for bit (a gap of exactly 0), so a raw log plot flickers to the floor;
    # a 25-step mean shows the level, and the dashed lines mark each format's rounding size
    window = min(25, len(steps))
    for name, color, eps in (("float32", F32, np.finfo(np.float32).eps), ("float64", F64, np.finfo(np.float64).eps)):
        gap = np.abs(np.array(runs[name]["ours"]["loss"]) - np.array(runs[name]["torch"]["loss"]))
        mean_gap = np.maximum(np.convolve(gap, np.ones(window) / window, mode="valid"), 1e-18)
        x = steps[window - 1:]
        right.semilogy(x, mean_gap, color=color, linewidth=2, label=f"{name} gap")
        right.annotate(f"{name} gap", (x[-1], mean_gap[-1]), xytext=(4, 0), textcoords="offset points",
                       color=INK, fontsize=9, va="center")
        rounding = eps * float(np.median(runs[name]["ours"]["loss"]))
        right.axhline(rounding, color=INK_2, linewidth=1, linestyle=(0, (2, 3)))
        right.annotate(f"{name} rounding (ε × loss)", (0, rounding), xytext=(2, 4), textcoords="offset points",
                       color=INK_2, fontsize=8)
    right.set_title(f"|engine loss − PyTorch loss|, {window}-step mean", color=INK, fontsize=11, loc="left")
    right.set_ylabel("absolute difference (log scale)", color=INK_2, fontsize=9)
    right.set_ylim(1e-18, 1e-4)
    right.legend(frameon=False, fontsize=9, labelcolor=INK, loc="center left")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="a text file; default: TinyShakespeare")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=HP["warmup"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replot", action="store_true",
                    help="redraw the figure and summaries from the saved results instead of training")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    saved_path = out / "results" / "engine_vs_torch.json"
    if args.replot:
        results = json.loads(saved_path.read_text(encoding="utf-8"))
        for run in results["runs"].values():
            run["summary"] = {**summarize(run["ours"], run["torch"]),
                              "final_max_abs_weight_diff": run["summary"]["final_max_abs_weight_diff"]}
        saved_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        plot(results["runs"], out / "docs" / "figures" / "engine_vs_torch.png")
        return results
    torch.set_num_threads(THREADS)

    text = Path(args.data).read_text(encoding="utf-8") if args.data else data.load_tinyshakespeare()
    tok = CharTokenizer.train(text)
    train_ids = np.array(tok.encode(data.split(text, 0.1)[0]))
    cfg = GPTConfig(tok.vocab_size, args.block_size, args.n_layer, args.n_head, args.d_model)
    state = GPT(cfg, rng=np.random.default_rng(args.seed)).state_dict()
    batches = make_batches(train_ids, args.steps, args.batch_size, args.block_size, args.seed + 1)
    hp = {**HP, "warmup": args.warmup}

    (out / "results").mkdir(parents=True, exist_ok=True)
    runs = {}
    results = {"config": {**vars(args), **dataclasses.asdict(cfg), **hp, "threads": THREADS,
                          "torch": torch.__version__, "numpy": np.__version__},
               "runs": runs}
    for name, dtype in (("float32", np.float32), ("float64", np.float64)):
        ours = run_ours(cfg, state, batches, hp, dtype)
        theirs = run_torch(cfg, state, batches, hp, dtype)
        weight_gap = max(float(np.abs(ours["state"][k] - theirs["state"][k]).max()) for k in state)
        runs[name] = {"ours": {k: ours[k] for k in ("loss", "seconds")},
                      "torch": {k: theirs[k] for k in ("loss", "seconds")},
                      "summary": {**summarize(ours, theirs), "final_max_abs_weight_diff": weight_gap}}
        s = runs[name]["summary"]
        print(f"{name}: max |loss gap| {s['max_abs_loss_diff']:.2e}, final weight gap {weight_gap:.2e}, "
              f"{s['ours_ms_per_step']:.0f} ms vs {s['torch_ms_per_step']:.0f} ms per step "
              f"({s['slowdown']:.1f}x)", flush=True)
        # written after each precision, so a run stopped part-way keeps what it finished
        saved_path.write_text(json.dumps(results, indent=1), encoding="utf-8")

    plot(runs, out / "docs" / "figures" / "engine_vs_torch.png")
    return results


if __name__ == "__main__":
    main()
