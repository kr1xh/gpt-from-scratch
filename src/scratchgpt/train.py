"""Train a GPT on TinyShakespeare (or any text file): python -m scratchgpt.train --name baseline

Writes results/<name>.json (config, loss every step, train/val loss at each eval; rewritten at each
eval so a stopped run keeps its numbers) and models/<name>.npz plus models/<name>.tokenizer.json.
"""
import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

# One BLAS thread per physical core: on this 4-core, 8-thread laptop 8 threads made a step 1.6x
# slower than 4. Only takes effect when this module loads NumPy first (python -m scratchgpt.train).
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max(1, (os.cpu_count() or 2) // 2)))

import numpy as np

from . import data, optim
from .nn import GPT, GPTConfig
from .tensor import no_grad
from .tokenizers import CharTokenizer


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--data", help="a text file; default: TinyShakespeare, downloaded into data/")
    ap.add_argument("--out-dir", default=".", help="results/ and models/ are created under it")
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--min-lr", type=float, default=None, help="default: lr / 10")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


BETAS = (0.9, 0.99)


def make_optimizers(params, lr: float, weight_decay: float) -> list:
    """AdamW with weight decay on the matrices only; biases and norm gains are left alone (nanoGPT)."""
    return [optim.AdamW([p for p in params if p.ndim >= 2], lr=lr, betas=BETAS, weight_decay=weight_decay),
            optim.AdamW([p for p in params if p.ndim < 2], lr=lr, betas=BETAS, weight_decay=0.0)]


def estimate_loss(model, ids, args, rng) -> float:
    with no_grad():
        losses = [model(*data.get_batch(ids, args.batch_size, args.block_size, rng))[1].item()
                  for _ in range(args.eval_iters)]
    return float(np.mean(losses))


def main(argv=None) -> dict:
    args = parse(argv)
    text = Path(args.data).read_text(encoding="utf-8") if args.data else data.load_tinyshakespeare()
    train_text, val_text = data.split(text, 0.1)
    # the character set comes from all the text: a character that only appears in validation would
    # otherwise fail to encode, and knowing a character exists leaks nothing about what follows it
    tok = CharTokenizer.train(text)
    train_ids = np.array(tok.encode(train_text))
    val_ids = np.array(tok.encode(val_text))

    cfg = GPTConfig(tok.vocab_size, args.block_size, args.n_layer, args.n_head, args.d_model)
    rng = np.random.default_rng(args.seed)
    model = GPT(cfg, rng=rng)
    params = model.parameters()
    opts = make_optimizers(params, args.lr, args.weight_decay)
    min_lr = args.lr / 10 if args.min_lr is None else args.min_lr

    out = Path(args.out_dir)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    metrics = {"config": {**vars(args), **dataclasses.asdict(cfg), "params": sum(p.size for p in params)},
               "train_loss": [], "evals": []}
    tok.save(out / "models" / f"{args.name}.tokenizer.json")

    start = time.perf_counter()
    for step in range(args.steps):
        lr = optim.cosine_lr(step, args.lr, args.warmup, args.steps, min_lr)
        for opt in opts:
            opt.lr = lr
        x, y = data.get_batch(train_ids, args.batch_size, args.block_size, rng)
        model.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optim.clip_grad_norm(params, args.grad_clip)
        for opt in opts:
            opt.step()
        metrics["train_loss"].append(loss.item())

        if step % args.eval_every == 0 or step == args.steps - 1:
            eval_rng = np.random.default_rng(1234)  # the same batches every eval, so evals compare
            row = {"step": step, "train_loss": estimate_loss(model, train_ids, args, eval_rng),
                   "val_loss": estimate_loss(model, val_ids, args, eval_rng), "lr": lr,
                   "seconds": round(time.perf_counter() - start, 2)}
            metrics["evals"].append(row)
            print(f"step {step:5d}  train {row['train_loss']:.4f}  val {row['val_loss']:.4f}  "
                  f"lr {lr:.2e}  {row['seconds']:.0f}s", flush=True)
            (out / "results" / f"{args.name}.json").write_text(json.dumps(metrics, indent=1))
            # saved at every eval, so a run stopped early keeps its latest weights
            np.savez(out / "models" / f"{args.name}.npz", __config__=json.dumps(dataclasses.asdict(cfg)),
                     **model.state_dict())

    return metrics


if __name__ == "__main__":
    main()
