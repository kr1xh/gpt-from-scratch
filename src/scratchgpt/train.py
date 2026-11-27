"""Train a GPT on TinyShakespeare (or any text file): python -m scratchgpt.train --name baseline

Writes results/<name>.json (config, loss every step, train/val loss at each eval; rewritten at each
eval so a stopped run keeps its numbers) and models/<name>.npz plus models/<name>.tokenizer.json.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

# One BLAS thread per physical core: on this 4-core, 8-thread laptop 8 threads made a step 1.6x
# slower than 4. Only takes effect when this module loads NumPy first (python -m scratchgpt.train).
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max(1, (os.cpu_count() or 2) // 2)))

import numpy as np

from . import data, optim
from .evaluate import bits_per_byte
from .nn import GPT, GPTConfig
from .tensor import no_grad
from .tokenizers import BPETokenizer, CharTokenizer


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--data", help="a text file; default: TinyShakespeare, downloaded into data/")
    ap.add_argument("--out-dir", default=".", help="results/ and models/ are created under it")
    ap.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    ap.add_argument("--vocab-size", type=int, default=512, help="BPE only: 256 byte ids plus merges")
    ap.add_argument("--cache-dir", default="data/cache", help="BPE only: trained tokenizers and encoded ids")
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--norm", choices=["layer", "rms"], default="layer")
    ap.add_argument("--mlp", choices=["gelu", "swiglu"], default="gelu")
    ap.add_argument("--pos", choices=["learned", "rope"], default="learned")
    ap.add_argument("--n-kv-head", type=int, default=None, help="fewer than --n-head: grouped-query attention")
    ap.add_argument("--mlp-hidden", type=int, default=None, help="default: 4d for gelu, 8d/3 for swiglu")
    ap.add_argument("--n-expert", type=int, default=0, help="0: dense; otherwise a mixture of experts per block")
    ap.add_argument("--top-k", type=int, default=2, help="experts per token")
    ap.add_argument("--moe-aux", type=float, default=0.01, help="weight of the load-balancing loss")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--min-lr", type=float, default=None, help="default: lr / 10")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--optimizer", choices=OPTIMIZERS, default="adamw")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


BETAS = (0.9, 0.99)


OPTIMIZERS = ("adamw", "adam", "sgd", "momentum", "muon")
EMBEDDINGS = ("wte.weight", "wpe.weight")  # tables, not maps; wte is also the output head


def make_optimizers(model, lr: float, weight_decay: float, optimizer: str = "adamw") -> list:
    """Weight decay goes on the matrices only; biases and norm gains are left alone (nanoGPT). With
    "muon", the hidden matrices get Muon, and the embeddings and vectors get AdamW (Jordan et al.)."""
    named = list(model.named_parameters())
    matrices = [p for n, p in named if p.ndim >= 2]
    vectors = [p for n, p in named if p.ndim < 2]
    if optimizer == "muon":
        hidden = [p for n, p in named if p.ndim == 2 and n not in EMBEDDINGS]
        tables = [p for n, p in named if n in EMBEDDINGS]
        return [optim.Muon(hidden, lr=lr, weight_decay=weight_decay),
                optim.AdamW(tables, lr=lr, betas=BETAS, weight_decay=weight_decay),
                optim.AdamW(vectors, lr=lr, betas=BETAS, weight_decay=0.0)]
    make = {
        "adamw": lambda ps, wd: optim.AdamW(ps, lr=lr, betas=BETAS, weight_decay=wd),
        "adam": lambda ps, wd: optim.Adam(ps, lr=lr, betas=BETAS, weight_decay=wd),  # L2 in the gradient
        "sgd": lambda ps, wd: optim.SGD(ps, lr=lr, weight_decay=wd),
        "momentum": lambda ps, wd: optim.SGD(ps, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd),
    }[optimizer]
    return [make(matrices, weight_decay), make(vectors, 0.0)]


def estimate_loss(model, ids, args, rng) -> float:
    with no_grad():
        losses = [model(*data.get_batch(ids, args.batch_size, args.block_size, rng))[1].item()
                  for _ in range(args.eval_iters)]
    return float(np.mean(losses))


def tokenize(text: str, args):
    """The tokenizer and both splits as ids. BPE is trained on the training split only (merges
    learned from validation text would leak it) and, being slow, is cached with its ids."""
    train_text, val_text = data.split(text, 0.1)
    if args.tokenizer == "char":
        # the character set comes from all the text: a character only in validation would otherwise
        # fail to encode, and knowing a character exists leaks nothing about what follows it
        tok = CharTokenizer.train(text)
        return tok, np.array(tok.encode(train_text)), np.array(tok.encode(val_text))
    key = f"bpe{args.vocab_size}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"
    cache = Path(args.cache_dir)
    try:
        with np.load(cache / f"{key}-ids.npz") as f:
            return BPETokenizer.load(cache / f"{key}.json"), f["train"], f["val"]
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):  # missing, or cut short by a stopped run: rebuild it
        pass
    tok = BPETokenizer.train(train_text, args.vocab_size)
    train_ids, val_ids = np.array(tok.encode(train_text)), np.array(tok.encode(val_text))
    cache.mkdir(parents=True, exist_ok=True)
    tok.save(cache / f"{key}.json")
    partial = cache / f"{key}-ids.partial"
    with open(partial, "wb") as f:
        np.savez(f, train=train_ids, val=val_ids)
    os.replace(partial, cache / f"{key}-ids.npz")  # the ids file only ever appears complete
    return tok, train_ids, val_ids


def main(argv=None) -> dict:
    args = parse(argv)
    text = Path(args.data).read_text(encoding="utf-8") if args.data else data.load_tinyshakespeare()
    tok, train_ids, val_ids = tokenize(text, args)

    cfg = GPTConfig(tok.vocab_size, args.block_size, args.n_layer, args.n_head, args.d_model,
                    norm=args.norm, mlp=args.mlp, pos=args.pos, n_kv_head=args.n_kv_head,
                    mlp_hidden=args.mlp_hidden, n_expert=args.n_expert, top_k=args.top_k)
    # separate streams: a model of another size draws more numbers at init, and must still see the
    # same batches in the same order for the same seed
    model = GPT(cfg, rng=np.random.default_rng([args.seed, 0]))
    rng = np.random.default_rng([args.seed, 1])
    params = model.parameters()
    opts = make_optimizers(model, args.lr, args.weight_decay, args.optimizer)
    min_lr = args.lr / 10 if args.min_lr is None else args.min_lr

    out = Path(args.out_dir)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    metrics = {"config": {**vars(args), **dataclasses.asdict(cfg), "params": sum(p.size for p in params)},
               "args": vars(args), "train_loss": [], "evals": []}
    tok.save(out / "models" / f"{args.name}.tokenizer.json")

    start = time.perf_counter()
    for step in range(args.steps):
        lr = optim.cosine_lr(step, args.lr, args.warmup, args.steps, min_lr)
        for opt in opts:
            opt.lr = lr
        x, y = data.get_batch(train_ids, args.batch_size, args.block_size, rng)
        model.zero_grad()
        _, loss = model(x, y)
        aux = model.aux_loss()
        (loss + args.moe_aux * aux if aux is not None else loss).backward()  # the logged loss stays the cross-entropy
        optim.clip_grad_norm(params, args.grad_clip)
        for opt in opts:
            opt.step()
        metrics["train_loss"].append(loss.item())
        if not np.isfinite(metrics["train_loss"][-1]):  # the weights are gone: every later step is NaN too
            metrics["diverged_at"] = step
            print(f"step {step:5d}  loss is {metrics['train_loss'][-1]}: stopping", flush=True)
            break

        if step % args.eval_every == 0 or step == args.steps - 1:
            eval_rng = np.random.default_rng(1234)  # the same batches every eval, so evals compare
            row = {"step": step, "train_loss": estimate_loss(model, train_ids, args, eval_rng),
                   "val_loss": estimate_loss(model, val_ids, args, eval_rng), "lr": lr,
                   "seconds": round(time.perf_counter() - start, 2)}
            if cfg.n_expert:  # share of assignments per expert, per layer, on the last validation batch
                row["expert_load"] = [load.tolist() for load in model.expert_load()]
            metrics["evals"].append(row)
            print(f"step {step:5d}  train {row['train_loss']:.4f}  val {row['val_loss']:.4f}  "
                  f"lr {lr:.2e}  {row['seconds']:.0f}s", flush=True)
            (out / "results" / f"{args.name}.json").write_text(json.dumps(metrics, indent=1))
            # saved at every eval, so a run stopped early keeps its latest weights
            np.savez(out / "models" / f"{args.name}.npz", __config__=json.dumps(dataclasses.asdict(cfg)),
                     **model.state_dict())

    # exact score on the whole validation split, comparable across tokenizers (see evaluate.py)
    result = bits_per_byte(model, tok, val_ids, args.batch_size)
    metrics["final"] = {"val_bits_per_byte": result["bits_per_byte"], "val_tokens": result["tokens"],
                        "val_bytes": result["bytes"], "seconds": round(time.perf_counter() - start, 2)}
    print(f"final  val {result['bits_per_byte']:.4f} bits per byte", flush=True)
    (out / "results" / f"{args.name}.json").write_text(json.dumps(metrics, indent=1))
    return metrics


if __name__ == "__main__":
    main()
