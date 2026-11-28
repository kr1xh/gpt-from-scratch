"""Grokking on modular addition (Power et al. 2022): a small transformer trained on part of the
table of a + b mod p first memorises its training pairs, and only much later, under weight decay,
gets the rest of the table right too.

The model reads the three tokens "a b =" (the "=" token is p) and predicts the sum at the "="
position. Training is full batch: every step sees every training pair.
"""
import numpy as np

from . import functional as F
from .nn import GPT, GPTConfig
from .tensor import no_grad
from .train import make_optimizers


def modular_addition(p: int, frac: float, seed: int = 0) -> dict:
    """Every pair (a, b) with a, b < p, split at random into a training part of round(frac * p^2)
    pairs and a test part with the rest."""
    a, b = np.divmod(np.arange(p * p), p)
    x = np.stack([a, b, np.full_like(a, p)], axis=1)
    y = (a + b) % p
    order = np.random.default_rng(seed).permutation(p * p)
    train, test = order[:round(frac * p * p)], order[round(frac * p * p):]
    return {"train_x": x[train], "train_y": y[train], "test_x": x[test], "test_y": y[test]}


def first_step(history: list, key: str, threshold: float):
    """The first logged step at which history[key] reached `threshold`, or None."""
    return next((h["step"] for h in history if h[key] >= threshold), None)


def _loss_and_accuracy(model, x, y):
    logits, _ = model(x)
    answer = logits[:, -1]  # the prediction made at the "=" token
    return F.cross_entropy(answer, y), float((answer.data.argmax(axis=-1) == y).mean())


def train(p: int = 97, frac: float = 0.4, steps: int = 10000, lr: float = 1e-3, weight_decay: float = 1.0,
          optimizer: str = "adamw", d_model: int = 128, n_head: int = 4, n_layer: int = 1, warmup: int = 10,
          eval_every: int = 100, seed: int = 0, log=None) -> list:
    """Full-batch training with a constant lr after a short warmup. Every `eval_every` steps (and
    at the last) it logs the loss and accuracy on both parts, measured before that step's update."""
    data = modular_addition(p, frac, seed)
    cfg = GPTConfig(vocab_size=p + 1, block_size=3, n_layer=n_layer, n_head=n_head, d_model=d_model)
    model = GPT(cfg, rng=np.random.default_rng([seed, 0]))
    opts = make_optimizers(model, lr, weight_decay, optimizer)
    history = []
    for step in range(steps):
        for opt in opts:
            opt.lr = lr * min(1.0, (step + 1) / warmup)
        model.zero_grad()
        loss, train_acc = _loss_and_accuracy(model, data["train_x"], data["train_y"])
        if step % eval_every == 0 or step == steps - 1:
            with no_grad():
                test_loss, test_acc = _loss_and_accuracy(model, data["test_x"], data["test_y"])
            history.append({"step": step, "train_loss": loss.item(), "train_acc": train_acc,
                            "test_loss": test_loss.item(), "test_acc": test_acc})
            if log:
                log(history[-1])
        loss.backward()
        for opt in opts:
            opt.step()
    return history
