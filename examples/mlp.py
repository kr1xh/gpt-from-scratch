"""Step 2 of the math path: an MLP learns two interleaved half-moons (not linearly separable).

python examples/mlp.py
"""
import numpy as np

from scratchgpt import functional as F
from scratchgpt import nn, optim
from scratchgpt.tensor import Tensor, no_grad


def moons(rng, n=1000, noise=0.1):
    t = rng.uniform(0, np.pi, n)
    y = rng.integers(0, 2, n)
    x = np.where(y[:, None] == 0,
                 np.stack([np.cos(t), np.sin(t)], 1),
                 np.stack([1 - np.cos(t), 0.5 - np.sin(t)], 1))
    return (x + rng.normal(0, noise, x.shape)).astype(np.float32), y


def train(seed: int = 0, steps: int = 400) -> float:
    rng = np.random.default_rng(seed)
    x, y = moons(rng)
    split = int(0.8 * len(x))
    model = nn.MLP([2, 32, 32, 2], rng=rng)
    opt = optim.Adam(model.parameters(), lr=0.01)
    xt = Tensor(x[:split])
    for _ in range(steps):
        loss = F.cross_entropy(model(xt), y[:split])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with no_grad():
        pred = model(Tensor(x[split:])).data.argmax(axis=1)
    return float((pred == y[split:]).mean())


if __name__ == "__main__":
    print(f"test accuracy: {train():.3f}")
