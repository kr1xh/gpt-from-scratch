"""Step 1 of the math path: logistic regression on two Gaussian blobs, trained with the engine.

python examples/logreg.py
"""
import numpy as np

from scratchgpt import functional as F
from scratchgpt import nn, optim
from scratchgpt.tensor import Tensor, concat, no_grad


def blobs(rng, n=400):
    y = rng.integers(0, 2, n)
    x = rng.normal(0, 1, (n, 2)) + np.where(y[:, None] == 1, 2.0, -2.0)
    return x.astype(np.float32), y.astype(np.float32)


def train(seed: int = 0, steps: int = 200) -> float:
    rng = np.random.default_rng(seed)
    x, y = blobs(rng)
    split = int(0.8 * len(x))
    model = nn.Linear(2, 1, rng=rng)
    opt = optim.SGD(model.parameters(), lr=0.5)
    xt, yt = Tensor(x[:split]), y[:split, None]
    for _ in range(steps):
        z = model(xt)
        # binary cross-entropy without log(sigmoid): sigmoid(z) = softmax([0, z])[1], and log_softmax
        # stays finite where log(1 - sigmoid(z)) would hit log(0) once the model is confident
        logp = F.log_softmax(concat([Tensor(np.zeros_like(z.data)), z], axis=1), axis=1)
        loss = -(yt * logp[:, 1:] + (1 - yt) * logp[:, :1]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with no_grad():
        pred = model(Tensor(x[split:])).data[:, 0] > 0
    return float((pred == (y[split:] == 1)).mean())


if __name__ == "__main__":
    print(f"test accuracy: {train():.3f}")
