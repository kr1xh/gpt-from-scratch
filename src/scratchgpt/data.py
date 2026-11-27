"""Get TinyShakespeare (downloaded once into data/, never committed) and cut it into batches."""
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

TINYSHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TINYSHAKESPEARE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def download(url: str, path, sha256: str | None = None) -> Path:
    """Fetch `url` to `path` unless it is already there; refuse a file whose hash is wrong,
    including a cached one (an interrupted earlier write, say)."""
    path = Path(path)
    if path.exists():
        body, source = path.read_bytes(), str(path)
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            body, source = response.read(), url
    if sha256 and hashlib.sha256(body).hexdigest() != sha256:
        raise ValueError(f"{source} does not have the expected sha256 {sha256}; delete it and retry")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return path


def load_tinyshakespeare(root="data") -> str:
    path = download(TINYSHAKESPEARE_URL, Path(root) / "tinyshakespeare.txt", TINYSHAKESPEARE_SHA256)
    return path.read_text(encoding="utf-8")


def split(seq, val_frac: float = 0.1):
    """The first part trains, the last `val_frac` validates (contiguous, so no text leaks across)."""
    cut = int(len(seq) * (1 - val_frac))
    return seq[:cut], seq[cut:]


def get_batch(ids: np.ndarray, batch_size: int, block_size: int, rng):
    """Random windows x of `block_size` tokens and their targets y, the same windows one step on."""
    if len(ids) < block_size + 1:
        raise ValueError(f"need at least {block_size + 1} tokens for a block of {block_size}")
    starts = rng.integers(0, len(ids) - block_size, batch_size)
    offsets = starts[:, None] + np.arange(block_size)
    return ids[offsets].astype(np.int64), ids[offsets + 1].astype(np.int64)
