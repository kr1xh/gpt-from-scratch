"""Tokenizers: one id per character, or byte-level byte-pair encoding (BPE) as in GPT-2.

Byte-level BPE starts from the 256 byte values, so any text can be encoded, even text the
tokenizer never saw. Training repeatedly merges the most frequent adjacent pair into a new id.
"""
import json
from collections import Counter
from pathlib import Path


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = chars
        self.index = {c: i for i, c in enumerate(chars)}

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.index[c] for c in text]
        except KeyError as e:
            raise ValueError(f"character {e.args[0]!r} is not in the vocabulary") from None

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"type": "char", "chars": self.chars}), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "CharTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["chars"])


def _merge(ids: list[int], pair: tuple[int, int], new: int) -> list[int]:
    out, i, a, b = [], 0, pair[0], pair[1]
    while i < len(ids):
        if ids[i] == a and i + 1 < len(ids) and ids[i + 1] == b:
            out.append(new)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges: dict[tuple[int, int], int]):
        self.merges = merges  # pair -> new id, in the order they were learned (that order is the rank)
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new in merges.items():
            self.vocab[new] = self.vocab[a] + self.vocab[b]

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BPETokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (one id per byte)")
        # ponytail: recounts every pair after each merge, O(len(text)) per merge (measured: about
        # 8 minutes for 1 MB and 768 merges; train once and save). Upgrade path: update the pair
        # counts incrementally around each merge.
        ids, merges = list(text.encode("utf-8")), {}
        for new in range(256, vocab_size):
            counts = Counter(zip(ids, ids[1:]))
            if not counts:
                break
            # most frequent pair; ties go to the smaller pair, so training is deterministic
            pair = min(counts, key=lambda p: (-counts[p], p))
            if counts[pair] < 2:  # merging a pair seen once saves nothing
                break
            merges[pair] = new
            ids = _merge(ids, pair, new)
        return cls(merges)

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) > 1:
            # merge the pair that was learned earliest, wherever it occurs, as training did
            pair = min(set(zip(ids, ids[1:])), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"type": "bpe", "merges": [list(p) for p in self.merges]}),
                              encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        pairs = json.loads(Path(path).read_text(encoding="utf-8"))["merges"]
        return cls({(a, b): 256 + i for i, (a, b) in enumerate(pairs)})
