"""Fail if datasets, checkpoints, run outputs or weights are about to be committed.

The one exception is the browser demo's weights in web/weights/, up to 8 MB per file.

Usage: python scripts/check_data_guard.py          (staged files)
       python scripts/check_data_guard.py --all    (every tracked file, used in CI)
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEMO_WEIGHTS = re.compile(r"^web/weights/[^/]+$")
DEMO_LIMIT = 8 * 1024 * 1024
BLOCKED = [
    (re.compile(r"(^|/)(data|results|models)/"), "data, results or models folder"),
    (re.compile(r"\.(npy|npz|pt|pth|ckpt|safetensors|bin)$", re.I), "checkpoint or weights"),
]


def check(paths: list[str], size_of=os.path.getsize) -> list[str]:
    problems = []
    for raw in paths:
        p = raw.replace("\\", "/")
        if DEMO_WEIGHTS.match(p):
            if size_of(raw) > DEMO_LIMIT:
                problems.append(f"{p}: demo weights over 8 MB")
            continue
        problems += [f"{p}: {why}" for rx, why in BLOCKED if rx.search(p)]
    return problems


def tracked(all_files: bool) -> list[str]:
    cmd = ["git", "ls-files"] if all_files else ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="check every tracked file, not just staged ones")
    paths = tracked(ap.parse_args(argv).all)
    problems = check(paths, size_of=lambda p: (REPO / p).stat().st_size)
    for problem in problems:
        print(f"blocked: {problem}", file=sys.stderr)
    if not problems:
        print("ok: no datasets, checkpoints or oversized weights")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
