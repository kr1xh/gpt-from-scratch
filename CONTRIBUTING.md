# Contributing

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu
git config core.hooksPath .githooks   # runs the data guard before every commit
python -m pytest -q
```

## Rules

- **Only NumPy and the standard library in `src/`.** PyTorch is a test dependency, used to check our results, never imported by the package.
- **Test first.** Every op, layer and optimizer gets a test that compares it with finite differences or PyTorch before the code is written.
- **One issue, one PR.** The PR says `Closes #n` and fills in the template, including the Ponytail rung: the simplest thing that worked.
- **No data in git.** Datasets, checkpoints and run outputs live in `data/`, `models/` and `results/` (ignored). The guard allows only the browser demo's weights in `web/weights/`, up to 8 MB per file.
- **Experiments must reproduce.** Every number in `REPORT.md` comes from `python -m scratchgpt.experiments <name>` and fits a laptop CPU (about 1 hour per run). Report seeds and what was held equal.
