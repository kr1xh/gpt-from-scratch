# gpt-from-scratch

A small language model stack written from scratch in NumPy, sized to run on a laptop CPU:

- an **autograd engine** (tensors, broadcasting, backward), checked against PyTorch
- a **GPT** that starts GPT-2 style and is upgraded to a modern block (RoPE, RMSNorm, SwiGLU, grouped-query attention, KV cache) and a Mixture-of-Experts layer
- optimizers up to **Muon**, plus quantization, speculative decoding and a browser demo
- a set of **measured experiments**, each answering one question in `REPORT.md`

Work in progress. The plan is in the [milestones](https://github.com/kr1xh/gpt-from-scratch/milestones).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu
python -m pytest -q
```

`src/` depends only on NumPy. PyTorch is installed for the tests, which use it as the reference answer.

## Train and sample

```bash
python -m scratchgpt.train --name baseline      # downloads TinyShakespeare into data/ on first run
python -m scratchgpt.sample --checkpoint models/baseline.npz --prompt "ROMEO:" --tokens 300 --top-k 10
```

The baseline is a GPT-2-style model with 4 layers, 4 heads, width 128 and a 128-character context (0.82M parameters). It trains for 1500 steps of 16 × 128 characters. `train.py` writes `results/<name>.json`, which records the config, the loss at every step and the train/val loss at each eval. It also writes `models/<name>.npz`, updated at every eval. Run `python -m scratchgpt.train --help` for the other settings.

On a 4-core laptop CPU the baseline took 34 minutes. Validation loss fell from 4.15 to 1.757 nats per character (2.54 bits), and the samples read like this:

```
ROMEO:
I am, sitlat thou had they? so are out pits, and those
To am and hop to yourst them spratse truestion: but,
That thus, for her.

MERDIONE:
Sir, well.
```

The speaker names, line breaks and most of the words are already in place. The later releases try to beat this number at the same budget.

## Results

[REPORT.md](REPORT.md) has the studies, one question per section. So far: training on this engine follows PyTorch to within float32 rounding over 500 steps (float64 agrees to 10⁻¹⁵), at 2–4× PyTorch's time per step.

## Roadmap

| Release | Contents |
|---|---|
| v0.1 Core | autograd engine, layers, SGD/Adam/AdamW, char and BPE tokenizers, GPT-2-style GPT; studies: engine vs PyTorch, char vs BPE |
| v0.2 Modern architecture | RoPE, RMSNorm, SwiGLU, GQA, KV cache; Muon; Mixture-of-Experts |
| v0.3 Science of learning | grokking; interpretability (induction heads, logit lens) |
| v0.4 Fast inference | int8/4-bit quantization, speculative decoding, browser demo |

## Credits

Written from the papers, not ported. Ideas and inspiration from:

- Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and [nanoGPT](https://github.com/karpathy/nanoGPT); Keller Jordan's [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) and Muon write-up
- Vaswani et al., *Attention Is All You Need* (2017); Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019); Su et al., *RoFormer* (RoPE, 2021); Zhang and Sennrich, *RMSNorm* (2019); Shazeer, *GLU Variants Improve Transformer* (2020); Ainslie et al., *GQA* (2023)
- Fedus et al., *Switch Transformers* (2021); Power et al., *Grokking* (2022); Olsson et al., *In-context Learning and Induction Heads* (2022); Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023)

TinyShakespeare is downloaded at run time and is not part of this repository.

## License

MIT
