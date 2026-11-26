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
- Vaswani et al., *Attention Is All You Need* (2017); Su et al., *RoFormer* (RoPE, 2021); Zhang and Sennrich, *RMSNorm* (2019); Shazeer, *GLU Variants Improve Transformer* (2020); Ainslie et al., *GQA* (2023)
- Fedus et al., *Switch Transformers* (2021); Power et al., *Grokking* (2022); Olsson et al., *In-context Learning and Induction Heads* (2022); Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023)

TinyShakespeare is downloaded at run time and is not part of this repository.

## License

MIT
