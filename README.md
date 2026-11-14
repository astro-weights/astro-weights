<p align="center">
  <img alt="astro-weights" src="https://avatars.githubusercontent.com/u/356708378?v=4" width="120" height="120">
</p>

<h1 align="center">astro-weights</h1>

<p align="center"><b>Agent-driven weight-space edits for open LLMs, exported as standard LoRA adapters.</b></p>

<p align="center">
  <a href="https://pypi.org/project/astro-weights/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/astro-weights"></a>
  <a href="https://pypi.org/project/astro-weights/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/astro-weights"></a>
  <a href="https://github.com/astro-weights/astro-weights/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/astro-weights/astro-weights/actions/workflows/ci.yml/badge.svg"></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/astro-weights/">PyPI</a> ·
  <a href="https://github.com/astro-weights/astro-weights/blob/main/CHANGELOG.md">Changelog</a> ·
  <a href="https://github.com/astro-weights/astro-weights/blob/main/CONTRIBUTING.md">Contributing</a>
</p>

The goal: an agent proposes small, targeted low-rank edits to a model's weights, and the ones that hold up under paired tests are saved as plain PEFT LoRA adapters. Today (0.0.1) the library covers the edit core: build rank-1 edits, save and load them as LoRA adapters, and apply them to transformers models. LoRA is the output format because every serving stack already loads it (transformers, PEFT, vLLM, llama.cpp and the rest), so an edit needs no custom runtime: ship the adapter, load it like any other, and drop it to get the base model back.

> **Status: pre-alpha.** 0.0.1 is on PyPI. The API will change without notice until 1.0.

## Install

```bash
pip install astro-weights
```

To apply edits to transformers models, add the `hf` extra (pulls in torch, transformers and peft):

```bash
pip install "astro-weights[hf]"
```

## Quick look

Build a rank-1 edit with numpy and save it as a LoRA adapter:

```python
import numpy as np
from astro_weights.edits import Edit, masked_rank1
from astro_weights.export import save_peft

W0 = np.random.default_rng(0).standard_normal((16, 48), dtype=np.float32)
u = np.eye(16, dtype=np.float32)[0]
mask = np.ones(48, dtype=np.float32)
delta = masked_rank1("model.layers.1.mlp.down_proj", W0, u, mask, alpha=0.5)
save_peft(Edit(deltas=(delta,), base_model="org/name"), "my-edit")
```

Summarize and verify the saved adapter:

```bash
astro-weights inspect my-edit
```

Apply it to a transformers model (`hf` extra), and remove it again:

```python
from astro_weights.export import load_peft
from astro_weights.runners.hf import apply, remove

peft_model = apply(model, load_peft("my-edit"))  # model: a transformers causal LM
model = remove(peft_model)  # base weights were never modified
```

## Development

```bash
git clone https://github.com/astro-weights/astro-weights
cd astro-weights
uv sync
uv run pytest
```

See [CONTRIBUTING.md](https://github.com/astro-weights/astro-weights/blob/main/CONTRIBUTING.md) for the full set of checks and how to send a pull request.

## Citation

If you use astro-weights in your work, please cite it. The metadata is in [CITATION.cff](https://github.com/astro-weights/astro-weights/blob/main/CITATION.cff); on GitHub, use **Cite this repository** in the sidebar.

## License

[Apache-2.0](https://github.com/astro-weights/astro-weights/blob/main/LICENSE).
