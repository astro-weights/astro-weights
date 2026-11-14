# astro-weights

Agent-driven weight-space edits for open LLMs. An agent proposes small, targeted edits to a model's weights, the edits are judged against paired statistical tests on held-out data, and the ones that hold up are exported as standard PEFT LoRA adapters (and GGUF) that existing tools already load.

> **Status: pre-alpha.** Nothing is released yet and the API will change without notice until 1.0.

## Install

```bash
pip install astro-weights
```

## Quick look

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

```bash
astro-weights inspect my-edit
```

To apply a saved edit to a transformers model:

```bash
pip install "astro-weights[hf]"
```

```python
from astro_weights.export import load_peft
from astro_weights.runners.hf import apply

peft_model = apply(model, load_peft("my-edit"))  # model: a transformers causal LM
```

## Development

```bash
git clone https://github.com/astro-weights/astro-weights
cd astro-weights
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE).
