# Save merged weights

`merged_weights` computes edited weight `W0 + B @ A` for each module in edit; model stays unchanged. `save_merged_tensors` writes result as safetensors with sha256 manifest; `load_merged_tensors` checks every hash, shape and dtype. `merged_weights` and `swap` need `hf` extra; save and load need numpy only.

## Compute, save, load

```python
import numpy as np
import torch
import transformers

from astro_weights.edits import Edit, LowRankDelta
from astro_weights.export import load_merged_tensors, save_merged_tensors
from astro_weights.runners.hf import merged_weights

torch.manual_seed(0)
config = transformers.LlamaConfig(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
)
model = transformers.LlamaForCausalLM(config).eval()

delta = LowRankDelta(
    "model.layers.1.mlp.down_proj",
    np.ones((32, 1), np.float32),
    np.ones((1, 64), np.float32),
)
edit = Edit(deltas=(delta,), base_model="tiny-llama")

merged = merged_weights(model, edit)  # model itself unchanged
save_merged_tensors(
    {name: w.float().numpy() for name, w in merged.items()},
    "my-merged",
    base_model="tiny-llama",
)
loaded = load_merged_tensors("my-merged")  # raises ValueError on hash mismatch
print({name: w.shape for name, w in loaded.items()})
```

Prints:

```text
{'model.layers.1.mlp.down_proj': (32, 64)}
```

- `merged_weights` needs plain base model. `PeftModel` raises `ValueError`. Sum runs in at least float32 (float64 weights stay float64), then casts back to weight's dtype; results live on CPU. Keys are model's module names, mapped like [`apply`](apply-across-model-classes.md) maps them.
- `save_merged_tensors` accepts float32 or float16 only. bf16 is not numpy dtype: pass `tensor.float().numpy()`.
- Directory holds `merged_tensors.safetensors` and `merged_manifest.json` (base model, sha256, shape and dtype per tensor). Pass `edit_manifest_sha256=` to link adapter manifest.
- `load_merged_tensors` raises `ValueError` on missing file, unreadable weights, malformed manifest, hash, shape or dtype mismatch, missing tensor or unexpected tensor. Manifest sits next to tensors, so matching hash only proves two files agree. Not signature; does not show weights authentic.

## Swap edit on live model

`swap` removes applied edit, then applies new one on base model. New edit's modules resolve first: bad edit raises before anything gets removed, `peft_model` stays usable. Continues example above:

```python
from astro_weights.runners.hf import apply, swap

other = Edit(
    deltas=(
        LowRankDelta(
            "model.layers.1.mlp.down_proj",
            np.full((32, 1), 0.5, np.float32),
            np.ones((1, 64), np.float32),
        ),
    ),
    base_model="tiny-llama",
)
peft_model = apply(model, edit)
peft_model = swap(peft_model, other)  # edit removed, other applied
```
