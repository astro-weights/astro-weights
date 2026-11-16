# Save merged weights

`merged_weights` computes the edited weight `W0 + B @ A` for each module in an edit and leaves the model unchanged. `save_merged_tensors` writes the result as safetensors with a sha256 manifest, and `load_merged_tensors` checks every hash, shape and dtype. `merged_weights` and `swap` need the `hf` extra; saving and loading need numpy only.

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

Output:

```text
{'model.layers.1.mlp.down_proj': (32, 64)}
```

- `merged_weights` needs the plain base model; a `PeftModel` raises `ValueError`. The sum runs in at least float32 (float64 weights stay float64) and is then cast back to the weight's dtype. The results live on CPU. The keys are the model's module names, mapped the way [`apply`](apply-across-model-classes.md) maps them.
- `save_merged_tensors` accepts float32 or float16 only. bf16 is not a numpy dtype, so pass `tensor.float().numpy()`.
- The directory holds `merged_tensors.safetensors` and `merged_manifest.json` (the base model, plus the sha256, shape and dtype of each tensor). Pass `edit_manifest_sha256=` to link the adapter manifest.
- `load_merged_tensors` raises `ValueError` on a missing file, unreadable weights, a malformed manifest, a hash, shape or dtype mismatch, a missing tensor or an unexpected tensor. The manifest sits next to the tensors, so a matching hash only proves that the two files agree. It is not a signature and does not show that the weights are authentic.

## Swap the edit on a live model

`swap` removes the applied edit, then applies the new one to the base model. The new edit's modules are resolved first, so a bad edit raises before anything is removed and `peft_model` stays usable. This continues the example above:

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
