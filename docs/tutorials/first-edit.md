# First edit

In this tutorial you build a rank-1 edit on a small random Llama, save it as a LoRA adapter, inspect it, apply it and remove it again. Everything runs on CPU in a few seconds. You need the `hf` extra:

```bash
pip install "astro-weights[hf]"
```

## 1. Make a model

A two-layer random Llama stands in for a real model, so nothing is downloaded.

```python
import numpy as np
import torch
import transformers

torch.manual_seed(0)
config = transformers.LlamaConfig(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=64,
)
model = transformers.LlamaForCausalLM(config).eval()
```

## 2. Build an edit

`masked_rank1` returns the low-rank form of `W = W0 - alpha * outer(u, mask * (u @ W0))`. It removes the direction `u` from the output of the masked input columns. `top_k_mask` picks those columns.

```python
from astro_weights.edits import Edit, masked_rank1, top_k_mask

MODULE = "model.layers.1.mlp.down_proj"
w0 = model.get_submodule(MODULE).weight.detach().numpy()
rng = np.random.default_rng(0)
u = rng.standard_normal(w0.shape[0]).astype(np.float32)
u /= np.linalg.norm(u)
mask = top_k_mask(rng.standard_normal(w0.shape[1]), 8)
delta = masked_rank1(MODULE, w0, u, mask, alpha=1.0)
edit = Edit(deltas=(delta,), base_model="tiny-llama")
```

## 3. Save it as a LoRA adapter

```python
from astro_weights.export import save_peft

save_peft(edit, "my-edit")
```

`my-edit/` now holds `adapter_config.json`, `adapter_model.safetensors` and a hash manifest. Any PEFT-aware stack can load it. `save_peft` writes the files to a temp directory first and then moves them in; if anything fails, the temp directory is removed. If `my-edit/` is not empty, `save_peft` raises `FileExistsError`. `save_peft(edit, "my-edit", overwrite=True)` replaces the adapter files and keeps any other files in the directory.

## 4. Inspect it

```bash
astro-weights inspect my-edit
```

This prints the base model, the rank, one row per module with its factor norms, and the manifest status. Exit code 1 means a hash problem; see [How-to: inspect an adapter](../how-to/inspect-adapter.md).

## 5. Apply and remove it

```python
from astro_weights.export import load_peft
from astro_weights.runners.hf import apply, remove

ids = torch.tensor([[1, 5, 9, 33, 100, 7, 64, 2]])
with torch.no_grad():
    before = model(ids).logits

peft_model = apply(model, load_peft("my-edit"))
with torch.no_grad():
    edited = peft_model(ids).logits

model = remove(peft_model)
with torch.no_grad():
    after = model(ids).logits

print("edit changed logits:", not torch.allclose(before, edited))
print("remove restored base:", torch.allclose(before, after))
```

`apply` wraps only the edited modules in LoRA layers. `remove` unloads them without merging, so the base weights never change.

## Next

- The other recipes (`rank1`, `column_shift`, `orthogonal_random`) are in the [Reference](../reference/index.md).
- Why edits are saved as adapters: [Explanation: why LoRA](../explanation/why-lora.md).
