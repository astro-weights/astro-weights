# Apply an adapter across model classes

The same checkpoint gets different module names depending on the class that loads it. The Qwen3.5 text-only class (`Qwen3_5ForCausalLM`) names its layers `model.layers.N...`, while the multimodal class (`Qwen3_5ForConditionalGeneration`) names them `model.language_model.layers.N...`. `apply` maps the edit's names to the model's names by itself, and `resolve_modules` shows that mapping without applying anything.

## Check the mapping, then apply

The edit below names its layer the multimodal way, and the model is text-only.

```python
import numpy as np
import torch
import transformers

from astro_weights.edits import Edit, LowRankDelta
from astro_weights.runners.hf import apply, resolve_modules

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
    "model.language_model.layers.1.mlp.down_proj",
    np.ones((32, 1), np.float32),
    np.ones((1, 64), np.float32),
)
edit = Edit(deltas=(delta,), base_model="tiny-llama")

print(resolve_modules(model, edit))
peft_model = apply(model, edit)
```

Output:

```text
{'model.language_model.layers.1.mlp.down_proj': 'model.layers.1.mlp.down_proj'}
```

## Matching rules

1. An exact name in the model wins.
2. Otherwise the name is matched from the last `layers.<int>` onward, among `nn.Linear` layers with the edit's shape.
3. No match, or more than one, raises `ValueError`. To fix an ambiguous match, rename the edit module to the model's exact name.
4. Two edit modules that map to one layer raise `ValueError`.
5. A name that matches a module that is not `nn.Linear` (for example GPT-2's `Conv1D`) raises `ValueError` naming the module type.
6. A model that already carries an adapter raises `ValueError`.

Every `ValueError` is raised before `apply` touches the model. After loading, `apply` raises `RuntimeError` if the adapter did not wrap every edit module or left unexpected keys; it unwraps the model again before raising.

## An adapter is already applied

`apply`, `resolve_modules` and `merged_weights` need the plain base model. This continues the example above:

```python
from astro_weights.runners.hf import remove

try:
    apply(peft_model, edit)
except ValueError as e:
    print(e)
model = remove(peft_model)
```

Output:

```text
an adapter is already applied to this model; use swap() to replace it, or remove() to get the plain base model back
```

## Rounding to the model dtype

`apply` casts the factors to each weight's dtype. `rounding_error(model, edit)` reports `|cast(B) @ cast(A) - B @ A|_F / |B @ A|_F` for each edit module: 0.0 for a float32 or float64 model, and `inf` when a factor overflows the dtype. It does not modify the model. `apply(..., max_rounding_error=...)` refuses an edit above that limit. This continues the example above:

```python
from astro_weights.runners.hf import rounding_error

rng = np.random.default_rng(0)
noisy = LowRankDelta(
    "model.layers.1.mlp.down_proj",
    rng.standard_normal((32, 1)).astype(np.float32),
    rng.standard_normal((1, 64)).astype(np.float32),
)
edit16 = Edit(deltas=(noisy,), base_model="tiny-llama")
model = model.to(torch.bfloat16)

print({m: f"{err:.2g}" for m, err in rounding_error(model, edit16).items()})
try:
    apply(model, edit16, max_rounding_error=1e-3)
except ValueError as e:
    print(e)
peft_model = apply(model, edit16, max_rounding_error=1e-2)
```

Output:

```text
{'model.layers.1.mlp.down_proj': '0.0022'}
model.layers.1.mlp.down_proj: rounding error 0.00221 of the factors in the model dtype exceeds max_rounding_error 0.001
```
