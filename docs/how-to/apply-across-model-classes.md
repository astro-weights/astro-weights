# Apply adapter across model classes

Same checkpoint gets different module names depending on class that loads it. Qwen3.5 text-only class (`Qwen3_5ForCausalLM`) names layers `model.layers.N...`; multimodal class (`Qwen3_5ForConditionalGeneration`) names them `model.language_model.layers.N...`. `apply` maps edit names to model names itself. `resolve_modules` shows mapping without applying.

## Check mapping, then apply

Edit below names its layer the multimodal way; model is text-only.

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

Prints:

```text
{'model.language_model.layers.1.mlp.down_proj': 'model.layers.1.mlp.down_proj'}
```

## Matching rules

1. Exact name in model wins.
2. Otherwise match name from last `layers.<int>` onward, among `nn.Linear` layers with edit's shape.
3. No match, or more than one: `ValueError`. Fix ambiguous match by renaming edit module to model's exact name.
4. Two edit modules mapping to one layer: `ValueError`.
5. Name matches module that is not `nn.Linear` (for example GPT-2 `Conv1D`): `ValueError` naming module type.
6. Model already carries adapter: `ValueError`.

Every `ValueError` comes before `apply` touches model. After loading, `apply` raises `RuntimeError` when adapter did not wrap every edit module or left unexpected keys, and unwraps model again first.

## Adapter already applied

`apply`, `resolve_modules` and `merged_weights` need plain base model. Continues example above:

```python
from astro_weights.runners.hf import remove

try:
    apply(peft_model, edit)
except ValueError as e:
    print(e)
model = remove(peft_model)
```

Prints:

```text
an adapter is already applied to this model; use swap() to replace it, or remove() to get the plain base model back
```

## Rounding to model dtype

`apply` casts factors to each weight's dtype. `rounding_error(model, edit)` reports per edit module `|cast(B) @ cast(A) - B @ A|_F / |B @ A|_F`: 0.0 for float32 or float64 model, `inf` when factor overflows dtype. Model not modified. `apply(..., max_rounding_error=...)` refuses edit above limit. Continues example above:

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

Prints:

```text
{'model.layers.1.mlp.down_proj': '0.0022'}
model.layers.1.mlp.down_proj: rounding error 0.00221 of the factors in the model dtype exceeds max_rounding_error 0.001
```
