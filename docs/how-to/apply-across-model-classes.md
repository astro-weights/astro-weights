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

After loading, `apply` raises `RuntimeError` when adapter did not wrap every edit module or left unexpected keys.
