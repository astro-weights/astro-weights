# Build an adapter from a recipe file

A recipe file is a small JSON file plus the `.npy` arrays it names. `astro-weights build` turns a recipe and a base model into a PEFT adapter, and `astro-weights export` writes a GGUF LoRA adapter and a model card from that adapter. Anyone with the recipe and the base model can rebuild a published adapter and compare. The core install is enough; GGUF output needs the `gguf` extra.

## Write a recipe

A tiny stand-in base model (one safetensors file) and a two-step recipe:

```python
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from astro_weights.edits import save_recipe

rng = np.random.default_rng(0)
module = "model.layers.1.mlp.down_proj"

# tiny stand-in for a base model directory
Path("base").mkdir(exist_ok=True)
W0 = rng.standard_normal((16, 48)).astype(np.float32)
save_file({module + ".weight": W0}, "base/model.safetensors")

# recipe plus the arrays it names, in one directory
Path("recipe").mkdir(exist_ok=True)
np.save("recipe/v.npy", rng.standard_normal(16).astype(np.float32))
np.save("recipe/b.npy", rng.standard_normal(16).astype(np.float32))
np.save("recipe/a.npy", rng.standard_normal(48).astype(np.float32))
save_recipe(
    "recipe/recipe.json",
    {
        "format": 1,
        "base_model": "org/tiny",
        "steps": [
            {
                "recipe": "project_out",
                "module": module,
                "params": {"strength": 0.7},
                "arrays": {"v": "v.npy"},
            },
            {
                "recipe": "rank1",
                "module": "model.layers.2.mlp.down_proj",
                "arrays": {"b": "b.npy", "a": "a.npy"},
            },
        ],
    },
)
```

`save_recipe` writes the JSON with sorted keys and a trailing newline.

## Build, inspect, export

```bash
astro-weights build recipe/recipe.json --base base --out my-adapter --card apache-2.0
astro-weights inspect my-adapter
astro-weights export my-adapter --gguf my-adapter.gguf --arch qwen35
```

Output:

```text
model.layers.1.mlp.down_proj  16x48  rank 1
model.layers.2.mlp.down_proj  16x48  rank 1
wrote my-adapter
base model: org/tiny
rank: 1
modules: 2
module                        out x in        |B|_F     |A|_F    |BA|_F
model.layers.1.mlp.down_proj  16x48             0.7       6.2      4.34
model.layers.2.mlp.down_proj  16x48            2.99      4.52      13.5
manifest: ok
wrote my-adapter.gguf
```

`my-adapter/` holds `adapter_config.json`, `adapter_model.safetensors`, `astro_manifest.json` and `README.md` (from `--card`). The base model has no weight for `layers.2`, but `rank1` does not read the base weight, so the build still works. Flags and exit codes are in the [CLI reference](../reference/cli.md).

## Recipe format

| Key | Value |
|---|---|
| `format` | `1` |
| `base_model` | non-empty string; written into the adapter |
| `steps` | non-empty list, one step per module |
| `steps[].recipe` | recipe name, from the table below |
| `steps[].module` | module name, for example `model.layers.1.mlp.down_proj` |
| `steps[].params` | scalars with the exact names below; floats must be finite |
| `steps[].arrays` | array name to `.npy` file name; a plain file name in the recipe directory, no subdirectory |

Only these recipes are allowed in a step. Their parameter and array names are the keyword arguments of the functions of the same name in `astro_weights.edits`.

| `recipe` | `params` | `arrays` | reads base weight |
|---|---|---|---|
| `rank1` | none | `b`, `a` | no |
| `masked_rank1` | `alpha` | `u`, `mask` | yes |
| `column_shift` | `n_in` (int) | `shift`, `idx`, `coef` | no |
| `project_out` | `strength` | `v` | yes |
| `scale_column` | `j` (int), `s` | none | yes |
| `scale_head` | `head` (int), `n_heads` (int), `s` | none | yes |
| `zero_head` | `head` (int), `n_heads` (int) | none | yes |

- An unknown or missing key in `params` or `arrays` raises `ValueError`, and so does a second step on the same module.
- `.npy` files load with `allow_pickle=False`; a path that resolves outside the recipe directory raises `ValueError`.
- A recipe from someone else is untrusted input. Loading applies size caps, never unpickles and never runs code; the full list is in [SECURITY.md](https://github.com/astro-weights/astro-weights/blob/main/SECURITY.md#untrusted-input).
- The built edit keeps the recipe plus the sha256 of every array (`meta["recipe"]["arrays_sha256"]`) in `astro_manifest.json`.

## Base weights

`build` reads `<module>.weight` from the `--base` directory. It follows `model.safetensors.index.json` when that file exists and reads `model.safetensors` otherwise. `read_base_weight(model_dir, key)` does the same in Python, `read_tensor(file, key)` reads one tensor from one file, and `tensor_sha256(file, key)` hashes the raw stored bytes. Only F32, F16 and BF16 are supported, and they are returned as float32. These functions need numpy only, no torch. Any problem (a missing key, an unsupported dtype, a truncated file, an unsafe shard name in the index) raises `ValueError`.

## The same in Python

`build_edit` plus `save_peft` do what `build` does. This continues the example above:

```python
from astro_weights.edits import build_edit, load_recipe
from astro_weights.export import read_base_weight, read_peft, save_peft

recipe = load_recipe("recipe/recipe.json")  # validates; arrays not loaded yet
edit = build_edit(
    recipe,
    lambda m: read_base_weight("base", m + ".weight"),
    "recipe",  # directory the .npy names resolve against
)
print(sorted(edit.meta["recipe"]["arrays_sha256"]))

save_peft(edit, "rebuilt")
rebuilt, problems = read_peft("rebuilt")
print(
    problems,
    rebuilt.meta["recipe"]["arrays_sha256"] == edit.meta["recipe"]["arrays_sha256"],
)
```

Output:

```text
['a.npy', 'b.npy', 'v.npy']
[] True
```

To check a shipped adapter against its recipe, compare `edit.meta["recipe"]["arrays_sha256"]` with the same field from `read_peft("my-adapter")`.
