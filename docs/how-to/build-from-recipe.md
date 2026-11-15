# Build adapter from recipe file

Recipe file: small JSON plus `.npy` arrays it names. `astro-weights build` turns recipe and base model into PEFT adapter; `astro-weights export` writes GGUF LoRA and model card from that adapter. Anyone with recipe and base model can rebuild published adapter and compare. Core install only; GGUF needs `gguf` extra.

## Write recipe

Tiny stand-in base model (one safetensors file) and two-step recipe:

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

`save_recipe` writes JSON with sorted keys and trailing newline.

## Build, inspect, export

```bash
astro-weights build recipe/recipe.json --base base --out my-adapter --card apache-2.0
astro-weights inspect my-adapter
astro-weights export my-adapter --gguf my-adapter.gguf --arch qwen35
```

Prints:

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

`my-adapter/` holds `adapter_config.json`, `adapter_model.safetensors`, `astro_manifest.json` and `README.md` (from `--card`). Base has no weight for `layers.2`; `rank1` does not read base weight, so build still works. Flags and exit codes: [CLI reference](../reference/cli.md).

## Recipe format

| Key | Value |
|---|---|
| `format` | `1` |
| `base_model` | non-empty string; goes into adapter |
| `steps` | non-empty list; one step per module |
| `steps[].recipe` | recipe name, table below |
| `steps[].module` | module name, e.g. `model.layers.1.mlp.down_proj` |
| `steps[].params` | scalars, exact names below; floats finite |
| `steps[].arrays` | array name to `.npy` file name; plain file name in recipe directory, no subdirectory |

Only these recipes allowed in a step. Same names as keyword arguments of functions in `astro_weights.edits`.

| `recipe` | `params` | `arrays` | reads base weight |
|---|---|---|---|
| `rank1` | none | `b`, `a` | no |
| `masked_rank1` | `alpha` | `u`, `mask` | yes |
| `column_shift` | `n_in` (int) | `shift`, `idx`, `coef` | no |
| `project_out` | `strength` | `v` | yes |
| `scale_column` | `j` (int), `s` | none | yes |
| `scale_head` | `head` (int), `n_heads` (int), `s` | none | yes |
| `zero_head` | `head` (int), `n_heads` (int) | none | yes |

- Unknown or missing key in `params` or `arrays`: `ValueError`. Second step on same module: `ValueError`.
- `.npy` loads with `allow_pickle=False`; path resolving outside recipe directory: `ValueError`.
- Built edit keeps recipe plus sha256 of every array (`meta["recipe"]["arrays_sha256"]`) in `astro_manifest.json`.

## Base weights

`build` reads `<module>.weight` from `--base` directory: follows `model.safetensors.index.json` when present, else reads `model.safetensors`. `read_base_weight(model_dir, key)` does same in Python; `read_tensor(file, key)` reads one tensor from one file; `tensor_sha256(file, key)` hashes raw stored bytes. F32, F16 and BF16 only, returned as float32. numpy only, no torch. Any problem (missing key, unsupported dtype, truncated file, unsafe shard name in index): `ValueError`.

## Same in Python

`build_edit` plus `save_peft` do what `build` does. Continues example above:

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

Prints:

```text
['a.npy', 'b.npy', 'v.npy']
[] True
```

To check shipped adapter against recipe, compare `edit.meta["recipe"]["arrays_sha256"]` with same field from `read_peft("my-adapter")`.
