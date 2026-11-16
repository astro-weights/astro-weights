# Inspect and verify an adapter

Check a saved adapter before you ship it or after you download it. A downloaded adapter is untrusted input; [SECURITY.md](https://github.com/astro-weights/astro-weights/blob/main/SECURITY.md#untrusted-input) lists what reading it checks and what it does not (a hash is not a signature).

## Command line

```bash
astro-weights inspect DIR
```

| Exit code | Meaning |
|---|---|
| 0 | the adapter reads and the hashes match (or there is no manifest) |
| 1 | the adapter reads, but there are `problem:` lines on stderr (for example a hash mismatch) |
| 2 | not a directory, `adapter_config.json` or `adapter_model.safetensors` missing, or unreadable |

On success it prints the base model, the rank, the module count and one row per module (`out x in`, `|B|_F`, `|A|_F`, `|BA|_F`), then `manifest: ok` or `manifest: none`.

## Python

```python
import numpy as np

from astro_weights.edits import Edit, rank1
from astro_weights.export import read_peft, save_peft, verify_peft

rng = np.random.default_rng(0)
delta = rank1(
    "model.layers.1.mlp.down_proj", rng.standard_normal(16), rng.standard_normal(48)
)
save_peft(Edit(deltas=(delta,), base_model="org/tiny"), "my-edit")

problems = verify_peft("my-edit")  # list of strings; empty when fine
edit, problems = read_peft("my-edit")  # edit plus problems, one pass
print(problems, edit.base_model)
```

Output:

```text
[] org/tiny
```

`read_peft` raises `ValueError` on adapters it cannot represent:

- `peft_type` is not `LORA`, or `use_dora`, `use_rslora` or `fan_in_fan_out` is true.
- `rank_pattern`, `alpha_pattern` or `modules_to_save` is set, or `bias` is other than `none`.
- `r` is not an int >= 1, `lora_alpha` is not a number, or `lora_alpha / r` is not finite.
- There are tensors for a module outside a list-form `target_modules`.
- The JSON or the manifest fields are malformed.

Hash problems come back in the list instead. `verify_peft` never raises: every problem, structural or hash, comes back as a string. `load_peft` raises when the list is not empty.
