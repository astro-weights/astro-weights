# Inspect and verify adapter

Check saved adapter before shipping or after download. Downloaded adapter is untrusted input: what reading checks and what it does not (hash is not signature) in [SECURITY.md](https://github.com/astro-weights/astro-weights/blob/main/SECURITY.md#untrusted-input).

## Command line

```bash
astro-weights inspect DIR
```

| Exit code | Meaning |
|---|---|
| 0 | adapter reads, hashes match (or no manifest) |
| 1 | adapter reads, but `problem:` lines on stderr (for example hash mismatch) |
| 2 | not directory, missing `adapter_config.json` or `adapter_model.safetensors`, or unreadable |

On success prints base model, rank, module count, one row per module (`out x in`, `|B|_F`, `|A|_F`, `|BA|_F`), then `manifest: ok` or `manifest: none`.

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

Prints:

```text
[] org/tiny
```

`read_peft` raises `ValueError` on adapters it cannot represent:

- `peft_type` not `LORA`; `use_dora`, `use_rslora` or `fan_in_fan_out` true.
- `rank_pattern`, `alpha_pattern` or `modules_to_save` set; `bias` other than `none`.
- `r` not int >= 1; `lora_alpha` not number, or `lora_alpha / r` not finite.
- tensors for module outside list-form `target_modules`.
- malformed JSON or manifest fields.

Hash problems come back in list. `verify_peft` never raises: every problem, structural or hash, comes back as string. `load_peft` raises when list not empty.
