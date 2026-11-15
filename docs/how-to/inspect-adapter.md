# Inspect and verify adapter

Check saved adapter before shipping or after download.

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
from astro_weights.export import read_peft, verify_peft

problems = verify_peft("my-edit")  # list of strings; empty when fine
edit, problems = read_peft("my-edit")  # edit plus problems, one pass
```

`read_peft` raises `ValueError` on structural problems (wrong `peft_type`, `use_dora`, `use_rslora` or `fan_in_fan_out` set). Hash problems come back in list. `load_peft` raises when list not empty.
