# CLI

```text
astro-weights [--version] {build,eval,export,inspect} ...
```

| Command | Does |
|---|---|
| `inspect DIR` | summarize and verify saved PEFT adapter; see [How-to: inspect adapter](../how-to/inspect-adapter.md) |
| `eval BASE.jsonl EDIT.jsonl [--boot N] [--seed N]` | compare two arms from per-item records, print [`format_report`](../how-to/compare-arms.md); `--boot` bootstrap resamples (default 10000), `--seed` bootstrap seed (default 0); exit code 2 on unreadable file or arms that do not pair; see [How-to: run eval end to end](../how-to/run-eval.md) |
| `build RECIPE.json --base MODEL_DIR --out DIR [--card LICENSE] [--force]` | build PEFT adapter from recipe file and base model weights; print one line per module (`out x in`, rank), then `wrote DIR`; see [How-to: build adapter from recipe file](../how-to/build-from-recipe.md) |
| `export DIR [--gguf OUT.gguf --arch ARCH] [--card LICENSE] [--force]` | verify adapter in `DIR`, then write GGUF LoRA adapter, model card or both; see [How-to: build adapter from recipe file](../how-to/build-from-recipe.md) |

## build

- `--base MODEL_DIR`: directory with `model.safetensors`, or `model.safetensors.index.json` plus shards. Read only for steps that need base weight.
- `--out DIR`: adapter directory. Non-empty `DIR` refused unless `--force`; `--force` keeps other files there (e.g. old `README.md`).
- `--card LICENSE`: also write `README.md` model card with that license.

## export

- Needs `--gguf`, `--card` or both. Neither: error.
- `--gguf OUT.gguf` needs non-empty `--arch` (written to `general.architecture`, e.g. `qwen35`) and `gguf` extra (`pip install "astro-weights[gguf]"`). Module mapping limits: [How-to: export GGUF](../how-to/export-gguf.md).
- `--card LICENSE`: write `README.md` model card into `DIR`.
- Existing `OUT.gguf` or `DIR/README.md` refused unless `--force`. `OUT.gguf` naming one of adapter's own files refused, even with `--force`.
- Adapter with hash problems: error, nothing written.

## Exit codes

`build` and `export`: 0 on success, 2 on any error. Error goes to stderr as line starting `error:`; read and parse errors name file.
