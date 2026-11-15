# CLI

```text
astro-weights [--version] {build,eval,export,inspect} ...
```

| Command | Does |
|---|---|
| `inspect DIR` | summarize and verify saved PEFT adapter; see [How-to: inspect adapter](../how-to/inspect-adapter.md) |
| `eval BASE.jsonl EDIT.jsonl [--boot N] [--seed N] [--floor A.jsonl B.jsonl] [--ignore-header]` | compare two arms from per-item records, print [`format_report`](../how-to/compare-arms.md); `--boot` bootstrap resamples (default 10000), `--seed` bootstrap seed (default 0), `--floor` two base-arm runs for noise floor, `--ignore-header` skips run-header check; see [How-to: run eval end to end](../how-to/run-eval.md) |
| `build RECIPE.json --base MODEL_DIR --out DIR [--card LICENSE] [--force]` | build PEFT adapter from recipe file and base model weights; print one line per module (`out x in`, rank), then `wrote DIR`; see [How-to: build adapter from recipe file](../how-to/build-from-recipe.md) |
| `export DIR [--gguf OUT.gguf --arch ARCH] [--card LICENSE] [--force]` | verify adapter in `DIR`, then write GGUF LoRA adapter, model card or both; see [How-to: build adapter from recipe file](../how-to/build-from-recipe.md) |

Every command takes `--json`: one JSON object on stdout instead of text.

## eval

- Run headers: when `BASE.header.json` and `EDIT.header.json` (beside `BASE.jsonl`, `EDIT.jsonl`) both exist, keys must match, `device` ignored. Difference: exit 1. One missing: no check. `--ignore-header`: no check.
- `--floor A.jsonl B.jsonl`: report adds `noise floor (score)` and `noise floor (tokens)` lines; see [How-to: compare arms](../how-to/compare-arms.md#noise-floor).
- `--json`: `PairedResult` fields plus `floor` (`null`, or `result` and `exceeds`); non-finite numbers as `null`.

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

## JSON output

| Command | Object |
|---|---|
| `inspect` | `base_model`, `rank`, `modules` (each `module`, `shape`, `rank`, `norm_B`, `norm_A`, `norm_BA`), `manifest` (bool), `problems` |
| `eval` | see [eval](#eval) |
| `build`, `export` | `written`: list of paths written |

## Exit codes

- All commands: 0 ok, 1 problems found or check failed, 2 usage error or unreadable input.
- `inspect`: 1 on hash problems (`problem:` lines on stderr, also with `--json`).
- `eval`: 1 when run headers differ; 2 on unreadable file or arms that do not pair.
- `build` and `export`: 0 on success, 2 on any error.
- Error goes to stderr as line starting `error:`; read and parse errors name file.
