# CLI

```text
astro-weights [--version] {build,eval,export,inspect} ...
```

| Command | Does |
|---|---|
| `inspect DIR` | summarize and verify a saved PEFT adapter; see [How-to: inspect an adapter](../how-to/inspect-adapter.md) |
| `eval BASE.jsonl EDIT.jsonl [--boot N] [--seed N] [--floor A.jsonl B.jsonl] [--ignore-header]` | compare two arms from per-item records and print the [`format_report`](../how-to/compare-arms.md); `--boot` sets the bootstrap resamples (default 10000), `--seed` the bootstrap seed (default 0), `--floor` takes two base-arm runs for a noise floor, `--ignore-header` skips the run-header check; see [How-to: run an eval end to end](../how-to/run-eval.md) |
| `build RECIPE.json --base MODEL_DIR --out DIR [--card LICENSE] [--force]` | build a PEFT adapter from a recipe file and the base model weights; print one line per module (`out x in`, rank), then `wrote DIR`; see [How-to: build an adapter from a recipe file](../how-to/build-from-recipe.md) |
| `export DIR [--gguf OUT.gguf --arch ARCH] [--card LICENSE] [--force]` | verify the adapter in `DIR`, then write a GGUF LoRA adapter, a model card or both; see [How-to: build an adapter from a recipe file](../how-to/build-from-recipe.md) |

Every command takes `--json`, which prints one JSON object on stdout instead of text. Run without a command, `astro-weights` prints the help and exits 2. `python -m astro_weights` runs the same command line, for example `python -m astro_weights inspect DIR`.

## eval

- Run headers: when `BASE.header.json` and `EDIT.header.json` (beside `BASE.jsonl` and `EDIT.jsonl`) both exist, their keys must match, ignoring `device`. A difference exits 1. If one is missing, or with `--ignore-header`, there is no check.
- `--floor A.jsonl B.jsonl` adds `noise floor (score)` and `noise floor (tokens)` lines to the report; see [How-to: compare two eval arms](../how-to/compare-arms.md#noise-floor).
- The report ends with a `WARNING: cap-hit censoring` block when an arm hits the token cap often or unevenly; see [How-to: compare two eval arms](../how-to/compare-arms.md#cap-hit-censoring).
- `--json` prints the `PairedResult` fields (including the cap-hit fields) plus `floor` (`null`, or `result` and `exceeds`), with non-finite numbers as `null`.

## build

- `--base MODEL_DIR`: a directory with `model.safetensors`, or with `model.safetensors.index.json` plus shards. It is read only for steps that need the base weight.
- `--out DIR`: the adapter directory. A non-empty `DIR` is refused unless `--force` is given; `--force` keeps other files there (for example an old `README.md`).
- `--card LICENSE`: also write a `README.md` model card with that license.

## export

- Needs `--gguf`, `--card` or both; with neither it is an error.
- `--gguf OUT.gguf` needs a non-empty `--arch` (written to `general.architecture`, for example `qwen35`) and the `gguf` extra (`pip install "astro-weights[gguf]"`). The module mapping limits are in [How-to: export GGUF and a model card](../how-to/export-gguf.md).
- `--card LICENSE`: write a `README.md` model card into `DIR`.
- An existing `OUT.gguf` or `DIR/README.md` is refused unless `--force` is given. An `OUT.gguf` that names one of the adapter's own files is refused even with `--force`.
- An adapter with hash problems is an error, and nothing is written.

## JSON output

| Command | Object |
|---|---|
| `inspect` | `base_model`, `rank`, `modules` (each `module`, `shape`, `rank`, `norm_B`, `norm_A`, `norm_BA`), `manifest` (bool), `problems` |
| `eval` | see [eval](#eval) |
| `build`, `export` | `written`: the list of paths written |

## Exit codes

- All commands: 0 means ok, 1 means problems were found or a check failed, 2 means a usage error or unreadable input. No command at all is a usage error (exit 2).
- `inspect` exits 1 on hash problems (`problem:` lines on stderr, also with `--json`).
- `eval` exits 1 when the run headers differ, and 2 on an unreadable file or arms that do not pair.
- `build` and `export` exit 0 on success and 2 on any error.
- Errors go to stderr as a line starting with `error:`; read and parse errors name the file.
