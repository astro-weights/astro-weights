# CLI

```text
astro-weights [--version] {build,compose,eval,export,inspect} ...
```

| Command | Does |
|---|---|
| `inspect DIR` | summarize and verify a saved PEFT adapter; see [How-to: inspect an adapter](../how-to/inspect-adapter.md) |
| `eval BASE.jsonl EDIT.jsonl [--boot N] [--seed N] [--floor A.jsonl B.jsonl] [--ignore-header]` | compare two arms from per-item records and print the [`format_report`](../how-to/compare-arms.md); `--boot` sets the bootstrap resamples (default 10000), `--seed` the bootstrap seed (default 0), `--floor` takes two base-arm runs for a noise floor, `--ignore-header` skips the run-header check; see [How-to: run an eval end to end](../how-to/run-eval.md) |
| `build RECIPE.json --base MODEL_DIR --out DIR [--card LICENSE] [--force]` | build a PEFT adapter from a recipe file and the base model weights; print one line per module (`out x in`, rank), then `wrote DIR`; see [How-to: build an adapter from a recipe file](../how-to/build-from-recipe.md) |
| `compose ADAPTER ADAPTER [ADAPTER...] -o OUT_DIR [--force]` | combine two or more PEFT adapters on the same base model into one adapter whose change is exactly their sum; print the module count and rank, then `wrote OUT_DIR` |
| `export DIR [--gguf OUT.gguf --arch ARCH] [--card LICENSE] [--splice BASE.gguf --base-model DIR --merged OUT.gguf [--out-type TYPE]] [--force]` | verify the adapter in `DIR`, then write a GGUF LoRA adapter, a merged GGUF, a model card or any mix of them; see [How-to: build an adapter from a recipe file](../how-to/build-from-recipe.md) and [How-to: use with Ollama](../how-to/use-with-ollama.md) |

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

## compose

- Takes two or more adapter directories and writes one adapter to `-o OUT_DIR`. Where several adapters change the same module, their factors are joined, so the result is exactly the sum of their changes and its rank is the sum of their ranks; the other modules are zero-padded to that rank. The adapter's `alpha / r` scaling is already applied.
- All adapters must name the same base model, and any base weight hashes they record must agree. Modules shared by two adapters must have the same shape.
- A non-empty `OUT_DIR` is refused unless `--force` is given; `--force` keeps other files there. `OUT_DIR` cannot be one of the input directories or inside one.
- Run `export --gguf` or `export --splice` on `OUT_DIR` to get one GGUF adapter or one merged GGUF. GGUF export maps the `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` and `down_proj` modules of each layer; any other module is an error.

## export

- Needs at least one of `--gguf`, `--card` and `--splice`; with none it is an error.
- `--gguf OUT.gguf` needs a non-empty `--arch` (written to `general.architecture`, for example `qwen35`) and the `gguf` extra (`pip install "astro-weights[gguf]"`). The module mapping limits are in [How-to: export GGUF and a model card](../how-to/export-gguf.md).
- `--card LICENSE`: write a `README.md` model card into `DIR`.
- `--splice BASE.gguf` writes a merged GGUF: `BASE.gguf` with the adapter folded in. It needs `--base-model DIR` (the base model as safetensors), `--merged OUT.gguf` (the file to write) and the `gguf` extra. `--base-model`, `--merged` and `--out-type` without `--splice` are an error. See [How-to: use with Ollama](../how-to/use-with-ollama.md).
- `--out-type` sets how the changed tensors are stored: `q8_0` (default), `f16` or `bf16`. With `q8_0`, a tensor whose row length is not divisible by 32 is an error (exit 2); use `--out-type f16`.
- The `--merged` file cannot be the `--splice` base file or the `--gguf` file, cannot sit inside `DIR`, and its directory must exist. `--gguf` cannot name the `--splice` base file.
- When the adapter records base weight hashes, a weight in `--base-model` that does not match is an error.
- An existing `OUT.gguf` or `DIR/README.md` is refused unless `--force` is given. An `OUT.gguf` that names one of the adapter's own files is refused even with `--force`.
- An adapter with hash problems is an error, and nothing is written.

## JSON output

| Command | Object |
|---|---|
| `inspect` | `base_model`, `rank`, `modules` (each `module`, `shape`, `rank`, `norm_B`, `norm_A`, `norm_BA`), `manifest` (bool), `problems` |
| `eval` | see [eval](#eval) |
| `compose` | `written` (the output directory), `rank`, `modules` (sorted names) and `sources` (the input directories, in order) |
| `build`, `export` | `written`: the list of paths written. `export --splice` adds `replaced` (the GGUF tensor names changed) and `out_type` |

## Exit codes

- All commands: 0 means ok, 1 means problems were found or a check failed, 2 means a usage error or unreadable input. No command at all is a usage error (exit 2).
- `inspect` exits 1 on hash problems (`problem:` lines on stderr, also with `--json`).
- `eval` exits 1 when the run headers differ, and 2 on an unreadable file or arms that do not pair.
- `build`, `compose` and `export` exit 0 on success and 2 on any error.
- Errors go to stderr as a line starting with `error:`; read and parse errors name the file.
