# CLI

```text
astro-weights [--version] {build,compose,eval,export,gate,gen,inspect,mcp,run,self-test} ...
```

| Command | Does |
|---|---|
| `inspect DIR` | summarize and verify a saved PEFT adapter; see [How-to: inspect an adapter](../how-to/inspect-adapter.md) |
| `eval BASE.jsonl EDIT.jsonl [--boot N] [--seed N] [--floor A.jsonl B.jsonl] [--ignore-header] [--arm-key KEY ...]` | compare two arms from per-item records and print the [`format_report`](../how-to/compare-arms.md); `--boot` sets the bootstrap resamples (default 10000), `--seed` the bootstrap seed (default 0), `--floor` takes two base-arm runs for a noise floor, `--ignore-header` skips the run-header check, `--arm-key KEY` (repeatable) names an `extra_body` key that may differ between the arms; see [How-to: run an eval end to end](../how-to/run-eval.md) |
| `build RECIPE.json --base MODEL_DIR --out DIR [--card LICENSE] [--force]` | build a PEFT adapter from a recipe file and the base model weights; print one line per module (`out x in`, rank), then `wrote DIR`; see [How-to: build an adapter from a recipe file](../how-to/build-from-recipe.md) |
| `compose ADAPTER ADAPTER [ADAPTER...] -o OUT_DIR [--base-model NAME] [--force]` | combine two or more PEFT adapters on the same base model into one adapter whose change is exactly their sum; print the module count and rank, then `wrote OUT_DIR` |
| `export DIR [--gguf OUT.gguf --arch ARCH] [--card LICENSE] [--splice BASE.gguf --base-model DIR --merged OUT.gguf [--out-type TYPE]] [--force]` | verify the adapter in `DIR`, then write a GGUF LoRA adapter, a merged GGUF, a model card or any mix of them; see [How-to: build an adapter from a recipe file](../how-to/build-from-recipe.md) and [How-to: use with Ollama](../how-to/use-with-ollama.md) |
| `gate GATE.toml [--out FILE] [--unseal] [--json]` | run the 0.3 gate end to end from recorded files: pre-screens, random controls on confirm, the sealed K=4 test split, regression suites; print the outcome with every check and arm, and write `verdict.json`; see [gate](#gate) and [How-to: run the 0.3 gate](../how-to/run-gate.md) |
| `gen ARM.toml [--resume] [--json]` | run one eval arm against an OpenAI-compatible server from a TOML file, writing per-item records and a run header as it goes, and continue it after a kill with `--resume`; see [gen](#gen) |
| `mcp RUN_DIR [--unsafe-kernel]` | serve the agent tools over stdio MCP for the run in `RUN_DIR`; see [mcp](#mcp) and [How-to: Claude Code](../how-to/agent-claude-code.md). New in 0.2 |
| `run CONFIG.toml [--resume \| --status] [--json]` | start a run from a TOML config: write `RUN_DIR/run.json`, then run the chosen editor, which starts the server; see [run](#run) and [Tutorial: first agent run](../tutorials/first-agent-run.md). New in 0.2 |
| `self-test --base-url URL --model NAME [--prompt TEXT] [--api-key-env VAR] [--json]` | send one prompt at temperature 0 under varied batch sizes (1, 4, 8) and shared-prefix fillers (0, 256, 1024 tokens) to an OpenAI-compatible server; print each output and whether all match; see [How-to: run the 0.3 gate](../how-to/run-gate.md#serving-self-test-and-noise-floor) |

Every command except `mcp` takes `--json`, which prints one JSON object on stdout instead of text. Run without a command, `astro-weights` prints the help and exits 2. `python -m astro_weights` runs the same command line, for example `python -m astro_weights inspect DIR`.

## eval

- Run headers: when `BASE.header.json` and `EDIT.header.json` (beside `BASE.jsonl` and `EDIT.jsonl`) both exist, their config keys must match (provenance keys such as `device`, run id and `base_url` are ignored; see [run headers](../how-to/run-eval.md#run-headers)). A difference exits 1. If one is missing, or with `--ignore-header`, there is no check.
- `--floor A.jsonl B.jsonl` adds `noise floor (score)` and `noise floor (tokens)` lines to the report; see [How-to: compare two eval arms](../how-to/compare-arms.md#noise-floor).
- The report ends with a `WARNING: cap-hit censoring` block when an arm hits the token cap often or unevenly; see [How-to: compare two eval arms](../how-to/compare-arms.md#cap-hit-censoring).
- `--json` prints the `PairedResult` fields (including the cap-hit fields) plus `floor` (`null`, or `result` and `exceeds`), with non-finite numbers as `null`.

## build

- `--base MODEL_DIR`: a directory with `model.safetensors`, or with `model.safetensors.index.json` plus shards. It is read only for steps that need the base weight.
- `--out DIR`: the adapter directory. A non-empty `DIR` is refused unless `--force` is given; `--force` keeps other files there (for example an old `README.md`).
- `--card LICENSE`: also write a `README.md` model card with that license.

## compose

- Takes two or more adapter directories and writes one adapter to `-o OUT_DIR`. Where several adapters change the same module, their factors are joined, so the result is exactly the sum of their changes and its rank is the sum of their ranks; the other modules are zero-padded to that rank. The adapter's `alpha / r` scaling is already applied.
- All adapters must name the same base model, and `--base-model NAME` writes `NAME` as the base model of every input first, for adapters that name one base differently (a hub name and a local path, say). Any base weight hashes they record must also agree. Modules shared by two adapters must have the same shape. The same layer named in the two Hugging Face layouts (`model.layers.N...` and `model.language_model.layers.N...`) counts as one module; the output uses the first adapter's layout. Other names, such as a vision tower's layers, are never renamed.
- A non-empty `OUT_DIR` is refused unless `--force` is given; `--force` keeps other files there. `OUT_DIR` cannot be one of the input directories or inside one.
- Run `export --gguf` or `export --splice` on `OUT_DIR` to get one GGUF adapter or one merged GGUF. GGUF export maps the `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` and `down_proj` modules of each layer; any other module is an error. `q_proj` and `k_proj` are exported only for the GGUF architectures `qwen2`, `qwen3`, `qwen3moe`, `qwen35` and `qwen35moe` (llama.cpp reorders their rows for other architectures, such as `llama`, so the change would be wrong); the other five work for any architecture. With `--splice` the architecture is read from the base GGUF.

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

## gate

`astro-weights gate GATE.toml` judges one edit from files that earlier runs wrote. It runs no model. The stages and rules are those of `eval.gate` (constraint style, no blended score); the command adds the file reading, the checks that arms are comparable, the sealed test scoring and `verdict.json`.

Order of work:

1. The run record must say the exported edits are the last that passed confirm (`exported_is_best` on its last `confirm_export` event). Missing or false: exit 2, nothing sealed.
2. `screens`: any failed pre-screen gives outcome `screened out`: exit 3, the test split is never read or scored, the seal ledger is not touched, and no `verdict.json` is written (the verdict schema has no screened-out value; the outcome is on stdout).
3. Confirm: the edit and every control against the base on the confirm split, then the control rule (target beats the controls' `q` quantile, each side effect inside the controls' range). If it fails, the test split is **not** scored: the outcome is `not significant` (target) or `regressed` (side effect), the MDE is the confirm split's, and the seal is untouched.
4. Test: only when stages 1 and 3 passed. Floor by `config_hash` of the test base arm's header, regression suites, and the whole gate are computed first; only then the seal is checked and written, through the same code as `eval --split test` (a second scoring of the same edit or records is refused unless `--unseal`, which the ledger records). The result is shown after the ledger line is written.

Refused with exit 2, nothing sealed: bad or unknown `GATE.toml` key; a records file that is unreadable, outside its split or incomplete; arms whose run headers disagree on any config key (a missing header counts; confirm and test arms may differ in `k` only); `k` of the manifest not 4; no stored floor for the test base arm's `config_hash`; fewer than 20 controls at `q = 0.95`; a seal refusal.

`GATE.toml` (paths are relative to the file's directory; every key is checked and an unknown key is an error):

| Key | Type | Meaning |
|---|---|---|
| `margin` | number | non-inferiority margin of the test benchmark, in score units, at most 0.02 |
| `margin_source` | string | where the margin came from (a pilot); required |
| `backend`, `build` | strings | serving engine and its version; go into `verdict.json` |
| `screens` | path | JSON list of `{"name", "passed", "value", "limit"}`; must hold `norm`, `perplexity` and `kl` |
| `run_record` | path | the agent run's `record.jsonl`; gives `candidates_tried`, `confirm_queries` and `exported_is_best` |
| `manifest` | path | split manifest; confirm and test records must lie in its splits; its `k` must be 4 |
| `boot`, `seed` | int | bootstrap resamples (default 10000) and seed (default 0) for every comparison |
| `arm_keys` | list of strings | `extra_body` keys allowed to differ between arms, as `eval --arm-key` |
| `max_cap_hit` | number in [0, 1] | optional; refuse (exit 2, no verdict) when any confirm or test arm has a larger share of records that hit `max_new_tokens`. Absent: no refusal. Cap-hit shares are printed and kept in `verdict.json` either way |
| `[footprint]` | table | name to number, copied to `verdict.json` |
| `[confirm]` | table | `base`, `edit` (confirm-split records), `controls` (a directory: every `*.jsonl` in it is one control arm, 20 or more), `target` (`tokens_diff`, better lower, or `diff`, better higher), `q` (default 0.95, at least 0.95), `side_effects` (names; `accuracy` is the `diff` of the confirm records, any other name needs a `[confirm.side.NAME]` table with `base`, `edit`, `controls`) |
| `[test]` | table | `base`, `edit` (test-split records, K = 4), `seal_dir` (ledger directory), `adapter` (the edit's PEFT directory, keyed by content), `floor` (directory of stored noise floors, `floor-<config_hash>.json`) |
| `[suites.NAME]` | tables | regression suite: `base`, `edit`, `margin`, `source` (the suite's own margin and where it came from) |
| `[arms]` | table, optional | extra test-split arms for the report: `prompt`, `scripted`, `edit_prompt` (shown as `edit+prompt`). The `prompt` and `edit_prompt` headers must carry a `system` prompt (made with `run_suites(..., system=...)`) that base, edit and scripted lack; every other header key must equal base's |

Every records file `X.jsonl` needs its `X.header.json`.

Output: the `format_gate` report (outcome, every check, MDE and its split, margin and sources, keep-original advice), `candidates tried` and `confirm queries`, then `format_arms` for the confirm split (the edit, with the controls summarised) and, when the test was scored, the test split (the edit plus any `[arms]`). Every row carries its MDE.

`verdict.json` (default: beside `GATE.toml`; `--out FILE` moves it) is the `eval.verdict.Verdict` record: verdict, backend, build, `config_hash` (of the test base arm's header, or of the confirm base arm's when the test was not scored), manifest sha256, candidates tried, confirm queries, `infra_errors` (the `retries` plus `failed` counts the HTTP runner wrote into the test base and edit headers; for a failed confirm, the confirm arms') and footprint. It holds no timestamp, so the same files give the same bytes. `--json` prints one object: the `GateResult` fields (`outcome`, `checks`, `reasons`, `mde`, `mde_split`, `margin`, `margin_source`, `advice`, `suite_margins`), `candidates_tried`, `confirm_queries`, `verdict_file` (or `null`) and `arms` (`confirm` and `test`, each `null` or the `arms_dict` form).

Exit codes: 0 confirmed, 1 regressed or not significant, 3 screened out, 2 bad input or refused.

## gen

```text
astro-weights gen [--resume] [--json] ARM.toml
```

`gen` runs one arm of an eval, one runner call per file, so a long run is its own batch job that can be killed and continued. It asks an OpenAI-compatible server (vLLM, llama.cpp) for the answers; there is no local-model (Hugging Face) kind. For that, call `runners.generate.generate_records` from Python.

`ARM.toml` has a `kind` and the keyword arguments of that kind's runner, under the same names and with the same meaning. Relative paths are read from the file's directory. An unknown key, a missing key or a bad type is an error (exit 2) that names the key.

| `kind` | Runner | Keys (required first) |
|---|---|---|
| `http` | `generate_records_http` with `run_header_http` for the header | `arm`, `base_url`, `model`, `max_new_tokens`, `out` (records JSONL), `items` (JSONL of `{"item_id", "prompt", "gold"}`), `grader` (`exact`, `choice` or `math`; `math` needs the `eval` extra). Optional: `seed` (0), `k` (1), `manifest` with `split`, `extra_body`, `serving_flags`, `model_id`, `system`, `concurrency` (1), `timeout` (120), `api_key_env` |
| `suites` | `run_suites` with `HttpBackend` | `arm`, `base_url`, `model`, `max_new_tokens`, `out_dir` (records go to `out_dir/<suite>/<arm>.jsonl`), `suites` (list of names). Optional: `seed`, `limit`, `cache_dir`, `suite_paths` (table of name to file), `extra_body`, `system`, `concurrency`, `api_key_env` |
| `bfcl` | `load_bfcl` then `run_bfcl`, header from `run_header_http` with `k = trials` | `arm`, `base_url`, `model`, `max_new_tokens`, `out`, `bfcl_root`. Optional: `trials` (1), `max_steps_per_turn` (20), `seed`, `extra_body`, `serving_flags`, `model_id`, `timeout`, `api_key_env` |

- `api_key_env` names an environment variable that holds the key; the key itself never goes in the file or the header.
- `http` with `manifest` and `split`: only that split's items of `items` are run (every id the split lists must be in the file). For `split = "test"`, `k` is the manifest's `k` (an explicit `k` of another value is refused). With `k` above 1 the arm runs seeds `seed` to `seed + k - 1` into `out`'s stem plus `.s0.jsonl`, `.s1.jsonl` and so on, then joins them into `out` with seed 0's header, as `tests/exit_0_3/README.md` makes its K = 4 test arms.
- Without `--resume`, a records file that is not empty is refused (exit 2) before any request. With `--resume`, the records already in the files stay as they are (byte for byte), a last line cut short by a crash is dropped and redone, and only the other items are asked. A file made under another config (`max_new_tokens`, `seed`, `extra_body`, `serving_flags`, `system`, `model_id`, `concurrency`, `k`, a suite's `limit`...) is refused with the differences, and nothing is written. `gen` keeps no state beyond the records and header files the runners write.
- Every record is written and fsynced when it is made, so a kill loses at most the requests in flight.
- Output: `ARM: done/total items, OUT`. `--json` prints `{"arm", "out", "done", "total"}` (`out` is the directory for `suites`; `total` is `done` for `suites`, whose item count comes from the suite files).
- Exit codes: 0 done; 2 bad config, missing extra or key variable, a non-empty file without `--resume`, or a resume refused; 1 a server error that outlasted the runner's retries, or a reply the runner cannot read (no message, no `usage`), once requests have begun (the records so far stay, `done` counts them, the error is on stderr: run again with `--resume`). Everything found before the first request is exit 2.

## mcp

Needs the `mcp` extra; without it, exit 2 and the install hint. `forward_pass` also needs the `hf` extra. Exit 2 when `run.json` is missing or bad, or differs from the run record on `seed`, `editor`, `model` or `limits`.

`RUN_DIR/run.json`:

| Key | Type | Default |
|---|---|---|
| `seed` | int | required |
| `editor` | non-empty string, a label for the record | required |
| `model` | string or `null` | required |
| `limits` | object with exactly `forward_passes`, `code_runs`, `tokens` (ints >= 0) and `dollars` (number >= 0) | required |
| `model_dir` | path to the model, relative to `RUN_DIR` or absolute | absent: `exec_python` and `forward_pass` say they are not available |
| `kernel` | object with `timeout_s` (number > 0), `mem_gib` (int >= 1, Linux only) and `allow_nested_userns` (bool: sandbox with bwrap older than 0.8, see [Explanation: agent budgets and sandbox](../explanation/agent-budgets-and-sandbox.md)) | `600.0`, `16`, `false` |
| `forward` | object with `device` (`auto`, `cpu`, `cuda`, `mps`; `auto` picks `cuda`, then `mps`, then `cpu`), `dtype` (`auto`, `float32`, `bfloat16`, `float16`) and `max_new_tokens` (1 to 32768) | `auto`, `auto`, `4096` |
| `export` | object with `rank` (int >= 1), `max_rel_residual` (number >= 0), `base_gguf` (path relative to `RUN_DIR` or absolute) and `gguf_out_type` (`q8_0`, `f16`, `bf16`) | `8`, `0.05`, absent, `q8_0` |

Tools:

| Tool | Does |
|---|---|
| `exec_python(code)` | start a cell in the sandboxed numpy kernel; returns a handle. One `code_runs` when it finishes |
| `forward_pass(request)` | start a teacher-forced or generate pass on the base or edited weights; returns a handle. One `forward_passes` when it finishes. Request keys are in the `forward_pass` docstring |
| `status(handle, wait_s=0)` | the job's state (`queued`, `running`, `done`, `failed`), its result and the budget; `wait_s` > 0 blocks until the job ends, at most 55 s |
| `finish(summary)` | end the run and export the stored edits to `RUN_DIR/export/` (adapter, plus a merged GGUF with `export.base_gguf`); works once |
| `report_usage`, `check_spend`, `report_refusal` | for editors: record a turn's tokens and dollars, check a dollar estimate, record a refusal |

State under `RUN_DIR`: `ledger.jsonl` (budget), `record.jsonl` (run history), `jobs/`, `edits/`, `forward/`, `kernel/`, `export/`. `--unsafe-kernel` runs agent code without an OS sandbox when none works; see [Explanation: agent budgets and sandbox](../explanation/agent-budgets-and-sandbox.md).

## run

```text
astro-weights run [--resume | --status] [--json] CONFIG.toml
```

`CONFIG.toml` is a TOML file. Relative paths are read from its directory. An unknown table or key, a missing required key or a bad type is an error (exit 2) that names the key.

| Table | Key | Type | Default |
|---|---|---|---|
| `[run]` | `dir` | path: the run directory | required |
| | `seed` | int | required |
| | `model_dir` | path to the model | absent: `exec_python` and `forward_pass` are off |
| `[limits]` | `forward_passes`, `code_runs`, `tokens` | int >= 0 | required |
| | `dollars` | number >= 0; must be > 0 for `hosted` | 20.0 for `hosted`, 0 for the others |
| `[editor]` | `kind` | `scripted`, `local` or `hosted` (`hosted` needs the `hosted` extra: `pip install "astro-weights[mcp,hf,hosted]"`) | required |
| `scripted` | `edit` | list of PEFT adapter directories, relative to the config: candidate edits applied and sent to `confirm` in order, then the last that passed is exported (needs `[confirm]`) | the fixed demo script |
| `local` | `base_url`, `model` | non-empty string | required |
| | `api_key_env` | the name of an environment variable | no key |
| | `max_turns` | int >= 1 | 40 |
| | `temperature` | number >= 0 | 0.0 |
| `hosted` | `model` | a model id in `astro_weights.mcp.editors.prices.PRICES`, or any id with `price` | required |
| | `price` | `"input,cache_write_5m,cache_write_1h,cache_read,output"`, dollars per million tokens; overrides the table | the table's price |
| | `max_output_tokens` | int >= 1 | 16000 |
| | `max_turns` | int >= 1 | 40 |
| | `fallback_base_url`, `fallback_model` | non-empty string | give both, or `no_fallback = true` |
| | `no_fallback` | bool | false |
| `[kernel]` | `timeout_s`, `mem_gib`, `allow_nested_userns` | as in [run.json](#mcp) | 600.0, 16, false |
| | `unsafe` | bool: pass `--unsafe-kernel` | false |
| `[forward]` | `device`, `dtype`, `max_new_tokens` | as in [run.json](#mcp) | `auto`, `auto`, 4096 |
| `[export]` | `rank` | int >= 1 | 8 |
| | `max_rel_residual` | number >= 0 | 0.05 |
| | `base_gguf` | path to the base model's GGUF; also write a merged GGUF | absent: adapter only |
| | `gguf_out_type` | `q8_0`, `f16` or `bf16` | `q8_0` |
| `[confirm]` | `items` | path to the confirm items, JSONL of `{"item_id", "prompt", "gold"}`, outside the run directory | the table is optional; with it, `items`, `grader`, `margin` and `max_queries` are required and `model_dir` is too |
| | `grader` | `exact`, `choice` or `math` | |
| | `margin` | number from 0 below 1 | |
| | `max_queries` | int >= 1 | |
| | `target` | `accuracy` (beat the best by more than `margin * n` items) or `tokens` (fewer total output tokens than the best, accuracy within `margin * n` of base); match the gate's `[confirm] target` (`tokens` for `tokens_diff`) | `accuracy` |
| | `batch_size` | int >= 1; confirm items generated at once, left-padded, for base and edit alike | 1 |

- `run CONFIG.toml` checks the config, the run directory, the extras the editor needs (`mcp`; `hf` with `model_dir`; `gguf` with `base_gguf`; `hosted` for `hosted`), that `model_dir` and `base_gguf` exist and that the key variable (`api_key_env`; `ANTHROPIC_API_KEY` for `hosted`) is set and not empty. Only then does it write `RUN_DIR/run.json` and `RUN_DIR/config.toml` (a copy of the config) and run the editor, which starts the server. The exit code is the editor's; any check that fails exits 2 and writes nothing.
- A key value given in place of a variable name in `api_key_env` is refused and never printed.
- For `hosted`, `[limits] dollars` is the cap passed to the editor as `--max-dollars`.
- Nothing in a run directory is ever deleted. A fresh start refuses a directory that is not empty, unless all it holds is what an aborted start of the same config wrote (`run.json`, `config.toml`). A directory that holds a run (`record.jsonl`, `ledger.jsonl`, `jobs/` or `edits/`) is refused without `--resume`.
- `--resume` goes on with the same ledger, record and edits; budget already used stays used. `seed`, `model_dir`, editor `kind`, editor `model` and `[limits]` must match the saved run, or it is refused with the differences. A finished run cannot be resumed. A resume rewrites `run.json` and adds `config.resume-<n>.toml`; the first `config.toml` is kept.
- `--status` prints the run directory, its state, editor, model and seed, the finish line, budget used / limit (with failure streaks), the stored edits and the last 10 events. It reads from disk and starts nothing. With `--json`, the same as one JSON object.
- On `finish` the server exports the stored edits: a PEFT adapter in `RUN_DIR/export/adapter`, plus a merged GGUF when `base_gguf` is set. The record gets an `export` event (or `export_failed`; the run still finishes).
- Editor details: [scripted](../tutorials/first-agent-run.md), [local](../how-to/agent-local-model.md), [hosted](../how-to/agent-hosted.md).

## JSON output

| Command | Object |
|---|---|
| `inspect` | `base_model`, `rank`, `modules` (each `module`, `shape`, `rank`, `norm_B`, `norm_A`, `norm_BA`), `manifest` (bool), `problems` |
| `eval` | see [eval](#eval) |
| `compose` | `written` (the output directory), `rank`, `modules` (sorted names) and `sources` (the input directories, in order) |
| `gate` | see [gate](#gate) |
| `gen` | `arm`, `out`, `done`, `total`; see [gen](#gen) |
| `build`, `export` | `written`: the list of paths written. `export --splice` adds `replaced` (the GGUF tensor names changed) and `out_type` |

## Exit codes

- All commands: 0 means ok, 1 means problems were found or a check failed, 2 means a usage error or unreadable input. No command at all is a usage error (exit 2).
- `inspect` exits 1 on hash problems (`problem:` lines on stderr, also with `--json`).
- `eval` exits 1 when the run headers differ, and 2 on an unreadable file or arms that do not pair.
- `self-test` exits 0 when every output matches, 1 when they differ, and 2 on a bad argument, an unset `--api-key-env` variable or a server error.
- `gen` exits 0 when every item has a record, 1 on a server error after retries or an unreadable reply once requests began, and 2 on bad input or a refused resume (see [gen](#gen)).
- `gate` exits 0 when confirmed, 1 when regressed or not significant, 3 when screened out, and 2 on bad input or a refusal (see [gate](#gate)).
- `build`, `compose` and `export` exit 0 on success and 2 on any error.
- Errors go to stderr as a line starting with `error:`; read and parse errors name the file.
