# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/). While the version is 0.x, any release may contain breaking changes; they are called out under **Changed** or **Removed**.

<!-- towncrier release notes start -->

## [0.3.0] - 2026-11-24

### Added

- A noise-floor result (`FloorResult`, `floor_from_records`, `save_floor`, `load_floor`) is stored under the run's `config_hash` and loaded only for that exact hash.
- Added the 0.3 exit test (`tests/test_exit_0_3.py`, `ASTRO_EXIT_DATA`, layout and commands in `tests/exit_0_3/README.md`): criteria 2 to 8 of PDR 0.3 on recorded files, plus a synthetic variant for CI.
- Docs: how-to "Run the 0.3 gate end to end" (splits and seal, self-test and noise floor, pre-screens, random controls, scripted and prompt arms, the `confirm` tool, regression suites, BFCL `multi_turn_base`, `astro-weights gate`) and explanation "Gate verdicts" (outcomes, keep-original advice, why the test split is sealed, MDE). CLI reference lists `self-test`; MCP reference lists the `confirm` object of `run.json`.
- Guards: edit norm range check, perplexity and last-token KL pre-screens (`edits.check_norms`, `runners.guards`).
- MCP `confirm` tool (with a `confirm` object in run.json): the agent learns only whether its stored edits beat the current best on held-out confirm items by the margin, at most `max_queries` times; `finish` reports whether the export is the last edit that passed (`exported_is_best`), and `RunRecord.confirm_counts` gives `candidates_tried` (distinct edit sets sent to confirm, not every save), `candidates_saved` (accepted `save_edit` calls) and `confirm_queries`.
- Multi-turn task records (`TaskRecord`, `read_tasks`, `write_tasks`) and `compare_tasks` / `format_tasks`: paired pass rate, tokens, tool calls and turns per task, trials averaged per task.
- Run headers split their keys into config (decide the outputs) and provenance (where and when). `check_headers` now compares config keys only (model identity, backend and build are config), so HTTP arms under different served names or URLs no longer need `--ignore-header`; new `config_hash`, `CONFIG_KEYS`, `PROVENANCE_KEYS`, and an `arm_keys` argument and `astro-weights eval --arm-key KEY` for arms on one server that differ in an `extra_body` key. `run_header` and `run_header_http` also record a run id, K, concurrency, library version and commit, numpy version, GPU model and count, llama.cpp build info, vLLM version, declared `serving_flags` and a `model_id` for HTTP arms. New `astro_weights.eval.Verdict` schema for `verdict.json`.
- Runners take a `system` prompt (`generate_records`, `generate_records_http`, `run_suites`; recorded in the run header as `system` and `system_sha256`) so the gate's `prompt` and `edit+prompt` arms pair with base on the item prompt, and `astro-weights gate` accepts those arms only when their header carries a system prompt that base, edit and scripted lack.
- Seeded dev / confirm / test splits (`astro_weights.eval.splits`: `assign_splits`, `SplitManifest` with `k` seeds per test item) and a sealed test split (`astro_weights.eval.seal`): `astro-weights eval --split test --manifest M --seal-dir D --adapter A` scores each edit once on a test set (same records, same factors, or the same `B @ A` within 1e-3 under probes seeded by the ledger's secret `seal.salt` are refused; a scored edit's records cannot come back as a base or floor arm), and `--unseal` scores it again with a line in the ledger saying so.
- `astro-weights export --gguf` and `--splice` print the sha256 of each GGUF they write, and return it under `sha256` with `--json`, so the exact file can be recorded with the results.
- `astro-weights gate GATE.toml` runs the 0.3 gate end to end from recorded files (pre-screens, random controls on confirm, the sealed K=4 test split through the same seal as `eval --split test`, regression suites), prints the outcome with every check and arm, and writes `verdict.json`; exit 0 confirmed, 1 regressed or not significant, 3 screened out, 2 bad input or refused.
- `astro_weights.eval.suites` loads the GSM8K, IFEval and MMLU-Pro regression suites at a pinned revision and sha256 (download or your own file), and `astro_weights.eval.ifeval` grades IFEval strictly; the `eval` extra now includes pyarrow and langdetect.
- `astro_weights.runners.suites_run.run_suites` runs the GSM8K, IFEval and MMLU-Pro regression suites for one arm (Hugging Face model or OpenAI-compatible server), grades each item with the suite grader and writes pairable records and a run header per suite.
- `compare_arms` and `format_arms` compare one base against many named arms (edit, random controls, prompt, scripted) in one table with the MDE on every row and a summary of the controls.
- `compare` and `astro-weights eval` add the 0.3 paired statistics: `k` samples per item with an exact sign test on per-item means, a paired bootstrap CI for the score difference, Wilson and Newcombe intervals for binary scores, per-stratum rows (`strata=`), and the minimum detectable difference on every result line.
- `control_rule` judges an edit against random controls: the target effect must strictly beat their q-quantile (at least 20 controls for q=0.95) and every side effect must stay inside the controls' range.
- `gate` turns pre-screens, random controls, the K=4 test split and regression suites into one constraint-style outcome (confirmed, not significant, regressed or screened out) with every check, the MDE, the recorded non-inferiority margin (capped at 2 pp) and keep-original advice; `format_gate` prints it.
- `generate_records_http` takes `concurrency`, `resume` and `header`: requests run in parallel while records stay in item order, a crashed run resumes without gaps or duplicates (refused under another config), and connection errors, timeouts, 429 and 5xx are retried and counted in the header's `infra_errors`.
- `random_like` and `random_controls` build seeded random edits with the modules, shapes, rank and per-module `|B @ A|_F` of an edit, optionally matched after rounding to bfloat16 or float16, as controls for a real edit.
- `runners.endpoint.chat_http` makes one tool-calling chat request, and `runners.bfcl` (`load_bfcl`, `run_task`, `run_bfcl`) runs BFCL `multi_turn_base` tasks from a pinned local checkout against an OpenAI-compatible server and scores them with BFCL's state and response checks into `TaskRecord`s.
- `runners.scripted_arm.scripted_arm` builds a closed-form baseline edit (difference-of-means direction, `project_out` on the writer modules of a given edit) to compare an agent's edit against.
- `self_test_http` and `astro-weights self-test` send one prompt under varied batch sizes and prefix-cache states and report whether the server returns a single output.
- `tokens_beyond_floor` is a one-sided check that an edit lowers output tokens by more than the base-vs-base noise floor.

### Changed

- `check_headers(base, edit, ignore=...)` now always skips the provenance keys and `ignore` adds to them, so the old `ignore=("device",)` still works; the default is `()`. `run_header_http` records the served name as `served_model` (provenance) instead of `model`, so an HTTP header written by 0.2 differs from a new one on `model`. New config keys (`model_name_or_path`, `model_id`, `backend`, and the torch, transformers, llama.cpp build and vLLM versions) are missing from old headers, so an old header differs from a new one on them. Rerun the arm, or pass `--ignore-header`.

### Fixed

- A tool call the server cannot parse (llama-server answers HTTP 500 "Failed to parse tool call") is now a BFCL turn without a call, and a conversation that outgrows the server context (HTTP 400 `exceed_context_size_error`) ends that task as failed; before, either one aborted `run_bfcl` and lost every record of the run.
- Export takes each edit's low-rank dW from the float edit the agent sent, not the copy rounded to the base dtype, so a small edit on a bf16 base no longer fails `max_rel_residual`; the adapter meta records `delta_source` ("float", or "rounded" for edits stored before this change) and `replay_max_abs` per tensor.
- The 0.3 exit test gates the edit against `control-01` to `control-20` and each of 21 controls against the other 20, so no gate pool contains the arm it gates.
- The `[confirm]` table of an `astro-weights run` config (items, grader, margin, max_queries) writes the `confirm` object of `run.json`, so a run started from TOML can use the `confirm` tool.
- The `exec_python` and `finish` tool descriptions now say that `finish` exports each edit at rank `export.rank` (default 8) and fails above `export.max_rel_residual` (default 0.05), so MCP clients such as Codex keep edits low rank.

## [0.2.2] - 2026-11-19

### Added

- GGUF export and splice accept `q_proj` and `k_proj` deltas for Gemma 3 and Gemma 4 models (`gemma3`, `gemma4`), whose llama.cpp converter keeps those rows in Hugging Face order.
- The hosted editor knows the prices of Claude Fable 5.2, 5.1 and 5, Claude Mythos 5.2, 5.1 and 5, Claude Opus 4.5 and Claude Sonnet 4.5, and takes a price for any other model id from `[editor] price` in the run config (or `--price`), so a new model works without waiting for a release.

### Fixed

- `grade_math` called from a worker thread now grades the answer; before, `math_verify`'s signal-based timeout raised there and every answer was marked wrong.

## [0.2.1] - 2026-11-18

### Added

- `forward_pass` runs on Apple GPUs: `forward.device` accepts `mps`, and `auto` now picks `mps` on a Mac without CUDA, where it used to pick the CPU. (Merged just before the `v0.2.1` tag, so it shipped in 0.2.1 but was missing from these notes at release.)

### Fixed

- The run record's `start` line now fills `library_commit`: the git commit of a `pip install git+...` install, or of an editable checkout (with `-dirty` when it has uncommitted changes). A wheel installed from PyPI carries no commit, so there it stays `null`: `library_version` names the release, and the release's `v` tag names the commit.

## [0.2.0] - 2026-11-18

0.2.0 final: everything in 0.2.0a1 below, plus the editors and the one-command run listed here.

Upgrading from 0.1.5: no public name was removed or changed, so 0.1.x code runs as before. The agent features are opt-in extras: `pip install "astro-weights[mcp,hf]"` for the MCP server and the scripted and local editors, and add `hosted` for the hosted Claude editor. The 0.1.5 model-card fix is included. The MCP server was run from Claude Code; the Codex setup page is not tested.

### Added

- A hosted editor, `python -m astro_weights.mcp.editors.hosted RUN_DIR --model ID` (the `hosted` extra), lets Claude drive the stdio MCP server on your own `ANTHROPIC_API_KEY`, checks every call against the run's dollar limit before making it (a run whose limit is above `--max-dollars`, $20 by default, is refused) and hands the run to a local model when Claude refuses.
- A local editor, `python -m astro_weights.mcp.editors.local RUN_DIR --base-url URL --model NAME`, lets any OpenAI-compatible model with tool calls drive the stdio MCP server, reporting its tokens to the ledger at no dollar cost.
- Report agent usage, spend check and refusal tools on the MCP server for editors.
- `astro-weights run CONFIG.toml` starts an agent run (run directory, budget, editor, export options) from one TOML file, never starts over a run that already exists, goes on with it under `--resume` with the budget used kept, prints its budget and last events with `--status`, and the run's `finish` now exports the stored edits as a PEFT adapter (and a spliced GGUF when `base_gguf` is set) to `RUN_DIR/export`.
- On Linux with bubblewrap older than 0.8 (RHEL 9 ships 0.6), `kernel.allow_nested_userns = true` lets the agent kernel run sandboxed without `--disable-userns`; the default still refuses, and the run record notes the weaker sandbox.

### Changed

- The `exec_python` tool description now lists the kernel helpers (`save_edit`, `load`, `forward_result` and the rest) and the edit size cap, so MCP clients such as Claude Code and Codex can use them without probing the kernel.

### Fixed

- A run config written with Windows line endings is now copied into the run directory byte for byte.
- Every text file the library reads or writes (recipes, adapter configs, manifests, model cards) now uses UTF-8, so non-ASCII text no longer depends on the Windows locale.

## [0.2.0a1] - 2026-11-17

Alpha: agent over MCP. Interfaces may change before 0.2.0.

### Added

- New `[mcp]` extra and `astro-weights mcp RUN_DIR`, a stdio MCP server with `exec_python`, `forward_pass`, `status` and `finish` tools, budgets for forward passes, code runs, tokens and dollars, and a run record.
- `exec_python` runs numpy code in a sandboxed kernel (bubblewrap on Linux, Seatbelt on macOS; elsewhere only with `--unsafe-kernel`); edits are checked and saved by the server.
- The MCP server's `forward_pass` tool runs the model (`hf` extra) on base or edited weights, in `teacher_forced` and `generate` modes, and agent code reads its arrays and texts with `forward_result(handle, key)`.
- Adds `runners.hooks.capture`, which reads hidden states, MLP inputs, attention and logprobs on chosen layers of a transformers model.
- A scripted editor, `python -m astro_weights.mcp.editors.scripted RUN_DIR`, drives the stdio MCP server end to end with a fixed edit-and-compare script and no model.
- Blog scaffold: `zensical.blog.toml` builds `docs/blog/` to https://astro-weights.dev/blog/. First post: https://astro-weights.dev/blog/0-1-on-real-gpus/.

### Changed

- The docs and blog load Google Analytics only after you accept the cookie banner; Cookie settings in the footer changes the choice.

## [0.1.5] - 2026-11-18

### Fixed

- Model cards for adapters saved with `model.language_model.layers` names now load the base with `AutoModelForImageTextToText`; the old `AutoModelForCausalLM` snippet loaded none of the adapter weights.

## [0.1.4] - 2026-11-17

### Added

- Added `astro_weights.edits.compose` and `astro-weights compose`, which combine several LoRA adapters on the same base model into one exact adapter; GGUF export now also maps `v_proj`, `gate_proj` and `up_proj` for every architecture, and `q_proj` and `k_proj` for the Qwen architectures whose llama.cpp conversion leaves them as they are.

## [0.1.3] - 2026-11-17

### Added

- Adapters can now be merged into a quantized base GGUF with `astro-weights export ADAPTER --splice BASE.gguf --base-model DIR --merged OUT.gguf`, giving one file that Ollama and LM Studio load directly.

## [0.1.2] - 2026-11-16

### Added

- `astro-weights eval` and `format_report` now print a warning when an arm hits the token cap often or unevenly, with the neither-capped comparison and loop counts, and `--json` carries the new fields.
- `generate_records` and `run_header` take `sampling=` (temperature, top_p, top_k, presence_penalty) for sampled decoding; the settings go into the run header so `astro-weights eval` rejects arms sampled differently.
- `python -m astro_weights` now runs the command line.

### Changed

- Docs and README now read as plain English, say plainly that 0.1 is a toolkit and the edit agent is still the goal, list every feature, cite the clustered-SE paper, and show an honest "apply" example that links the first-edit tutorial. The code of conduct contact is now conduct@astro-weights.dev, and the `mcp` package keyword is gone.
- Docs moved to https://astro-weights.dev/docs/; old URLs redirect.
- Docs open in dark mode by default and pick up a light starfield from the landing page.
- New tagline: "Agent-driven weight edits. Democratizing AI for all." Docs header logo and site name link to the landing page.
- The docs site at https://astro-weights.dev/docs/ now loads Google Tag Manager for visitor analytics.

### Fixed

- Importing the runners or GGUF export without the `hf` or `gguf` extra now says which extra to install instead of a bare `No module named 'torch'`.
- Model cards for edited GGUF models no longer say `base_model_relation: quantized` (now `finetune`), and every card states that the base model's weights were modified by a low-rank edit.
- Running `astro-weights` with no subcommand now prints help and exits 2 instead of 0, and every option has a help string.

## [0.1.1] - 2026-11-15

### Changed

- Docs moved to https://astro-weights.dev; old github.io URLs redirect.
- The logo astronaut has larger eyes, a slimmer face and a softer smile.

## [0.1.0] - 2026-11-15

### Added

- Eval reports now show the minimum detectable effect and a noise-floor check (`astro-weights eval --floor A.jsonl B.jsonl`), arms are checked against their `X.header.json` run headers (`--ignore-header` to skip), and every subcommand takes `--json`.
- New `rounding_error(model, edit)` reports how much each delta changes when its factors are cast to the model's weight dtype (for example bfloat16), or `inf` when a factor overflows that dtype, and `apply(..., max_rounding_error=...)` refuses an edit whose error is above the limit before touching the model.
- `compare` now uses cluster-robust standard errors when records carry `cluster_id`: the score SE, CI and MDEs and the token SE and MDE use the clustered SE (with a G/(G-1) factor), the token bootstrap resamples clusters, and `PairedResult.n_clusters` and the report show the cluster count. Records without `cluster_id` give the same results as before.
- `generate_records` takes `out`, `resume` and `batch_size`: records are written to a JSONL file as they are made, a crashed run resumes from that file, and items can be generated in left-padded batches. `run_header` records `batch_size`.
- from_dense converts a dense dW matrix to a LowRankDelta by truncated SVD and returns its relative residual, with an optional bound.
- generate_records_http and run_header_http (astro_weights.runners.endpoint) run an eval arm against an OpenAI-compatible server (llama-server, vLLM) with the standard library only, so base and edit served under two model names can be compared without torch.

### Changed

- The `hf` extra now requires `torch>=2.4`, the lowest version that works with `transformers>=5.4`.
- `read_peft`, `load_peft` and `verify_peft` now reject adapters they cannot represent (non-positive or non-integer `r`, `rank_pattern`, `alpha_pattern`, `modules_to_save`, `bias` other than `none`, tensors outside a list-form `target_modules`, malformed JSON or manifest fields) with a `ValueError`, and `verify_peft` never raises.
- save_peft now refuses a non-empty directory unless `overwrite=True`, and writes the adapter atomically through a temp directory.

### Fixed

- The run header from `run_header` records the model class under an applied adapter instead of `PeftModel`, so `astro-weights eval` no longer rejects a base arm and an `apply`-ed edit arm as run under different settings. ([#46](https://github.com/astro-weights/astro-weights/pull/46))
- GGUF reading now raises `ValueError` for any malformed file, including duplicate keys, and rejects absurd header counts, array lengths, files with more than 2,097,152 kv array items in total (the reader costs about 1 KB per item), and non-regular files such as FIFOs up front instead of hanging or exhausting memory.
- LowRankDelta now rejects NaN or infinite factors, including float64 values that overflow when cast to float32.
- Recipe loading now raises `ValueError` instead of `IndexError`, `OverflowError`, `tokenize.TokenError` or an out-of-memory crash on malformed `.npy` side files (wrong rank or dtype, a header claiming more data than the file holds, a file over 256 MiB), on integer parameters too large for a float, and on a `column_shift` `n_in` above 1048576.
- Recipes now raise `ValueError` instead of `IndexError` or `TypeError` when an array has the wrong dtype or number of dimensions: `column_shift` needs an integer `idx` and real `shift` and `coef` (not bool or complex), and `rank1`, `masked_rank1` (a bool `mask` still works), `project_out`, `top_k_mask` and `orthogonal_random` need real 1-D arrays.
- `apply`, `resolve_modules` and `merged_weights` now say an adapter is already applied (use `swap()` or `remove()`) instead of "no nn.Linear matches", name the module type when the target is not `nn.Linear` (such as GPT-2's `Conv1D`), and `merged_weights` rejects non-floating-point weights; a failed `apply` no longer leaves LoRA layers on the model.
- `compare` computes the exact McNemar p-value in linear time, so large evals with tens of thousands of discordant items no longer hang.
- `read_jsonl` now raises `ValueError` (not `OverflowError` or `RecursionError`) for an out-of-range score or deeply nested JSON.
- `read_peft` and `load_peft` now raise `ValueError` for an adapter config missing `r` or `lora_alpha` and for a weights file that is not valid safetensors, instead of leaking `KeyError` and `SafetensorError`.

## [0.0.9] - 2026-11-15

### Added

- A recipe file (`load_recipe`, `build_edit`, `save_recipe`) rebuilds an edit from a small JSON description plus the `.npy` arrays it names, so a published adapter can be checked.
- Add `astro_weights.export.base`: `read_tensor`, `read_base_weight` and `tensor_sha256` read one F32, F16 or BF16 weight of a base model from its safetensors files with numpy only, no torch.
- Adds `astro-weights build` (recipe file plus base model to a PEFT adapter) and `astro-weights export` (GGUF LoRA adapter and model card), and exports `build_edit`, `load_recipe`, `save_recipe`, `read_base_weight`, `read_tensor` and `tensor_sha256`.

## [0.0.8] - 2026-11-15

### Added

- Added `astro_weights.export.gguf` (optional `gguf` extra): write an edit as a llama.cpp LoRA adapter GGUF and read it back, and splice replacement tensors into an existing GGUF.
- `astro_weights.export.card` writes a Hugging Face model card (`model_card`, `write_card`) for an adapter or GGUF edit, with a per-module table of shape, rank and Frobenius norm.

## [0.0.7] - 2026-11-15

### Added

- Added `astro_weights.eval` graders (`grade_choice`, `grade_math`, `grade_exact`), `runners.generate` to produce per-item eval records with fixed greedy settings, and an `astro-weights eval BASE.jsonl EDIT.jsonl` command.

### Fixed

- Recipes, merged-tensor loading, `swap` and the direction gate now reject bad input (non-finite values, non-integer indices, tampered shapes or dtypes, `draws < 1`) with a `ValueError`, and a failed `swap` leaves the original model usable.

## 0.0.6 - 2026-11-15 (never released; these changes shipped in 0.0.7)

### Added

- Added `astro_weights.eval`: paired comparison of two eval arms from per-item JSONL records (paired SE and CI, exact McNemar, bootstrap token CI), numpy only.

## [0.0.5] - 2026-11-15

### Added

- Add `astro_weights.directions`: difference-of-means directions (`direction_from_means`, `directions_from_pairs`, `neuron_scores`, `stability`) and `direction_gate`, a split-half and shuffled-sign check that a direction is real.

## 0.0.4 - 2026-11-15 (never released; these changes shipped in 0.0.5)

### Added

- `runners.hf.swap` switches the applied edit on a live model, `merged_weights` computes the edited weights without touching the model, and `save_merged_tensors` / `load_merged_tensors` store them as hash-checked safetensors.

## 0.0.3 - 2026-11-15 (never released; these changes shipped in 0.0.5)

### Added

- Docs site at https://astro-weights.github.io/astro-weights/: tutorial, how-to, API reference built from docstrings, explanation. ([#14](https://github.com/astro-weights/astro-weights/pull/14))
- Added `LowRankDelta.padded(rank)` and `Edit.from_deltas(...)` to combine deltas of different rank in one edit by exact zero padding.
- New edit recipes `project_out`, `scale_column`, `scale_head` and `zero_head` in `astro_weights.edits`.

## [0.0.2] - 2026-11-15

### Added

- `read_peft(dir)` returns the edit and any verification problems in one pass. ([#10](https://github.com/astro-weights/astro-weights/pull/10))
- `astro_weights.runners.hf.resolve_modules(model, edit)` returns which model layer each edit module maps to, so a mapping can be checked before applying. ([#13](https://github.com/astro-weights/astro-weights/pull/13))

### Changed

- `astro-weights inspect` reads the adapter once and computes `|BA|_F` without building the dense matrix. ([#10](https://github.com/astro-weights/astro-weights/pull/10))
- New logo, based on the Noto Emoji astronaut (Apache-2.0), replacing the unlicensed emoji avatar. README has a new tagline: "Small course corrections for open LLMs." ([#11](https://github.com/astro-weights/astro-weights/pull/11))

### Fixed

- `verify_peft` reports a corrupt weights file as a problem instead of raising `SafetensorError`. ([#10](https://github.com/astro-weights/astro-weights/pull/10))
- `apply` now finds the edit's layers when the model was loaded under a different class than the adapter was built with: `model.layers.N...` (text-only) and `model.language_model.layers.N...` (multimodal) name the same layers. An ambiguous or missing match raises instead of guessing, and `apply` raises if the adapter did not attach to every layer. ([#13](https://github.com/astro-weights/astro-weights/pull/13))

## [0.0.1] - 2026-11-14

First release. Pre-alpha: the API will change.

### Added
- Project skeleton: packaging, CI, release workflow.
- `Edit` and `LowRankDelta` types.
- Edit recipes: `rank1`, `masked_rank1`, `column_shift`, `orthogonal_random`.
- PEFT LoRA save/load with a hash manifest.
- `apply`/`remove` on transformers models (`hf` extra).
- `astro-weights inspect` CLI.

[0.3.0]: https://github.com/astro-weights/astro-weights/releases/tag/v0.3.0
[0.2.2]: https://github.com/astro-weights/astro-weights/releases/tag/v0.2.2
[0.2.1]: https://github.com/astro-weights/astro-weights/releases/tag/v0.2.1
[0.2.0]: https://github.com/astro-weights/astro-weights/releases/tag/v0.2.0
[0.2.0a1]: https://github.com/astro-weights/astro-weights/releases/tag/v0.2.0a1
[0.1.5]: https://github.com/astro-weights/astro-weights/releases/tag/v0.1.5
[0.1.4]: https://github.com/astro-weights/astro-weights/releases/tag/v0.1.4
[0.1.3]: https://github.com/astro-weights/astro-weights/releases/tag/v0.1.3
[0.1.2]: https://github.com/astro-weights/astro-weights/releases/tag/v0.1.2
[0.1.1]: https://github.com/astro-weights/astro-weights/releases/tag/v0.1.1
[0.1.0]: https://github.com/astro-weights/astro-weights/releases/tag/v0.1.0
[0.0.9]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.9
[0.0.8]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.8
[0.0.7]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.7
[0.0.5]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.5
[0.0.2]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.2
[0.0.1]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.1
