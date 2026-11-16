# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/). While the version is 0.x, any release may contain breaking changes; they are called out under **Changed** or **Removed**.

<!-- towncrier release notes start -->

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

[0.1.2]: https://github.com/astro-weights/astro-weights/releases/tag/v0.1.2
[0.1.1]: https://github.com/astro-weights/astro-weights/releases/tag/v0.1.1
[0.1.0]: https://github.com/astro-weights/astro-weights/releases/tag/v0.1.0
[0.0.9]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.9
[0.0.8]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.8
[0.0.7]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.7
[0.0.5]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.5
[0.0.2]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.2
[0.0.1]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.1
