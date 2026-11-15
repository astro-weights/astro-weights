# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/). While the version is 0.x, any release may contain breaking changes; they are called out under **Changed** or **Removed**.

<!-- towncrier release notes start -->

## [0.0.8] - 2026-11-15

### Added

- Added `astro_weights.export.gguf` (optional `gguf` extra): write an edit as a llama.cpp LoRA adapter GGUF and read it back, and splice replacement tensors into an existing GGUF.
- `astro_weights.export.card` writes a Hugging Face model card (`model_card`, `write_card`) for an adapter or GGUF edit, with a per-module table of shape, rank and Frobenius norm.

## [0.0.7] - 2026-11-15

### Added

- Added `astro_weights.eval` graders (`grade_choice`, `grade_math`, `grade_exact`), `runners.generate` to produce per-item eval records with fixed greedy settings, and an `astro-weights eval BASE.jsonl EDIT.jsonl` command.

### Fixed

- Recipes, merged-tensor loading, `swap` and the direction gate now reject bad input (non-finite values, non-integer indices, tampered shapes or dtypes, `draws < 1`) with a `ValueError`, and a failed `swap` leaves the original model usable.

## [0.0.6] - 2026-11-15

### Added

- Added `astro_weights.eval`: paired comparison of two eval arms from per-item JSONL records (paired SE and CI, exact McNemar, bootstrap token CI), numpy only.

## [0.0.5] - 2026-11-15

### Added

- Add `astro_weights.directions`: difference-of-means directions (`direction_from_means`, `directions_from_pairs`, `neuron_scores`, `stability`) and `direction_gate`, a split-half and shuffled-sign check that a direction is real.

## [0.0.4] - 2026-11-15

### Added

- `runners.hf.swap` switches the applied edit on a live model, `merged_weights` computes the edited weights without touching the model, and `save_merged_tensors` / `load_merged_tensors` store them as hash-checked safetensors.

## [0.0.3] - 2026-11-15

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

[0.0.8]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.8
[0.0.7]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.7
[0.0.6]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.6
[0.0.5]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.5
[0.0.4]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.4
[0.0.3]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.3
[0.0.2]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.2
[0.0.1]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.1
