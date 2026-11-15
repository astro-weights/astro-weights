# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/). While the version is 0.x, any release may contain breaking changes; they are called out under **Changed** or **Removed**.

<!-- towncrier release notes start -->

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

[0.0.3]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.3
[0.0.2]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.2
[0.0.1]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.1
