# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/). While the version is 0.x, any release may contain breaking changes; they are called out under **Changed** or **Removed**.

<!-- towncrier release notes start -->

## [0.0.1] - 2026-11-14

First release. Pre-alpha: the API will change.

### Added
- Project skeleton: packaging, CI, release workflow.
- `Edit` and `LowRankDelta` types.
- Edit recipes: `rank1`, `masked_rank1`, `column_shift`, `orthogonal_random`.
- PEFT LoRA save/load with a hash manifest.
- `apply`/`remove` on transformers models (`hf` extra).
- `astro-weights inspect` CLI.

[0.0.1]: https://github.com/astro-weights/astro-weights/releases/tag/v0.0.1
