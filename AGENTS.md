# AGENTS.md

Instructions for coding agents and human contributors. `CLAUDE.md` is a symlink to this file.

astro-weights: agent-driven weight-space edits for open LLMs, exported as standard LoRA adapters. Pre-alpha; the API may change in any 0.x release.

## Layout

- `src/astro_weights/edits/`: `Edit`, `LowRankDelta` and the numpy recipes (`rank1`, `masked_rank1`, `column_shift`, `orthogonal_random`, `top_k_mask`).
- `src/astro_weights/export/`: PEFT LoRA adapter I/O (`save_peft`, `load_peft`, `read_peft`, `verify_peft`).
- `src/astro_weights/runners/`: apply edits to live models (`hf.py`: `apply`, `remove`).
- `src/astro_weights/cli.py`: the `astro-weights` command.
- Layer rule, enforced by import-linter: `cli > runners > export > edits`. A lower layer never imports a higher one.
- Core dependencies are numpy and safetensors only. torch, transformers and peft are imported only inside `runners/`, and only through the `hf` extra.

## Commands

```bash
uv sync                  # install the package and dev tools
uv run ruff check        # lint
uv run ruff format       # format
uv run mypy src tests    # type check (strict)
uv run lint-imports      # layer contract
uv run pytest            # core tests
uv run --extra hf pytest # tests that need torch/transformers/peft
```

Run all of them before opening a PR; CI runs the same set.

## Conventions

- mypy strict. Small, typed functions.
- No new dependency without a stated reason in the PR. Core stays numpy + safetensors.
- Every behavior change comes with a test.
- GPU-only tests carry `@pytest.mark.gpu`. CI skips them; maintainers run them by hand before a release.

## Workflow

- `main` is protected: PRs only, squash merge. Required checks: `lint`, `test` (3.12, 3.13, 3.14), `test-hf`, `zizmor`.
- Sign off every commit (DCO): `git commit -s`.
- Add a `CHANGELOG.md` entry under `## [Unreleased]` for anything a user would notice.
- GitHub Actions are pinned by commit SHA, and zizmor must pass on workflow changes.

## Release

1. Bump the version with `uv version <x.y.z>` (or `uv version --bump patch`).
2. Move the `[Unreleased]` entries in `CHANGELOG.md` under the new version, with the date.
3. Open a PR and merge it.
4. Push tag `vX.Y.Z` on `main`. `release.yml` checks the tag matches the version, builds, publishes to TestPyPI, then waits for a maintainer to approve the `pypi` environment.

## Security

- Never commit secrets, tokens or credentials.
- Report vulnerabilities through GitHub private vulnerability reporting (see `SECURITY.md`), not public issues.
