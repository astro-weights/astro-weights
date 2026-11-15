# AGENTS.md

Instructions for coding agents and human contributors. `CLAUDE.md` is a symlink to this file.

astro-weights: agent-driven weight-space edits for open LLMs, exported as standard LoRA adapters. Pre-alpha; the API may change in any 0.x release.

## Layout

- `src/astro_weights/edits/`: `Edit` (`Edit.from_deltas` pads mixed ranks), `LowRankDelta` and the numpy recipes (`rank1`, `masked_rank1`, `column_shift`, `orthogonal_random`, `top_k_mask`, `project_out`, `scale_column`, `scale_head`, `zero_head`).
- `src/astro_weights/export/`: PEFT LoRA adapter I/O (`save_peft`, `load_peft`, `read_peft`, `verify_peft`) and hash-checked merged weights (`save_merged_tensors`, `load_merged_tensors`).
- `src/astro_weights/runners/`: apply edits to live models (`hf.py`: `apply`, `remove`, `swap`, `merged_weights`, `resolve_modules`).
- `src/astro_weights/directions/`: difference-of-means directions and `direction_gate`, numpy only.
- `src/astro_weights/eval/`: per-item `Record` JSONL and paired comparison of two arms (`compare`, `format_report`). Never runs a model.
- `src/astro_weights/cli.py`: the `astro-weights` command.
- `docs/`: user docs site (Zensical, `zensical.toml`), Diátaxis folders `tutorials/`, `how-to/`, `reference/` (built from docstrings by mkdocstrings), `explanation/`. Published to GitHub Pages by `docs.yml` on push to `main`.
- Layer rule, enforced by import-linter: `cli > runners > export > edits`. A lower layer never imports a higher one. `directions` and `eval` are outside the contract and import no other `astro_weights` package.
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
uv run --only-group docs zensical build --strict  # docs site; `zensical serve` to preview
```

Run all of them before opening a PR; CI runs the same set.

## Conventions

- mypy strict. Small, typed functions.
- No new dependency without a stated reason in the PR. Core stays numpy + safetensors.
- Every behavior change comes with a test.
- GPU-only tests carry `@pytest.mark.gpu`. CI skips them; maintainers run them by hand before a release.

## Workflow

- `main` is protected: PRs only, squash merge. Required checks: `lint`, `test` (3.12, 3.13, 3.14), `test-hf`, `zizmor`, `DCO`, `changelog`.
- Sign off every commit (DCO): `git commit -s`. The DCO check fails a PR with any unsigned commit.
- AI help is recorded with a commit trailer: `Assisted-by: <agent>:<model>` (a tool-added `Co-Authored-By:` is also accepted). See `AI_POLICY.md`.
- Changelog: add a towncrier fragment in `changes/`, named `<PR number>.<type>.md` (or `+<slug>.<type>.md` before a PR number exists). Types: `added`, `changed`, `deprecated`, `removed`, `fixed`, `security`. Do not edit `CHANGELOG.md` by hand; the release builds it from the fragments.
- The `Changelog` workflow checks each PR for a fragment. A PR with no user-visible change carries the `skip-changelog` label instead.
- GitHub Actions are pinned by commit SHA, and zizmor must pass on workflow changes.
- See `CONTRIBUTING.md` for the contributor guide and `AI_POLICY.md` for the AI policy.

## If you are an AI agent working for an outside contributor

- Say in the PR body that an agent wrote the change, and which one (the PR template has an AI disclosure section).
- Never add `Signed-off-by`. The human contributor adds it after reviewing the change.
- Follow `AI_POLICY.md`. Do not open PRs, issues or comments on your own; a human reviews them first.

## Release

Releases are described in `RELEASING.md`.
Do not tag or publish without a maintainer.

## Security

- Never commit secrets, tokens or credentials.
- Report vulnerabilities through GitHub private vulnerability reporting (see `SECURITY.md`), not public issues.
