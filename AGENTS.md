# AGENTS.md

Instructions for coding agents and human contributors. `CLAUDE.md` is a symlink to this file.

astro-weights: agent-driven weight edits. Democratizing AI for all. Edits to open LLMs are exported as standard LoRA adapters. Pre-alpha; the API may change in any 0.x release.

## Layout

- `src/astro_weights/edits/`: `Edit` (`Edit.from_deltas` pads mixed ranks), `LowRankDelta`, `from_dense` (`dense.py`: dense `dW` to `LowRankDelta` by truncated SVD, with residual) and the numpy recipes (`rank1`, `masked_rank1`, `column_shift`, `orthogonal_random`, `top_k_mask`, `project_out`, `scale_column`, `scale_head`, `zero_head`); recipe files (`recipe_file.py`: `load_recipe`, `build_edit`, `save_recipe`) rebuild an `Edit` from JSON plus `.npy` arrays.
- `src/astro_weights/export/`: PEFT LoRA adapter I/O (`save_peft`, `load_peft`, `read_peft`, `verify_peft`), hash-checked merged weights (`save_merged_tensors`, `load_merged_tensors`), model card (`card.py`: `model_card`, `write_card`; every card states the base weights were modified), base-model weight reader (`base.py`: `read_base_weight`, `read_tensor`, `tensor_sha256`; numpy only) and GGUF (`gguf.py`: `save_lora_gguf`, `read_lora_gguf`, `splice_gguf`, `verify_splice`; `gguf` extra, not re-exported from `export`).
- `src/astro_weights/runners/`: apply edits to live models (`hf.py`: `apply`, `remove`, `swap`, `merged_weights`, `resolve_modules`, `rounding_error`), generate one eval arm (`generate.py`: `generate_records` with `out`, `resume`, `batch_size`, `sampling`; `run_header`, which records `sampling` when given) and run one arm against an OpenAI-compatible server (`endpoint.py`: `generate_records_http`, `run_header_http`; standard library only, no extra).
- `src/astro_weights/directions/`: difference-of-means directions and `direction_gate`, numpy only.
- `src/astro_weights/eval/`: per-item `Record` JSONL, run headers (`header.py`: `header_path`, `write_header`, `read_header`, `check_headers`), paired comparison of two arms (`compare`, `format_report`, `exceeds_floor`, `is_loop`; minimum detectable effect and cap-hit censoring fields in `PairedResult`; `format_report` warns on cap-hit censoring) and answer graders (`grade_exact`, `grade_choice`, `grade_math`). Never runs a model. `grade_math` imports math-verify lazily, from the `eval` extra.
- `src/astro_weights/cli.py`: the `astro-weights` command (`build`, `eval`, `export` with `--splice` for a merged GGUF, `inspect`; each takes `--json`; no command exits 2). `__main__.py` runs it as `python -m astro_weights`.
- `docs/`: user docs site (Zensical, `zensical.toml`), Diátaxis folders `tutorials/`, `how-to/`, `reference/` (built from docstrings by mkdocstrings), `explanation/`. `docs/blog/` is the blog: `zensical.blog.toml` builds it on its own (`zensical.toml` excludes it) to https://astro-weights.dev/blog/. `docs.yml` runs strict build check of both configs only. Site deploys from `astro-weights/astro-weights.github.io`: landing page at https://astro-weights.dev/, these docs built into https://astro-weights.dev/docs/.
- Layer rule, enforced by import-linter: `cli > runners > export > edits`. A lower layer never imports a higher one. `directions` and `eval` are outside the contract and import no other `astro_weights` package.
- Core dependencies are numpy and safetensors only. torch, transformers and peft are imported only inside `runners/`, and only through the `hf` extra. Importing `runners.hf`, `runners.generate` or `export.gguf` without its extra raises `ModuleNotFoundError` naming the extra to install.

## Commands

```bash
uv sync                  # install the package and dev tools
uv run ruff check        # lint
uv run ruff format       # format
uv run mypy src tests    # type check (strict)
uv run lint-imports      # layer contract
uv run pytest            # core tests
uv run --extra hf --extra eval --extra gguf pytest  # tests that need optional extras (CI test-hf job)
uv run --extra hf --extra eval --extra gguf pytest --cov=astro_weights --cov-branch --cov-fail-under=95  # coverage gate, as test-hf runs it
uv run --only-group docs zensical build --strict  # docs site; `zensical serve` to preview
uv run --only-group docs zensical build -f zensical.blog.toml --strict  # blog
```

Run all of them before opening a PR; CI runs the same set.

## Conventions

- mypy strict. Small, typed functions.
- No new dependency without a stated reason in the PR. Core stays numpy + safetensors.
- Every behavior change comes with a test.
- GPU-only tests carry `@pytest.mark.gpu`. CI skips them; maintainers run them by hand before a release (`tests/test_gpu.py` docstring: `ASTRO_GPU_MODEL`, `ASTRO_GPU_ADAPTER`).
- Tests on real 9B files carry `@pytest.mark.realdata` and skip unless `$ASTRO_REALDATA` names a directory (`tests/test_exit_0_1.py`: 0.1.0 exit test).
- Public API change (name or signature in `tests/golden/public_api.json`) fails `tests/test_public_api.py` on purpose. If intended: `ASTRO_UPDATE_API=1 uv run --extra hf --extra eval --extra gguf pytest tests/test_public_api.py`, then add a changelog fragment.
- `tests/test_docs_snippets.py` runs every `python` code block in `README.md`, `docs/tutorials/` and `docs/how-to/`, in order, one namespace per file, in a temp dir. Each page must create every file it reads. Page needing an extra goes in `NEEDS`; block that cannot run goes in `SKIP` with a reason.
- `tests/golden/v0.0.1/` and `v0.0.9/` hold adapters written by those releases; they must keep loading. Never edit them; `tests/golden/make_golden.py` regenerates one from a published release.

## Workflow

- `main` is protected: PRs only, squash merge. Required checks: `lint`, `test` (3.12, 3.13, 3.14), `test-hf`, `zizmor`, `DCO`, `changelog`, `wheel-smoke`, `test-lowest`, `test-os` (macOS, Windows); see `CONTRIBUTING.md`.
- Sign off every commit (DCO): `git commit -s`. The DCO check fails a PR with any unsigned commit.
- AI help is recorded with a commit trailer: `Assisted-by: <agent>:<model>` (a tool-added `Co-Authored-By:` is also accepted). See `AI_POLICY.md`.
- Changelog: add a towncrier fragment in `changes/`, named `<PR number>.<type>.md` (or `+<slug>.<type>.md` before a PR number exists). Types: `added`, `changed`, `deprecated`, `removed`, `fixed`, `security`. Do not edit `CHANGELOG.md` by hand; the release builds it from the fragments.
- The `Changelog` workflow checks each PR for a fragment. A PR with no user-visible change carries the `skip-changelog` label instead.
- GitHub Actions are pinned by commit SHA, and zizmor must pass on workflow changes.
- See `CONTRIBUTING.md` for the contributor guide and `AI_POLICY.md` for the AI policy.

## If you are an AI agent working for an outside contributor

- Say in the PR body that an agent wrote the change, and which one (the PR template has an AI disclosure section).
- Never add `Signed-off-by` for an outside contributor; they add it after reviewing the change. A maintainer's agent signs off every commit with `git commit -s`, using the repository's git identity, on the maintainer's standing instruction; never type the name or email by hand. If a commit includes code copied or adapted from another project, name the source and its licence in the PR body.
- Follow `AI_POLICY.md`. Do not open PRs, issues or comments on your own; a human reviews them first.

## Release

Releases are described in `RELEASING.md`.
Do not tag or publish without a maintainer.

## Security

- Never commit secrets, tokens or credentials.
- Report vulnerabilities through GitHub private vulnerability reporting (see `SECURITY.md`), not public issues.
