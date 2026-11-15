# Contributing to astro-weights

Thanks for helping. The project is early, so open an issue before starting large changes.

## Setup

Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/astro-weights/astro-weights
cd astro-weights
uv sync
uv run pre-commit install   # optional: uvx pre-commit install
```

## Checks

```bash
uv run ruff check
uv run ruff format
uv run mypy src
uv run pytest
```

Docs changes: `uv run --only-group docs zensical build --strict` must pass; preview with `uv run --only-group docs zensical serve`.

CI runs the same checks on Python 3.12 to 3.14. Tests that need a GPU are marked `@pytest.mark.gpu`; CI skips them, and maintainers run them by hand before each release. Tests that need the real 9B files are marked `@pytest.mark.realdata` and skip unless `$ASTRO_REALDATA` is set.

Required CI jobs beyond the checks above: `wheel-smoke` builds the wheel, installs it alone in a clean venv and runs the CLI; `test-lowest` runs the tests with `--resolution lowest-direct`, so a dependency floor in `pyproject.toml` (for example `torch>=2.4` in the `hf` extra) must really work; `test-os` runs the core tests on macOS and Windows. All three are required checks on `main` (`test-os` counts as two: macOS and Windows). A weekly `Canary` workflow tests against peft and transformers `main`; it is not a required check, and a red run means upstream moved, not that a PR broke.

`test-hf` also enforces coverage: `--cov=astro_weights --cov-branch --cov-fail-under=95`, so branch coverage below 95% fails it. Run the same locally with `uv run --extra hf --extra eval --extra gguf pytest --cov=astro_weights --cov-branch --cov-fail-under=95`.

### Tests that guard the public surface

- `tests/test_public_api.py` compares every public name and signature with `tests/golden/public_api.json`. An intended API change needs the snapshot updated by hand: `ASTRO_UPDATE_API=1 uv run --extra hf --extra eval --extra gguf pytest tests/test_public_api.py`, plus a changelog fragment.
- `tests/test_docs_snippets.py` runs every `python` code block in `README.md`, `docs/tutorials/` and `docs/how-to/`. Blocks of one page run in order in one namespace, in a temp directory, so a page must create every file it reads. A page that needs an extra is listed in `NEEDS`; a block that cannot run is listed in `SKIP` with a reason.
- `tests/test_golden.py` loads adapters written by released versions (`tests/golden/v0.0.1/`, `tests/golden/v0.0.9/`). Do not edit them; a change that breaks them breaks users' saved adapters.

## Pull requests

- Keep each PR to one change.
- Add a changelog fragment under `changes/` for anything a user would notice (see [Changelog fragments](#changelog-fragments)). Do not edit `CHANGELOG.md` directly.
- Sign off every commit (see below).
- Read the [AI tools](#ai-tools) section if an AI tool helped you.

## Branching

There is no long-lived development branch. `main` is the development branch.

- Branch from `main`, keep the branch short-lived, and open a PR back to `main`.
- PRs are squash-merged and the branch is deleted on merge.
- Releases are tags `vX.Y.Z` on `main`.
- A maintenance branch `X.Y.x` is cut from a release tag only when an older minor needs a fix. The fix lands on `main` first and is cherry-picked back.

## Changelog fragments

The changelog is built from [towncrier](https://towncrier.readthedocs.io/) news fragments. Each user-visible change adds one small file in `changes/`:

- Name: `<PR number>.<type>.md`. If you do not have a PR number yet, use `+<slug>.<type>.md`, for example `+lora-dtype.fixed.md`.
- Types: `added`, `changed`, `deprecated`, `removed`, `fixed`, `security`.
- Content: one or two sentences, written for users.

Example, `changes/42.fixed.md`:

```markdown
`save_peft` no longer fails when a delta has rank 0.
```

At release time the maintainer runs towncrier, which writes the fragments into `CHANGELOG.md` and deletes them. A PR with no user-visible change (docs typo, CI, refactor) gets the `skip-changelog` label instead of a fragment.

## Versioning and support

astro-weights follows [SemVer](https://semver.org/) with 0.x rules:

- A minor bump (0.1 to 0.2) is a milestone or a breaking change.
- A patch bump (0.1.0 to 0.1.1) may carry any backward-compatible change, including new features.
- From 1.0, a patch bump is fixes only.
- Public API is deprecated with a `DeprecationWarning` for at least one minor release before it is removed.
- The version is set by hand with `uv version`. There are no nightly releases.

Python 3.12, 3.13 and 3.14 are supported. Python 3.12 stays supported until 0.1. After that the project follows [SPEC 0](https://scientific-python.org/specs/spec-0000/): a Python version is dropped 3 years after its release.

Maintainers: see [RELEASING.md](RELEASING.md) for the release steps.

## Issue triage

- New issues get the `needs triage` label. A maintainer removes it after reading the issue.
- `needs info` or `needs repro` means we are waiting on the reporter. Issues with no reply for a few weeks may be closed.
- `good first issue` is reserved for people learning the codebase without AI help. Please do these by hand.
- `good second issue` is for contributors who have done one already. AI tools are allowed there under the [AI policy](AI_POLICY.md).

## Developer Certificate of Origin

This project uses the [Developer Certificate of Origin](https://developercertificate.org/) (DCO) instead of a CLA. By signing off a commit you certify that you wrote it, or otherwise have the right to submit it under the project's Apache-2.0 license. Sign off with:

```bash
git commit -s
```

which adds a line like `Signed-off-by: Your Name <you@example.com>` to the commit message.

The DCO check is enforced by a GitHub App and is a required status check. Every commit in a PR needs a sign-off, or the PR cannot merge. To fix a missing sign-off, run `git rebase --signoff main` and force-push your branch.

## AI tools

AI tools are allowed, with rules. In short:

- Disclose the tool and how much it did in the PR template. Add an `Assisted-by: <agent>:<model>` trailer to commits it helped with.
- You must understand and be able to explain every change in your own words. Your agent must never add `Signed-off-by` for you.
- No autonomous agents. PRs, issues or comments opened by an agent without human review are closed.

Read the full [AI_POLICY.md](AI_POLICY.md).

## Reporting security issues

Do not open a public issue. See [SECURITY.md](SECURITY.md).

## Code of conduct

Everyone taking part in this project is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
