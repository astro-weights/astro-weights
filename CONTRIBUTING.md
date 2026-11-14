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

CI runs the same checks on Python 3.12 to 3.14. Tests that need a GPU are marked `@pytest.mark.gpu`; CI skips them, and maintainers run them by hand before each release.

## Pull requests

- Keep each PR to one change.
- Add an entry under `## [Unreleased]` in `CHANGELOG.md` for anything a user would notice.
- Sign off every commit (see below).

## Developer Certificate of Origin

This project uses the [Developer Certificate of Origin](https://developercertificate.org/) (DCO) instead of a CLA. By signing off a commit you certify that you wrote it, or otherwise have the right to submit it under the project's Apache-2.0 license. Sign off with:

```bash
git commit -s
```

which adds a line like `Signed-off-by: Your Name <you@example.com>` to the commit message.

## Reporting security issues

Do not open a public issue. See [SECURITY.md](SECURITY.md).
