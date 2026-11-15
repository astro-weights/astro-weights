# Releasing

Steps for maintainers. Contributors do not need this file.

Replace `X.Y.Z` with the new version. See [Versioning and support](CONTRIBUTING.md#versioning-and-support) for how to pick it.

## Release steps

1. Branch from `main`: `git switch main && git pull && git switch -c release/vX.Y.Z`.
2. Set the version: `uv version X.Y.Z`.
3. Build the changelog: `uv run towncrier build --version X.Y.Z --yes`. This writes the fragments from `changes/` into `CHANGELOG.md` and deletes them. Then add the link for the new version at the bottom of `CHANGELOG.md`.
4. Open a PR with the labels `release` and `skip-changelog`. Wait for the required checks, then squash merge.
5. On the updated `main`, create an annotated tag and push it: `git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z`. A tag ruleset protects `v*`, so only repo admins can create these tags.
6. `release.yml` checks that the tag matches the version, builds, publishes to TestPyPI, then waits for a maintainer to approve the `pypi` environment. Check the TestPyPI page, then approve.
7. Verify the published package in a fresh venv: `python -m venv /tmp/v && /tmp/v/bin/pip install astro-weights==X.Y.Z`.
8. After `pypi` succeeds, `release.yml` job `github-release` publishes the GitHub release from the tag: notes from the changelog section, wheel and sdist attached. Check it on the Releases page.

Releases are immutable. Assets cannot change after the release is published, and a tag cannot be moved. If something is wrong, fix forward with a patch release.

## Maintenance branches

There is no long-lived dev branch. A maintenance branch is cut only when an older minor needs a fix.

1. Land the fix on `main` first.
2. Create the branch from the release tag of the minor: `git switch -c X.Y.x vX.Y.0`, then push it.
3. Cherry-pick the fix onto `X.Y.x` (`git cherry-pick -x <sha>`) and open a PR against that branch.
4. Release from the branch with the same steps as above, with patch version `X.Y.(Z+1)`.

## Yanking

If a release is broken, yank it on PyPI (project page, Manage, Options, Yank). Never delete a release: deleted versions cannot be reused and break pinned installs. Ship a patch release with the fix, and say in its changelog entry what was wrong.
