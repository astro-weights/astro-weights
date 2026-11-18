# Releasing

Steps for maintainers. Contributors do not need this file.

Replace `X.Y.Z` with the new version. See [Versioning and support](CONTRIBUTING.md#versioning-and-support) for how to pick it.

## Release steps

1. Branch from `main`: `git switch main && git pull && git switch -c bump/vX.Y.Z`. Do not name it `release/...`: the `release/*` ruleset is for maintenance branches.
2. Set the version: `uv version X.Y.Z`.
3. Build the changelog: `uv run towncrier build --version X.Y.Z --yes`. This writes the fragments from `changes/` into `CHANGELOG.md` and deletes them. Then add the link for the new version at the bottom of `CHANGELOG.md`.
4. Open a draft PR with the labels `release` and `skip-changelog`. It stays a draft until the release's exit criteria have passed; only then mark it ready, wait for the required checks, and squash merge. Anything that merges to `main` after the PR's last commit needs a refresh: merge `main` in and fold new fragments into the version's section.
5. On the updated `main`, check that the CI run for the merge commit is green, including `test-os` (macOS, Windows). Those jobs run only after merge and do not block it. Then create an annotated tag and push it: `git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z`. A tag ruleset protects `v*`, so only repo admins can create these tags.
6. `release.yml` checks that the tag matches the version, builds, publishes to TestPyPI, then waits for a maintainer to approve the `pypi` environment. Check the TestPyPI page, then approve.
7. Verify the published package in a fresh venv: `python -m venv /tmp/v && /tmp/v/bin/pip install astro-weights==X.Y.Z`.
8. After `pypi` succeeds, `release.yml` job `github-release` publishes the GitHub release from the tag: notes from the changelog section, wheel, sdist and the provenance bundle `astro_weights-X.Y.Z.intoto.jsonl` attached. Check it on the Releases page, then check the provenance: `gh release download vX.Y.Z -p '*.whl' -p '*.tar.gz' -D /tmp/vX.Y.Z && gh attestation verify /tmp/vX.Y.Z/* -R astro-weights/astro-weights`.

Releases are immutable. Assets cannot change after the release is published, and a tag cannot be moved. If something is wrong, fix forward with a patch release.

## Maintenance branches

There is no long-lived dev branch. While the project is at 0.x, only the newest minor gets patches (see [SECURITY.md](SECURITY.md#supported-versions)); patch releases for it come from `main` until `main` moves to the next minor. A maintenance branch is cut only after that, when the older minor needs a fix. When a minor stops getting patches, its maintenance branch is locked (read-only).

1. Land the fix on `main` first.
2. Create the branch from the release tag of the minor: `git switch -c release/X.Y.x vX.Y.0`, then push it.
3. Cherry-pick the fix onto `release/X.Y.x` (`git cherry-pick -x <sha>`) and open a PR against that branch.
4. Release from the branch with the same steps as above, with patch version `X.Y.(Z+1)`.

## Yanking

If a release is broken, yank it on PyPI (project page, Manage, Options, Yank). Never delete a release: deleted versions cannot be reused and break pinned installs. Ship a patch release with the fix, and say in its changelog entry what was wrong.
