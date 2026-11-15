## Summary

<!-- What changes and why. A few sentences. -->

## Linked issue

<!-- Non-trivial PRs need an issue first. Use "Closes #123". -->
Closes #

## Checks run

- [ ] `uv run ruff check`
- [ ] `uv run ruff format`
- [ ] `uv run mypy src tests`
- [ ] `uv run lint-imports`
- [ ] `uv run pytest`
- [ ] `uv run --extra hf pytest` (if `runners/` was touched)

## Changelog

<!-- See CONTRIBUTING.md for fragment naming: changes/<PR number>.<type>.md -->
- [ ] Added a fragment in `changes/`, or this PR has the `skip-changelog` label

## AI disclosure

<!-- See AI_POLICY.md. Say which tools you used and roughly what they produced. Write "none" if no AI was used. -->
Tools used:

What they produced:

- [ ] I have reviewed and understand every line of this change.

## DCO

- [ ] Every commit is signed off by me (`git commit -s`). My agent did not add `Signed-off-by`.
