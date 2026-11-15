# Changelog fragments

Each user-visible change gets one file here: `<PR number>.<type>.md`. Use `+<slug>.<type>.md` if there is no PR number yet.

Types: `added`, `changed`, `deprecated`, `removed`, `fixed`, `security`.

The file holds one sentence a user would understand, for example: `verify_peft` reports a corrupt weights file as a problem instead of raising.

PRs with no user-visible change get the `skip-changelog` label instead. The release step runs `towncrier build` and deletes the fragments.
