# AI policy

## How this project uses AI
Most of astro-weights is written by AI coding agents (currently Claude Code) working
under the direction of a maintainer. A maintainer reviews every change before it is
merged and signs off on it under the Developer Certificate of Origin. Agent-written
commits carry an `Assisted-by:` or `Co-Authored-By:` trailer naming the tool.

## Contributing with AI tools
You may use AI tools. You are responsible for what you submit.

- **Disclose.** Fill in the AI section of the pull request template: which tools you
  used and roughly how much of the change they produced. Add a trailer such as
  `Assisted-by: Claude:claude-opus-5-5` to your commits.
- **Understand it.** You must be able to explain the change in your own words and
  answer review questions without asking the tool.
- **Sign off yourself.** Only a human can certify the DCO. Your agent must not add
  `Signed-off-by` for you; add it yourself after you have reviewed the change.
- **Open an issue first** for anything larger than a small fix.
- **Write to people yourself.** Issues, PR descriptions and comments must be written
  or fully edited by you. Translation and grammar tools are fine.
- **Good first issues are for learning.** Please do them without AI assistance.
- **No autonomous agents.** Pull requests, issues or comments opened by an agent
  without a human reviewing them first will be closed. If you run an agent, you are
  accountable for everything it does here.
- **Licensing.** Do not submit code you know or suspect was reproduced from another
  project. Make sure your tool's terms let you contribute its output under Apache-2.0.

## Security reports
Report vulnerabilities privately through GitHub's "Report a vulnerability" button
(see SECURITY.md). If AI helped you find or write up the issue, say so. Confirm the
problem reproduces before reporting. There is no bug bounty. Fabricated reports lead
to a block.

## Enforcement
Maintainers may ask you to explain a change, close contributions that do not follow
this policy without detailed review, and block accounts that repeatedly ignore it.
Using AI is never on its own a reason to reject a contribution.
