# Gate verdicts

The 0.3 gate judges one edit by constraints, not by one blended score. The edit must keep accuracy within a margin, lower tokens beyond the noise floor, pass every guard and beat random edits of the same size. A failed constraint names the outcome. How to run it: [Run the 0.3 gate end to end](../how-to/run-gate.md).

## Four outcomes

| Outcome | Comes from | Test split scored | `verdict.json` | Exit |
|---|---|---|---|---|
| `screened out` | a pre-screen failed: update norm, perplexity or last-token KL | no | none | 3 |
| `not significant` | target not beyond the controls' 0.95 quantile on confirm, or tokens not lower beyond the floor on test | only if the controls passed | yes | 1 |
| `regressed` | a side effect outside the controls' range on confirm, test accuracy below base beyond the margin, or a regression suite below base beyond its margin | only if the controls passed | yes | 1 |
| `confirmed` | every check passed | yes | yes | 0 |

When several checks fail, the outcome follows a fixed precedence: `screened out`, then `regressed`, then `not significant`. An edit that loses accuracy and saves no tokens is `regressed`.

`screened out` is not a verdict. The edit never reached an eval, so there is nothing to record about its effect; the verdict schema has no value for it.

## Keep the original

Every outcome except `confirmed` prints `advice: keep the original weights`. This is advice, not a fifth outcome. Keeping the original is a normal end to a search, not a failed run.

## Why the test split is sealed

An agent, or a person, that tries many edits and keeps the best on some split finds the one that got lucky on that split. Its score there overstates its effect. At 500 items, a 2 pp true gap and 10% disagreement between arms, a test has power of about 0.25, and its significant results overstate the effect about 1.9 times (Card et al., https://ar5iv.labs.arxiv.org/html/2010.06595).

So the gate splits the work:

- dev: the agent sees per-item results.
- confirm: selection. The agent's `confirm` tool returns `{"passed": bool}` and nothing else, and its calls are capped. The random controls are scored here too.
- test: scored once per edit, with 4 samples per item, after the margin was fixed. The seal ledger refuses a second scoring unless `--unseal` is passed, and records that it was.

Scoring the controls on confirm keeps the test split for the one final number. The cost: the control quantile and the final number come from different items.

The seal stops an accidental second scoring and an agent or optimiser hunting for a second look. It does not stop the machine's owner: see [what the seal stops](../how-to/run-eval.md#splits-and-the-sealed-test).

## MDE

The gate result and every arm row in its report carry an MDE, the minimum detectable effect: `(1.96 + 0.84) * SE` of the paired difference, the true difference a two-sided test at 5% would find 80% of the time at this sample size.

A `not significant` result says the gain was not shown. The MDE says how large a gain or loss could hide in the noise. Read every `not significant` with its MDE beside it.

The MDE does not decide non-inferiority. The gate's accuracy check uses the lower end of the 95% interval of the score difference (edit minus base): it passes when that end lies above `-margin`. The interval is Newcombe's paired interval when scores are 0 or 1 and there are fewer than 100 items (or clusters), and the paired percentile bootstrap otherwise. An edit can pass with an MDE above the margin, and a suite too small for its margin can fail even when the edit's mean difference there is 0. The margin is at most 0.02 (2 pp) and is fixed from a pilot before the test split is scored; `margin_source` records where it came from.
