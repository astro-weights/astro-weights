---
date: 2026-11-24
slug: 27b-accuracy-correction
categories:
  - research
---

# We got the 27B accuracy claim wrong

Since launch, our Qwen3.8-27B card has said the terse adapter makes answers about 30% shorter at the same accuracy. That's wrong. When both models get room to finish, the adapter is less accurate than the base model on every set we rechecked, by about 2 to 4 points.

<!-- more -->

## What went wrong

Our evals stopped every answer at 4,096 tokens, and an answer cut off before it ends is graded wrong. The base model writes longer answers, so it hit that limit more often: 5.0% to 8.3% of its answers on our four sampled sets, against 0.7% to 3.9% for the adapter. The limit took more points from the base model than from the adapter, and the two came out level.

## The rerun

We reran every answer that hit the limit, for both models, with a 16,384-token limit. No base answer hit the new limit; 3 adapter answers on the greedy MATH-500 run still did. On that greedy run we reran both models on all 33 problems where either one hit the limit. Everything else is the published records, paired problem by problem. Thinking off.

| Set | Answers (problems) | Base | Adapter | Change, points [95% CI] | Tokens |
|---|---|---|---|---|---|
| MATH-500 confirm split | 976 (244) | 97.6% | 95.4% | −2.25 [−3.69, −0.92] | −33.8% |
| MATH fresh, set 1 | 1,000 (500) | 96.6% | 94.7% | −1.90 [−3.20, −0.60] | −34.8% |
| MATH fresh, set 2 | 1,000 (1,000) | 96.9% | 94.6% | −2.30 [−3.60, −1.10] | −34.6% |
| MMLU-Pro | 300 (300) | 83.7% | 79.7% | −4.00 [−7.00, −1.00] | −39.2% |
| MATH-500, greedy | 500 (500) | 93.0% | 90.0% | −3.00 [−5.00, −1.20] | −25.4% |

CIs are a bootstrap over problems. Every one is below zero.

So the adapter still cuts answers by a quarter to two fifths, and it costs accuracy. If you use it, you're trading some correct answers for shorter ones.

## What we haven't rerun

GSM8K, IFEval, thinking mode (which already ran with a 16k limit) and the code benchmarks. Several of our other results used the same 4,096-token limit: the 0.2 agent test, the stacking post and the 9B adapter's evals. Until we rerun them, read their accuracy numbers with this in mind.

## What we changed

The model card, the site and our Product Hunt page now carry these numbers. The adapter file itself hasn't changed. From now on our evals report how many answers hit the limit next to every accuracy number, and we rerun those answers before we call accuracy unchanged.

Thanks for reading the fine print with us. If you spot something else off, open an issue.
