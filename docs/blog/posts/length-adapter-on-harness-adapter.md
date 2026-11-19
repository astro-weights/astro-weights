---
date: 2026-11-19
slug: length-adapter-on-harness-adapter
categories:
  - research
---

# Our length adapter on top of a Hugging Face harness adapter

Hugging Face released `harness-hub/Qwen3.8-27B-pi-adapted`, a rank-32 LoRA on Qwen3.8-27B trained for agent work. Our terse adapter (rank 1, `down_proj` only) targets answer length on same base. Question: does length cut survive when both adapters are active?

<!-- more -->

## Setup
Both adapters loaded in PEFT, both active, base unmerged. Compared pi-adapted + terse against pi-adapted alone on same problems: greedy, thinking off, one sample, one seed. Control: random rank-1 edit (same layers, same norms) on top of pi-adapted. Tokens: paired bootstrap 95% CI per problem. Pass@1: exact McNemar.

## Result
| Set | n | Output tokens [95% CI, tokens per problem] | Accuracy / pass@1 change (p) | Random edit on top |
|---|---|---|---|---|
| MATH-500 | 500 | −29.1% [−417.2, −304.0] | +0.20 pp (1.0) | +2.6%, not significant |
| HumanEval+ | 164 | −13.8% [−65.6, −17.9] | −0.61 pp (1.0) | −1.5%, not significant |
| MBPP+ | 378 | −27.1% [−64.7, −36.4] | −1.59 pp (0.26) | −2.7%, not significant |
| BigCodeBench | 1,129 | −15.9% [−70.8, −56.1] | +0.27 pp (0.84) | +2.2% (slightly longer) |
| LiveCodeBench | 341 | −19.9% [−710.5, −456.5] | +2.64 pp (0.34) | −2.3%, not significant |

Cut on top of pi-adapted about same as on bare base (within 4.4% on every set). pi-adapted alone changed length by +5.7% at most.

## What this doesn't show
- Agent behaviour. pi-adapted is trained for pi harness; these are single-turn tasks. Terminal-Bench and SWE-bench Pro untested: they need sandboxed runs we haven't done.
- LiveCodeBench pass@1: over half of answers hit 4,096-token cap, and shorter answers hit it less, so pass@1 there isn't clean accuracy reading.
- One seed, greedy. New arms ran on different GPUs from reused base arms; cross-GPU noise floor not measured.

Our adapter: https://huggingface.co/glitch401/qwen3.8-27B-terse. Their adapter: https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted.
