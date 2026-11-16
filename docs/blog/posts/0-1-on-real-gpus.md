---
date: 2026-11-16
slug: 0-1-on-real-gpus
categories:
  - research
---
# 0.1 on real GPUs: what held, what didn't

astro-weights makes small edits to open model's weights and exports them as standard LoRA adapters. Until 2026-11-15, 9B numbers came from one 3090 and quantized GGUF file; 27B numbers came from research harness, not from library. On 2026-11-15 we took 0.1 code path to HPC cluster and reran MATH-500 and GSM8K at bf16 with paired statistics, full sets where noted, and added Python code tasks.

Short version. 27B adapter did what we said: about 30% fewer tokens on MATH-500 with no measured accuracy change (n=500). 9B adapter did not hold everywhere: about 10% fewer tokens on held-out MATH-500 (n=244) with no measured accuracy change there, but on full GSM8K (n=1319) it loses about 2 points and saves no tokens. Our earlier 300-item GSM8K check was too small to see that loss. Source file name sits beside every number; raw files are not public yet.

<!-- more -->

## How runs were made

- Two adapters under test: public "terse" adapters for Qwen3.5-9B (https://huggingface.co/glitch401/qwen3.5-9B-terse) and Qwen3.8-27B (https://huggingface.co/glitch401/qwen3.8-27B-terse), same files as on Hugging Face (sha256 match). Each one is meant to make model answer in fewer tokens. Below, "9B" and "27B" mean these two.
- Library commit `e4bca76` (0.0.9, one commit before 0.1.0). GPU test suite also rerun on tag `v0.1.0` (`74729f5`). Paired evals were not rerun on `v0.1.0`, so read "0.1 code path", not "0.1.0 release build".
- 9B on one H100 80GB, 27B on one H200. bf16, HF transformers, thinking off, batch 16, one sample per item.
- "Paired" means base model and edited model answer same items. Accuracy difference comes with 95% CI from `astro-weights eval`. Token CI is bootstrap 95%, given in tokens per item, not percent.
- "card sampling" means sampling settings from model card: temp 1.0, top_p 0.95, top_k 20, presence penalty 1.5, cap 4096 tokens. Presence penalty was approximated over last 64 tokens by wrapper outside library, because library's `generate_records` is greedy only (issue #53).
- "cap-hit" is share of answers cut off at token cap. It sits beside accuracy numbers below for reason given in last section.

Math rows:

```
env MODEL=<snapshot> ADAPTER=<adapter> ITEMS=<bench dir>/<set>.jsonl OUT=<dir> [SPLIT=<set>:confirm] [SAMPLE=card] MAX_NEW=4096 .venv/bin/python paired_eval2.py
.venv/bin/astro-weights eval <dir>/base.jsonl <dir>/edit.jsonl
```

Code rows: same with `ITEMS=code_<set>.jsonl MODE=code MAX_NEW=2048`, then `evalplus.sanitize`, `evalplus.evaluate --dataset <set>`, `code_bench.py report`. Rows marked "greedy 1024" used older `paired_eval.py`; exact shell line for those was not logged.

Source file paths below are relative to `aw-hpc/` on cluster, with raw `base.jsonl` and `edit.jsonl` beside each report. Raw files are not public yet.

## What held

### 27B: shorter answers, no measured accuracy change

| eval | n | decoding | accuracy base / edit | accuracy diff (95% CI) | tokens | source |
|---|---|---|---|---|---|---|
| MATH-500, full | 500 | greedy, cap 4096 | 0.8980 / 0.8920 | −0.60 pp [−2.64, +1.44] | 1222.5 / 853.0, −30.2% [−424.7, −315.4] | `x27/math500-full-greedy4k/report.txt` |
| MATH-500, held-out split | 244 | card sampling | 0.9098 / 0.9057 | −0.41 pp [−3.53, +2.71] | −26.4% [−402.0, −264.6] | `x27/math500-confirm-card/report.txt` |
| GSM8K, held-out split | 300 | card sampling | 0.9700 / 0.9700 | +0.00 pp [−1.31, +1.31] | −30.2% [−178.0, −115.5] | `x27/gsm8k-confirm-card/report.txt` |
| GSM8K, full | 1319 | greedy, cap 1024 | 0.9538 / 0.9621 | +0.83 pp [+0.03, +1.63] | −22.5% [−99.4, −87.7] | `h3b/gsm8k/report.txt` |

Limits on these rows. Held-out MATH-500 split has 244 items and cannot see loss under about 3 pp. Full GSM8K row ran greedy with cap 1024, where base model hit cap more often than edited one (cap-hit 0.027 / 0.005), so its +0.83 pp is partly cap effect (McNemar p=0.061). We read that row as "no loss", not as gain. Full GSM8K with card sampling was not run on 27B.

These numbers match our earlier record for same adapter: MATH-500 held-out +0.00 pp [−1.71, +1.71] with −29.9% tokens, GSM8K held-out −0.33 pp [−2.06, +1.40] with −27.6% (`wsp/results/g27-dconf-fable27c-diffs.txt`). That record was vLLM at bf16 with same LoRA, not GGUF, so this compares two bf16 LoRA paths.

### 27B on Python code tasks

Edit was never tuned or checked on code before this run. Token cut carried over and pass@1 stayed inside noise.

| eval | n | pass@1 base / edit | items lost / gained | McNemar p | tokens | source |
|---|---|---|---|---|---|---|
| HumanEval+ | 164 | 0.9146 / 0.9085 | 3 / 2 | 1.0 | −10.9% [−51.9, −12.2] | `x27/humaneval/code_report.json` |
| MBPP+ | 378 | 0.7725 / 0.7672 | 9 / 7 | 0.80 | −31.0% [−75.6, −38.3] | `x27/mbpp/code_report.json` |

Greedy, cap 2048, one sample. Report file has no CI on pass@1 difference, and 164 items cannot rule out loss of 2 to 3 pp.

### Adapter costs almost no GPU memory, and removing it returns GPU memory to start value

| model | GiB base / applied / removed | source |
|---|---|---|
| 9B | 16.6778 / 16.6781 / 16.6778 | `h2/gsm8k/stats.json` |
| 27B | 50.0978 / 50.1019 / 50.0978 | `h3b/gsm8k/stats.json` |

Measured with `torch.cuda.memory_allocated`, one run each.

## 9B: one result held, one did not

These two results belong together.

On held-out MATH-500 (n=244, card sampling) 9B adapter uses about 10% fewer tokens with no measured accuracy change: accuracy 0.8975 / 0.8934, −0.41 pp [−3.08, +2.26]; tokens 1301.6 / 1170.9, −10.0% [−210.8, −52.3]; cap-hit 0.066 / 0.049 (`x9/math500-confirm-card/report.txt`). Our earlier GGUF record said −10.5%.

On full GSM8K (n=1319, card sampling) same adapter loses about 2 points and saves no tokens: accuracy 0.9545 / 0.9340, −2.05 pp [−3.20, −0.89], McNemar p=0.00073, 44 items lost and 17 gained. Mean tokens 473.8 / 479.4, +1.2% [−25.3, +37.8], not significant. Cap-hit 0.007 / 0.012 (`x9/gsm8k-full-card/report.txt`, `breakdown.txt`). 30 of 44 lost items hit no cap, so loss is not only cut-off answers. Median answer is still shorter, 391 / 376 tokens.

One caveat on that loss: it was measured on bf16 base plus LoRA. Published GGUF has not been rerun at n=1319.

## What did not hold

### Our 300-item GSM8K check was too small

Earlier 9B GSM8K claim rested on 300 items with CI of ±2.62 pp (record: `wsp/results/p5-reg-diffs.txt`, wsp commit `bed9329`). Same 300 item ids on cluster with card sampling give −1.00 pp [−4.27, +2.27] and tokens −6.3% [−97.5, +37.5], not significant (`x9/gsm8k-confirm-card/report.txt`). Set that size cannot see 2 pp loss. Full set can, and did.

### 9B token cut does not show on full MATH-500 with greedy decoding, nor on code

| eval | n | accuracy or pass@1 base / edit | difference | tokens | source |
|---|---|---|---|---|---|
| MATH-500, full, greedy cap 4096 | 500 | 0.8860 / 0.8740 | −1.20 pp [−3.42, +1.02], 19 lost / 13 gained, p=0.38 | 1222.1 / 1182.0, −3.3% [−95.4, +14.8] | `x9/math500-full-greedy4k/report.txt` |
| HumanEval+ | 164 | 0.8476 / 0.8293 | 10 lost / 7 gained, p=0.63 | −2.4% [−34.5, +26.4] | `x9/humaneval/code_report.json` |
| MBPP+ | 378 | 0.6323 / 0.6005 | 28 lost / 16 gained, p=0.096 | −5.6% [−44.6, +12.0] | `x9/mbpp/code_report.json` |

Every token CI here includes zero. On code: no saving, and pass@1 is lower but inside noise.

### 9B loops more under plain greedy decoding

Stress setting, not card's setting: greedy, no presence penalty, cap 1024, GSM8K n=1319. Cap-hit 0.020 / 0.038, accuracy −2.58 pp [−3.80, −1.36]. 34 of 51 lost items are answers where only edited model hit cap (`h2/gsm8k/report.txt`, `h2/gsm8k/breakdown.txt`).

### GPU test suite: 7 of 7 on 27B, 6 of 7 on 9B

On 27B on H200, suite passes 7 of 7 in 369.63 s (`h3a/gpu.log`).

On 9B on H100, one of 7 tests fails, on `e4bca76` and on `v0.1.0`. Test `apply_vs_merged` compares adapter applied at runtime against adapter merged into weights and needs agreement above 0.97; it got 0.9333. Cause is one exact bf16 logit tie (13.0 / 13.0) that flips argmax at 1 of 15 positions. Largest gap between applied and merged logits is 0.1875, against 5.1875 between applied and base. Weights are correct; test is too strict for bf16. Same test passes on 3090 and on 27B. Issue #51 (`h1/gpu.log`, `h1-v010/gpu.log`, `h1/diag.log`).

### Export check on 27B: 47 of 49 layers inside test bound

Layers 39 and 49 exceed bound `2 × rel_residual + 1e-4` (0.00434 against 0.00198; 0.00547 against 0.00236). Cause is open: loose fit on two layers, or bound too tight. Adapter still reproduces earlier record above. Test also hardcodes 5 modules, issue #52 (`h4/result.json`).

### Our first runs showed fake +14.8 pp

First 27B MATH-500 run used cap of 1024 tokens: accuracy 0.5980 / 0.7460, cap-hit 0.398 / 0.222 (`h3b/math500/report.txt`). Base model ran out of tokens far more often than edited one, and cut-off answers count as wrong. Same set at cap 4096: −0.60 pp (`x27/math500-full-greedy4k/report.txt`). Since then, cap-hit rate goes beside every accuracy number. Issue #54.

## Confounds on every number here

1. No random-direction control arm on cluster. Control exists only in older 9B GGUF record.
2. One edit per model and one generation run per row. Run-to-run variance under sampling was not measured.
3. Thinking off only.
4. Jobs ran in interactive OnDemand sessions, not `sbatch`.

## Not run

- Other model families. LiveCodeBench, SWE-bench Verified.
- IFEval, MMLU-Pro, MATH fresh sets.
- Thinking on, on cluster. GGUF path at full-set size. 27B full GSM8K with card sampling.
- Any paired eval on `v0.1.0` or `0.1.1` build.

## About the base models

Qwen3.5-9B and Qwen3.8-27B are models from the Qwen team at Alibaba Cloud, released under Apache-2.0. The terse adapters are unofficial community derivatives, not affiliated with or endorsed by the Qwen team or Alibaba Cloud.

## Acknowledgement

The authors acknowledge the High Performance Computing at The University of Texas at Dallas (HPC@UTD) for providing the computing resources and support.

