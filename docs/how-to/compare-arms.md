# Compare two eval arms

`astro_weights.eval` compares a base arm and an edit arm on the same items, using per-item records. It never runs a model: you generate the outputs, grade them and write one `Record` per item, seed and arm. It needs numpy only. To generate records from a model, see [Run an eval end to end](run-eval.md).

## Compare

```python
from astro_weights.eval import Record, compare, format_report, prompt_hash

prompts = {f"q{i}": f"question {i}" for i in range(8)}


def record(item_id: str, arm: str, score: float, tokens: int) -> Record:
    return Record(
        item_id=item_id,
        seed=0,
        arm=arm,
        prompt_hash=prompt_hash(prompts[item_id]),
        output="...",
        score=score,
        n_output_tokens=tokens,
        hit_cap=False,
    )


base = [record(k, "base", 1.0 if i < 5 else 0.0, 300) for i, k in enumerate(prompts)]
edit = [record(k, "edit", 1.0 if i < 6 else 0.0, 250) for i, k in enumerate(prompts)]

result = compare(base, edit)
print(format_report(result))
```

Output:

```text
n=8 items  base=0.6250  edit=0.7500
score diff=+0.1250  SE=0.1250  95% CI [-0.1200, +0.3700]; not significant; smallest detectable difference at this n is 0.3500
McNemar exact p=1  (base-only right=0, edit-only right=1)
tokens: base=300.0  edit=250.0  diff=-50.0 (-16.7%)  bootstrap 95% CI [-50.0, -50.0]
cap-hit rate: base=0.000  edit=0.000
WARNING: only 8 items (<100); intervals are unreliable
```

## What `compare` does

- `pair` joins the arms on `(item_id, seed)`. It raises `ValueError` when the keys differ, a key repeats within an arm, or `prompt_hash` differs between the arms.
- Scores and token counts are first averaged over seeds for each item; every statistic then runs over items.
- The score difference gets a paired SE and a 95% CI of `diff ± 1.96 * SE`.
- The exact McNemar p-value is computed only when every per-item score is 0 or 1; otherwise `mcnemar_p` is `None`.
- The token difference gets a bootstrap 95% CI, with `n_boot=10000` resamples and `seed=0` by default.
- `small_n` is `True` under 100 items, and `format_report` then adds a warning line.
- Minimum detectable effect: `score_mde = (1.96 + 0.84) * se` and `tokens_mde = (1.96 + 0.84) * tokens_se` (a 5% two-sided test at 80% power). When the 95% CI contains 0, the report line ends with `not significant; smallest detectable difference at this n is ...`.

## Noise floor

Run the base arm twice under the same settings; `compare` of those two runs is the noise floor. `exceeds_floor(result, floor)` is `True` for a metric when its CI excludes 0 and `|diff| > |floor diff| + 1.96 * floor SE`. A NaN anywhere gives `False`. `format_report(result, floor)` adds one line per metric. This continues the example above:

```python
from astro_weights.eval import exceeds_floor

# base arm run again; token counts differ by 1 on half the items
rerun = [
    record(k, "base", 1.0 if i < 5 else 0.0, 300 + i % 2) for i, k in enumerate(prompts)
]
floor = compare(base, rerun)
print(exceeds_floor(result, floor))
print(format_report(result, floor).splitlines()[-3:-1])
```

Output:

```text
{'score': False, 'tokens': True}
['noise floor (score): |diff|=0.1250 does not exceed |floor diff| + 1.96 SE = 0.0000', 'noise floor (tokens): |diff|=50.0 exceeds |floor diff| + 1.96 SE = 0.9']
```

On the command line: `astro-weights eval BASE.jsonl EDIT.jsonl --floor A.jsonl B.jsonl`; see [Run an eval end to end](run-eval.md#compare-arms).

## Clustered items

Items that share a source (the same passage, the same task family) are not independent. Set `cluster_id` on every record of both arms, with the same id for an item across arms and seeds. `compare` then uses a cluster-robust SE for the score SE, CI and MDE and for the token SE and MDE, and the token bootstrap resamples whole clusters. The clustered SE follows section 2.2 of Evan Miller, [Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations](https://arxiv.org/abs/2411.00640) (2024), with the usual `G / (G - 1)` small-sample factor added; that factor is not in the paper. This continues the example above:

```python
from dataclasses import replace


def clustered(records: list[Record]) -> list[Record]:
    # two items per cluster: q0 and q1 share c0, q2 and q3 share c1, ...
    return [replace(r, cluster_id=f"c{int(r.item_id[1:]) // 2}") for r in records]


cres = compare(clustered(base), clustered(edit))
print(cres.n_clusters)
print(format_report(cres))
```

Output:

```text
4
n=8 items in 4 clusters  base=0.6250  edit=0.7500
score diff=+0.1250  SE=0.1250  95% CI [-0.1200, +0.3700]; not significant; smallest detectable difference at this n is 0.3500
McNemar exact p=1  (base-only right=0, edit-only right=1); ignores clustering
tokens: base=300.0  edit=250.0  diff=-50.0 (-16.7%)  bootstrap 95% CI [-50.0, -50.0]
cap-hit rate: base=0.000  edit=0.000
WARNING: only 4 clusters (<100); intervals are unreliable
```

- `cluster_id` on some records but not all, or one item with two ids, raises `ValueError`.
- With fewer than 2 clusters, the SE, CI and MDE are NaN.
- `small_n` is also `True` under 100 clusters. The McNemar line ends with `; ignores clustering`, because McNemar always treats items as independent.
- Without `cluster_id`, items are treated as independent.

## Cap-hit censoring

A record with `hit_cap=True` stopped at `max_new_tokens`. A capped answer usually has no final answer, so it is graded wrong and its token count is cut short; the score and token diffs over all items are then censored by the cap. `format_report` adds a `WARNING: cap-hit censoring` block when either arm's cap-hit rate is above 5%, or when the exact McNemar p-value on `hit_cap` between the arms is below 0.05. The block shows the comparison over the items where neither arm hit the cap, and counts loops: capped outputs whose last 400 characters are one chunk of at most 100 characters repeated 3 or more times (`is_loop`). The same numbers are in `PairedResult` (`cap_hit_p`, `n_uncapped`, `uncapped_diff`, `uncapped_tokens_diff`, `loops_base`, `loops_edit`) and in `astro-weights eval --json`. This continues the example above:

```python
# the edit arm loops on two items and hits the cap there
looped = [
    replace(r, output="and so on. " * 40, score=0.0, n_output_tokens=512, hit_cap=True)
    if r.item_id in ("q0", "q1")
    else r
    for r in edit
]
capped = compare(base, looped)
print(capped.cap_hit_edit, capped.n_uncapped, capped.loops_edit)
print("\n".join(format_report(capped).splitlines()[-7:]))
```

Output:

```text
0.25 6 2
WARNING: cap-hit censoring
  cap-hit rate: base=0.0%  edit=25.0%  (McNemar exact p=0.5)
  A capped answer usually has no final answer, so it is graded wrong and its token count is cut short.
  The score and token diffs over all items above are censored by the cap.
  items where neither arm hit the cap: n=6 items  score diff=+0.1667  tokens diff=-50.0
  loops (capped, tail repeats one chunk): base=0  edit=2
  Raise max_new_tokens and rerun; pick it from a short base-arm pilot so that cap-hit is near 0.
```

To avoid the warning, pick `max_new_tokens` from a short pilot of the base arm: raise it until almost no output hits the cap, then run both arms with that value. Arms should also use the same decoding; see [Run an eval end to end](run-eval.md#greedy-or-sampled).

## Store records

`write_jsonl(path, records)` writes one JSON object per line. `read_jsonl(path)` reads them back and skips blank lines. It raises `ValueError` with the line number on invalid JSON, missing or unknown keys, a wrong value type or a non-finite score. `prompt_hash(text)` is the SHA-256 hex digest of the UTF-8 prompt.
