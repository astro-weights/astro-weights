# Compare two eval arms

`astro_weights.eval` compares base arm and edit arm on same items from per-item records. Never runs model: caller generates outputs, grades them, writes one `Record` per item, seed and arm. Numpy only.

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

Prints:

```text
n=8 items  base=0.6250  edit=0.7500
score diff=+0.1250  SE=0.1250  95% CI [-0.1200, +0.3700]
McNemar exact p=1  (base-only right=0, edit-only right=1)
tokens: base=300.0  edit=250.0  diff=-50.0 (-16.7%)  bootstrap 95% CI [-50.0, -50.0]
cap-hit rate: base=0.000  edit=0.000
WARNING: only 8 items (<100); intervals are unreliable
```

## What `compare` does

- `pair` joins arms on `(item_id, seed)`. Raises `ValueError` when keys differ, key repeats within arm, or `prompt_hash` differs between arms.
- Scores and token counts get averaged over seeds per item first; every statistic runs over items.
- Score difference: paired SE, 95% CI `diff ± 1.96 * SE`.
- Exact McNemar p-value only when every per-item score is 0 or 1; otherwise `mcnemar_p` is `None`.
- Token difference: bootstrap 95% CI, `n_boot=10000` resamples, `seed=0` by default.
- `small_n` is `True` under 100 items; `format_report` then adds warning line.

## Store records

`write_jsonl(path, records)` writes one JSON object per line. `read_jsonl(path)` reads them back, skips blank lines, raises `ValueError` with line number on invalid JSON, missing or unknown keys, wrong value type or non-finite score. `prompt_hash(text)` is SHA-256 hex digest of UTF-8 prompt.
