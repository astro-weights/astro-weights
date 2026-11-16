# Find and gate a direction

`astro_weights.directions` turns residual activations from paired prompts into a unit direction, then reports whether that direction stands out from noise. It needs numpy only; collecting the activations is up to you. A direction is a `[hidden]` vector that points from set `b` to set `a`.

## Gate it, then project it out

```python
import numpy as np

from astro_weights.directions import direction_from_means, direction_gate
from astro_weights.edits import project_out

rng = np.random.default_rng(0)
hidden = 64
# Residuals at one layer, one row per prompt pair: set a (trait on), set b (trait off).
res_b = rng.standard_normal((40, hidden))
res_a = res_b + 0.2 * rng.standard_normal((40, hidden))
res_a[:, 3] += 1.0  # planted direction

gate = direction_gate(res_a - res_b, seed=0)
print(gate.split_half_cos > gate.null_975, gate.heldout_auc)

u = direction_from_means(res_a.mean(axis=0), res_b.mean(axis=0))
W0 = rng.standard_normal((hidden, 128)).astype(np.float32)  # weight writing to residual
delta = project_out("model.layers.10.mlp.down_proj", W0, u)
print(delta.rank)
```

Output:

```text
True 1.0
1
```

## Read the gate

`direction_gate` takes pair differences `[P, H]` (a minus b). It needs at least 4 pairs, finite values only and `draws >= 1`, and raises `ValueError` otherwise. Under about 20 pairs the results mean nothing: the halves are tiny and `null_975` is near 1. `seed` picks the random split into two halves. The function returns `GateResult` statistics, not a pass/fail verdict; the field meanings are in the [reference](../reference/index.md). A statistic is NaN when the mean it needs has zero norm. Pure noise gives a split-half cosine near 0 and a held-out AUC near 0.5. The package's own tests count a planted direction as passing when `split_half_cos > null_975` (`tests/test_directions.py`).

`directions_from_pairs(res_a, res_b)` gives one direction per layer, `[L, H]`, from per-pair residual means `[P, L, H]`.

## Remove a direction from a layer

`project_out(module, W0, v, strength=1.0)` returns a rank-1 `LowRankDelta` with `W = W0 - strength * outer(vhat, vhat @ W0)`. The rows of `W0` must match the length of `v`, so the weight must be one that writes to the residual, for example `mlp.down_proj.weight` with shape `[hidden, intermediate]`. The delta saves as a LoRA adapter like any other.

The same module has `scale_column`, `scale_head` and `zero_head`. A `scale_head` delta has a rank equal to the head size; to combine deltas of different rank, use `Edit.from_deltas`, which zero-pads each one to the largest rank.
