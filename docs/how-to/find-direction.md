# Find and gate direction

`astro_weights.directions` turns residual activations from paired prompts into unit direction, then reports whether direction stands out from noise. Numpy only: collecting activations is caller's job. Direction is `[hidden]` vector pointing from set `b` to set `a`.

## Gate, then project out

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

Prints:

```text
True 1.0
1
```

## Read gate

`direction_gate` takes pair differences `[P, H]` (a minus b), at least 4 pairs. `seed` picks random split into two halves. Returns `GateResult` statistics, no pass/fail verdict; field meanings in [reference](../reference/index.md). Pure noise gives split-half cosine near 0 and held-out AUC near 0.5. Package tests count planted direction as passing when `split_half_cos > null_975` (`tests/test_directions.py`).

`directions_from_pairs(res_a, res_b)` gives one direction per layer, `[L, H]`, from per-pair residual means `[P, L, H]`.

## Remove direction from layer

`project_out(module, W0, v, strength=1.0)` returns rank-1 `LowRankDelta` with `W = W0 - strength * outer(vhat, vhat @ W0)`. `W0` rows must match length of `v`: weight writes to residual, for example `mlp.down_proj.weight`, `[hidden, intermediate]`. Delta saves as LoRA adapter like any other.

Same module has `scale_column`, `scale_head` and `zero_head`. `scale_head` delta has rank equal to head size; combine deltas of different rank with `Edit.from_deltas`, which zero-pads each to largest rank.
