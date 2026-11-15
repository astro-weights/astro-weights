# Convert dense update to low-rank delta

`from_dense` turns dense `dW` (for example `W_tuned - W0`) into `LowRankDelta` by truncated SVD and returns relative residual `|dW - B @ A|_F / |dW|_F`, measured on stored float32 factors. numpy only.

```python
import numpy as np

from astro_weights.edits import Edit, from_dense

rng = np.random.default_rng(0)
module = "model.layers.1.mlp.down_proj"
# stand-in for a dense update: rank 2 plus small noise
dW = rng.standard_normal((32, 2)) @ rng.standard_normal((2, 64))
dW += 0.01 * rng.standard_normal((32, 64))

delta, rel = from_dense(module, dW, rank=2)
print(delta.shape, delta.rank, f"{rel:.4f}")

try:
    from_dense(module, dW, rank=1, max_rel_residual=0.05)
except ValueError as e:
    print(e)

edit = Edit(deltas=(delta,), base_model="tiny-llama")
```

Prints:

```text
(32, 64) 2 0.0078
model.layers.1.mlp.down_proj: rel_residual 0.6071 exceeds max_rel_residual 0.05 at rank 1
```

- `max_rel_residual` set: residual above it raises `ValueError`, so rank too low for update fails loudly.
- `dW` not 2-D, not finite or all zero: `ValueError`. `rank` not int, or outside `[1, min(dW.shape)]`: `ValueError`.
- Save `edit` as adapter like any other: [Tutorial: first edit](../tutorials/first-edit.md).
