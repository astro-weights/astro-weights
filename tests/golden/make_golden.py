"""Write the golden adapter fixture with whichever astro-weights is installed.

Run with a published release, e.g.:

    uv run --no-project --with astro-weights==0.0.1 python make_golden.py out_dir

Only uses API that exists in 0.0.1. Also writes the expected dense delta of each
module to out_dir/expected_<module>.npy.
"""

import sys
from pathlib import Path

import numpy as np

from astro_weights import Edit, LowRankDelta
from astro_weights.export import save_peft

MODULES = ("model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj")
BASE = "golden/tiny-base"
SHA = {"model.layers.0.mlp.down_proj.weight": "ab" * 32}
META = {"recipe": "golden", "seed": 7}


def main(out: str) -> None:
    rng = np.random.default_rng(7)
    deltas = tuple(
        LowRankDelta(
            m,
            rng.standard_normal((4, 2)).astype(np.float32),
            rng.standard_normal((2, 8)).astype(np.float32),
        )
        for m in MODULES
    )
    out_dir = Path(out)
    save_peft(Edit(deltas, BASE, base_sha256=SHA, meta=META), out_dir)
    for d in deltas:
        np.save(out_dir / f"expected_{d.module}.npy", d.dense())


if __name__ == "__main__":
    main(sys.argv[1])
