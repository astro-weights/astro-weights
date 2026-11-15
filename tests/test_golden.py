"""Adapters written by released versions must keep loading (manifest format 1)."""

from pathlib import Path

import numpy as np
import pytest

from astro_weights.cli import main
from astro_weights.export import load_peft, verify_peft

GOLDEN = Path(__file__).parent / "golden"
VERSIONS = ["v0.0.1", "v0.0.9"]
MODULES = ("model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj")


@pytest.mark.parametrize("version", VERSIONS)
def test_golden_adapter_loads(version: str) -> None:
    path = GOLDEN / version
    assert verify_peft(path) == []
    edit = load_peft(path)
    assert edit.base_model == "golden/tiny-base"
    assert edit.base_sha256 == {"model.layers.0.mlp.down_proj.weight": "ab" * 32}
    assert edit.meta == {"recipe": "golden", "seed": 7}
    assert [d.module for d in edit.deltas] == list(MODULES)
    for d in edit.deltas:
        expected = np.load(path / f"expected_{d.module}.npy")
        assert d.rank == 2
        assert d.shape == (4, 8)
        # factors are stored verbatim: the loader must not alter a single bit
        assert d.B.tobytes() == np.load(path / f"expected_{d.module}_B.npy").tobytes()
        assert d.A.tobytes() == np.load(path / f"expected_{d.module}_A.npy").tobytes()
        # the product depends on the BLAS in use, so only compare to tolerance
        np.testing.assert_allclose(d.dense(), expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize("version", VERSIONS)
def test_golden_adapter_inspect(version: str) -> None:
    assert main(["inspect", str(GOLDEN / version)]) == 0
