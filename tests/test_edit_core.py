from pathlib import Path

import numpy as np
import pytest

from astro_weights.edits import Edit, LowRankDelta
from astro_weights.export import load_peft, save_peft


def rand_delta(module: str, rank: int, seed: int = 0) -> LowRankDelta:
    rng = np.random.default_rng(seed)
    return LowRankDelta(
        module, rng.standard_normal((6, rank)), rng.standard_normal((rank, 10))
    )


def test_padded_keeps_dense_and_shapes() -> None:
    d = rand_delta("m", 2)
    p = d.padded(5)
    assert p.rank == 5
    assert p.B.shape == (6, 5)
    assert p.A.shape == (5, 10)
    assert p.shape == d.shape
    assert np.array_equal(p.dense(), d.dense())
    assert not p.B[:, 2:].any()
    assert not p.A[2:].any()


def test_padded_same_rank_returns_self() -> None:
    d = rand_delta("m", 2)
    assert d.padded(2) is d


def test_padded_down_raises() -> None:
    with pytest.raises(ValueError, match="m: cannot pad rank 3 down to 2"):
        rand_delta("m", 3).padded(2)


def test_from_deltas_mixed_ranks() -> None:
    a, b = rand_delta("a", 1, 1), rand_delta("b", 3, 2)
    edit = Edit.from_deltas(iter([a, b]), "base", {"w": "x"}, {"k": 1})
    assert edit.rank == 3
    assert [d.module for d in edit.deltas] == ["a", "b"]
    assert np.array_equal(edit.deltas[0].dense(), a.dense())
    assert np.array_equal(edit.deltas[1].dense(), b.dense())
    assert edit.base_model == "base"
    assert edit.base_sha256 == {"w": "x"}
    assert edit.meta == {"k": 1}


def test_from_deltas_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one delta"):
        Edit.from_deltas([], "base")


def test_mixed_ranks_still_rejected_by_edit() -> None:
    with pytest.raises(ValueError, match="mixed ranks"):
        Edit((rand_delta("a", 1), rand_delta("b", 2)), "base")


def test_padded_edit_peft_roundtrip(tmp_path: Path) -> None:
    a, b = (
        rand_delta("model.layers.0.mlp.down_proj", 1, 1),
        rand_delta("model.layers.1.mlp.down_proj", 3, 2),
    )
    edit = Edit.from_deltas([a, b], "base")
    save_peft(edit, tmp_path)
    got = {d.module: d for d in load_peft(tmp_path).deltas}
    assert load_peft(tmp_path).rank == 3
    assert np.array_equal(got[a.module].dense(), a.dense())
    assert np.array_equal(got[b.module].dense(), b.dense())


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_factor_rejected(bad: float) -> None:
    b = np.ones((4, 1))
    a = np.ones((1, 3))
    b[0, 0] = bad
    with pytest.raises(ValueError, match=r"mod\.x.*finite"):
        LowRankDelta("mod.x", b, a)
    with pytest.raises(ValueError, match=r"mod\.x.*finite"):
        LowRankDelta("mod.x", np.ones((4, 1)), b.reshape(1, 4))


def test_float32_overflow_rejected_without_warning() -> None:
    import warnings

    b = np.full((4, 1), 1e300)
    a = np.ones((1, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=r"mod\.x.*finite"):
            LowRankDelta("mod.x", b, a)


def test_empty_module_name_message() -> None:
    with pytest.raises(ValueError, match=r"^module name must be non-empty$"):
        LowRankDelta("", np.ones((2, 1)), np.ones((1, 2)))


def test_edit_without_deltas_message() -> None:
    with pytest.raises(ValueError, match=r"^an edit needs at least one delta$"):
        Edit((), "base")


def test_from_deltas_without_deltas_message() -> None:
    with pytest.raises(ValueError, match=r"^an edit needs at least one delta$"):
        Edit.from_deltas([], "base")


def test_duplicate_module_names_message() -> None:
    d = rand_delta("m", 1)
    with pytest.raises(ValueError, match=r"^duplicate module names in edit$"):
        Edit((d, d), "base")


def test_dense_is_float32_product() -> None:
    d = rand_delta("m", 2)
    assert d.dense().dtype == np.float32
    assert np.allclose(d.dense(), d.B @ d.A, atol=1e-5)
