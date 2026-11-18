from pathlib import Path

import numpy as np
import pytest

from astro_weights.edits import Edit, LowRankDelta, compose
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


def _edit(mods: dict[str, int], seed: int = 0, base: str = "base") -> Edit:
    ds = [rand_delta(m, r, seed + i) for i, (m, r) in enumerate(mods.items())]
    return Edit.from_deltas(ds, base)


def test_compose_disjoint_modules() -> None:
    a, b = _edit({"x": 2}, 1), _edit({"y": 2}, 2)
    c = compose(a, b)
    assert {d.module for d in c.deltas} == {"x", "y"}
    got = {d.module: d for d in c.deltas}
    assert np.array_equal(got["x"].dense(), a.deltas[0].dense())
    assert np.array_equal(got["y"].dense(), b.deltas[0].dense())
    assert c.base_model == "base"


def test_compose_overlap_sums_dense() -> None:
    a, b = _edit({"x": 2, "y": 1}, 1), _edit({"x": 3}, 7)
    c = compose(a, b)
    got = {d.module: d for d in c.deltas}
    want = a.deltas[0].dense() + b.deltas[0].dense()
    np.testing.assert_allclose(got["x"].dense(), want, rtol=1e-5, atol=1e-5)
    assert got["x"].B.dtype == np.float32
    assert c.rank == 5
    assert got["y"].rank == 5  # padded
    assert np.array_equal(got["y"].dense(), a.deltas[1].dense())


def test_compose_three_way() -> None:
    es = [_edit({"x": 1}, s) for s in (1, 2, 3)]
    c = compose(*es)
    want = sum(e.deltas[0].dense() for e in es)
    np.testing.assert_allclose(c.deltas[0].dense(), want, rtol=1e-5, atol=1e-5)
    assert c.rank == 3


def test_compose_meta_and_sha() -> None:
    a = Edit.from_deltas([rand_delta("x", 1)], "b", {"x.weight": "h1"}, {"n": 1})
    b = Edit.from_deltas(
        [rand_delta("y", 1)], "b", {"x.weight": "h1", "y.weight": "h2"}
    )
    c = compose(a, b)
    assert c.base_sha256 == {"x.weight": "h1", "y.weight": "h2"}
    assert c.meta == {"composed_from": [{"n": 1}, {}]}


def test_compose_one_edit_raises() -> None:
    with pytest.raises(ValueError, match="at least two"):
        compose(_edit({"x": 1}))
    with pytest.raises(ValueError, match="at least two"):
        compose()


def test_compose_different_base_model_raises() -> None:
    with pytest.raises(ValueError, match=r"'one'.*'two'"):
        compose(_edit({"x": 1}, base="one"), _edit({"y": 1}, base="two"))


def test_compose_conflicting_sha_raises() -> None:
    a = Edit.from_deltas([rand_delta("x", 1)], "b", {"x.weight": "h1"})
    b = Edit.from_deltas([rand_delta("y", 1)], "b", {"x.weight": "h2"})
    with pytest.raises(ValueError, match=r"x\.weight.*different bases"):
        compose(a, b)


def test_compose_shape_mismatch_raises() -> None:
    a = Edit.from_deltas([rand_delta("x", 1)], "b")
    small = LowRankDelta("x", np.ones((4, 1)), np.ones((1, 10)))
    with pytest.raises(ValueError, match=r"x: shape mismatch"):
        compose(a, Edit.from_deltas([small], "b"))


def test_compose_peft_alpha_folded(tmp_path: Path) -> None:
    """A PEFT adapter with alpha != r composes with its scaling applied."""
    import json

    from safetensors.numpy import save_file

    rng = np.random.default_rng(9)
    mod = "model.layers.0.self_attn.q_proj"
    b3 = rng.standard_normal((6, 2)).astype(np.float32)
    a3 = rng.standard_normal((2, 10)).astype(np.float32)
    ext = tmp_path / "ext"
    ext.mkdir()
    save_file(
        {
            f"base_model.model.{mod}.lora_A.weight": a3,
            f"base_model.model.{mod}.lora_B.weight": b3,
        },
        str(ext / "adapter_model.safetensors"),
    )
    (ext / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "base",
                "r": 2,
                "lora_alpha": 8,
                "target_modules": ["q_proj"],
            }
        )
    )
    ours = Edit.from_deltas([rand_delta(mod, 1, 4)], "base")
    save_peft(ours, tmp_path / "ours")
    c = compose(load_peft(ext), load_peft(tmp_path / "ours"))
    want = 4.0 * (b3 @ a3) + ours.deltas[0].dense()
    np.testing.assert_allclose(c.deltas[0].dense(), want, rtol=1e-5, atol=1e-5)
