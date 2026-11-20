import numpy as np
import pytest

from astro_weights.edits import (
    Edit,
    LowRankDelta,
    compose,
    random_controls,
    random_like,
)
from astro_weights.edits.controls import Dtype, _round


def _edit(rank: int, shapes: dict[str, tuple[int, int]], seed: int = 1) -> Edit:
    rng = np.random.default_rng(seed)
    deltas = [
        LowRankDelta(
            m, rng.standard_normal((o, rank)), 0.1 * rng.standard_normal((rank, i))
        )
        for m, (o, i) in shapes.items()
    ]
    return Edit(tuple(deltas), "base/x", {"w": "abc"}, {"name": "parent"})


SHAPES = {"layers.0.q": (64, 48), "layers.1.v": (32, 48)}


def _norms(e: Edit) -> dict[str, float]:
    return {
        d.module: float(np.linalg.norm(d.dense().astype(np.float64))) for d in e.deltas
    }


@pytest.mark.parametrize("rank", [1, 4])
def test_norm_shape_rank_match(rank: int) -> None:
    e = _edit(rank, SHAPES)
    c = random_like(e, 7)
    assert [d.module for d in c.deltas] == [d.module for d in e.deltas]
    assert c.rank == e.rank
    assert [d.shape for d in c.deltas] == [d.shape for d in e.deltas]
    assert c.base_model == e.base_model
    assert c.base_sha256 == e.base_sha256
    for m, n in _norms(e).items():
        assert _norms(c)[m] == pytest.approx(n, rel=1e-6)
    assert c.meta["norm_match"] == "factor"


def test_compose_output() -> None:
    e = compose(_edit(1, SHAPES, 1), _edit(2, SHAPES, 2))
    assert e.rank == 3
    c = random_like(e, 0)
    assert c.rank == 3
    for m, n in _norms(e).items():
        assert _norms(c)[m] == pytest.approx(n, rel=1e-6)


def test_seeds() -> None:
    e = _edit(2, SHAPES)
    a, b, c = random_like(e, 3), random_like(e, 3), random_like(e, 4)
    for x, y, z in zip(a.deltas, b.deltas, c.deltas, strict=True):
        assert np.array_equal(x.B, y.B)
        assert np.array_equal(x.A, y.A)
        assert not np.array_equal(x.B, z.B)


def test_uncorrelated_with_edit() -> None:
    e = _edit(1, {"m": (256, 256)})
    c = random_like(e, 0)
    x, y = e.deltas[0].dense().ravel(), c.deltas[0].dense().ravel()
    assert abs(x @ y / np.linalg.norm(x) / np.linalg.norm(y)) < 0.05


def test_controls() -> None:
    e = _edit(1, SHAPES)
    cs = random_controls(e, n=4, seed=5)
    assert [c.meta["control"] for c in cs] == [0, 1, 2, 3]
    assert cs[0].meta["random_control_of"] == {"name": "parent"}
    assert cs[0].meta["controls"] == 4
    assert not np.array_equal(cs[0].deltas[0].B, cs[1].deltas[0].B)
    again = random_controls(e, n=4, seed=5)
    assert np.array_equal(cs[2].deltas[0].A, again[2].deltas[0].A)
    assert len(random_controls(e)) == 20
    with pytest.raises(ValueError, match="at least 1"):
        random_controls(e, n=0)


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_rounded_norm(dtype: Dtype) -> None:
    rng = np.random.default_rng(0)
    w0 = {
        m: _round(rng.standard_normal(s).astype(np.float32), dtype)
        for m, s in SHAPES.items()
    }
    e = _edit(2, SHAPES)
    c = random_like(e, 1, dtype=dtype, base=lambda m: w0[m])
    assert c.meta["norm_match"] == f"rounded:{dtype}"
    for d, k in zip(e.deltas, c.deltas, strict=True):
        want = np.linalg.norm(_round(w0[d.module] + d.dense(), dtype) - w0[d.module])
        got = np.linalg.norm(_round(w0[d.module] + k.dense(), dtype) - w0[d.module])
        assert got == pytest.approx(want, rel=0.01)


def test_rounded_norm_small_delta_lost_in_rounding() -> None:
    # a tiny dW on large W0 mostly rounds away; the control must still match that
    w0 = np.full((16, 16), 64.0, dtype=np.float32)
    e = Edit((LowRankDelta("m", np.full((16, 1), 0.05), np.full((1, 16), 1.0)),), "b")
    c = random_like(e, 0, dtype="bfloat16", base=lambda _m: w0)
    want = np.linalg.norm(_round(w0 + e.deltas[0].dense(), "bfloat16") - w0)
    got = np.linalg.norm(_round(w0 + c.deltas[0].dense(), "bfloat16") - w0)
    assert got == pytest.approx(want, rel=0.01)


def test_dtype_without_base_and_errors() -> None:
    e = _edit(1, SHAPES)
    assert random_like(e, 0, dtype="bfloat16").meta["norm_match"] == "factor"
    with pytest.raises(ValueError, match="needs a dtype"):
        random_like(e, 0, base=lambda _m: np.zeros((64, 48), np.float32))
    with pytest.raises(ValueError, match="dtype must be"):
        random_like(e, 0, dtype="int8")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="base weight"):
        random_like(e, 0, dtype="float16", base=lambda _m: np.zeros((2, 2), np.float32))


def test_zero_delta() -> None:
    e = Edit((LowRankDelta("m", np.zeros((4, 1)), np.ones((1, 3))),), "b")
    c = random_like(e, 0)
    assert not c.deltas[0].dense().any()
    z = random_like(e, 0, dtype="float16", base=lambda _m: np.ones((4, 3), np.float32))
    assert not z.deltas[0].dense().any()


def test_round_bfloat16_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    x = np.random.default_rng(0).standard_normal(1000).astype(np.float32) * 3
    assert np.array_equal(
        _round(x, "bfloat16"), torch.from_numpy(x).bfloat16().float().numpy()
    )
