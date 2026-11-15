import numpy as np
import pytest

from astro_weights.edits import (
    Edit,
    LowRankDelta,
    column_shift,
    masked_rank1,
    orthogonal_random,
    project_out,
    rank1,
    scale_column,
    scale_head,
    top_k_mask,
    zero_head,
)

RNG = np.random.default_rng(0)
OUT, IN = 6, 10


def _w0() -> np.ndarray:
    return RNG.standard_normal((OUT, IN)).astype(np.float32)


def test_masked_rank1_matches_dense() -> None:
    w0, u = _w0(), RNG.standard_normal(OUT).astype(np.float32)
    mask = top_k_mask(RNG.standard_normal(IN), 3)
    alpha = 0.7
    d = masked_rank1("m", w0, u, mask, alpha)
    dense = w0.astype(np.float64) - alpha * np.outer(
        u, mask * (u @ w0.astype(np.float64))
    )
    np.testing.assert_allclose(d.dense(), dense - w0, atol=1e-6)
    changed = np.abs(d.dense()).sum(axis=0) > 0
    assert not changed[mask == 0].any()


def test_masked_rank1_full_mask_projects_out_u() -> None:
    w0, u = _w0(), RNG.standard_normal(OUT).astype(np.float32)
    u /= np.linalg.norm(u)
    d = masked_rank1("m", w0, u, np.ones(IN, dtype=np.float32), 1.0)
    atol = 1e-5 * float(np.abs(u @ w0).max())
    np.testing.assert_allclose(u @ (w0 + d.dense()), 0.0, atol=atol)


def test_masked_rank1_bad_shapes() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        masked_rank1("m", _w0(), np.ones(OUT + 1), np.ones(IN), 1.0)


def test_column_shift() -> None:
    shift = RNG.standard_normal(OUT)
    idx, coef = np.array([1, 4, 7]), np.array([0.5, -1.0, 2.0])
    d = column_shift("m", shift, idx, coef, IN)
    c = np.zeros(IN)
    c[idx] = coef
    np.testing.assert_allclose(d.dense(), np.outer(shift, c), atol=1e-6)
    outside = np.setdiff1d(np.arange(IN), idx)
    assert not d.dense()[:, outside].any()


@pytest.mark.parametrize(
    ("idx", "coef"),
    [
        ([1, 1], [1.0, 2.0]),
        ([1, IN], [1.0, 2.0]),
        ([-1, 2], [1.0, 2.0]),
        ([1, 2], [1.0]),
    ],
)
def test_column_shift_invalid(idx: list[int], coef: list[float]) -> None:
    with pytest.raises(ValueError, match=r"idx|same length"):
        column_shift("m", np.ones(OUT), np.array(idx), np.array(coef), IN)


def test_top_k_mask() -> None:
    mask = top_k_mask(np.array([0.1, 5.0, -2.0, 3.0, 0.0]), 2)
    assert mask.dtype == np.float32
    np.testing.assert_array_equal(mask, [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(top_k_mask(np.arange(4.0), 10), np.ones(4))


def test_orthogonal_random() -> None:
    u = RNG.standard_normal(32)
    r = orthogonal_random(u, 1)
    assert abs(float(r @ u)) < 1e-6
    assert abs(float(np.linalg.norm(r)) - 1.0) < 1e-9
    assert r.shape == u.shape
    np.testing.assert_array_equal(r, orthogonal_random(u, 1))
    assert not np.allclose(r, orthogonal_random(u, 2))


def test_rank1_and_delta_properties() -> None:
    d = rank1("m", np.ones(OUT), np.ones(IN))
    assert (d.rank, d.shape) == (1, (OUT, IN))
    assert d.B.dtype == d.A.dtype == np.float32
    assert d.B.flags.c_contiguous
    assert d.dense().dtype == np.float32


def test_delta_validation() -> None:
    with pytest.raises(ValueError, match="2-D"):
        LowRankDelta("m", np.ones(3), np.ones((1, 4)))
    with pytest.raises(ValueError, match="rank mismatch"):
        LowRankDelta("m", np.ones((3, 2)), np.ones((1, 4)))
    with pytest.raises(ValueError, match="rank mismatch"):
        LowRankDelta("m", np.ones((3, 0)), np.ones((0, 4)))
    with pytest.raises(ValueError, match="non-empty"):
        LowRankDelta("", np.ones((3, 1)), np.ones((1, 4)))


def test_edit_validation() -> None:
    a = LowRankDelta("a", np.ones((3, 1)), np.ones((1, 4)))
    b = LowRankDelta("b", np.ones((3, 1)), np.ones((1, 4)))
    c = LowRankDelta("c", np.ones((3, 2)), np.ones((2, 4)))
    assert Edit((a, b), "base").rank == 1
    with pytest.raises(ValueError, match="at least one"):
        Edit((), "base")
    with pytest.raises(ValueError, match="duplicate"):
        Edit((a, a), "base")
    with pytest.raises(ValueError, match="mixed ranks"):
        Edit((a, c), "base")


def _w0_heads() -> np.ndarray:
    return RNG.standard_normal((6, 8)).astype(np.float32)


def test_project_out_matches_direct() -> None:
    w0, v = _w0_heads(), RNG.standard_normal(6).astype(np.float32)
    vh = v.astype(np.float64) / np.linalg.norm(v.astype(np.float64))
    for strength in (1.0, 0.4):
        d = project_out("m", w0, v, strength)
        direct = w0 - strength * np.outer(vh, vh @ w0.astype(np.float64))
        np.testing.assert_allclose(w0 + d.dense(), direct, atol=1e-6)
        assert d.rank == 1
    d = project_out("m", w0, v)
    np.testing.assert_allclose(vh @ (w0 + d.dense()), 0.0, atol=1e-5)


def test_project_out_errors() -> None:
    w0 = _w0_heads()
    with pytest.raises(ValueError, match="m: shape mismatch"):
        project_out("m", w0, np.ones(5, dtype=np.float32))
    with pytest.raises(ValueError, match="m: shape mismatch"):
        project_out("m", w0[0], np.ones(6, dtype=np.float32))
    with pytest.raises(ValueError, match="m: v must be non-zero"):
        project_out("m", w0, np.zeros(6, dtype=np.float32))


def test_scale_column_matches_direct() -> None:
    w0 = _w0_heads()
    d = scale_column("m", w0, 3, 2.5)
    direct = w0.copy()
    direct[:, 3] *= 2.5
    np.testing.assert_allclose(w0 + d.dense(), direct, atol=1e-6)
    assert d.rank == 1
    assert not scale_column("m", w0, 3, 1.0).dense().any()


def test_scale_column_errors() -> None:
    w0 = _w0_heads()
    for j in (-1, 8):
        with pytest.raises(ValueError, match="m: column"):
            scale_column("m", w0, j, 2.0)
    with pytest.raises(ValueError, match="m: W0 must be 2-D"):
        scale_column("m", w0[0], 0, 2.0)


def test_scale_head_matches_direct() -> None:
    w0 = _w0_heads()
    d = scale_head("m", w0, 2, 4, 1.7)
    direct = w0.copy()
    direct[:, 4:6] *= 1.7
    np.testing.assert_allclose(w0 + d.dense(), direct, atol=1e-6)
    assert d.rank == 2
    assert not scale_head("m", w0, 2, 4, 1.0).dense().any()


def test_zero_head_matches_direct() -> None:
    w0 = _w0_heads()
    d = zero_head("m", w0, 1, 4)
    direct = w0.copy()
    direct[:, 2:4] = 0.0
    np.testing.assert_allclose(w0 + d.dense(), direct, atol=1e-6)
    assert d.rank == 2


def test_scale_head_errors() -> None:
    w0 = _w0_heads()
    with pytest.raises(ValueError, match="m: n_in=8 not divisible"):
        scale_head("m", w0, 0, 3, 2.0)
    with pytest.raises(ValueError, match="m: n_in=8 not divisible"):
        scale_head("m", w0, 0, 0, 2.0)
    for head in (-1, 4):
        with pytest.raises(ValueError, match="m: head"):
            scale_head("m", w0, head, 4, 2.0)
        with pytest.raises(ValueError, match="m: head"):
            zero_head("m", w0, head, 4)
    with pytest.raises(ValueError, match="m: W0 must be 2-D"):
        scale_head("m", w0[0], 0, 4, 2.0)


def test_project_out_rejects_non_finite() -> None:
    w0 = RNG.standard_normal((OUT, IN))
    for bad in (np.nan, np.inf):
        v = np.ones(OUT)
        v[0] = bad
        with pytest.raises(ValueError, match="finite"):
            project_out("m", w0, v)
    with pytest.raises(ValueError, match="finite"):
        project_out("m", w0, np.ones(OUT), strength=float("nan"))


@pytest.mark.parametrize("s", [float("nan"), float("inf"), -float("inf")])
def test_scale_recipes_reject_non_finite_s(s: float) -> None:
    w0 = RNG.standard_normal((OUT, IN))
    with pytest.raises(ValueError, match="finite"):
        scale_column("m", w0, 1, s)
    with pytest.raises(ValueError, match="finite"):
        scale_head("m", w0, 0, 2, s)


@pytest.mark.parametrize("bad", [1.5, True, "1", None])
def test_scale_recipes_reject_non_int_indices(bad: object) -> None:
    w0 = RNG.standard_normal((OUT, IN))
    with pytest.raises(ValueError, match="must be an int"):
        scale_column("m", w0, bad, 2.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an int"):
        scale_head("m", w0, bad, 2, 2.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an int"):
        scale_head("m", w0, 0, bad, 2.0)  # type: ignore[arg-type]


def test_scale_recipes_accept_numpy_int_index() -> None:
    w0 = RNG.standard_normal((OUT, IN))
    assert scale_column("m", w0, np.int64(2), 2.0).shape == (OUT, IN)  # type: ignore[arg-type]
    assert scale_head("m", w0, np.int32(1), 2, 2.0).shape == (OUT, IN)  # type: ignore[arg-type]
