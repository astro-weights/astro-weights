import numpy as np
import pytest

from astro_weights.edits import Edit, from_dense


def test_exact_low_rank_recovered() -> None:
    rng = np.random.default_rng(0)
    dW = rng.standard_normal((12, 3)) @ rng.standard_normal((3, 9))
    delta, rel = from_dense("m", dW, 3)
    assert delta.module == "m"
    assert delta.rank == 3
    assert delta.shape == (12, 9)
    assert rel < 1e-6
    assert np.allclose(delta.dense(), dW, atol=1e-4)


def test_residual_matches_eckart_young() -> None:
    rng = np.random.default_rng(1)
    dW = rng.standard_normal((20, 15))
    rank = 4
    s = np.linalg.svd(dW, compute_uv=False)
    expected = np.sqrt(np.sum(s[rank:] ** 2)) / np.linalg.norm(dW)
    _, rel = from_dense("m", dW, rank)
    assert rel == pytest.approx(expected, rel=1e-4)


def test_residual_uses_stored_float32_factors() -> None:
    rng = np.random.default_rng(2)
    dW = rng.standard_normal((8, 6))
    delta, rel = from_dense("m", dW, 2)
    direct = np.linalg.norm(
        dW - delta.B.astype(np.float64) @ delta.A.astype(np.float64)
    ) / np.linalg.norm(dW)
    assert rel == pytest.approx(direct, rel=1e-12)


def test_bound_enforced() -> None:
    rng = np.random.default_rng(3)
    dW = rng.standard_normal((10, 10))
    _, rel = from_dense("m", dW, 2)
    from_dense("m", dW, 2, max_rel_residual=rel + 1e-3)
    with pytest.raises(ValueError, match="exceeds") as exc:
        from_dense("m", dW, 2, max_rel_residual=0.1)
    assert "0.1" in str(exc.value)
    assert f"{rel:.4g}" in str(exc.value)


def test_full_rank_is_near_exact() -> None:
    dW = np.random.default_rng(4).standard_normal((5, 7))
    _, rel = from_dense("m", dW, 5, max_rel_residual=1e-5)
    assert rel < 1e-5


@pytest.mark.parametrize(
    "dW",
    [np.zeros((4, 4)), np.ones(4), np.ones((2, 3, 4)), np.full((4, 4), np.nan)],
)
def test_bad_matrix_rejected(dW: np.ndarray) -> None:
    with pytest.raises(ValueError, match="m:"):
        from_dense("m", dW, 1)


def test_inf_matrix_rejected() -> None:
    dW = np.ones((4, 4))
    dW[1, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        from_dense("m", dW, 1)


@pytest.mark.parametrize("rank", [0, -1, 6, 1.0, True, "2"])
def test_bad_rank_rejected(rank: object) -> None:
    dW = np.random.default_rng(5).standard_normal((5, 8))
    with pytest.raises(ValueError, match="rank"):
        from_dense("m", dW, rank)  # type: ignore[arg-type]


def test_works_in_edit_from_deltas() -> None:
    rng = np.random.default_rng(6)
    d1, _ = from_dense("a", rng.standard_normal((6, 8)), 1)
    d2, _ = from_dense("b", rng.standard_normal((6, 8)), 3)
    edit = Edit.from_deltas([d1, d2], "base")
    assert edit.rank == 3


def test_residual_equal_to_threshold_is_accepted() -> None:
    rng = np.random.default_rng(3)
    dW = rng.standard_normal((10, 8))
    _, rel = from_dense("m", dW, 2)
    from_dense("m", dW, 2, max_rel_residual=rel)  # not strictly greater: no raise
    with pytest.raises(ValueError, match="exceeds max_rel_residual"):
        from_dense("m", dW, 2, max_rel_residual=rel * 0.999)


def test_float32_input_residual_is_measured_in_float64() -> None:
    # Large float32 entries: a float32 norm overflows to inf and the residual reads 0.
    rng = np.random.default_rng(4)
    dW = (1e20 * rng.standard_normal((6, 5))).astype(np.float32)
    s = np.linalg.svd(dW.astype(np.float64), compute_uv=False)
    expected = np.sqrt(np.sum(s[2:] ** 2)) / np.linalg.norm(s)
    _, rel = from_dense("m", dW, 2)
    assert rel == pytest.approx(expected, rel=1e-4)
