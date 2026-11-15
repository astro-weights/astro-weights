"""Tests for astro_weights.directions."""

import numpy as np
import pytest

from astro_weights.directions import (
    GateResult,
    core,
    direction_from_means,
    direction_gate,
    directions_from_pairs,
    neuron_scores,
    stability,
)

H, P = 64, 200


def _planted() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    v = rng.standard_normal(H)
    v /= np.linalg.norm(v)
    diffs = 3.0 * v + rng.standard_normal((P, H))
    return v, diffs


def test_direction_from_means_unit_and_oriented() -> None:
    u = direction_from_means(np.array([3.0, 4.0]), np.array([0.0, 0.0]))
    assert u.dtype == np.float32
    np.testing.assert_allclose(u, [0.6, 0.8], atol=1e-6)
    np.testing.assert_allclose(
        direction_from_means(np.zeros(2), np.array([3.0, 4.0])), [-0.6, -0.8], atol=1e-6
    )


def test_direction_from_means_zero_raises() -> None:
    with pytest.raises(ValueError, match="no direction"):
        direction_from_means(np.ones(3), np.ones(3))


def test_neuron_scores_hand_example() -> None:
    u = np.array([1.0, 2.0])
    W = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])  # [hidden=2, intermediate=3]
    act = np.array([2.0, -1.0, 0.5])
    out = neuron_scores(u, W, act)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [2.0, -2.0, 2.0])


def test_stability() -> None:
    u = np.array([1.0, 0.0])
    diffs = np.array([[2.0, 0.0], [0.0, 3.0], [-1.0, 0.0]])
    assert stability(u, diffs) == pytest.approx(0.0)
    assert stability(u, diffs[:1]) == pytest.approx(1.0)


def test_directions_from_pairs() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 3, 8))
    b = rng.standard_normal((5, 3, 8))
    out = directions_from_pairs(a, b)
    assert out.shape == (3, 8)
    np.testing.assert_allclose(
        out[1], direction_from_means(a[:, 1].mean(0), b[:, 1].mean(0))
    )
    with pytest.raises(ValueError, match="share shape"):
        directions_from_pairs(a, b[:4])


def test_planted_direction_recovered_and_gate_passes() -> None:
    v, diffs = _planted()
    u = direction_from_means(diffs.mean(0), np.zeros(H))
    assert float(u @ v) > 0.95
    g = direction_gate(diffs)
    assert g.split_half_cos > 0.8
    assert g.heldout_auc > 0.9
    assert g.split_half_cos > g.null_975


def test_noise_fails_gate() -> None:
    diffs = np.random.default_rng(2).standard_normal((P, H))
    g = direction_gate(diffs)
    assert abs(g.split_half_cos) < 0.3
    assert abs(g.heldout_auc - 0.5) < 0.2


def test_gate_deterministic_and_seeded() -> None:
    _, diffs = _planted()
    assert direction_gate(diffs, seed=3) == direction_gate(diffs, seed=3)
    assert isinstance(direction_gate(diffs), GateResult)
    assert direction_gate(diffs, seed=3) != direction_gate(diffs, seed=4)


def test_gate_errors() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        direction_gate(np.ones((3, 8)))
    with pytest.raises(ValueError, match=r"\[P, H\]"):
        direction_gate(np.ones(8))


def test_gate_rejects_bad_draws() -> None:
    _, diffs = _planted()
    for draws in (0, -3):
        with pytest.raises(ValueError, match="draws must be >= 1"):
            direction_gate(diffs, draws=draws)
    assert isinstance(direction_gate(diffs, draws=1), GateResult)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_gate_rejects_non_finite(bad: float) -> None:
    _, diffs = _planted()
    diffs[3, 5] = bad
    with pytest.raises(ValueError, match="finite"):
        direction_gate(diffs)


def test_gate_zero_half_mean_is_nan() -> None:
    diffs = np.zeros((8, 5))
    r = direction_gate(diffs)
    for v in (r.split_half_cos, r.null_975, r.heldout_auc, r.per_pair_cos):
        assert np.isnan(v)


def test_gate_zero_half_a_only_is_nan() -> None:
    v = np.arange(1.0, 6.0)
    diffs = np.stack([v, v, -v, -v] + [2 * v] * 4)
    # find a seed whose half A is balanced (mean exactly 0) and half B is not
    for seed in range(200):
        perm = np.random.default_rng(seed).permutation(8)
        a, b = diffs[perm[:4]], diffs[perm[4:]]
        if not a.mean(axis=0).any() and b.mean(axis=0).any():
            break
    else:
        pytest.fail("no seed gives a cancelling half A")
    r = direction_gate(diffs, seed=seed)
    assert np.isnan(r.split_half_cos)
    assert np.isnan(r.heldout_auc)
    assert np.isnan(r.per_pair_cos)


def test_from_means_shape_checks() -> None:
    with pytest.raises(ValueError, match="1-D and equal shape"):
        direction_from_means(np.ones(4), np.ones(1))
    with pytest.raises(ValueError, match="1-D and equal shape"):
        direction_from_means(np.ones((2, 2)), np.zeros((2, 2)))


def test_neuron_scores_shape_checks() -> None:
    w = np.ones((4, 6))
    with pytest.raises(ValueError, match="act_contrast"):
        neuron_scores(np.ones(4), w, np.ones(1))
    with pytest.raises(ValueError, match="hidden"):
        neuron_scores(np.ones(3), w, np.ones(6))


def test_directions_from_pairs_errors_name_layer() -> None:
    res_a = np.random.default_rng(0).standard_normal((3, 4, 5))
    res_b = res_a - 1.0
    res_b[:, 2] = res_a[:, 2]  # layer 2 has equal means
    with pytest.raises(ValueError, match="layer 2"):
        directions_from_pairs(res_a, res_b)
    with pytest.raises(ValueError, match="at least 1 pair"):
        directions_from_pairs(np.zeros((0, 4, 5)), np.zeros((0, 4, 5)))


def test_direction_from_means_unit_for_tiny_difference() -> None:
    u = direction_from_means(np.array([1e-14, 0.0]), np.zeros(2))
    assert np.isclose(np.linalg.norm(u), 1.0)


# --- gaps found by mutation testing ---


def test_direction_from_means_equal_message_is_exact() -> None:
    with pytest.raises(
        ValueError, match=r"^mean_a and mean_b are equal: no direction$"
    ):
        direction_from_means(np.ones(3), np.ones(3))


def test_direction_from_means_float16_does_not_overflow() -> None:
    # float16 norm overflows past 65504; the maths must run in float64
    a = np.array([300.0, 400.0], dtype=np.float16)
    u = direction_from_means(a, np.zeros(2, dtype=np.float16))
    np.testing.assert_allclose(u, [0.6, 0.8], atol=1e-3)


def test_neuron_scores_float16_does_not_overflow() -> None:
    u = np.array([1.0, 1.0], dtype=np.float16)
    w = np.full((2, 3), 300.0, dtype=np.float16)
    act = np.full(3, 300.0, dtype=np.float16)
    np.testing.assert_allclose(neuron_scores(u, w, act), [180000.0] * 3)


def test_neuron_scores_act_contrast_error_shows_actual_shape() -> None:
    with pytest.raises(ValueError, match=r"act_contrast must be \[6\]; got \(1,\)"):
        neuron_scores(np.ones(4), np.ones((4, 6)), np.ones(1))


def test_stability_does_not_assume_unit_u() -> None:
    diffs = np.array([[2.0, 0.0], [0.0, 3.0], [-1.0, 1.0]])
    expected = stability(np.array([1.0, 0.0]), diffs)
    assert stability(np.array([5.0, 0.0]), diffs) == pytest.approx(expected)
    assert stability(np.array([0.2, 0.0]), diffs) == pytest.approx(expected)


def test_stability_float16_does_not_overflow() -> None:
    u = np.array([300.0, 0.0], dtype=np.float16)
    diffs = np.array([[300.0, 0.0], [0.0, 300.0]], dtype=np.float16)
    assert stability(u, diffs) == pytest.approx(0.5)


def test_stability_of_a_zero_row_counts_as_zero() -> None:
    u = np.array([1.0, 0.0])
    diffs = np.array([[1.0, 0.0], [0.0, 0.0]])
    assert stability(u, diffs) == pytest.approx(0.5)


def test_directions_from_pairs_accepts_one_pair() -> None:
    a = np.array([[[3.0, 4.0]]])  # [P=1, L=1, H=2]
    out = directions_from_pairs(a, np.zeros_like(a))
    np.testing.assert_allclose(out, [[0.6, 0.8]], atol=1e-6)


def test_directions_from_pairs_zero_pairs_message_is_exact() -> None:
    with pytest.raises(ValueError, match=r"^need at least 1 pair; got 0$"):
        directions_from_pairs(np.zeros((0, 4, 5)), np.zeros((0, 4, 5)))


def test_gate_defaults_are_seed_0_and_20_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(9)
    diffs = 0.3 * rng.standard_normal(12) + rng.standard_normal((30, 12))
    assert direction_gate(diffs) == direction_gate(diffs, seed=0, draws=20)

    calls = []
    real_unit = core._unit

    def spy(v: np.ndarray) -> np.ndarray:
        calls.append(1)
        return real_unit(v)

    monkeypatch.setattr(core, "_unit", spy)
    direction_gate(diffs)
    default_calls = len(calls)
    calls.clear()
    direction_gate(diffs, draws=20)
    assert default_calls == len(calls)  # one _unit per shuffle: default draws is 20
    calls.clear()
    direction_gate(diffs, draws=21)
    assert len(calls) == default_calls + 1


def test_gate_accepts_exactly_four_pairs() -> None:
    r = direction_gate(np.random.default_rng(0).standard_normal((4, 6)))
    assert isinstance(r, GateResult)


def test_gate_non_finite_message_is_exact() -> None:
    d = np.ones((6, 3))
    d[0, 0] = np.nan
    with pytest.raises(
        ValueError, match=r"^pair_diffs must be finite \(no NaN or inf\)$"
    ):
        direction_gate(d)


def test_gate_is_scale_invariant() -> None:
    _, diffs = _planted()
    base = direction_gate(diffs, seed=2)
    for scale in (1e-3, 7.0, 1e4):
        got = direction_gate(diffs * scale, seed=2)
        for name in ("split_half_cos", "null_975", "heldout_auc", "per_pair_cos"):
            assert getattr(got, name) == pytest.approx(getattr(base, name), abs=1e-9)


def test_gate_float16_input_does_not_overflow() -> None:
    diffs = np.full((6, 2), 300.0, dtype=np.float16)
    r = direction_gate(diffs)
    assert r.split_half_cos == pytest.approx(1.0)
    assert r.per_pair_cos == pytest.approx(1.0)


def test_gate_null_uses_a_sign_shuffle_that_can_cancel() -> None:
    # One-dimensional identical pairs: a +-1 sign shuffle cancels exactly (null 0) when
    # half the signs flip; any other sign set (say {-1, 2}) never cancels.
    d = np.ones((4, 1))
    nulls = [direction_gate(d, seed=s, draws=2).null_975 for s in range(20)]
    assert 0.0 in nulls
    assert 1.0 in nulls


def test_gate_null_975_is_a_high_quantile_not_the_minimum() -> None:
    # With 20 draws null_975 is the 19th smallest: 0 only if >= 19 of 20 shuffles cancel.
    d = np.ones((4, 1))
    nulls = [direction_gate(d, seed=s, draws=20).null_975 for s in range(10)]
    assert nulls == [1.0] * 10


def test_gate_null_975_index_is_int_draws_times_0975_minus_one() -> None:
    # draws=3 -> index 1 (the median of three), not index 0. Four identical 1-D pairs
    # cancel with probability 3/8 per shuffle, so null_975 is 1 with probability ~0.68
    # at index 1 but ~0.24 at index 0. Statistical bounds, not a pinned stream.
    d = np.ones((4, 1))
    nulls = [direction_gate(d, seed=s, draws=3).null_975 for s in range(100)]
    assert nulls.count(1.0) >= 50


def test_gate_zero_half_b_only_makes_split_half_cos_nan() -> None:
    x = np.array([1.0, 0.0])
    diffs = np.stack([x, x, x, -x])  # the half that holds -x averages to zero
    seen = 0
    for seed in range(20):
        r = direction_gate(diffs, seed=seed)
        if not np.isnan(r.heldout_auc):  # half A is non-zero, so half B is the zero one
            seen += 1
            assert np.isnan(r.split_half_cos)
    assert seen > 0


def test_gate_null_975_is_nan_only_when_the_full_mean_is_zero() -> None:
    diffs = np.array([[0.5, 0.0], [0.5, 0.0], [0.5, 0.0], [0.5, 0.0]])
    assert not np.isnan(direction_gate(diffs).null_975)  # mean norm 0.5, not zero
    assert np.isnan(direction_gate(np.array([[1.0], [-1.0], [1.0], [-1.0]])).null_975)


def test_gate_orthogonal_pair_does_not_count_as_positive() -> None:
    x, y = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    diffs = np.stack([x, x, x, y])
    aucs = {direction_gate(diffs, seed=s).heldout_auc for s in range(20)}
    assert 0.5 in aucs  # half A = {x, x}, half B = {x, y}: y has zero projection
