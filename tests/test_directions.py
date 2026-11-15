"""Tests for astro_weights.directions."""

import numpy as np
import pytest

from astro_weights.directions import (
    GateResult,
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
