"""Difference-of-means directions and the split-half gate (numpy only).

Orientation: a direction ``u`` is ``[hidden]`` (residual space). ``W_down`` is
``mlp.down_proj.weight``, ``[hidden, intermediate]``, so ``u @ W_down`` is
``[intermediate]``. Directions point from set ``b`` to set ``a``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


def _unit(v: np.ndarray) -> np.ndarray:
    """Normalize in float64; a zero vector stays zero."""
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), _EPS)


def direction_from_means(mean_a: np.ndarray, mean_b: np.ndarray) -> np.ndarray:
    """Unit vector ``normalize(mean_a - mean_b)``, float32; points b -> a."""
    d = np.asarray(mean_a, dtype=np.float64) - np.asarray(mean_b, dtype=np.float64)
    if not np.linalg.norm(d) > 0:
        raise ValueError("mean_a and mean_b are equal: no direction")
    return _unit(d).astype(np.float32)


def neuron_scores(
    u: np.ndarray, W_down: np.ndarray, act_contrast: np.ndarray
) -> np.ndarray:
    """``(u @ W_down) * act_contrast``, shape ``[intermediate]``.

    ``act_contrast`` is the mean difference of the ``down_proj`` input, ``[intermediate]``.
    """
    u64 = np.asarray(u, dtype=np.float64)
    scores = (u64 @ np.asarray(W_down, dtype=np.float64)) * np.asarray(
        act_contrast, dtype=np.float64
    )
    return scores.astype(np.float32)


def _cosines(rows: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Cosine of each row of ``rows`` with ``u``; zero rows give 0."""
    rows = np.asarray(rows, dtype=np.float64)
    norms = np.maximum(np.linalg.norm(rows, axis=-1), _EPS)
    cos: np.ndarray = (rows @ _unit(u)) / norms
    return cos


def stability(u: np.ndarray, pair_diffs: np.ndarray) -> float:
    """Mean cosine of each row of ``pair_diffs`` ``[n_pairs, hidden]`` with ``u``."""
    return float(_cosines(pair_diffs, u).mean())


def directions_from_pairs(res_a: np.ndarray, res_b: np.ndarray) -> np.ndarray:
    """Per-layer directions ``[L, H]`` from per-pair residual means ``[P, L, H]``."""
    if res_a.ndim != 3 or res_a.shape != res_b.shape:
        raise ValueError(
            f"res_a and res_b must share shape [P, L, H]; got {res_a.shape}, {res_b.shape}"
        )
    return np.stack(
        [
            direction_from_means(res_a[:, i].mean(axis=0), res_b[:, i].mean(axis=0))
            for i in range(res_a.shape[1])
        ]
    )


@dataclass(frozen=True)
class GateResult:
    """Statistics of :func:`direction_gate`.

    ``split_half_cos``: cosine between the mean directions of two random halves.
    ``null_975``: 97.5th percentile of ``|cos|`` for sign-shuffled pairs vs the full direction.
    ``heldout_auc``: fraction of half-B pairs with positive projection on the half-A direction.
    ``per_pair_cos``: mean cosine of half-B pairs with the half-A direction.
    """

    split_half_cos: float
    null_975: float
    heldout_auc: float
    per_pair_cos: float


def direction_gate(
    pair_diffs: np.ndarray, seed: int = 0, draws: int = 20
) -> GateResult:
    """Is the direction of one layer real? ``pair_diffs`` is ``[P, H]``, pairs oriented a - b.

    Pure noise gives split-half cosine near 0 and AUC near 0.5; the random-direction
    cosine floor is ``1/sqrt(H)``.
    """
    d = np.asarray(pair_diffs, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"pair_diffs must be [P, H]; got shape {d.shape}")
    n = d.shape[0]
    if n < 4:
        raise ValueError(f"need at least 4 pairs; got {n}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    half_a, half_b = d[perm[: n // 2]], d[perm[n // 2 :]]
    ua, ub = _unit(half_a.mean(axis=0)), _unit(half_b.mean(axis=0))
    u = _unit(d.mean(axis=0))
    nulls = []
    for _ in range(draws):
        signs = rng.integers(0, 2, size=(n, 1)) * 2 - 1
        nulls.append(abs(float(_unit((d * signs).mean(axis=0)) @ u)))
    null = sorted(nulls)[max(int(0.975 * draws) - 1, 0)]
    # Pairs are oriented, so AUC is the fraction with positive projection.
    auc = float(((half_b @ ua) > 0).mean())
    return GateResult(
        split_half_cos=float(ua @ ub),
        null_975=null,
        heldout_auc=auc,
        per_pair_cos=float(_cosines(half_b, ua).mean()),
    )
