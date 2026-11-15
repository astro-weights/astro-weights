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
    """Normalize in float64; a zero vector stays zero.

    Vectors with norm below ``_EPS`` (1e-12) are not unit length, like torch ``F.normalize``.
    """
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), _EPS)


def direction_from_means(mean_a: np.ndarray, mean_b: np.ndarray) -> np.ndarray:
    """Unit vector ``normalize(mean_a - mean_b)``, float32; points b -> a."""
    a = np.asarray(mean_a, dtype=np.float64)
    b = np.asarray(mean_b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError(
            f"mean_a and mean_b must be 1-D and equal shape; got {a.shape}, {b.shape}"
        )
    d = a - b
    norm = float(np.linalg.norm(d))
    if not norm > 0:
        raise ValueError("mean_a and mean_b are equal: no direction")
    return (d / norm).astype(np.float32)


def neuron_scores(
    u: np.ndarray, W_down: np.ndarray, act_contrast: np.ndarray
) -> np.ndarray:
    """``(u @ W_down) * act_contrast``, shape ``[intermediate]``.

    ``act_contrast`` is the mean difference of the ``down_proj`` input, ``[intermediate]``.
    """
    u64 = np.asarray(u, dtype=np.float64)
    if W_down.ndim != 2 or u64.shape != (W_down.shape[0],):
        raise ValueError(
            f"u must be [hidden] for W_down {W_down.shape}; got {u64.shape}"
        )
    if np.shape(act_contrast) != (W_down.shape[1],):
        raise ValueError(
            f"act_contrast must be [{W_down.shape[1]}]; got {np.shape(act_contrast)}"
        )
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
    if res_a.shape[0] < 1:
        raise ValueError("need at least 1 pair; got 0")
    rows = []
    for i in range(res_a.shape[1]):
        try:
            rows.append(
                direction_from_means(res_a[:, i].mean(axis=0), res_b[:, i].mean(axis=0))
            )
        except ValueError as e:
            raise ValueError(f"layer {i}: {e}") from e
    return np.stack(rows)


@dataclass(frozen=True)
class GateResult:
    """Statistics of :func:`direction_gate`.

    ``split_half_cos``: cosine between the mean directions of two random halves.
    ``null_975``: upper quantile of ``|cos|`` for sign-shuffled pairs vs the full direction.
    It is element ``int(0.975 * draws) - 1`` of the sorted draws, so with the default 20
    draws it is the 19th of 20 values (about the 95th percentile, not a true 97.5th).
    ``heldout_auc``: fraction of half-B pairs with positive projection on the half-A direction.
    ``per_pair_cos``: mean cosine of half-B pairs with the half-A direction.

    A statistic is NaN when a mean it needs has zero norm (for example the half-A mean).
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
    cosine floor is ``1/sqrt(H)``. Needs ``draws >= 1`` and finite input. Results with fewer
    than about 20 pairs are not meaningful: the halves are tiny and ``null_975`` is near 1.
    """
    d = np.asarray(pair_diffs, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"pair_diffs must be [P, H]; got shape {d.shape}")
    if draws < 1:
        raise ValueError(f"draws must be >= 1; got {draws}")
    if not np.isfinite(d).all():
        raise ValueError("pair_diffs must be finite (no NaN or inf)")
    n = d.shape[0]
    if n < 4:
        raise ValueError(f"need at least 4 pairs; got {n}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    half_a, half_b = d[perm[: n // 2]], d[perm[n // 2 :]]
    mean_a, mean_b, mean = half_a.mean(axis=0), half_b.mean(axis=0), d.mean(axis=0)
    ua, ub, u = _unit(mean_a), _unit(mean_b), _unit(mean)
    nulls = []
    for _ in range(draws):
        signs = rng.integers(0, 2, size=(n, 1)) * 2 - 1
        nulls.append(abs(float(_unit((d * signs).mean(axis=0)) @ u)))
    null = sorted(nulls)[max(int(0.975 * draws) - 1, 0)]
    nan = float("nan")
    zero_a, zero_b = not np.linalg.norm(mean_a) > 0, not np.linalg.norm(mean_b) > 0
    # Pairs are oriented, so AUC is the fraction with positive projection.
    return GateResult(
        split_half_cos=nan if zero_a or zero_b else float(ua @ ub),
        null_975=nan if not np.linalg.norm(mean) > 0 else null,
        heldout_auc=nan if zero_a else float(((half_b @ ua) > 0).mean()),
        per_pair_cos=nan if zero_a else float(_cosines(half_b, ua).mean()),
    )
