"""Edit recipes that produce rank-1 LowRankDeltas (numpy only)."""

import numpy as np

from astro_weights.edits.core import LowRankDelta


def rank1(module: str, b: np.ndarray, a: np.ndarray) -> LowRankDelta:
    """dW = outer(b, a)."""
    return LowRankDelta(module, b[:, None], a[None, :])


def top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """float32 0/1 vector with ones at the k largest scores."""
    mask = np.zeros(len(scores), dtype=np.float32)
    if k > 0:
        mask[np.argsort(scores)[-k:]] = 1.0
    return mask


def masked_rank1(
    module: str, W0: np.ndarray, u: np.ndarray, mask: np.ndarray, alpha: float
) -> LowRankDelta:
    """Low-rank form of W = W0 - alpha * outer(u, mask * (u @ W0))."""
    if W0.ndim != 2 or u.shape != (W0.shape[0],) or mask.shape != (W0.shape[1],):
        raise ValueError(
            f"{module}: shape mismatch, W0 {W0.shape}, u {u.shape}, mask {mask.shape}"
        )
    u64 = u.astype(np.float64)
    a = -alpha * mask.astype(np.float64) * (u64 @ W0.astype(np.float64))
    return LowRankDelta(module, u64[:, None], a[None, :])


def column_shift(
    module: str, shift: np.ndarray, idx: np.ndarray, coef: np.ndarray, n_in: int
) -> LowRankDelta:
    """Bake a constant shift into input columns: W = W0 + outer(shift, c), c[idx] = coef."""
    if shift.ndim != 1 or idx.ndim != 1 or coef.shape != idx.shape:
        raise ValueError(
            f"{module}: shift must be 1-D; idx and coef must be 1-D, same length"
        )
    if len(idx) and (idx.min() < 0 or idx.max() >= n_in):
        raise ValueError(f"{module}: idx out of range for n_in={n_in}")
    if len(np.unique(idx)) != len(idx):
        raise ValueError(f"{module}: idx must be unique")
    c = np.zeros(n_in, dtype=np.float64)
    c[idx] = coef
    return LowRankDelta(module, shift[:, None], c[None, :])


def orthogonal_random(u: np.ndarray, seed: int) -> np.ndarray:
    """Seeded unit vector orthogonal to u (random-edit control)."""
    u64 = u.astype(np.float64)
    r = np.random.default_rng(seed).standard_normal(len(u64))
    r -= (r @ u64) / (u64 @ u64) * u64
    return r / np.linalg.norm(r)
