"""Edit recipes that produce rank-1 LowRankDeltas (numpy only)."""

import math

import numpy as np

from astro_weights.edits.core import LowRankDelta


def _check_finite(module: str, name: str, x: float) -> None:
    if not math.isfinite(x):
        raise ValueError(f"{module}: {name} must be finite, got {x}")


def _check_index(module: str, name: str, x: object) -> None:
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
        raise ValueError(f"{module}: {name} must be an int, got {x!r}")  # noqa: TRY004 (spec: ValueError)


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


def project_out(
    module: str, W0: np.ndarray, v: np.ndarray, strength: float = 1.0
) -> LowRankDelta:
    """Remove direction v from the layer's output: W = W0 - strength * outer(vhat, vhat @ W0)."""
    if W0.ndim != 2 or v.shape != (W0.shape[0],):
        raise ValueError(f"{module}: shape mismatch, W0 {W0.shape}, v {v.shape}")
    v64 = v.astype(np.float64)
    _check_finite(module, "strength", strength)
    norm = np.linalg.norm(v64)
    if not np.isfinite(norm):
        raise ValueError(f"{module}: v must be finite")
    if norm == 0.0:
        raise ValueError(f"{module}: v must be non-zero")
    vhat = v64 / norm
    a = vhat @ W0.astype(np.float64)
    return LowRankDelta(module, -strength * vhat[:, None], a[None, :])


def scale_column(module: str, W0: np.ndarray, j: int, s: float) -> LowRankDelta:
    """Multiply input column j by s: dW = (s - 1) * outer(W0[:, j], e_j)."""
    if W0.ndim != 2:
        raise ValueError(f"{module}: W0 must be 2-D, got {W0.ndim}-D")
    _check_index(module, "column", j)
    _check_finite(module, "s", s)
    if not 0 <= j < W0.shape[1]:
        raise ValueError(f"{module}: column {j} out of range for n_in={W0.shape[1]}")
    e = np.zeros(W0.shape[1], dtype=np.float64)
    e[j] = 1.0
    b = (s - 1.0) * W0[:, j].astype(np.float64)
    return LowRankDelta(module, b[:, None], e[None, :])


def scale_head(
    module: str, W0: np.ndarray, head: int, n_heads: int, s: float
) -> LowRankDelta:
    """Multiply the input columns of one attention head's block by s."""
    if W0.ndim != 2:
        raise ValueError(f"{module}: W0 must be 2-D, got {W0.ndim}-D")
    _check_index(module, "head", head)
    _check_index(module, "n_heads", n_heads)
    _check_finite(module, "s", s)
    n_in = W0.shape[1]
    if n_heads < 1 or n_in % n_heads != 0:
        raise ValueError(f"{module}: n_in={n_in} not divisible by n_heads={n_heads}")
    if not 0 <= head < n_heads:
        raise ValueError(f"{module}: head {head} out of range for n_heads={n_heads}")
    size = n_in // n_heads
    lo = head * size
    a = np.zeros((size, n_in), dtype=np.float64)
    a[np.arange(size), lo + np.arange(size)] = 1.0
    b = (s - 1.0) * W0[:, lo : lo + size].astype(np.float64)
    return LowRankDelta(module, b, a)


def zero_head(module: str, W0: np.ndarray, head: int, n_heads: int) -> LowRankDelta:
    """Zero one attention head's input columns (scale_head with s = 0)."""
    return scale_head(module, W0, head, n_heads, 0.0)
