"""Convert a dense dW matrix to a LowRankDelta with a known error."""

import numpy as np

from astro_weights.edits.core import LowRankDelta


def from_dense(
    module: str,
    dW: np.ndarray,
    rank: int,
    max_rel_residual: float | None = None,
) -> tuple[LowRankDelta, float]:
    """Truncated SVD of dW to `rank`; returns (delta, rel_residual).

    rel_residual = |dW - B @ A|_F / |dW|_F, measured on the stored float32 factors.
    Raises ValueError if it exceeds `max_rel_residual`.
    """
    if dW.ndim != 2:
        raise ValueError(f"{module}: dW must be 2-D, got {dW.ndim}-D")
    w = np.asarray(dW, dtype=np.float64)
    if not np.isfinite(w).all():
        raise ValueError(f"{module}: dW must be finite")
    norm = float(np.linalg.norm(w))
    if norm == 0.0:
        raise ValueError(f"{module}: dW is all zero")
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise ValueError(f"{module}: rank must be an int, got {rank!r}")  # noqa: TRY004 (spec: ValueError)
    if not 1 <= rank <= min(w.shape):
        raise ValueError(f"{module}: rank must be in [1, {min(w.shape)}], got {rank}")
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    delta = LowRankDelta(module, u[:, :rank] * s[:rank], vt[:rank])
    approx = delta.B.astype(np.float64) @ delta.A.astype(np.float64)
    rel = float(np.linalg.norm(w - approx)) / norm
    if max_rel_residual is not None and rel > max_rel_residual:
        raise ValueError(
            f"{module}: rel_residual {rel:.4g} exceeds max_rel_residual "
            f"{max_rel_residual:.4g} at rank {rank}"
        )
    return delta, rel
