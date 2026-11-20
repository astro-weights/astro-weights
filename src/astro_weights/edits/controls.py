"""Norm-matched random control edits (numpy only).

A control has the same modules, shapes and rank as the edit it mirrors, random
seeded factors, and the same per-module `|B @ A|_F`. If an edit beats its
controls, the direction mattered and not only the size of the change.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np

from astro_weights.edits.core import Edit, LowRankDelta

Dtype = Literal["bfloat16", "float16"]

_BISECT_STEPS = 60
_TOL = 1e-3  # relative norm error at which the rounded-norm search stops


def _norm_ba(B: np.ndarray, A: np.ndarray) -> float:
    """|BA|_F without the dense matrix: sqrt(trace((B^T B)(A A^T))), in float64."""
    b, a = B.astype(np.float64), A.astype(np.float64)
    return float(np.sqrt(max(np.sum((b.T @ b) * (a @ a.T)), 0.0)))


def _round(x: np.ndarray, dtype: Dtype) -> np.ndarray:
    """`x` rounded to `dtype` (nearest even), returned as float32."""
    if dtype == "float16":
        with np.errstate(over="ignore"):
            return x.astype(np.float16).astype(np.float32)
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    u = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(
        0xFFFF0000
    )
    return u.view(np.float32)


def _rounded_norm(w0: np.ndarray, delta: np.ndarray, dtype: Dtype) -> float:
    """`|round(W0 + delta) - W0|_F`, the change that survives the dtype."""
    return float(np.linalg.norm(_round(w0 + delta, dtype).astype(np.float64) - w0))


def _scale_for_rounded(
    d: LowRankDelta, B: np.ndarray, A: np.ndarray, w0: np.ndarray, dtype: Dtype
) -> float:
    """Scale `s` so `|round(W0 + s*B@A) - W0|` is within ~0.1% of the original's."""
    w0 = np.asarray(w0, dtype=np.float32)
    if w0.shape != d.shape:
        raise ValueError(f"{d.module}: base weight {w0.shape}, delta {d.shape}")
    w064 = w0.astype(np.float64)
    target = _rounded_norm(w064, d.dense(), dtype)
    if target == 0.0:
        return 0.0
    rand = np.asarray(B @ A, dtype=np.float32)

    def f(s: float) -> float:
        return _rounded_norm(w064, s * rand, dtype)

    lo, hi = 0.0, target / max(float(np.linalg.norm(rand)), 1e-30)
    for _ in range(_BISECT_STEPS):  # f(0) = 0 < target; grow hi until it passes
        if f(hi) >= target:
            break
        lo, hi = hi, hi * 2
    s = hi
    for _ in range(_BISECT_STEPS):
        s = (lo + hi) / 2
        got = f(s)
        if abs(got - target) <= _TOL * target:
            break
        lo, hi = (s, hi) if got < target else (lo, s)
    return s


def _control(
    edit: Edit,
    seed: np.random.SeedSequence,
    dtype: Dtype | None,
    base: Callable[[str], np.ndarray] | None,
) -> Edit:
    if dtype not in (None, "bfloat16", "float16"):
        raise ValueError(f"dtype must be 'bfloat16' or 'float16', got {dtype!r}")
    if base is not None and dtype is None:
        raise ValueError("base needs a dtype to round to")
    rounded = dtype is not None and base is not None
    deltas = []
    for d, ss in zip(edit.deltas, seed.spawn(len(edit.deltas)), strict=True):
        rng = np.random.default_rng(ss)
        # float32 first: LowRankDelta stores float32, so match the norm of what is stored
        B = rng.standard_normal(d.B.shape).astype(np.float32)
        A = rng.standard_normal(d.A.shape).astype(np.float32)
        if rounded:
            assert dtype is not None
            assert base is not None
            s = _scale_for_rounded(d, B, A, base(d.module), dtype)
        else:
            got = _norm_ba(B, A)
            s = _norm_ba(d.B, d.A) / got if got else 0.0
        B = (B * np.float32(s)).astype(np.float32)
        deltas.append(LowRankDelta(d.module, B, A))
    return Edit(
        tuple(deltas),
        edit.base_model,
        dict(edit.base_sha256),
        {
            "random_control_of": dict(edit.meta),
            "norm_match": f"rounded:{dtype}" if rounded else "factor",
        },
    )


def random_like(
    edit: Edit,
    seed: int,
    *,
    dtype: Dtype | None = None,
    base: Callable[[str], np.ndarray] | None = None,
) -> Edit:
    """A random edit with the modules, shapes and rank of `edit` and equal per-module `|B @ A|_F`.

    Factors are seeded standard normals, scaled so each module's Frobenius norm
    of `B @ A` matches the original's (computed without the dense matrix). The
    same seed gives the same edit.

    With `dtype` ("bfloat16" or "float16") and `base` (module name -> W0, as
    float32), the norm is matched after rounding: `|round(W0 + dW) - W0|_F` of
    the control equals the original's within 0.1%, found by bisection on the
    scale. `dtype` without `base` falls back to the factor norm;
    `meta["norm_match"]` says which was used (`"factor"` or `"rounded:<dtype>"`).
    """
    return _control(edit, np.random.SeedSequence(seed), dtype, base)


def random_controls(
    edit: Edit,
    n: int = 20,
    seed: int = 0,
    *,
    dtype: Dtype | None = None,
    base: Callable[[str], np.ndarray] | None = None,
) -> list[Edit]:
    """`n` independent norm-matched random controls of `edit`.

    Child seeds come from `seed`, so the list is reproducible. Each control's
    `meta` has `control` (its index, 0-based), `controls` (n) and
    `random_control_of` (the parent's meta); `dtype` and `base` are as in
    `random_like`.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    out = []
    for i, ss in enumerate(np.random.SeedSequence(seed).spawn(n)):
        c = _control(edit, ss, dtype, base)
        c.meta.update({"control": i, "controls": n})
        out.append(c)
    return out
