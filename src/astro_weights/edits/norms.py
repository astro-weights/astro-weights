"""Edit norm range check: a cheap pre-screen before any eval."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from astro_weights.edits.core import Edit, LowRankDelta


def delta_norm(d: LowRankDelta) -> float:
    """|BA|_F without the dense matrix: sqrt(trace((B^T B)(A A^T))), in float64."""
    b, a = d.B.astype(np.float64), d.A.astype(np.float64)
    return float(np.sqrt(max(np.sum((b.T @ b) * (a @ a.T)), 0.0)))


def edit_norms(edit: Edit) -> dict[str, float]:
    """`delta_norm` per module, keyed by the edit's module name."""
    return {d.module: delta_norm(d) for d in edit.deltas}


@dataclass(frozen=True)
class NormCheck:
    """Result of `check_norms`: `outside` names the modules out of range."""

    norms: dict[str, float]
    outside: tuple[str, ...]
    passed: bool


def _range(r: tuple[float, float], key: str) -> tuple[float, float]:
    lo, hi = float(r[0]), float(r[1])
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi):
        raise ValueError(f"{key}: range must be finite with 0 <= low <= high, got {r}")
    return lo, hi


def _lookup(ranges: dict[str, tuple[float, float]], module: str) -> tuple[float, float]:
    """The range for `module`: exact name, else the longest matching suffix."""
    if module in ranges:
        return ranges[module]
    hits = [k for k in ranges if module.endswith("." + k)]
    if not hits:
        raise ValueError(f"{module}: no norm range matches it")
    return ranges[max(hits, key=len)]


def check_norms(
    edit: Edit,
    ranges: Mapping[str, tuple[float, float]] | tuple[float, float],
) -> NormCheck:
    """Check each module's ||BA||_F against an inclusive `(low, high)` range.

    `ranges` is one range for every module, or a mapping from a module name or
    a dotted suffix (`"mlp.down_proj"`) to its range; an exact name beats a
    suffix, a longer suffix beats a shorter one. A module that no key matches
    raises `ValueError`, as does a range that is not finite, negative or
    reversed. The ranges are data (what a sane edit looks like for the model),
    not defaults of the library.
    """
    if isinstance(ranges, tuple):
        one = _range(ranges, "ranges")
        table: dict[str, tuple[float, float]] = {}
    else:
        one = None
        table = {k: _range(v, k) for k, v in ranges.items()}
    norms = edit_norms(edit)
    outside = []
    for module, n in norms.items():
        lo, hi = one or _lookup(table, module)
        if not lo <= n <= hi:
            outside.append(module)
    return NormCheck(norms, tuple(outside), not outside)
