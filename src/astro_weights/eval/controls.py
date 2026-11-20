"""The control rule: an edit passes only if it beats random controls on the target.

Controls are paired results of random edits of the same size (PDR 0.3 section 4.4). The
target passes when the edit's effect is strictly better, in the wanted direction, than the
`q`-quantile of the controls' effects (empirical nearest-rank quantile, no parametric test).
Each side effect passes when the edit's `diff` lies inside `[min, max]` of the controls'
`diff` values (inclusive).

Ties: the target test is strict, so an edit that only equals the quantile fails. Side-effect
bounds are inclusive, so an edit equal to the control min or max is inside.

This module judges the edit against random controls only. Control-vs-control checks (one
leave-one-out control always wins somewhere) belong to W5, not here.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from astro_weights.eval.paired import PairedResult

Direction = Literal["higher", "lower"]


@dataclass(frozen=True)
class ControlVerdict:
    """Outcome of `control_rule`.

    `target_rank` is how many controls the edit strictly beats on the target (out of
    `n_controls`); `target_quantile` is the controls' `q`-quantile of goodness, given in the
    metric's own units (for `direction="lower"` that is the `1 - q` quantile of the raw values). `side_effects` maps each side-effect name to True when the edit
    is inside the controls' range. `outside` names the ones that are not.
    """

    passed: bool
    target_beats_quantile: bool
    target_rank: int
    target_quantile: float
    q: float
    n_controls: int
    side_effects: Mapping[str, bool]
    outside: tuple[str, ...]


def min_controls(q: float) -> int:
    """Fewest controls for the `q`-quantile to mean anything: 20 for q=0.95."""
    return math.ceil(1 / (1 - q) - 1e-9)


def _values(results: Sequence[PairedResult], metric: str, what: str) -> list[float]:
    out: list[float] = []
    for r in results:
        v = getattr(r, metric, None)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"{what}: {metric!r} is not a numeric PairedResult field")  # noqa: TRY004 (caller arg, not a type bug)
        out.append(float(v))
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{what}: {metric} is NaN or infinite")
    return out


def _check_n(n: int, q: float, what: str) -> None:
    need = min_controls(q)
    if n < need:
        raise ValueError(f"{what}: {n} controls, need at least {need} for q={q}")


def control_rule(
    edit: PairedResult,
    controls: Sequence[PairedResult],
    *,
    target: str,
    direction: Direction,
    side_effects: Mapping[str, tuple[PairedResult, Sequence[PairedResult]]]
    | None = None,
    q: float = 0.95,
) -> ControlVerdict:
    """Apply the control rule.

    `target` names the `PairedResult` field to judge (`"diff"` for accuracy, `"tokens_diff"`
    for length); `direction` says which way is better (`"higher"` or `"lower"`). Neither has a
    default. `side_effects` maps a name to the edit's result and its controls' results on
    that side metric; their `diff` is checked against the controls' range. Raises
    `ValueError` for `q` outside (0, 1), fewer than `min_controls(q)` controls (target or
    any side effect), or NaN/infinite values.
    """
    if not 0 < q < 1:
        raise ValueError(f"q must be in (0, 1), got {q}")
    if direction not in ("higher", "lower"):
        raise ValueError(f"direction must be 'higher' or 'lower', got {direction!r}")
    _check_n(len(controls), q, "target")
    sign = 1.0 if direction == "higher" else -1.0
    (edit_v,) = _values([edit], target, "edit")
    ctrl = _values(controls, target, "controls")
    good = sorted(sign * v for v in ctrl)
    quantile = good[math.ceil(q * len(good) - 1e-9) - 1]
    rank = sum(g < sign * edit_v for g in good)
    beats = sign * edit_v > quantile

    sides: dict[str, bool] = {}
    for name, (s_edit, s_ctrls) in side_effects.items() if side_effects else ():
        _check_n(len(s_ctrls), q, f"side effect {name!r}")
        (e,) = _values([s_edit], "diff", f"side effect {name!r} edit")
        c = _values(s_ctrls, "diff", f"side effect {name!r} controls")
        sides[name] = min(c) <= e <= max(c)
    outside = tuple(n for n, inside in sides.items() if not inside)
    return ControlVerdict(
        passed=beats and not outside,
        target_beats_quantile=beats,
        target_rank=rank,
        target_quantile=sign * quantile,
        q=q,
        n_controls=len(controls),
        side_effects=sides,
        outside=outside,
    )
