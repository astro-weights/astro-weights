"""One base, many arms (edit, random controls, prompt, scripted) in one report (numpy only)."""

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from astro_weights.eval.paired import PairedResult, compare
from astro_weights.eval.records import Record

_KINDS = ("edit", "control", "prompt", "scripted")


@dataclass(frozen=True)
class ArmsResult:
    """`compare(base, arm)` per arm on the same items, in the order the arms were given.

    `results` and `kinds` share keys (arm names). `kinds[name]` is one of "edit", "control",
    "prompt", "scripted".
    """

    results: dict[str, PairedResult]
    kinds: dict[str, str]

    @property
    def n_arms(self) -> int:
        return len(self.results)

    def controls(self) -> dict[str, PairedResult]:
        """The arms tagged "control", in order."""
        return {n: r for n, r in self.results.items() if self.kinds[n] == "control"}


def _same_items(base: Sequence[Record], arm: str, records: Sequence[Record]) -> None:
    b = {(r.item_id, r.seed) for r in base}
    a = {(r.item_id, r.seed) for r in records}
    if a == b:
        return
    items_b, items_a = {i for i, _ in b}, {i for i, _ in a}
    x: list[object]
    y: list[object]
    if items_a != items_b:
        what, x, y = "item set", sorted(items_b - items_a), sorted(items_a - items_b)
    else:
        what, x, y = "seed set", sorted(b - a), sorted(a - b)
    raise ValueError(
        f"arm {arm!r} {what} differs from base: {len(x)} only in base (e.g. {x[:3]}), "
        f"{len(y)} only in {arm!r} (e.g. {y[:3]})"
    )


def compare_arms(
    base: Sequence[Record],
    arms: Mapping[str, Sequence[Record]],
    kinds: Mapping[str, str] | None = None,
    *,
    n_boot: int = 10000,
    seed: int = 0,
    strata: Mapping[str, str] | None = None,
) -> ArmsResult:
    """Run `compare(base, arm)` for each named arm; raise if an arm's items or seeds differ from base.

    `kinds` maps arm name to "edit", "control", "prompt" or "scripted"; arms it omits are "edit".
    `n_boot`, `seed` and `strata` go to `compare` unchanged.
    """
    if not arms:
        raise ValueError("no arms to compare")
    kinds = kinds or {}
    if unknown := sorted(set(kinds) - set(arms)):
        raise ValueError(f"kinds names arms that were not given: {unknown}")
    if bad := {n: k for n, k in kinds.items() if k not in _KINDS}:
        raise ValueError(f"unknown arm kinds {bad}; use one of {_KINDS}")
    for name, records in arms.items():
        _same_items(base, name, records)
    return ArmsResult(
        {
            n: compare(base, r, n_boot=n_boot, seed=seed, strata=strata)
            for n, r in arms.items()
        },
        {n: kinds.get(n, "edit") for n in arms},
    )


def _finite(x: float) -> float | None:
    return x if math.isfinite(x) else None


def _spread(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _shown(result: ArmsResult, full_controls: bool) -> list[str]:
    ctl = result.controls()
    return [n for n in result.results if full_controls or n not in ctl]


def arms_dict(result: ArmsResult, full_controls: bool = False) -> dict[str, object]:
    """JSON-ready form of `format_arms`; NaN becomes None. Same rows, same order."""
    rows = []
    for n in _shown(result, full_controls):
        r = result.results[n]
        rows.append(
            {
                "arm": n,
                "kind": result.kinds[n],
                "n_items": r.n_items,
                "diff": _finite(r.diff),
                "ci95": [_finite(r.ci95[0]), _finite(r.ci95[1])],
                "ci95_method": r.ci95_method,
                "score_mde": _finite(r.score_mde),
                "tokens_diff": _finite(r.tokens_diff),
                "tokens_mde": _finite(r.tokens_mde),
            }
        )
    out: dict[str, object] = {"n_arms": result.n_arms, "arms": rows}
    if ctl := result.controls():
        out["controls"] = {
            "n": len(ctl),
            "diff": {
                k: _finite(v)
                for k, v in _spread([r.diff for r in ctl.values()]).items()
            },
            "score_mde": {
                k: _finite(v)
                for k, v in _spread([r.score_mde for r in ctl.values()]).items()
            },
        }
    return out


def _spread_text(values: list[float], fmt: str) -> str:
    return "  ".join(f"{k}={v:{fmt}}" for k, v in _spread(values).items())


def format_arms(result: ArmsResult, full_controls: bool = False) -> str:
    """One table, one row per arm: n, score diff, 95% CI, score MDE, tokens diff.

    The MDE is on every row. Controls are summarised below the table (count and min/median/max
    of their diffs and MDEs); `full_controls` also gives each control its own row.
    """
    rows = [("arm", "kind", "n", "diff", "95% CI", "method", "MDE", "tokens diff")]
    for n in _shown(result, full_controls):
        r = result.results[n]
        rows.append(
            (
                n,
                result.kinds[n],
                str(r.n_items),
                f"{r.diff:+.4f}",
                f"[{r.ci95[0]:+.4f}, {r.ci95[1]:+.4f}]",
                r.ci95_method,
                f"{r.score_mde:.4f}",
                f"{r.tokens_diff:+.1f}",
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = [
        "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip()
        for r in rows
    ]
    if ctl := list(result.controls().values()):
        lines.append(
            f"controls: n={len(ctl)}  diff {_spread_text([r.diff for r in ctl], '+.4f')}"
            f"  MDE {_spread_text([r.score_mde for r in ctl], '.4f')}"
        )
    return "\n".join(lines)
