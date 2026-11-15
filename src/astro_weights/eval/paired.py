"""Paired comparison of two eval arms (numpy only)."""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from astro_weights.eval.records import Record

_Key = tuple[str, int]
_SMALL_N = 100
_BOOT_CHUNK = 500  # bootstrap rows drawn at a time; bounds memory at chunk * n ints
_Z = 1.96  # two-sided 5%
_Z_POWER = 0.84  # 80% power


@dataclass(frozen=True)
class PairedResult:
    """Paired statistics; scores and token counts are per-item means over seeds."""

    n_items: int
    mean_base: float
    mean_edit: float
    diff: float
    se: float
    ci95: tuple[float, float]
    correlation: float
    mcnemar_p: float | None
    n_base_only: int
    n_edit_only: int
    tokens_base: float
    tokens_edit: float
    tokens_diff: float
    tokens_diff_pct: float
    tokens_ci95: tuple[float, float]
    cap_hit_base: float
    cap_hit_edit: float
    small_n: bool
    score_mde: (
        float  # minimum detectable score difference at this n (5% test, 80% power)
    )
    tokens_se: float
    tokens_mde: float


def _index(records: Sequence[Record], arm: str) -> dict[_Key, Record]:
    out: dict[_Key, Record] = {}
    for r in records:
        key = (r.item_id, r.seed)
        if key in out:
            raise ValueError(f"{arm} arm repeats (item_id, seed) = {key!r}")
        out[key] = r
    return out


def pair(base: Sequence[Record], edit: Sequence[Record]) -> list[tuple[Record, Record]]:
    """Join two arms on (item_id, seed), sorted; raise if they do not line up."""
    b, e = _index(base, "base"), _index(edit, "edit")
    if b.keys() != e.keys():
        only_b, only_e = sorted(b.keys() - e.keys()), sorted(e.keys() - b.keys())
        raise ValueError(
            "arms do not have the same (item_id, seed) keys: "
            f"{len(only_b)} only in base (e.g. {only_b[:3]}), "
            f"{len(only_e)} only in edit (e.g. {only_e[:3]})"
        )
    pairs = [(b[k], e[k]) for k in sorted(b)]
    bad = [p[0].item_id for p in pairs if p[0].prompt_hash != p[1].prompt_hash]
    if bad:
        raise ValueError(
            f"arms were not given the same prompts: prompt_hash differs for {len(bad)} pair(s), "
            f"e.g. item_id {bad[0]!r}"
        )
    return pairs


def _per_item(
    pairs: list[tuple[Record, Record]],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Mean over seeds per item: base scores, edit scores, base tokens, edit tokens."""
    acc: dict[str, list[list[float]]] = defaultdict(lambda: [[], [], [], []])
    for b, e in pairs:
        cols = acc[b.item_id]
        cols[0].append(b.score)
        cols[1].append(e.score)
        cols[2].append(float(b.n_output_tokens))
        cols[3].append(float(e.n_output_tokens))
    rows = [[sum(c) / len(c) for c in acc[k]] for k in sorted(acc)]
    arr = np.array(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def _paired_se(d: npt.NDArray[np.float64]) -> float:
    if d.size < 2:
        return math.nan
    return float(d.std(ddof=1) / math.sqrt(d.size))


def _pearson(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial (p = 0.5) p-value on b, c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    # Sum P(X = k), k <= m, downward from k = m: terms shrink geometrically, so stop once negligible.
    term = math.exp(
        math.lgamma(n + 1)
        - math.lgamma(m + 1)
        - math.lgamma(n - m + 1)
        - n * math.log(2)
    )
    tail = 0.0
    for k in range(m, 0, -1):
        tail += term
        term *= k / (n - k + 1)
        if term < tail * 1e-18:
            break
    else:
        tail += term
    return float(min(1.0, 2 * tail))


def compare(
    base: Sequence[Record], edit: Sequence[Record], n_boot: int = 10000, seed: int = 0
) -> PairedResult:
    """Paired comparison of two arms; scores and tokens are averaged over seeds per item first."""
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    pairs = pair(base, edit)
    if not pairs:
        raise ValueError("no records to compare")
    sb, se_, tb, te = _per_item(pairs)
    if not (np.isfinite(sb).all() and np.isfinite(se_).all()):
        raise ValueError(
            "scores must be finite (found NaN or inf); grade all records first"
        )
    n = sb.size
    d = se_ - sb
    diff = float(d.mean())
    se = _paired_se(d)

    binary = bool(np.isin(np.concatenate([sb, se_]), (0.0, 1.0)).all())
    n_base_only = int(((sb == 1) & (se_ == 0)).sum()) if binary else 0
    n_edit_only = int(((sb == 0) & (se_ == 1)).sum()) if binary else 0

    dt = te - tb
    rng = np.random.default_rng(seed)
    # Seed fixes the result for a given _BOOT_CHUNK (draws are made chunk by chunk).
    boot = np.concatenate(
        [
            dt[rng.integers(0, n, size=(min(_BOOT_CHUNK, n_boot - i), n))].mean(axis=1)
            for i in range(0, n_boot, _BOOT_CHUNK)
        ]
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    tokens_base = float(tb.mean())
    tokens_diff = float(dt.mean())
    tokens_se = _paired_se(dt)

    return PairedResult(
        n_items=n,
        mean_base=float(sb.mean()),
        mean_edit=float(se_.mean()),
        diff=diff,
        se=se,
        ci95=(diff - 1.96 * se, diff + 1.96 * se),
        correlation=_pearson(sb, se_),
        mcnemar_p=_mcnemar_exact(n_base_only, n_edit_only) if binary else None,
        n_base_only=n_base_only,
        n_edit_only=n_edit_only,
        tokens_base=tokens_base,
        tokens_edit=float(te.mean()),
        tokens_diff=tokens_diff,
        tokens_diff_pct=100 * tokens_diff / tokens_base if tokens_base else math.nan,
        tokens_ci95=(float(lo), float(hi)),
        cap_hit_base=sum(b.hit_cap for b, _ in pairs) / len(pairs),
        cap_hit_edit=sum(e.hit_cap for _, e in pairs) / len(pairs),
        small_n=n < _SMALL_N,
        score_mde=(_Z + _Z_POWER) * se,
        tokens_se=tokens_se,
        tokens_mde=(_Z + _Z_POWER) * tokens_se,
    )


def _excludes_zero(ci: tuple[float, float]) -> bool:
    return ci[0] > 0 or ci[1] < 0


def _beyond(diff: float, floor_diff: float, floor_se: float) -> bool:
    return abs(diff) > abs(floor_diff) + _Z * floor_se  # False if any is nan


def exceeds_floor(result: PairedResult, floor: PairedResult) -> dict[str, bool]:
    """Whether `result` is significant and larger than the base-vs-base noise `floor`.

    `floor` is `compare(base_run_1, base_run_2)`: the same arm run twice under the same
    settings. True for a metric when its CI excludes 0 and |diff| > |floor diff| + 1.96
    floor SE. NaN anywhere gives False.
    """
    return {
        "score": _excludes_zero(result.ci95)
        and _beyond(result.diff, floor.diff, floor.se),
        "tokens": _excludes_zero(result.tokens_ci95)
        and _beyond(result.tokens_diff, floor.tokens_diff, floor.tokens_se),
    }


def _mde_note(ci: tuple[float, float], mde: float, fmt: str) -> str:
    if ci[0] <= 0 <= ci[1] and math.isfinite(mde):
        return f"; not significant; smallest detectable difference at this n is {mde:{fmt}}"
    return ""


def _floor_line(
    name: str, diff: float, floor_diff: float, floor_se: float, exceeds: bool, fmt: str
) -> str:
    bar = abs(floor_diff) + _Z * floor_se
    verdict = "exceeds" if exceeds else "does not exceed"
    return (
        f"noise floor ({name}): |diff|={abs(diff):{fmt}} {verdict} "
        f"|floor diff| + 1.96 SE = {bar:{fmt}}"
    )


def format_report(r: PairedResult, floor: PairedResult | None = None) -> str:
    """A few plain lines summarising a `PairedResult`; `floor` adds noise-floor verdicts."""
    lines = [
        f"n={r.n_items} items  base={r.mean_base:.4f}  edit={r.mean_edit:.4f}",
        f"score diff={r.diff:+.4f}  SE={r.se:.4f}  95% CI [{r.ci95[0]:+.4f}, {r.ci95[1]:+.4f}]"
        + _mde_note(r.ci95, r.score_mde, ".4f"),
    ]
    if r.mcnemar_p is not None:
        lines.append(
            f"McNemar exact p={r.mcnemar_p:.4g}  "
            f"(base-only right={r.n_base_only}, edit-only right={r.n_edit_only})"
        )
    lines += [
        (
            f"tokens: base={r.tokens_base:.1f}  edit={r.tokens_edit:.1f}  "
            f"diff={r.tokens_diff:+.1f} ({r.tokens_diff_pct:+.1f}%)  "
            f"bootstrap 95% CI [{r.tokens_ci95[0]:+.1f}, {r.tokens_ci95[1]:+.1f}]"
            + _mde_note(r.tokens_ci95, r.tokens_mde, ".1f")
        ),
        f"cap-hit rate: base={r.cap_hit_base:.3f}  edit={r.cap_hit_edit:.3f}",
    ]
    if floor is not None:
        v = exceeds_floor(r, floor)
        lines.append(
            _floor_line("score", r.diff, floor.diff, floor.se, v["score"], ".4f")
        )
        lines.append(
            _floor_line(
                "tokens",
                r.tokens_diff,
                floor.tokens_diff,
                floor.tokens_se,
                v["tokens"],
                ".1f",
            )
        )
    if r.small_n:
        lines.append(
            f"WARNING: only {r.n_items} items (<{_SMALL_N}); intervals are unreliable"
        )
    return "\n".join(lines)
