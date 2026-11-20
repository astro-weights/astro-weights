"""Paired comparison of two eval arms (numpy only)."""

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from astro_weights.eval.records import Record

_Key = tuple[str, int]
_SMALL_N = 100
_BOOT_CHUNK = 500  # bootstrap rows drawn at a time; bounds memory at chunk * n ints
_Z = 1.96  # two-sided 5%
_Z_POWER = 0.84  # 80% power
_CAP_WARN_RATE = 0.05  # warn when either arm's cap-hit rate is above this
_CAP_WARN_P = 0.05  # ... or the cap-hit McNemar p-value is below this
_LOOP_TAIL = 400  # characters of a capped output checked for a repeating tail
_LOOP_MAX_PERIOD = 100
_LOOP_MIN_REPEATS = 3


@dataclass(frozen=True)
class PairedResult:
    """Paired statistics; scores and token counts are per-item means over seeds.

    `n_clusters` is the number of distinct `cluster_id`s, or None when records carry none.
    When set, `se`, `ci95`, `tokens_se`, both MDEs and `tokens_ci95` account for clustering,
    and `small_n` also flags fewer than 100 clusters. McNemar always treats items as independent.

    `ci95` is `diff +- 1.96 SE` (`ci95_method` "normal") except for binary scores on
    fewer than 100 independent items with K = 1, where it is Newcombe's interval
    (`ci95_method` "newcombe": no normal approximation, PDR 0.3 section 4.2); `exceeds_floor`
    uses it. `ci95_normal` always holds the `diff +- 1.96 SE` value (None on old constructors).

    The cap fields come last. `cap_hit_p` is the exact McNemar p-value on `hit_cap` between the
    arms (1.0 when no record differs). `n_uncapped`, `uncapped_diff` and `uncapped_tokens_diff`
    cover the items with at least one (item, seed) pair where neither arm hit the cap, using only
    those pairs; the diffs are None when `n_uncapped` is 0. `loops_base` and `loops_edit` count
    capped records whose output ends in one short chunk repeated (see `is_loop`).
    """

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
    n_clusters: int | None = None
    cap_hit_p: float = 1.0
    n_uncapped: int = 0
    uncapped_diff: float | None = None
    uncapped_tokens_diff: float | None = None
    loops_base: int = 0
    loops_edit: int = 0
    k: int = 1
    score_ci95_boot: tuple[float, float] = (math.nan, math.nan)
    sign_test_p: float | None = None
    base_ci95_wilson: tuple[float, float] | None = None
    edit_ci95_wilson: tuple[float, float] | None = None
    newcombe_ci95: tuple[float, float] | None = None
    strata: tuple["StratumResult", ...] = ()
    ci95_normal: tuple[float, float] | None = None
    ci95_method: str = "normal"


@dataclass(frozen=True)
class StratumResult:
    """Paired score statistics on the items of one stratum (see `compare(strata=...)`).

    `ci95_method` says how `ci95` was made: "newcombe" (binary scores, independent items,
    under 100 items: Newcombe's hybrid-score interval for the paired difference, built from
    Wilson intervals), "normal" (the same items at 100 or more: paired SE), or "bootstrap"
    (graded scores, K > 1 or `cluster_id`: percentile bootstrap over items or clusters).
    The Wilson intervals for each arm's score are set only with "newcombe" or "normal".
    `score_mde` is (1.96 + 0.84) * SE, as on `PairedResult`.
    """

    stratum: str
    n_items: int
    mean_base: float
    mean_edit: float
    diff: float
    ci95: tuple[float, float]
    ci95_method: str
    se: float
    score_mde: float
    small_n: bool
    n_base_only: int = 0
    n_edit_only: int = 0
    base_ci95_wilson: tuple[float, float] | None = None
    edit_ci95_wilson: tuple[float, float] | None = None


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


def _clusters(pairs: list[tuple[Record, Record]]) -> list[str] | None:
    """Cluster id per item in sorted item order; None when no record has one."""
    recs = [r for p in pairs for r in p]
    if all(r.cluster_id is None for r in recs):
        return None
    out: dict[str, str] = {}
    for r in recs:
        if r.cluster_id is None:
            raise ValueError(
                "cluster_id is set on some records but not all "
                f"(missing on item_id {r.item_id!r}, seed {r.seed}, {r.arm!r} arm); "
                "set it on every record of both arms or on none"
            )
        prev = out.setdefault(r.item_id, r.cluster_id)
        if prev != r.cluster_id:
            raise ValueError(
                f"item_id {r.item_id!r} has conflicting cluster_id "
                f"({prev!r} vs {r.cluster_id!r}) across arms or seeds"
            )
    return [out[k] for k in sorted(out)]


def _paired_se(d: npt.NDArray[np.float64]) -> float:
    if d.size < 2:
        return math.nan
    return float(d.std(ddof=1) / math.sqrt(d.size))


def _clustered_se(
    d: npt.NDArray[np.float64], g: npt.NDArray[np.intp], n_g: int
) -> float:
    """Cluster-robust SE of mean(d); `g` maps each item to a cluster in 0..n_g-1.

    sqrt(G / (G - 1) * sum_c (sum_{i in c} (d_i - dbar))^2) / n, with G = n_g clusters
    (Miller 2024, arXiv 2411.00640, plus the usual G / (G - 1) small-sample factor). With one
    item per cluster this is exactly std(d, ddof=1) / sqrt(n). NaN for G < 2.
    """
    if n_g < 2:
        return math.nan
    s = np.bincount(g, weights=d - d.mean(), minlength=n_g)
    return float(math.sqrt(n_g / (n_g - 1) * float(s @ s)) / d.size)


def _boot_means(
    x: npt.NDArray[np.float64],
    g: npt.NDArray[np.intp] | None,
    n_g: int,
    n_boot: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Bootstrap means of x, resampling items (g None) or whole clusters."""
    if g is None:
        n = x.size
        # Seed fixes the result for a given _BOOT_CHUNK (draws are made chunk by chunk).
        return np.concatenate(
            [
                x[rng.integers(0, n, size=(min(_BOOT_CHUNK, n_boot - i), n))].mean(
                    axis=1
                )
                for i in range(0, n_boot, _BOOT_CHUNK)
            ]
        )
    sums = np.bincount(g, weights=x, minlength=n_g)
    sizes = np.bincount(g, minlength=n_g).astype(np.float64)
    chunks = []
    for i in range(0, n_boot, _BOOT_CHUNK):
        idx = rng.integers(0, n_g, size=(min(_BOOT_CHUNK, n_boot - i), n_g))
        chunks.append(sums[idx].sum(axis=1) / sizes[idx].sum(axis=1))
    return np.concatenate(chunks)


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


def _sign_counts(d: npt.NDArray[np.float64]) -> tuple[int, int]:
    """(items where base is higher, items where edit is higher) for d = edit - base."""
    return int((d < 0).sum()), int((d > 0).sum())


def _wilson(k: float, n: int) -> tuple[float, float]:
    """Wilson score interval (95%) for k successes in n trials."""
    p = k / n
    z2 = _Z * _Z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = _Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _newcombe(
    sb: npt.NDArray[np.float64], se_: npt.NDArray[np.float64]
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Wilson CIs for base and edit, and Newcombe's method 10 CI for edit - base.

    Newcombe (1998, Stat. Med. 17:2635) "Improved confidence intervals for the difference
    between binomial proportions based on paired data": hybrid score interval from the two
    Wilson intervals and the correlation phi of the paired binary scores (0 if either
    marginal is 0 or 1). Scores must be 0/1 per item. No normal approximation.
    """
    n = sb.size
    p_b, p_e = float(sb.mean()), float(se_.mean())
    lb, ub = _wilson(float(sb.sum()), n)
    le, ue = _wilson(float(se_.sum()), n)
    phi = _pearson(sb, se_)
    phi = 0.0 if math.isnan(phi) else phi
    lo_a, lo_b = p_e - le, ub - p_b
    hi_a, hi_b = ue - p_e, p_b - lb
    diff = p_e - p_b
    lo = diff - math.sqrt(max(0.0, lo_a**2 + lo_b**2 - 2 * phi * lo_a * lo_b))
    hi = diff + math.sqrt(max(0.0, hi_a**2 + hi_b**2 - 2 * phi * hi_a * hi_b))
    return (lb, ub), (le, ue), (lo, hi)


def _boot_ci(
    x: npt.NDArray[np.float64],
    g: npt.NDArray[np.intp] | None,
    n_g: int,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of mean(x); NaN for a single cluster."""
    if g is not None and n_g < 2:
        return math.nan, math.nan
    lo, hi = np.percentile(_boot_means(x, g, n_g, n_boot, rng), [2.5, 97.5])
    return float(lo), float(hi)


def _stratum(
    name: str,
    sb: npt.NDArray[np.float64],
    se_: npt.NDArray[np.float64],
    labels: list[str] | None,
    binary_items: bool,
    n_boot: int,
    rng: np.random.Generator,
) -> StratumResult:
    n = sb.size
    d = se_ - sb
    g: npt.NDArray[np.intp] | None = None
    n_g = n
    if labels is not None:
        uniq, g = np.unique(np.array(labels), return_inverse=True)
        n_g = uniq.size
    se = _paired_se(d) if g is None else _clustered_se(d, g, n_g)
    diff = float(d.mean())
    wilson: tuple[tuple[float, float], tuple[float, float]] | None = None
    if binary_items and g is None:
        wb, we, nc = _newcombe(sb, se_)
        wilson = (wb, we)
    if wilson is not None and n < _SMALL_N:
        method, ci = "newcombe", nc
    elif wilson is not None:
        method, ci = "normal", (diff - _Z * se, diff + _Z * se)
    else:
        method, ci = "bootstrap", _boot_ci(d, g, n_g, n_boot, rng)
    base_only, edit_only = _sign_counts(d) if binary_items else (0, 0)
    return StratumResult(
        stratum=name,
        n_items=n,
        mean_base=float(sb.mean()),
        mean_edit=float(se_.mean()),
        diff=diff,
        ci95=ci,
        ci95_method=method,
        se=se,
        score_mde=(_Z + _Z_POWER) * se,
        small_n=min(n, n_g) < _SMALL_N,
        n_base_only=base_only,
        n_edit_only=edit_only,
        base_ci95_wilson=None if wilson is None else wilson[0],
        edit_ci95_wilson=None if wilson is None else wilson[1],
    )


def _strata_rows(
    strata: Mapping[str, str],
    items: list[str],
    sb: npt.NDArray[np.float64],
    se_: npt.NDArray[np.float64],
    labels: list[str] | None,
    binary_items: bool,
    n_boot: int,
    seed: int,
) -> tuple[StratumResult, ...]:
    missing = [i for i in items if i not in strata]
    if missing:
        raise ValueError(
            f"strata has no entry for {len(missing)} item_id(s), e.g. {missing[0]!r}"
        )
    names = sorted({strata[i] for i in items})
    rows = []
    for idx, name in enumerate(names):
        pick = [j for j, i in enumerate(items) if strata[i] == name]
        rows.append(
            _stratum(
                name,
                sb[pick],
                se_[pick],
                None if labels is None else [labels[j] for j in pick],
                binary_items,
                n_boot,
                np.random.default_rng((seed, 2, idx)),
            )
        )
    return tuple(rows)


def is_loop(output: str) -> bool:
    """True when the last 400 characters are one chunk of at most 100 repeated 3 or more times."""
    tail = output[-_LOOP_TAIL:]
    return any(
        len(tail) >= _LOOP_MIN_REPEATS * p and tail[p:] == tail[:-p]
        for p in range(1, _LOOP_MAX_PERIOD + 1)
    )


def _uncapped(
    pairs: list[tuple[Record, Record]],
) -> tuple[int, float | None, float | None]:
    """Items, score diff and token diff over the pairs where neither arm hit the cap."""
    ok = [p for p in pairs if not (p[0].hit_cap or p[1].hit_cap)]
    if not ok:
        return 0, None, None
    sb, se_, tb, te = _per_item(ok)
    return sb.size, float((se_ - sb).mean()), float((te - tb).mean())


def compare(
    base: Sequence[Record],
    edit: Sequence[Record],
    n_boot: int = 10000,
    seed: int = 0,
    strata: Mapping[str, str] | None = None,
) -> PairedResult:
    """Paired comparison of two arms; scores and tokens are averaged over seeds per item first.

    With K > 1 seeds per item (`k` is the most any item has) the statistics are over items.
    Binary scores then get `sign_test_p`, the exact binomial test on the items whose mean
    score differs (the discordant counts are those items); McNemar is for K = 1 only.
    `score_ci95_boot` is a percentile bootstrap CI of the score difference, resampling items
    (clusters when `cluster_id` is set), seeded; `tokens_ci95` is the same for tokens.
    For binary scores on independent items the Wilson CIs of each arm and Newcombe's CI of
    the difference are filled in; use them rather than `ci95` when `small_n`.
    `strata` maps every item_id to a stratum name and adds one `StratumResult` per stratum.

    If records carry `cluster_id` (every record of both arms, the same id for an item across
    arms and seeds), SEs, the score CI and the MDEs use the cluster-robust SE (see
    `_clustered_se`) and the token bootstrap resamples clusters. With fewer than 2 clusters
    these are NaN. Without `cluster_id` items are treated as independent.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    pairs = pair(base, edit)
    if not pairs:
        raise ValueError("no records to compare")
    labels = _clusters(pairs)
    sb, se_, tb, te = _per_item(pairs)
    if not (np.isfinite(sb).all() and np.isfinite(se_).all()):
        raise ValueError(
            "scores must be finite (found NaN or inf); grade all records first"
        )
    n = sb.size
    d = se_ - sb
    diff = float(d.mean())
    g: npt.NDArray[np.intp] | None = None
    n_g = n
    if labels is not None:
        uniq, g = np.unique(np.array(labels), return_inverse=True)
        n_g = uniq.size
    se = _paired_se(d) if g is None else _clustered_se(d, g, n_g)

    binary = bool(np.isin(np.concatenate([sb, se_]), (0.0, 1.0)).all())
    binary_records = all(
        p[0].score in (0.0, 1.0) and p[1].score in (0.0, 1.0) for p in pairs
    )
    n_base_only, n_edit_only = _sign_counts(d) if binary or binary_records else (0, 0)
    k = max(Counter(b.item_id for b, _ in pairs).values())
    items = sorted({b.item_id for b, _ in pairs})
    wb = we = nc = None
    if binary and g is None:
        wb, we, nc = _newcombe(sb, se_)
    ci_normal = (diff - _Z * se, diff + _Z * se)
    use_newcombe = nc is not None and k == 1 and n < _SMALL_N
    score_boot = _boot_ci(d, g, n_g, n_boot, np.random.default_rng((seed, 1)))
    strata_rows = (
        ()
        if strata is None
        else _strata_rows(strata, items, sb, se_, labels, binary, n_boot, seed)
    )

    dt = te - tb
    boot = _boot_means(dt, g, n_g, n_boot, np.random.default_rng(seed))
    if g is not None and n_g < 2:
        lo = hi = math.nan  # one cluster: no between-cluster spread to resample
    else:
        lo, hi = np.percentile(boot, [2.5, 97.5])
    tokens_base = float(tb.mean())
    tokens_diff = float(dt.mean())
    tokens_se = _paired_se(dt) if g is None else _clustered_se(dt, g, n_g)

    n_uncapped, uncapped_diff, uncapped_tokens_diff = _uncapped(pairs)
    return PairedResult(
        n_items=n,
        mean_base=float(sb.mean()),
        mean_edit=float(se_.mean()),
        diff=diff,
        se=se,
        ci95=nc if use_newcombe and nc is not None else ci_normal,
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
        small_n=min(n, n_g) < _SMALL_N,
        score_mde=(_Z + _Z_POWER) * se,
        tokens_se=tokens_se,
        tokens_mde=(_Z + _Z_POWER) * tokens_se,
        n_clusters=None if g is None else n_g,
        cap_hit_p=_mcnemar_exact(
            sum(b.hit_cap and not e.hit_cap for b, e in pairs),
            sum(e.hit_cap and not b.hit_cap for b, e in pairs),
        ),
        n_uncapped=n_uncapped,
        uncapped_diff=uncapped_diff,
        uncapped_tokens_diff=uncapped_tokens_diff,
        loops_base=sum(b.hit_cap and is_loop(b.output) for b, _ in pairs),
        loops_edit=sum(e.hit_cap and is_loop(e.output) for _, e in pairs),
        k=k,
        score_ci95_boot=score_boot,
        sign_test_p=_mcnemar_exact(n_base_only, n_edit_only)
        if binary_records
        else None,
        base_ci95_wilson=wb,
        edit_ci95_wilson=we,
        newcombe_ci95=nc,
        strata=strata_rows,
        ci95_normal=ci_normal,
        ci95_method="newcombe" if use_newcombe else "normal",
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
    """Always the MDE; adds the not-significant reading when the CI contains 0."""
    out = f"  MDE={mde:{fmt}}"
    if ci[0] <= 0 <= ci[1] and math.isfinite(mde):
        out += f"; not significant; smallest detectable difference at this n is {mde:{fmt}}"
    return out


def _span(ci: tuple[float, float], fmt: str) -> str:
    return f"[{ci[0]:{fmt}}, {ci[1]:{fmt}}]"


def _stratum_line(r: StratumResult) -> str:
    return (
        f"stratum {r.stratum}: n={r.n_items}  base={r.mean_base:.4f}  edit={r.mean_edit:.4f}  "
        f"diff={r.diff:+.4f}  95% CI {_span(r.ci95, '+.4f')} ({r.ci95_method})"
        + _mde_note(r.ci95, r.score_mde, ".4f")
        + (
            "  WARNING: under 100 items"
            + (
                "; percentile bootstrap under-covers at small n"
                if r.ci95_method == "bootstrap"
                else ""
            )
            if r.small_n
            else ""
        )
    )


def _floor_line(
    name: str, diff: float, floor_diff: float, floor_se: float, exceeds: bool, fmt: str
) -> str:
    bar = abs(floor_diff) + _Z * floor_se
    verdict = "exceeds" if exceeds else "does not exceed"
    return (
        f"noise floor ({name}): |diff|={abs(diff):{fmt}} {verdict} "
        f"|floor diff| + 1.96 SE = {bar:{fmt}}"
    )


def _cap_warning(r: PairedResult) -> list[str]:
    """Warning block when cap-hit is high or unequal; empty otherwise."""
    if (
        max(r.cap_hit_base, r.cap_hit_edit) <= _CAP_WARN_RATE
        and r.cap_hit_p >= _CAP_WARN_P
    ):
        return []
    if r.n_uncapped and r.uncapped_diff is not None:
        both = (
            f"n={r.n_uncapped} items  score diff={r.uncapped_diff:+.4f}  "
            f"tokens diff={r.uncapped_tokens_diff:+.1f}"
        )
    else:
        both = "n=0 items"
    return [
        "",
        "WARNING: cap-hit censoring",
        (
            f"  cap-hit rate: base={r.cap_hit_base:.1%}  edit={r.cap_hit_edit:.1%}  "
            f"(McNemar exact p={r.cap_hit_p:.4g})"
        ),
        (
            "  A capped answer usually has no final answer, so it is graded wrong and its "
            "token count is cut short."
        ),
        "  The score and token diffs over all items above are censored by the cap.",
        f"  items where neither arm hit the cap: {both}",
        f"  loops (capped, tail repeats one chunk): base={r.loops_base}  edit={r.loops_edit}",
        (
            "  Raise max_new_tokens and rerun; pick it from a short base-arm pilot so that "
            "cap-hit is near 0."
        ),
    ]


def format_report(r: PairedResult, floor: PairedResult | None = None) -> str:
    """A few plain lines summarising a `PairedResult`; `floor` adds noise-floor verdicts."""
    clusters = "" if r.n_clusters is None else f" in {r.n_clusters} clusters"
    mde = f"  MDE={r.score_mde:.4f}"
    lines = [
        f"n={r.n_items} items{clusters}  K={r.k}  base={r.mean_base:.4f}  edit={r.mean_edit:.4f}",
        f"score diff={r.diff:+.4f}  SE={r.se:.4f}  95% CI {_span(r.ci95, '+.4f')}"
        f" ({r.ci95_method})  corr={r.correlation:.3f}"
        + _mde_note(r.ci95, r.score_mde, ".4f"),
        f"score diff bootstrap 95% CI {_span(r.score_ci95_boot, '+.4f')}{mde}",
    ]
    if r.newcombe_ci95 is not None and r.base_ci95_wilson and r.edit_ci95_wilson:
        lines.append(
            f"Wilson 95% CI: base {_span(r.base_ci95_wilson, '.4f')}  "
            f"edit {_span(r.edit_ci95_wilson, '.4f')}; "
            f"Newcombe 95% CI of diff {_span(r.newcombe_ci95, '+.4f')}{mde}"
        )
    ignores = "; ignores clustering" if r.n_clusters is not None else ""
    if r.k > 1 and r.sign_test_p is not None:
        lines.append(
            f"sign test exact p={r.sign_test_p:.4g}  K={r.k} samples per item, "
            f"items with higher mean score: base={r.n_base_only}, edit={r.n_edit_only}"
            + ignores
            + mde
        )
    elif r.mcnemar_p is not None:
        lines.append(
            f"McNemar exact p={r.mcnemar_p:.4g}  "
            f"(base-only right={r.n_base_only}, edit-only right={r.n_edit_only})"
            + ignores
            + mde
        )
    lines += [
        (
            f"tokens: base={r.tokens_base:.1f}  edit={r.tokens_edit:.1f}  "
            f"diff={r.tokens_diff:+.1f} ({r.tokens_diff_pct:+.1f}%)  "
            f"bootstrap 95% CI {_span(r.tokens_ci95, '+.1f')}"
            + _mde_note(r.tokens_ci95, r.tokens_mde, ".1f")
        ),
        f"cap-hit rate: base={r.cap_hit_base:.3f}  edit={r.cap_hit_edit:.3f}",
    ]
    lines += [_stratum_line(s) for s in r.strata]
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
        few = (
            f"{r.n_items} items"
            if r.n_clusters is None or r.n_items <= r.n_clusters
            else f"{r.n_clusters} clusters"
        )
        lines.append(f"WARNING: only {few} (<{_SMALL_N}); intervals are unreliable")
    return "\n".join(lines + _cap_warning(r))
