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
    """Paired statistics; scores and token counts are per-item means over seeds.

    `n_clusters` is the number of distinct `cluster_id`s, or None when records carry none.
    When set, `se`, `ci95`, `tokens_se`, both MDEs and `tokens_ci95` account for clustering,
    and `small_n` also flags fewer than 100 clusters. McNemar always treats items as independent.
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


def compare(
    base: Sequence[Record], edit: Sequence[Record], n_boot: int = 10000, seed: int = 0
) -> PairedResult:
    """Paired comparison of two arms; scores and tokens are averaged over seeds per item first.

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
    n_base_only = int(((sb == 1) & (se_ == 0)).sum()) if binary else 0
    n_edit_only = int(((sb == 0) & (se_ == 1)).sum()) if binary else 0

    dt = te - tb
    boot = _boot_means(dt, g, n_g, n_boot, np.random.default_rng(seed))
    if g is not None and n_g < 2:
        lo = hi = math.nan  # one cluster: no between-cluster spread to resample
    else:
        lo, hi = np.percentile(boot, [2.5, 97.5])
    tokens_base = float(tb.mean())
    tokens_diff = float(dt.mean())
    tokens_se = _paired_se(dt) if g is None else _clustered_se(dt, g, n_g)

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
        small_n=min(n, n_g) < _SMALL_N,
        score_mde=(_Z + _Z_POWER) * se,
        tokens_se=tokens_se,
        tokens_mde=(_Z + _Z_POWER) * tokens_se,
        n_clusters=None if g is None else n_g,
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
    clusters = "" if r.n_clusters is None else f" in {r.n_clusters} clusters"
    lines = [
        f"n={r.n_items} items{clusters}  base={r.mean_base:.4f}  edit={r.mean_edit:.4f}",
        f"score diff={r.diff:+.4f}  SE={r.se:.4f}  95% CI [{r.ci95[0]:+.4f}, {r.ci95[1]:+.4f}]"
        + _mde_note(r.ci95, r.score_mde, ".4f"),
    ]
    if r.mcnemar_p is not None:
        lines.append(
            f"McNemar exact p={r.mcnemar_p:.4g}  "
            f"(base-only right={r.n_base_only}, edit-only right={r.n_edit_only})"
            + ("; ignores clustering" if r.n_clusters is not None else "")
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
        few = (
            f"{r.n_items} items"
            if r.n_clusters is None or r.n_items <= r.n_clusters
            else f"{r.n_clusters} clusters"
        )
        lines.append(f"WARNING: only {few} (<{_SMALL_N}); intervals are unreliable")
    return "\n".join(lines)
