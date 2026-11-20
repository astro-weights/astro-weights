"""0.3 paired statistics: K > 1 sign test, bootstrap CI, Wilson/Newcombe, strata, MDE."""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from astro_weights.cli import main
from astro_weights.eval import Record, compare, format_report, write_jsonl
from astro_weights.eval.paired import _newcombe, _wilson


def _recs(
    base: list[list[float]], edit: list[list[float]], cluster: list[str] | None = None
) -> tuple[list[Record], list[Record]]:
    """Item k has len(base[k]) seeds; scores given per item."""
    b, e = [], []
    for k, (sb, se) in enumerate(zip(base, edit, strict=True)):
        c = None if cluster is None else cluster[k]
        for s, (x, y) in enumerate(zip(sb, se, strict=True)):
            b.append(Record(f"i{k}", s, "base", f"p{k}", "o", x, 100, False, c))
            e.append(Record(f"i{k}", s, "edit", f"p{k}", "o", y, 120, False, c))
    return b, e


def _binom_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def test_wilson_known_values() -> None:
    lo, hi = _wilson(8, 10)
    assert (lo, hi) == pytest.approx((0.4902, 0.9433), abs=1e-4)
    lo, hi = _wilson(0, 20)
    assert lo == 0.0
    assert hi == pytest.approx(0.1611, abs=1e-4)
    assert _wilson(5, 5)[1] == 1.0


def test_newcombe_hand_computed_and_phi_zero_case() -> None:
    # a=4 (both), b=3 (base only), c=1 (edit only), d=2: n=10, p_b=.7, p_e=.5
    sb = np.array([1] * 4 + [1] * 3 + [0] + [0] * 2, dtype=float)
    se = np.array([1] * 4 + [0] * 3 + [1] + [0] * 2, dtype=float)
    wb, we, (lo, hi) = _newcombe(sb, se)
    phi = (4 * 2 - 3 * 1) / math.sqrt(7 * 3 * 5 * 5)
    lb, ub = _wilson(7, 10)
    le, ue = _wilson(5, 10)
    want_lo = -0.2 - math.sqrt(
        (0.5 - le) ** 2 + (ub - 0.7) ** 2 - 2 * phi * (0.5 - le) * (ub - 0.7)
    )
    want_hi = -0.2 + math.sqrt(
        (ue - 0.5) ** 2 + (0.7 - lb) ** 2 - 2 * phi * (ue - 0.5) * (0.7 - lb)
    )
    assert (wb, we) == ((lb, ub), (le, ue))
    assert (lo, hi) == pytest.approx((want_lo, want_hi))
    assert -1 <= lo < -0.2 < hi <= 1
    # constant base arm: phi = 0, so the interval is the two Wilson arms added in quadrature
    _, _, (lo1, hi1) = _newcombe(np.ones(10), se)
    lb1, ub1 = _wilson(10, 10)
    assert lo1 == pytest.approx(-0.5 - math.hypot(0.5 - le, ub1 - 1.0))
    assert hi1 == pytest.approx(-0.5 + math.hypot(ue - 0.5, 1.0 - lb1))


def test_newcombe_coverage_simulation() -> None:
    rng = np.random.default_rng(5)
    probs = [
        0.45,
        0.15,
        0.10,
        0.30,
    ]  # both, base only, edit only, neither: true diff -0.05
    reps, n, hit_n, hit_w = 2000, 80, 0, 0
    for _ in range(reps):
        c = rng.choice(4, size=n, p=probs)
        sb = np.isin(c, (0, 1)).astype(float)
        se = np.isin(c, (0, 2)).astype(float)
        wb, _, nc = _newcombe(sb, se)
        hit_n += nc[0] <= -0.05 <= nc[1]
        hit_w += wb[0] <= 0.60 <= wb[1]
    assert 0.93 <= hit_n / reps <= 0.97
    assert 0.93 <= hit_w / reps <= 0.98


def test_bootstrap_ci_coverage_simulation() -> None:
    rng = np.random.default_rng(21)
    reps, n, hit = 400, 100, 0
    for _ in range(reps):
        b = rng.normal(0, 1, n)
        e = b + rng.exponential(1, n) - 1 + rng.normal(0, 0.5, n)  # true diff 0, skewed
        base, edit = _recs([[x] for x in b], [[x] for x in e])
        lo, hi = compare(base, edit, n_boot=400).score_ci95_boot
        hit += lo <= 0 <= hi
    assert 0.91 <= hit / reps <= 0.99


def test_bootstrap_ci_clustered_wider_and_seeded() -> None:
    rng = np.random.default_rng(3)
    eff = np.repeat(rng.normal(0, 1, 30), 4)
    d = eff + rng.normal(0, 0.3, 120)
    cl = [f"c{k // 4}" for k in range(120)]
    base, edit = _recs([[0.0]] * 120, [[float(x)] for x in d], cl)
    r = compare(base, edit, n_boot=2000, seed=4)
    # repr, not ==: the base arm is constant, so correlation is nan, and since
    # Python 3.13 dataclass == compares fields one by one (nan != nan)
    assert repr(r) == repr(compare(base, edit, n_boot=2000, seed=4))
    nb, ne = _recs([[0.0]] * 120, [[float(x)] for x in d])
    u = compare(nb, ne, n_boot=2000, seed=4)
    wr, wu = (x.score_ci95_boot[1] - x.score_ci95_boot[0] for x in (r, u))
    assert wr > 1.5 * wu
    assert r.score_ci95_boot[0] < r.diff < r.score_ci95_boot[1]


def test_bootstrap_single_cluster_nan_and_single_item() -> None:
    base, edit = _recs([[0.0], [1.0]], [[1.0], [1.0]], ["c", "c"])
    assert all(math.isnan(x) for x in compare(base, edit, n_boot=10).score_ci95_boot)


def test_sign_test_k4_matches_exact_binomial() -> None:
    # 10 items, K=4: item means differ on 8 items (7 edit higher, 1 base higher), 2 tie.
    base = [[0, 0, 1, 0]] * 7 + [[1, 1, 1, 0], [1, 0, 1, 0], [0, 0, 0, 0]]
    edit = [[1, 0, 1, 1]] * 7 + [[0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 0, 0]]
    b, e = _recs(base, edit)  # type: ignore[arg-type]
    r = compare(b, e, n_boot=50)
    assert r.k == 4
    assert (r.n_base_only, r.n_edit_only) == (1, 7)
    assert r.sign_test_p == pytest.approx(_binom_two_sided(1, 7))
    assert r.sign_test_p == pytest.approx(0.0703125)
    assert r.mcnemar_p is None  # per-item means are fractional
    text = format_report(r)
    assert "sign test exact p=0.07031" in text
    assert "K=4" in text
    assert "McNemar" not in text


@pytest.mark.parametrize(("nb", "ne"), [(0, 0), (0, 5), (3, 3), (2, 9), (12, 1)])
def test_sign_test_matches_reference(nb: int, ne: int) -> None:
    base = [[1.0, 1.0]] * nb + [[0.0, 0.0]] * ne + [[1.0, 0.0]]
    edit = [[0.0, 1.0]] * nb + [[1.0, 0.0]] * ne + [[1.0, 0.0]]
    b, e = _recs(base, edit)
    r = compare(b, e, n_boot=10)
    assert (r.n_base_only, r.n_edit_only) == (nb, ne)
    assert r.sign_test_p == pytest.approx(_binom_two_sided(nb, ne))


def test_k1_binary_sign_test_equals_mcnemar() -> None:
    b, e = _recs([[1], [1], [0], [0], [1], [0]], [[1], [0], [1], [1], [1], [1]])
    r = compare(b, e, n_boot=10)
    assert r.k == 1
    assert r.sign_test_p == r.mcnemar_p
    assert "McNemar" in format_report(r)
    assert "sign test" not in format_report(r)


def test_graded_scores_have_no_sign_test() -> None:
    b, e = _recs([[0.2], [0.5], [0.9]], [[0.3], [0.4], [1.0]])
    r = compare(b, e, n_boot=10)
    assert r.sign_test_p is None
    assert r.newcombe_ci95 is None
    assert r.base_ci95_wilson is None
    assert (r.n_base_only, r.n_edit_only) == (0, 0)


def test_new_fields_binary_k1_and_old_constructor() -> None:
    b, e = _recs([[1], [1], [0], [0], [1], [0]], [[1], [0], [1], [1], [1], [1]])
    r = compare(b, e, n_boot=200)
    assert r.k == 1
    assert r.newcombe_ci95 is not None
    assert r.newcombe_ci95[0] < r.diff < r.newcombe_ci95[1]
    assert r.base_ci95_wilson == pytest.approx(_wilson(3, 6))
    assert r.edit_ci95_wilson == pytest.approx(_wilson(5, 6))
    assert r.strata == ()
    old = {
        f.name: getattr(r, f.name)
        for f in dataclasses.fields(r)
        if f.name
        not in {
            "k",
            "score_ci95_boot",
            "sign_test_p",
            "base_ci95_wilson",
            "edit_ci95_wilson",
            "newcombe_ci95",
            "strata",
        }
    }
    r2 = type(r)(**old)
    assert (r2.k, r2.sign_test_p, r2.strata) == (1, None, ())


def test_clustered_binary_has_no_newcombe() -> None:
    b, e = _recs([[1], [0], [1], [0]], [[0], [0], [1], [1]], ["a", "a", "b", "b"])
    r = compare(b, e, n_boot=10)
    assert r.newcombe_ci95 is None
    assert "ignores clustering" in format_report(r)


def test_uneven_k_reports_max() -> None:
    b, e = _recs([[1, 1, 0], [0]], [[1, 0, 0], [1]])
    assert compare(b, e, n_boot=10).k == 3


def _strata_arms(
    n_a: int, n_b: int
) -> tuple[list[Record], list[Record], dict[str, str]]:
    rng = np.random.default_rng(1)
    n = n_a + n_b
    sb = rng.integers(0, 2, n).astype(float)
    se = np.where(rng.random(n) < 0.3, 1 - sb, sb)
    b, e = _recs([[x] for x in sb], [[x] for x in se])
    return b, e, {f"i{k}": ("small" if k < n_a else "big") for k in range(n)}


def test_strata_rows_methods_and_mde() -> None:
    b, e, strata = _strata_arms(40, 150)
    r = compare(b, e, n_boot=100, strata=strata)
    assert [s.stratum for s in r.strata] == ["big", "small"]
    big, small = r.strata
    assert (big.n_items, small.n_items) == (150, 40)
    assert (big.ci95_method, small.ci95_method) == ("normal", "newcombe")
    assert (big.small_n, small.small_n) == (False, True)
    assert small.base_ci95_wilson is not None
    assert small.ci95 == pytest.approx(
        _newcombe(
            np.array([x.score for x in b if int(x.item_id[1:]) < 40]),
            np.array([x.score for x in e if int(x.item_id[1:]) < 40]),
        )[2]
    )
    for s in r.strata:
        assert s.score_mde == pytest.approx(2.8 * s.se)
        assert s.ci95[0] < s.diff < s.ci95[1]
        assert s.n_base_only + s.n_edit_only > 0
    assert sum(s.n_items for s in r.strata) == r.n_items


def test_strata_graded_and_k_gt_1_use_bootstrap() -> None:
    rng = np.random.default_rng(2)
    base = [[float(x) for x in rng.random(2)] for _ in range(30)]
    edit = [[float(x) for x in rng.random(2)] for _ in range(30)]
    b, e = _recs(base, edit)
    strata = {f"i{k}": "ab"[k % 2] for k in range(30)}
    r = compare(b, e, n_boot=300, seed=1, strata=strata)
    assert {s.ci95_method for s in r.strata} == {"bootstrap"}
    assert all(s.base_ci95_wilson is None for s in r.strata)
    assert r == compare(b, e, n_boot=300, seed=1, strata=strata)
    # K = 4 binary samples: item means are fractional, so bootstrap too
    bb, ee = _recs([[1, 0, 1, 1]] * 6, [[0, 0, 1, 1]] * 6)
    r2 = compare(bb, ee, n_boot=50, strata={f"i{k}": "s" for k in range(6)})
    assert r2.strata[0].ci95_method == "bootstrap"


def test_strata_clustered_uses_cluster_bootstrap() -> None:
    b, e = _recs([[1], [0], [1], [0]], [[0], [0], [1], [1]], ["a", "a", "b", "b"])
    r = compare(b, e, n_boot=50, strata={f"i{k}": "s" for k in range(4)})
    assert r.strata[0].ci95_method == "bootstrap"
    assert r.strata[0].base_ci95_wilson is None


def test_strata_missing_item_rejected() -> None:
    b, e = _recs([[1], [0]], [[0], [1]])
    with pytest.raises(ValueError, match="no entry for 1 item_id"):
        compare(b, e, n_boot=10, strata={"i0": "a"})


def test_strata_newcombe_coverage_simulation() -> None:
    rng = np.random.default_rng(9)
    probs = [0.45, 0.15, 0.10, 0.30]
    reps, n, hit = 400, 60, 0
    for _ in range(reps):
        c = rng.choice(4, size=n, p=probs)
        sb = np.isin(c, (0, 1)).astype(float)
        se = np.isin(c, (0, 2)).astype(float)
        b, e = _recs([[x] for x in sb], [[x] for x in se])
        s = compare(b, e, n_boot=1, strata={f"i{k}": "s" for k in range(n)}).strata[0]
        assert s.ci95_method == "newcombe"
        hit += s.ci95[0] <= -0.05 <= s.ci95[1]
    assert 0.91 <= hit / reps <= 0.99


def test_report_has_every_field_and_mde_when_significant() -> None:
    n = 60
    b, e = _recs(
        [[float(k % 2)] for k in range(n)], [[float(k % 2) + 1.0] for k in range(n)]
    )
    r = compare(b, e, n_boot=50, strata={f"i{k}": "ab"[k % 2] for k in range(n)})
    text = format_report(r)
    assert "not significant" not in text
    for want in (
        "n=60 items",
        "K=1",
        "base=",
        "SE=",
        "corr=",
        "MDE=",
        "bootstrap 95% CI",
        "stratum a:",
        "stratum b:",
    ):
        assert want in text
    # MDE on the score line, the tokens line and each stratum row
    assert all(
        "MDE=" in ln
        for ln in text.splitlines()
        if ln.startswith(("score diff=", "tokens:", "stratum"))
    )


def test_report_binary_small_prints_newcombe_and_wilson() -> None:
    b, e = _recs([[1], [1], [0], [0], [1], [0]], [[1], [0], [1], [1], [1], [1]])
    text = format_report(compare(b, e, n_boot=50))
    assert "Wilson 95% CI: base [" in text
    assert "Newcombe 95% CI of diff [" in text
    assert "(base-only right=" in text


def _files(tmp_path: Path, k: int) -> tuple[Path, Path]:
    n = 12
    base = [[float(i % 2)] * k for i in range(n)]
    edit = [[1.0] * k for _ in range(n)]
    b, e = _recs(base, edit)
    write_jsonl(tmp_path / "b.jsonl", b)
    write_jsonl(tmp_path / "e.jsonl", e)
    return tmp_path / "b.jsonl", tmp_path / "e.jsonl"


def test_cli_text_and_json_show_new_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    b, e = _files(tmp_path, 3)
    assert main(["eval", str(b), str(e), "--boot", "50"]) == 0
    out = capsys.readouterr().out
    assert "K=3" in out
    assert "MDE=" in out
    assert "bootstrap 95% CI" in out
    assert "corr=" in out
    assert main(["eval", str(b), str(e), "--boot", "50", "--json"]) == 0
    j = json.loads(capsys.readouterr().out)
    assert j["k"] == 3
    assert j["score_mde"] is not None
    assert j["tokens_mde"] is not None
    assert len(j["score_ci95_boot"]) == 2
    assert j["strata"] == []
    assert j["sign_test_p"] == pytest.approx(_binom_two_sided(0, 6))
    assert (j["n_base_only"], j["n_edit_only"]) == (0, 6)


def test_newcombe_published_example() -> None:
    # Newcombe 1998, Stat. Med. 17:2635, method 10: a=36, b=12, c=2, d=0 -> (0.0569, 0.3404)
    # (edit = row 1 = a + c right; base = column 1 = a + b right; b = base only).
    sb = np.array([1.0] * 36 + [1.0] * 12 + [0.0] * 2)
    se = np.array([1.0] * 36 + [0.0] * 12 + [1.0] * 2)
    lo, hi = _newcombe(se, sb)[2]  # the paper's P1 - P2 with P1 = the 38/50 arm
    assert (lo, hi) == pytest.approx((0.0569, 0.3404), abs=1e-4) or (
        -hi,
        -lo,
    ) == pytest.approx((0.0569, 0.3404), abs=1e-4)


def test_small_binary_ci95_is_newcombe_and_normal_kept() -> None:
    b, e = _recs([[1], [1], [0], [0], [1], [0]], [[1], [0], [1], [1], [1], [1]])
    r = compare(b, e, n_boot=10)
    assert (r.ci95_method, r.ci95) == ("newcombe", r.newcombe_ci95)
    assert r.ci95_normal == pytest.approx((r.diff - 1.96 * r.se, r.diff + 1.96 * r.se))
    assert r.ci95 != r.ci95_normal
    assert "(newcombe)" in format_report(r)


@pytest.mark.parametrize("which", ["n100", "graded", "k2", "clustered"])
def test_ci95_stays_normal_otherwise(which: str) -> None:
    rng = np.random.default_rng(0)
    n = 100 if which == "n100" else 30
    k = 2 if which == "k2" else 1
    sb = rng.integers(0, 2, (n, k)).astype(float)
    se = rng.integers(0, 2, (n, k)).astype(float)
    if which == "graded":
        sb, se = sb * 0.5, se * 0.7
    cl = [f"c{i // 2}" for i in range(n)] if which == "clustered" else None
    b, e = _recs(sb.tolist(), se.tolist(), cl)
    r = compare(b, e, n_boot=10)
    assert r.ci95_method == "normal"
    assert r.ci95 == r.ci95_normal


def test_exceeds_floor_uses_newcombe_at_small_n() -> None:
    # 3 edit-only items of 8: the CLT interval excludes 0 (lower end +0.016), Newcombe does not.
    from astro_weights.eval import exceeds_floor

    b, e = _recs([[0.0]] * 8, [[1.0]] * 3 + [[0.0]] * 5)
    r = compare(b, e, n_boot=10)
    assert r.ci95_normal is not None
    assert r.ci95_normal[0] > 0
    assert r.ci95[0] < 0
    floor = compare(*_recs([[0.0]] * 8, [[0.0]] * 8), n_boot=10)
    assert exceeds_floor(r, floor)["score"] is False


def test_mde_on_every_result_line() -> None:
    b, e = _recs([[1], [1], [0], [0], [1], [0]], [[1], [0], [1], [1], [1], [1]])
    text = format_report(compare(b, e, n_boot=50))
    for prefix in (
        "score diff=",
        "score diff bootstrap",
        "Wilson 95% CI",
        "McNemar",
        "tokens:",
    ):
        (line,) = (ln for ln in text.splitlines() if ln.startswith(prefix))
        assert "MDE=" in line, prefix
    b4, e4 = _recs([[0, 0, 1, 0]] * 7, [[1, 0, 1, 1]] * 7)
    text = format_report(compare(b4, e4, n_boot=50))
    (line,) = (ln for ln in text.splitlines() if ln.startswith("sign test"))
    assert "MDE=" in line


def test_small_bootstrap_stratum_is_flagged() -> None:
    rng = np.random.default_rng(4)
    b, e = _recs(
        [[float(x)] for x in rng.random(30)], [[float(x)] for x in rng.random(30)]
    )
    r = compare(b, e, n_boot=50, strata={f"i{k}": "s" for k in range(30)})
    s = r.strata[0]
    assert (s.ci95_method, s.small_n) == ("bootstrap", True)
    assert json.loads(json.dumps(dataclasses.asdict(r)))["strata"][0]["small_n"] is True
    assert "percentile bootstrap under-covers at small n" in format_report(r)


def test_small_stratum_bootstrap_coverage_simulation() -> None:
    # Measured, skewed d, true diff 0: n=20 89.2%, n=30 88.8-93.3% (two seeds, 400-600 reps),
    # n=50 93.8%: the percentile bootstrap under-covers a little below ~50 items.
    rng = np.random.default_rng(31)
    reps, n, hit = 400, 30, 0
    for _ in range(reps):
        x = rng.normal(0, 1, n)
        y = x + rng.exponential(1, n) - 1 + rng.normal(0, 0.5, n)
        b, e = _recs([[v] for v in x], [[v] for v in y])
        s = compare(b, e, n_boot=300, strata={f"i{k}": "s" for k in range(n)}).strata[0]
        assert s.ci95_method == "bootstrap"
        hit += s.ci95[0] <= 0 <= s.ci95[1]
    assert 0.85 <= hit / reps <= 0.99
