import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from astro_weights.eval import (
    PairedResult,
    Record,
    compare,
    exceeds_floor,
    format_report,
    is_loop,
    pair,
    prompt_hash,
    read_jsonl,
    write_jsonl,
)
from astro_weights.eval.paired import _mcnemar_exact


def rec(
    item: str,
    arm: str,
    score: float,
    seed: int = 0,
    tokens: int = 100,
    cap: bool = False,
    ph: str = "p",
) -> Record:
    return Record(item, seed, arm, ph + item, "out", score, tokens, cap, None)


def arms(
    base_scores: list[float], edit_scores: list[float]
) -> tuple[list[Record], list[Record]]:
    ids = [f"i{k}" for k in range(len(base_scores))]
    return (
        [rec(i, "base", s) for i, s in zip(ids, base_scores, strict=True)],
        [rec(i, "edit", s) for i, s in zip(ids, edit_scores, strict=True)],
    )


def test_hand_computed_example() -> None:
    base, edit = arms([1, 1, 0, 0, 1, 0], [1, 0, 1, 1, 1, 1])
    r = compare(base, edit, n_boot=100)
    # d = [0, -1, 1, 1, 0, 1]: mean 1/3, var (ddof=1) 2/3, se = sqrt(2/3 / 6) = 1/3
    assert r.n_items == 6
    assert r.diff == pytest.approx(1 / 3, abs=1e-9)
    assert r.se == pytest.approx(1 / 3, abs=1e-9)
    assert r.ci95[0] == pytest.approx(1 / 3 - 1.96 / 3, abs=1e-9)
    assert r.ci95[1] == pytest.approx(1 / 3 + 1.96 / 3, abs=1e-9)
    # discordant: 1 base-only, 3 edit-only; p = 2 * (C(4,0) + C(4,1)) / 2**4
    assert (r.n_base_only, r.n_edit_only) == (1, 3)
    expected_p = 2 * (math.comb(4, 0) + math.comb(4, 1)) / 2**4
    assert r.mcnemar_p == pytest.approx(expected_p, abs=1e-9)
    assert r.mean_base == pytest.approx(0.5)
    assert r.mean_edit == pytest.approx(5 / 6)
    assert r.small_n
    assert not math.isnan(r.correlation)


def test_synthetic_400_items_ci_covers_truth() -> None:
    rng = np.random.default_rng(1)
    n = 400
    s_base = rng.normal(0.5, 0.2, n)
    s_edit = s_base + 0.1 + rng.normal(0, 0.1, n)
    t_base = rng.normal(1000, 100, n).round()
    t_edit = (t_base - 50 + rng.normal(0, 30, n)).round()
    base = [
        rec(f"i{k}", "base", float(s_base[k]), tokens=int(t_base[k])) for k in range(n)
    ]
    edit = [
        rec(f"i{k}", "edit", float(s_edit[k]), tokens=int(t_edit[k])) for k in range(n)
    ]
    r = compare(base, edit, seed=3)
    assert not r.small_n
    assert r.ci95[0] < 0.1 < r.ci95[1]
    assert r.tokens_ci95[0] < -50 < r.tokens_ci95[1]
    assert r.mcnemar_p is None
    assert r.tokens_diff_pct < 0
    assert r.correlation > 0.5


def test_identical_arms() -> None:
    base, _ = arms([1, 0, 1, 0, 1], [0] * 5)
    edit = [Record(**{**r.__dict__, "arm": "edit"}) for r in base]
    r = compare(base, edit, n_boot=50)
    assert r.diff == 0
    assert r.se == 0
    assert r.mcnemar_p == 1.0
    assert (r.n_base_only, r.n_edit_only) == (0, 0)
    assert r.tokens_diff == 0
    assert r.correlation == pytest.approx(1.0)


def test_constant_arm_correlation_nan() -> None:
    base, edit = arms([1, 1, 1, 1], [1, 0, 1, 0])
    assert math.isnan(compare(base, edit, n_boot=10).correlation)


def test_multi_seed_items_averaged_per_item() -> None:
    base = [
        rec("a", "base", s, seed=k, tokens=t)
        for k, (s, t) in enumerate([(1, 100), (0, 200)])
    ]
    base += [rec("b", "base", 0, seed=k, tokens=300) for k in range(2)]
    edit = [rec("a", "edit", 1, seed=k, tokens=100) for k in range(2)]
    edit += [rec("b", "edit", s, seed=k, tokens=300) for k, s in enumerate([1, 0])]
    r = compare(base, edit, n_boot=20)
    assert r.n_items == 2
    assert r.mean_base == pytest.approx(0.25)  # a: 0.5, b: 0
    assert r.mean_edit == pytest.approx(0.75)  # a: 1, b: 0.5
    assert r.diff == pytest.approx(0.5)
    assert r.mcnemar_p is None  # averaged scores are not all 0/1
    assert r.tokens_base == pytest.approx(225)  # a: 150, b: 300
    assert r.tokens_edit == pytest.approx(200)
    assert r.tokens_diff == pytest.approx(-25)
    assert r.tokens_diff_pct == pytest.approx(-100 / 9)


def test_cap_hit_rates() -> None:
    base = [rec(f"i{k}", "base", 1, cap=k < 2) for k in range(4)]
    edit = [rec(f"i{k}", "edit", 1, cap=k == 0) for k in range(4)]
    r = compare(base, edit, n_boot=10)
    assert (r.cap_hit_base, r.cap_hit_edit) == (0.5, 0.25)


def test_pair_sorted_and_joined() -> None:
    base = [rec("b", "base", 1), rec("a", "base", 0, seed=1), rec("a", "base", 0)]
    edit = [rec("a", "edit", 1), rec("b", "edit", 1), rec("a", "edit", 1, seed=1)]
    keys = [(b.item_id, b.seed) for b, _ in pair(base, edit)]
    assert keys == [("a", 0), ("a", 1), ("b", 0)]
    assert all(b.item_id == e.item_id and e.arm == "edit" for b, e in pair(base, edit))


def test_pair_key_mismatch() -> None:
    base, edit = arms([1, 0], [1, 0])
    with pytest.raises(ValueError, match=r"same .item_id, seed. keys"):
        pair(base, edit[:1])
    with pytest.raises(ValueError, match=r"same .item_id, seed. keys"):
        pair(base[:1], edit)


def test_pair_duplicate_key() -> None:
    base, edit = arms([1, 0], [1, 0])
    with pytest.raises(ValueError, match="base arm repeats"):
        pair([*base, base[0]], edit)
    with pytest.raises(ValueError, match="edit arm repeats"):
        pair(base, [*edit, edit[0]])


def test_pair_prompt_hash_mismatch() -> None:
    base, edit = arms([1, 0], [1, 0])
    edit[1] = rec("i1", "edit", 0, ph="other")
    with pytest.raises(ValueError, match="not given the same prompts"):
        pair(base, edit)


def test_compare_empty() -> None:
    with pytest.raises(ValueError, match="no records"):
        compare([], [])


def test_prompt_hash() -> None:
    assert (
        prompt_hash("")
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert prompt_hash("é") != prompt_hash("e")


def test_jsonl_round_trip_lossless(tmp_path: Path) -> None:
    records = [
        Record(
            "a", 0, "base", prompt_hash("x"), 'héllo\n"wörld" 日本', 0.1 + 0.2, 7, False
        ),
        Record("b", 3, "edit", prompt_hash("y"), "", 1.0, 0, True, "c1"),
    ]
    path = tmp_path / "r.jsonl"
    write_jsonl(path, records)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    assert read_jsonl(path) == records


def _write(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "bad.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _good(**over: object) -> str:
    d: dict[str, object] = {
        "item_id": "a",
        "seed": 0,
        "arm": "base",
        "prompt_hash": "h",
        "output": "o",
        "score": 1,
        "n_output_tokens": 3,
        "hit_cap": False,
    }
    d.update(over)
    return json.dumps(d)


def test_read_jsonl_accepts_int_score_and_blank_lines(tmp_path: Path) -> None:
    (r,) = read_jsonl(_write(tmp_path, _good(), ""))
    assert r.score == 1.0
    assert isinstance(r.score, float)
    assert r.cluster_id is None


def test_read_jsonl_missing_key(tmp_path: Path) -> None:
    d = json.loads(_good())
    del d["score"]
    with pytest.raises(ValueError, match=r"line 2: missing keys: score"):
        read_jsonl(_write(tmp_path, _good(), json.dumps(d)))


def test_read_jsonl_unknown_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"line 1: unknown keys: extra"):
        read_jsonl(_write(tmp_path, _good(extra=1)))


def test_read_jsonl_bad_json(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"line 2: invalid JSON"):
        read_jsonl(_write(tmp_path, _good(), "{not json"))


def test_read_jsonl_not_object(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"line 1: expected a JSON object"):
        read_jsonl(_write(tmp_path, "[1, 2]"))


@pytest.mark.parametrize(
    "over",
    [{"seed": "0"}, {"seed": True}, {"score": "1"}, {"hit_cap": 1}, {"cluster_id": 5}],
)
def test_read_jsonl_wrong_type(tmp_path: Path, over: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"line 1: wrong value type"):
        read_jsonl(_write(tmp_path, _good(**over)))


def test_format_report() -> None:
    base, edit = arms([1, 1, 0, 0, 1, 0], [1, 0, 1, 1, 1, 1])
    text = format_report(compare(base, edit, n_boot=50))
    assert "n=6" in text
    assert "McNemar" in text
    assert "WARNING" in text
    assert "cap-hit" in text
    assert "tokens" in text


def test_format_report_no_mcnemar_no_warning() -> None:
    n = 120
    base = [rec(f"i{k}", "base", k / n) for k in range(n)]
    edit = [rec(f"i{k}", "edit", k / n + 0.01 * (k % 3)) for k in range(n)]
    r = compare(base, edit, n_boot=50)
    assert isinstance(r, PairedResult)
    text = format_report(r)
    assert "McNemar" not in text
    assert "WARNING" not in text


def test_constant_fractional_arm_correlation_nan() -> None:
    # 3 seeds, 1 of 3 right on every item: mean 1/3 is not exactly representable
    base = [
        rec(f"i{k}", "base", float(s == 0), seed=s) for k in range(10) for s in range(3)
    ]
    edit = [
        rec(f"i{k}", "edit", float(k % 2), seed=s) for k in range(10) for s in range(3)
    ]
    assert math.isnan(compare(base, edit, n_boot=10).correlation)
    base2, edit2 = arms([0.3] * 10, [float(k) for k in range(10)])
    assert math.isnan(compare(base2, edit2, n_boot=10).correlation)


@pytest.mark.parametrize("n_boot", [0, -5])
def test_n_boot_must_be_positive(n_boot: int) -> None:
    base, edit = arms([1, 0], [0, 1])
    with pytest.raises(ValueError, match="n_boot must be >= 1"):
        compare(base, edit, n_boot=n_boot)


def test_bootstrap_chunked_matches_expected_and_is_seeded() -> None:
    n_boot = 1234  # not a multiple of the chunk size
    base = [rec(f"i{k}", "base", 1, tokens=100) for k in range(30)]
    edit = [rec(f"i{k}", "edit", 1, tokens=100 + k) for k in range(30)]
    a = compare(base, edit, n_boot=n_boot, seed=3)
    # repr, not ==: correlation is nan here and nan != nan field by field (3.13+)
    assert repr(a) == repr(compare(base, edit, n_boot=n_boot, seed=3))
    assert a.tokens_ci95[0] < a.tokens_diff < a.tokens_ci95[1]
    assert a.tokens_ci95 != compare(base, edit, n_boot=n_boot, seed=4).tokens_ci95


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_compare_rejects_non_finite_scores(bad: float) -> None:
    base, edit = arms([1, 0, 1], [1, 1, 0])
    edit[1] = rec("i1", "edit", bad)
    with pytest.raises(ValueError, match="finite"):
        compare(base, edit, n_boot=10)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_read_jsonl_rejects_non_finite_score(tmp_path: Path, token: str) -> None:
    line = _good().replace('"score": 1,', f'"score": {token},')
    assert token in line
    with pytest.raises(ValueError, match=r"line 2: score must be finite"):
        read_jsonl(_write(tmp_path, _good(), line))


def test_uneven_seed_counts_per_item() -> None:
    base = [
        rec("a", "base", 1, seed=0),
        rec("a", "base", 0, seed=1),
        rec("b", "base", 0),
    ]
    edit = [
        rec("a", "edit", 0, seed=0),
        rec("a", "edit", 0, seed=1),
        rec("b", "edit", 1),
    ]
    r = compare(base, edit, n_boot=10)
    assert r.n_items == 2
    assert r.mean_base == pytest.approx(0.25)  # a: 0.5, b: 0
    assert r.mean_edit == pytest.approx(0.5)  # a: 0, b: 1
    assert r.diff == pytest.approx(0.25)  # d = [-0.5, 1]


def _mcnemar_reference(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    return float(min(1.0, 2 * tail / 2**n))


def test_mcnemar_exact_matches_reference_small() -> None:
    for b in range(31):
        for c in range(31):
            assert _mcnemar_exact(b, c) == pytest.approx(
                _mcnemar_reference(b, c), abs=1e-12
            )


def test_mcnemar_exact_known_values() -> None:
    assert _mcnemar_exact(3, 3) == 1.0
    assert _mcnemar_exact(0, 5) == pytest.approx(0.0625)
    assert _mcnemar_exact(0, 0) == 1.0


def test_mcnemar_exact_large_is_fast_and_close_to_normal_approximation() -> None:
    b, c = 20000, 20800
    start = time.perf_counter()
    p = _mcnemar_exact(b, c)
    assert time.perf_counter() - start < 1.0
    z = (abs(b - c) - 1) / math.sqrt(b + c)
    approx = math.erfc(z / math.sqrt(2))
    assert 0.0 <= p <= 1.0
    assert p == pytest.approx(approx, rel=0.05)


def test_mcnemar_exact_extremes_stay_in_unit_interval() -> None:
    assert _mcnemar_exact(0, 100000) == 0.0
    assert _mcnemar_exact(50000, 50000) == 1.0


def test_mde_is_2_8_se() -> None:
    base, edit = arms([1, 1, 0, 0, 1, 0], [1, 0, 1, 1, 1, 1])
    r = compare(base, edit, n_boot=50)
    assert r.score_mde == pytest.approx(2.8 * r.se)
    assert r.tokens_mde == pytest.approx(2.8 * r.tokens_se)


def test_mde_nan_for_single_item() -> None:
    base, edit = arms([1], [0])
    r = compare(base, edit, n_boot=10)
    assert math.isnan(r.score_mde)
    assert math.isnan(r.tokens_mde)


def test_report_mde_only_when_ci_contains_zero() -> None:
    base, edit = arms([1, 1, 0, 0, 1, 0], [1, 0, 1, 1, 1, 1])
    r = compare(base, edit, n_boot=50)
    assert "not significant; smallest detectable difference at this n is" in (
        format_report(r)
    )
    n = 40
    base = [rec(f"i{k}", "base", float(k % 2), tokens=100 + k) for k in range(n)]
    edit = [
        rec(f"i{k}", "edit", float(k % 2) + 1.0 + 0.01 * (k % 3), tokens=300 + k)
        for k in range(n)
    ]
    text = format_report(compare(base, edit, n_boot=50))
    assert "not significant" not in text


def _noisy(shift: float, wiggle: float, n: int = 60) -> list[Record]:
    return [
        rec(f"i{k}", "x", 0.5 + shift + wiggle * ((k * 7) % 5 - 2), tokens=100 + k)
        for k in range(n)
    ]


def test_exceeds_floor_true_and_false() -> None:
    base = _noisy(0.0, 0.0)
    big = _noisy(0.5, 0.05)
    small = _noisy(0.0, 0.05)
    floor = compare(_noisy(0.0, 0.1), _noisy(0.0, 0.11), n_boot=50)
    assert exceeds_floor(compare(base, big, n_boot=50), floor)["score"]
    assert not exceeds_floor(compare(base, small, n_boot=50), floor)["score"]
    out = format_report(compare(base, big, n_boot=50), floor=floor)
    assert "noise floor (score)" in out
    assert "noise floor (tokens)" in out
    assert "does not exceed" in out  # tokens did not move


def test_exceeds_floor_requires_significance_and_size() -> None:
    base, edit = arms([1, 1, 0, 0, 1, 0], [1, 0, 1, 1, 1, 1])
    r = compare(base, edit, n_boot=50)  # CI contains 0
    zero_floor = compare(base, base_copy(base), n_boot=50)
    assert not exceeds_floor(r, zero_floor)["score"]


def base_copy(recs: list[Record]) -> list[Record]:
    return [rec(r.item_id, "base2", r.score, ph="p") for r in recs]


def test_exceeds_floor_nan_is_false() -> None:
    b1, e1 = arms([1], [0])
    nan_r = compare(b1, e1, n_boot=10)
    base, edit = arms([0] * 30, [1] * 30)
    r = compare(base, edit, n_boot=50)
    assert exceeds_floor(r, nan_r) == {"score": False, "tokens": False}
    assert exceeds_floor(nan_r, r) == {"score": False, "tokens": False}


# --- clustered standard errors ---------------------------------------------------------


def _golden_arms(binary: bool) -> tuple[list[Record], list[Record]]:
    rng = np.random.default_rng(7)
    base, edit = [], []
    for k in range(37):
        for s in range(1 if binary else 2):
            if binary:
                b, e = float(rng.integers(0, 2)), float(rng.integers(0, 2))
            else:
                b, e = float(rng.random()), float(rng.random())
            tb, te = int(rng.integers(50, 500)), int(rng.integers(50, 500))
            base.append(Record(f"i{k}", s, "base", f"p{k}", "o", b, tb, tb > 450))
            edit.append(Record(f"i{k}", s, "edit", f"p{k}", "o", e, te, te > 450))
    return base, edit


# Captured from compare() before cluster support was added; must not drift.
_GOLDEN = {
    True: PairedResult(
        n_items=37,
        mean_base=0.5945945945945946,
        mean_edit=0.5405405405405406,
        diff=-0.05405405405405406,
        se=0.11589802099341087,
        ci95=(-0.28121417520113934, 0.17310606709303125),
        correlation=0.011941628680530631,
        mcnemar_p=0.8145294189453137,
        n_base_only=10,
        n_edit_only=8,
        tokens_base=282.0,
        tokens_edit=263.3783783783784,
        tokens_diff=-18.62162162162162,
        tokens_diff_pct=-6.603411922560858,
        tokens_ci95=(-79.78175675675675, 40.41621621621616),
        cap_hit_base=0.13513513513513514,
        cap_hit_edit=0.08108108108108109,
        small_n=True,
        score_mde=0.32451445878155044,
        tokens_se=30.772031862910826,
        tokens_mde=86.16168921615031,
    ),
    False: PairedResult(
        n_items=37,
        mean_base=0.45059862551470625,
        mean_edit=0.553578546965348,
        diff=0.10297992145064187,
        se=0.03982564718293352,
        ci95=(0.024921652972092173, 0.18103818992919157),
        correlation=0.2290555006288351,
        mcnemar_p=None,
        n_base_only=0,
        n_edit_only=0,
        tokens_base=263.7432432432432,
        tokens_edit=280.3783783783784,
        tokens_diff=16.635135135135137,
        tokens_diff_pct=6.307321822001334,
        tokens_ci95=(-20.37162162162162, 51.50844594594594),
        cap_hit_base=0.08108108108108109,
        cap_hit_edit=0.0945945945945946,
        small_n=True,
        score_mde=0.11151181211221385,
        tokens_se=18.939781772154085,
        tokens_mde=53.031388962031436,
    ),
}


@pytest.mark.parametrize("binary", [True, False])
def test_unclustered_results_bit_identical_to_before(binary: bool) -> None:
    r = compare(*_golden_arms(binary), n_boot=1234, seed=5)
    want = _GOLDEN[binary]
    # correlation goes through BLAS dot products: last-bit differences across platforms
    assert r.correlation == pytest.approx(want.correlation, rel=1e-12)
    # every other field: exact float equality
    # (the cap fields were added later and are checked in their own tests)
    cap = {
        f: getattr(want, f)
        for f in (
            "cap_hit_p",
            "n_uncapped",
            "uncapped_diff",
            "uncapped_tokens_diff",
            "loops_base",
            "loops_edit",
        )
    }
    assert dataclasses.replace(r, correlation=want.correlation, **cap) == want
    assert r.n_clusters is None


def _with_clusters(recs: list[Record], cluster: dict[str, str]) -> list[Record]:
    return [dataclasses.replace(r, cluster_id=cluster[r.item_id]) for r in recs]


def _three_clusters() -> tuple[list[Record], list[Record]]:
    # d per item: A = [1, 1], B = [0], C = [0, 1, 0]; tokens follow 10 * d.
    d = {"a1": 1, "a2": 1, "b1": 0, "c1": 0, "c2": 1, "c3": 0}
    cl = {k: k[0].upper() for k in d}
    base = [rec(k, "base", 0.0, tokens=100) for k in d]
    edit = [rec(k, "edit", float(v), tokens=100 + 10 * v) for k, v in d.items()]
    return _with_clusters(base, cl), _with_clusters(edit, cl)


def test_clustered_hand_computed_three_clusters() -> None:
    r = compare(*_three_clusters(), n_boot=200)
    # dbar = 1/2; cluster residual sums A = 1, B = -1/2, C = -1/2; sum of squares 3/2.
    # SE = sqrt(3/2 * 3/2) / 6 = 1/4 (G = 3, factor G/(G-1) = 3/2).
    assert r.n_clusters == 3
    assert r.n_items == 6
    assert r.diff == pytest.approx(0.5, abs=1e-12)
    assert r.se == pytest.approx(0.25, abs=1e-12)
    assert r.ci95 == pytest.approx((0.5 - 1.96 * 0.25, 0.5 + 1.96 * 0.25), abs=1e-12)
    assert r.score_mde == pytest.approx(2.8 * 0.25, abs=1e-12)
    assert r.tokens_se == pytest.approx(2.5, abs=1e-12)
    assert r.tokens_mde == pytest.approx(2.8 * 2.5, abs=1e-12)
    # McNemar is still reported on items: 0 base-only, 3 edit-only.
    assert (r.n_base_only, r.n_edit_only) == (0, 3)
    assert r.mcnemar_p == pytest.approx(0.25)
    assert r.small_n


def test_clustered_singletons_match_unclustered_se() -> None:
    rng = np.random.default_rng(11)
    n = 150
    sb, sd = rng.normal(0.5, 0.2, n), rng.normal(0.05, 0.3, n)
    tb, td = rng.integers(100, 900, n), rng.integers(-80, 80, n)
    base = [rec(f"i{k}", "base", float(sb[k]), tokens=int(tb[k])) for k in range(n)]
    edit = [
        rec(f"i{k}", "edit", float(sb[k] + sd[k]), tokens=int(tb[k] + td[k]))
        for k in range(n)
    ]
    own = {f"i{k}": f"c{k}" for k in range(n)}
    u = compare(base, edit, n_boot=50)
    c = compare(_with_clusters(base, own), _with_clusters(edit, own), n_boot=50)
    assert c.n_clusters == n
    assert u.n_clusters is None
    for name in ("se", "score_mde", "tokens_se", "tokens_mde"):
        assert abs(getattr(c, name) - getattr(u, name)) < 1e-12, name
    assert np.allclose(c.ci95, u.ci95, rtol=0, atol=1e-12)
    assert c.diff == u.diff
    assert c.small_n == u.small_n


def test_clustered_bootstrap_resamples_clusters() -> None:
    # Cluster X: one item with token diff 0; cluster Y: nine items with diff 10.
    # Resampling 2 clusters gives a mean in {0, 9, 10} only.
    ids = [f"i{k}" for k in range(10)]
    cl = {i: ("X" if i == "i0" else "Y") for i in ids}
    base = [rec(i, "base", 0.0, tokens=100) for i in ids]
    edit = [rec(i, "edit", 0.0, tokens=100 if i == "i0" else 110) for i in ids]
    r = compare(_with_clusters(base, cl), _with_clusters(edit, cl), n_boot=4000)
    assert r.tokens_ci95 == (0.0, 10.0)
    u = compare(base, edit, n_boot=4000)
    assert u.tokens_ci95[0] > 0  # item bootstrap: some draws include i0, most don't


def test_single_cluster_gives_nan() -> None:
    base, edit = arms([0, 1, 1, 0], [1, 1, 0, 1])
    one = {r.item_id: "only" for r in base}
    r = compare(_with_clusters(base, one), _with_clusters(edit, one), n_boot=20)
    assert r.n_clusters == 1
    for v in (r.se, *r.ci95, r.score_mde, r.tokens_se, r.tokens_mde, *r.tokens_ci95):
        assert math.isnan(v)


def test_partial_cluster_id_rejected() -> None:
    base, edit = arms([0, 1, 1], [1, 1, 0])
    base = _with_clusters(base, {"i0": "a", "i1": "a", "i2": "b"})
    with pytest.raises(ValueError, match="some records but not all"):
        compare(base, edit)
    edit = _with_clusters(edit, {"i0": "a", "i1": "a", "i2": "b"})
    edit[2] = dataclasses.replace(edit[2], cluster_id=None)
    with pytest.raises(ValueError, match="'i2'"):
        compare(base, edit)


def test_arms_disagreeing_on_cluster_rejected() -> None:
    base, edit = arms([0, 1, 1], [1, 1, 0])
    base = _with_clusters(base, {"i0": "a", "i1": "a", "i2": "b"})
    edit = _with_clusters(edit, {"i0": "a", "i1": "b", "i2": "b"})
    with pytest.raises(ValueError, match="item_id 'i1' has conflicting cluster_id"):
        compare(base, edit)


def test_seeds_disagreeing_on_cluster_rejected() -> None:
    base = [
        dataclasses.replace(rec("i0", "base", 1, seed=s), cluster_id=c)
        for s, c in ((0, "a"), (1, "b"))
    ]
    edit = [dataclasses.replace(r, arm="edit") for r in base]
    with pytest.raises(ValueError, match="item_id 'i0' has conflicting cluster_id"):
        compare(base, edit)


def test_format_report_clustered() -> None:
    text = format_report(compare(*_three_clusters(), n_boot=50))
    assert "n=6 items in 3 clusters" in text
    assert "McNemar exact p=" in text
    assert "ignores clustering" in text
    assert "WARNING: only 3 clusters" in text
    plain = format_report(compare(*arms([0, 1, 1, 0], [1, 1, 1, 0]), n_boot=50))
    assert "clusters" not in plain
    assert "ignores clustering" not in plain


def test_small_n_counts_clusters() -> None:
    n = 120
    base, edit = arms([float(k % 2) for k in range(n)], [1.0] * n)
    pairs_of = {f"i{k}": f"c{k // 2}" for k in range(n)}  # 60 clusters of 2
    r = compare(_with_clusters(base, pairs_of), _with_clusters(edit, pairs_of))
    assert r.small_n
    assert not compare(base, edit).small_n
    assert "WARNING: only 60 clusters" in format_report(r)


def _sim_arms(
    rng: np.random.Generator, n_clusters: int, size: int
) -> tuple[list[Record], list[Record]]:
    # True difference 0; d = cluster effect (var 1) + item noise (var 1), ICC 0.5.
    d = np.repeat(rng.normal(0, 1, n_clusters), size) + rng.normal(
        0, 1, n_clusters * size
    )
    base, edit = [], []
    for k, dk in enumerate(d):
        c = f"c{k // size}"
        base.append(Record(f"i{k}", 0, "base", "p", "o", 0.0, 100, False, c))
        edit.append(Record(f"i{k}", 0, "edit", "p", "o", float(dk), 100, False, c))
    return base, edit


def test_clustered_ci_coverage_simulation() -> None:
    rng = np.random.default_rng(2024)
    reps, covered_c, covered_u = 400, 0, 0
    for _ in range(reps):
        base, edit = _sim_arms(rng, 200, 5)
        c = compare(base, edit, n_boot=1)
        covered_c += c.ci95[0] <= 0 <= c.ci95[1]
        nb = [dataclasses.replace(r, cluster_id=None) for r in base]
        ne = [dataclasses.replace(r, cluster_id=None) for r in edit]
        u = compare(nb, ne, n_boot=1)
        covered_u += u.ci95[0] <= 0 <= u.ci95[1]
    assert 0.92 <= covered_c / reps <= 0.98
    assert covered_u / reps < 0.90


def test_is_loop_repeating_and_not() -> None:
    assert is_loop("some reasoning. " + "so the answer is 4. " * 30)
    assert is_loop("x" * 500)
    assert not is_loop("so the answer is 4. " * 2)  # only 2 repeats in the tail
    text = "".join(f"step {k}: {k * k} then " for k in range(80))
    assert not is_loop(text)
    assert not is_loop("")


def test_is_loop_period_limit() -> None:
    assert is_loop("head" + ("b" * 99 + "c") * 4)  # period 100
    assert not is_loop("".join(chr(0x4E00 + k) for k in range(150)) * 3)  # period 150


def _cap_arms(
    n: int, base_cap: set[int], edit_cap: set[int], loop: bool = False
) -> tuple[list[Record], list[Record]]:
    spin = "I will check it again. " * 40

    def one(arm: str, k: int, cap: bool) -> Record:
        out = spin if cap and loop else "answer"
        score = 0.0 if cap else float(k % 2)
        tokens = 1000 if cap else 100 + k
        return Record(f"i{k}", 0, arm, f"h{k}", out, score, tokens, cap, None)

    return (
        [one("base", k, k in base_cap) for k in range(n)],
        [one("edit", k, k in edit_cap) for k in range(n)],
    )


def test_cap_stats_uncapped_subset_and_loops() -> None:
    base, edit = _cap_arms(10, {0, 1, 2, 3}, {0}, loop=True)
    r = compare(base, edit, n_boot=10)
    assert r.n_uncapped == 6  # items 4..9
    assert r.uncapped_diff == pytest.approx(0.0)  # same scores where neither capped
    assert r.uncapped_tokens_diff == pytest.approx(0.0)
    assert r.diff == pytest.approx(0.2)  # capped base items 1 and 3 would be right
    assert (r.loops_base, r.loops_edit) == (4, 1)
    # base-only capped = 3, edit-only = 0: p = 2 * 0.5**3
    assert r.cap_hit_p == pytest.approx(0.25)


def test_cap_stats_no_loops_for_short_capped_output() -> None:
    base, edit = _cap_arms(4, {0}, set())
    r = compare(base, edit, n_boot=10)
    assert (r.loops_base, r.loops_edit) == (0, 0)


def test_cap_stats_all_capped_has_no_uncapped_row() -> None:
    base, edit = _cap_arms(4, {0, 1, 2, 3}, {0, 1, 2, 3})
    r = compare(base, edit, n_boot=10)
    assert r.n_uncapped == 0
    assert r.uncapped_diff is None
    assert r.uncapped_tokens_diff is None
    assert r.cap_hit_p == 1.0
    assert "n=0 items" in format_report(r)


def test_pairedresult_old_constructor_still_works() -> None:
    base, edit = _cap_arms(4, set(), set())
    r = compare(base, edit, n_boot=10)
    old = {
        f.name: getattr(r, f.name)
        for f in dataclasses.fields(r)
        if f.name
        not in {
            "cap_hit_p",
            "n_uncapped",
            "uncapped_diff",
            "uncapped_tokens_diff",
            "loops_base",
            "loops_edit",
        }
    }
    r2 = PairedResult(**old)
    assert (r2.cap_hit_p, r2.n_uncapped, r2.uncapped_diff, r2.loops_base) == (
        1.0,
        0,
        None,
        0,
    )


def test_report_cap_warning_high_rate() -> None:
    base, edit = _cap_arms(20, {0, 1, 2, 3, 4, 5}, {0, 1}, loop=True)
    text = format_report(compare(base, edit, n_boot=10))
    assert "WARNING: cap-hit censoring" in text
    assert "base=30.0%  edit=10.0%" in text
    assert "censored by the cap" in text
    assert "items where neither arm hit the cap: n=14 items" in text
    assert "base=6  edit=2" in text
    assert "max_new_tokens" in text
    assert "base-arm pilot" in text


def test_report_cap_warning_on_unequal_rates_alone() -> None:
    # rates 4.5% and 0 (both <= 5%) but 9 base-only caps: McNemar p = 2 * 0.5**9 < 0.05
    base, edit = _cap_arms(200, set(range(9)), set())
    r = compare(base, edit, n_boot=10)
    assert max(r.cap_hit_base, r.cap_hit_edit) <= 0.05
    assert r.cap_hit_p < 0.05
    assert "WARNING: cap-hit censoring" in format_report(r)


def test_report_no_cap_warning_unchanged() -> None:
    # at the threshold (5%) and balanced: no block, report is the 0.1.1 text
    base, edit = _cap_arms(20, {0}, {1})
    r = compare(base, edit, n_boot=10)
    text = format_report(r)
    assert "cap-hit censoring" not in text
    assert text.splitlines()[-1].startswith(("cap-hit rate:", "WARNING: only"))
    base, edit = _cap_arms(20, set(), set())
    assert "cap-hit censoring" not in format_report(compare(base, edit, n_boot=10))
