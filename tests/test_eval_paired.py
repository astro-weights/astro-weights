import json
import math
from pathlib import Path

import numpy as np
import pytest

from astro_weights.eval import (
    PairedResult,
    Record,
    compare,
    format_report,
    pair,
    prompt_hash,
    read_jsonl,
    write_jsonl,
)


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
