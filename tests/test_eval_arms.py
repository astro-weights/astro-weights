"""compare_arms / format_arms: one base, many arms, one report."""

import json
import statistics
from typing import Any

import pytest

from astro_weights.eval import Record, arms_dict, compare, compare_arms, format_arms


def _arm(
    name: str, scores: list[float], toks: int = 100, seeds: int = 1
) -> list[Record]:
    return [
        Record(f"i{k}", s, name, f"p{k}", "o", x, toks, False)
        for k, x in enumerate(scores)
        for s in range(seeds)
    ]


BASE = _arm("base", [1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
ARMS = {
    "edit": _arm("edit", [1, 1, 1, 1, 1, 1, 0, 0, 1, 1], 120),
    "ctl0": _arm("ctl0", [1, 0, 1, 0, 1, 0, 0, 0, 1, 0], 100),
    "ctl1": _arm("ctl1", [0, 0, 1, 0, 1, 1, 0, 1, 1, 0], 90),
    "ctl2": _arm("ctl2", [1, 0, 1, 1, 1, 1, 0, 0, 0, 0], 110),
    "prompt": _arm("prompt", [1, 0, 0, 0, 1, 1, 1, 0, 1, 0], 150),
}
KINDS = {
    "edit": "edit",
    "ctl0": "control",
    "ctl1": "control",
    "ctl2": "control",
    "prompt": "prompt",
}


def test_rows_match_compare_and_order_is_kept() -> None:
    res = compare_arms(BASE, ARMS, KINDS, n_boot=200)
    assert res.n_arms == 5
    assert list(res.results) == list(ARMS)
    for name, recs in ARMS.items():
        want = compare(BASE, recs, n_boot=200)
        got = res.results[name]
        assert (got.diff, got.ci95, got.score_mde, got.tokens_diff, got.n_items) == (
            want.diff,
            want.ci95,
            want.score_mde,
            want.tokens_diff,
            want.n_items,
        )
    assert res.kinds["prompt"] == "prompt"


def test_default_kind_is_edit_and_bad_kinds_refused() -> None:
    assert compare_arms(BASE, {"a": ARMS["edit"]}, n_boot=50).kinds == {"a": "edit"}
    with pytest.raises(ValueError, match="unknown arm kinds"):
        compare_arms(BASE, {"a": ARMS["edit"]}, {"a": "wat"}, n_boot=50)
    with pytest.raises(ValueError, match="not given"):
        compare_arms(BASE, {"a": ARMS["edit"]}, {"b": "edit"}, n_boot=50)
    with pytest.raises(ValueError, match="no arms"):
        compare_arms(BASE, {})


def test_mismatched_item_set_names_the_arm() -> None:
    bad = _arm("x", [1] * 9)  # i9 missing
    with pytest.raises(
        ValueError, match=r"arm 'oops' item set differs.*1 only in base"
    ):
        compare_arms(BASE, {"edit": ARMS["edit"], "oops": bad}, n_boot=50)


def test_mismatched_seed_set_names_the_arm() -> None:
    base = _arm("base", [1, 0, 1, 0], seeds=2)
    bad = [
        r for r in _arm("e", [1, 1, 1, 1], seeds=2) if (r.item_id, r.seed) != ("i0", 1)
    ]
    with pytest.raises(ValueError, match="arm 'e' seed set differs"):
        compare_arms(base, {"e": bad}, n_boot=50)


def test_every_row_has_mde_and_controls_summary_is_right() -> None:
    res = compare_arms(BASE, ARMS, KINDS, n_boot=200)
    out = format_arms(res)
    lines = out.splitlines()
    assert lines[0].split()[:2] == ["arm", "kind"]
    assert [ln.split()[0] for ln in lines[1:-1]] == ["edit", "prompt"]
    ctl = [res.results[n] for n in ("ctl0", "ctl1", "ctl2")]
    for n in ("edit", "prompt"):
        assert f"{res.results[n].score_mde:.4f}" in out
    diffs = [r.diff for r in ctl]
    want = (
        f"controls: n=3  diff min={min(diffs):+.4f}  median={statistics.median(diffs):+.4f}"
        f"  max={max(diffs):+.4f}"
    )
    assert lines[-1].startswith(want)
    assert "ctl0" not in out


def test_full_controls_adds_rows_in_order_each_with_mde() -> None:
    res = compare_arms(BASE, ARMS, KINDS, n_boot=200)
    lines = format_arms(res, full_controls=True).splitlines()
    assert [ln.split()[0] for ln in lines[1:-1]] == list(ARMS)
    for ln, name in zip(lines[1:-1], ARMS, strict=True):
        assert f"{res.results[name].score_mde:.4f}" in ln
    assert format_arms(res, True) == format_arms(res, True)  # stable


def test_no_controls_no_summary_line() -> None:
    res = compare_arms(BASE, {"edit": ARMS["edit"]}, n_boot=50)
    assert "controls:" not in format_arms(res)


def test_dict_form_is_json() -> None:
    res = compare_arms(BASE, ARMS, KINDS, n_boot=200)
    d: Any = arms_dict(res)
    assert json.loads(json.dumps(d)) == d
    assert d["n_arms"] == 5
    assert [a["arm"] for a in d["arms"]] == ["edit", "prompt"]
    assert d["controls"]["n"] == 3
    full: Any = arms_dict(res, True)
    assert len(full["arms"]) == 5
