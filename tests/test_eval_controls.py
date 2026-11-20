import dataclasses
import math

import pytest

from astro_weights.eval import PairedResult, control_rule, min_controls

_BASE = PairedResult(
    n_items=100, mean_base=0.5, mean_edit=0.5, diff=0.0, se=0.0, ci95=(0.0, 0.0),
    correlation=0.0, mcnemar_p=None, n_base_only=0, n_edit_only=0, tokens_base=100.0,
    tokens_edit=100.0, tokens_diff=0.0, tokens_diff_pct=0.0, tokens_ci95=(0.0, 0.0),
    cap_hit_base=0.0, cap_hit_edit=0.0, small_n=False, score_mde=0.0, tokens_se=0.0,
    tokens_mde=0.0,
)  # fmt: skip


def pr(diff: float = 0.0, tokens_diff: float = 0.0) -> PairedResult:
    return dataclasses.replace(_BASE, diff=diff, tokens_diff=tokens_diff)


def ctrls(values: list[float]) -> list[PairedResult]:
    return [pr(v) for v in values]


SPREAD = [float(i) for i in range(-10, 10)]  # integers: exact ties


def test_min_controls() -> None:
    assert min_controls(0.95) == 20
    assert min_controls(0.9) == 10


def test_clear_winner_passes_higher() -> None:
    v = control_rule(pr(20.0), ctrls(SPREAD), target="diff", direction="higher")
    assert v.passed
    assert v.target_beats_quantile
    assert v.target_rank == 20
    assert v.n_controls == 20
    assert v.target_quantile == 8.0  # 19th of 20
    assert v.side_effects == {}
    assert v.outside == ()


def test_clear_winner_passes_lower() -> None:
    ctl = [pr(tokens_diff=t) for t in range(-10, 10)]
    v = control_rule(pr(tokens_diff=-200), ctl, target="tokens_diff", direction="lower")
    assert v.passed
    assert v.target_rank == 20
    assert v.target_quantile == -9  # 2nd lowest of the 20, in token units


def test_inside_spread_fails() -> None:
    v = control_rule(pr(0.0), ctrls(SPREAD), target="diff", direction="higher")
    assert not v.passed
    assert not v.target_beats_quantile
    assert 0 < v.target_rank < 19


def test_wrong_direction_fails() -> None:
    v = control_rule(pr(20.0), ctrls(SPREAD), target="diff", direction="lower")
    assert not v.passed
    assert v.target_rank == 0


def test_tie_at_quantile_fails_but_just_above_passes() -> None:
    q = 8.0
    assert not control_rule(
        pr(q), ctrls(SPREAD), target="diff", direction="higher"
    ).passed
    assert control_rule(
        pr(q + 0.5), ctrls(SPREAD), target="diff", direction="higher"
    ).passed


def test_side_effect_outside_named() -> None:
    se = {
        "ok": (pr(0.0), ctrls(SPREAD)),
        "bad": (pr(-50.0), ctrls(SPREAD)),
    }
    v = control_rule(
        pr(20.0), ctrls(SPREAD), target="diff", direction="higher", side_effects=se
    )
    assert v.target_beats_quantile
    assert not v.passed
    assert v.outside == ("bad",)
    assert v.side_effects == {"ok": True, "bad": False}


def test_side_effect_bounds_inclusive() -> None:
    se = {"edge": (pr(max(SPREAD)), ctrls(SPREAD))}
    v = control_rule(
        pr(20.0), ctrls(SPREAD), target="diff", direction="higher", side_effects=se
    )
    assert v.passed


def test_too_few_controls() -> None:
    with pytest.raises(ValueError, match="19 controls, need at least 20"):
        control_rule(pr(1), ctrls(SPREAD[:19]), target="diff", direction="higher")
    with pytest.raises(ValueError, match=r"side effect 'x'.*need at least 20"):
        control_rule(
            pr(1),
            ctrls(SPREAD),
            target="diff",
            direction="higher",
            side_effects={"x": (pr(0), ctrls(SPREAD[:5]))},
        )


def test_nan_and_bad_args() -> None:
    nan = math.nan
    with pytest.raises(ValueError, match="edit"):
        control_rule(pr(nan), ctrls(SPREAD), target="diff", direction="higher")
    with pytest.raises(ValueError, match="controls"):
        control_rule(
            pr(1), ctrls([*SPREAD[:19], nan]), target="diff", direction="higher"
        )
    with pytest.raises(ValueError, match="side effect 'x' edit"):
        control_rule(
            pr(1),
            ctrls(SPREAD),
            target="diff",
            direction="higher",
            side_effects={"x": (pr(nan), ctrls(SPREAD))},
        )
    with pytest.raises(ValueError, match="q must"):
        control_rule(pr(1), ctrls(SPREAD), target="diff", direction="higher", q=1.0)
    with pytest.raises(ValueError, match="direction"):
        control_rule(pr(1), ctrls(SPREAD), target="diff", direction="up")  # type: ignore[arg-type]


def test_verdict_equality_has_no_nan() -> None:
    a = control_rule(pr(20.0), ctrls(SPREAD), target="diff", direction="higher")
    b = control_rule(pr(20.0), ctrls(SPREAD), target="diff", direction="higher")
    assert a == b


@pytest.mark.parametrize("target", ["dif", "mcnemar_p"])
def test_unknown_or_non_numeric_target_is_a_value_error(target: str) -> None:
    edit = pr(1.0)
    with pytest.raises(ValueError, match="numeric PairedResult field"):
        control_rule(edit, ctrls([0.0] * 20), target=target, direction="higher")
