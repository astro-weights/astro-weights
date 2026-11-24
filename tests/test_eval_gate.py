import dataclasses
import itertools
import math

import numpy as np
import pytest

from astro_weights.eval import (
    ControlVerdict,
    FloorResult,
    GateInputs,
    PairedResult,
    Record,
    Screen,
    compare,
    control_rule,
    floor_from_records,
    format_gate,
    gate,
)
from astro_weights.eval.gate import OUTCOMES, ni_ci

_BASE = PairedResult(
    n_items=200, mean_base=0.5, mean_edit=0.5, diff=0.0, se=0.01, ci95=(-0.02, 0.02),
    correlation=0.0, mcnemar_p=None, n_base_only=0, n_edit_only=0, tokens_base=100.0,
    tokens_edit=50.0, tokens_diff=-50.0, tokens_diff_pct=-50.0, tokens_ci95=(-60.0, -40.0),
    cap_hit_base=0.0, cap_hit_edit=0.0, small_n=False, score_mde=0.028, tokens_se=5.0,
    tokens_mde=14.0, k=4, score_ci95_boot=(-0.01, 0.01),
)  # fmt: skip
_FLOOR = FloorResult(
    config_hash="0" * 64, n_items=200, k=4, accuracy_delta=0.0, accuracy_se=0.01,
    flip_rate=0.0, tokens_delta=1.0, tokens_se=2.0, created="2026-11-23T00:00:00+00:00",
    library_version=None,
)  # fmt: skip
SCREENS = ("norm", "perplexity", "kl")


def screens(fails: tuple[str, ...] = ()) -> tuple[Screen, ...]:
    return tuple(Screen(n, n not in fails, 1.0, 2.0) for n in SCREENS)


def controls(target: bool = True, side: bool = True) -> ControlVerdict:
    return ControlVerdict(
        passed=target and side, target_beats_quantile=target,
        target_rank=20 if target else 10, target_quantile=-30.0, q=0.95, n_controls=20,
        side_effects={"accuracy": side}, outside=() if side else ("accuracy",),
    )  # fmt: skip


def scored(acc: bool = True, tokens: bool = True) -> PairedResult:
    return dataclasses.replace(
        _BASE,
        score_ci95_boot=(-0.01, 0.01) if acc else (-0.03, 0.0),
        tokens_diff=-50.0 if tokens else 0.0,
        tokens_ci95=(-60.0, -40.0) if tokens else (-5.0, 5.0),
    )


def suite(ok: bool) -> tuple[PairedResult, float, str]:
    return (
        dataclasses.replace(_BASE, score_ci95_boot=(-0.01 if ok else -0.05, 0.0)),
        0.02,
        f"pilot {'ok' if ok else 'bad'}",
    )


def full(**kw: object) -> GateInputs:
    args: dict[str, object] = {
        "screens": screens(),
        "margin": 0.02,
        "margin_source": "pilot dev run 2026-11-20",
        "confirm": _BASE,
        "controls": controls(),
        "test": scored(),
        "floor": _FLOOR,
        "k_test": 4,
        "suites": {"gsm8k": suite(True), "ifeval": suite(True)},
    }
    args.update(kw)
    return GateInputs(**args)  # type: ignore[arg-type]


# ---- truth table ----------------------------------------------------------------------

LATER = ("target", "side", "accuracy", "tokens", "gsm8k", "ifeval")
_ON_FAIL = {
    "target": "not significant",
    "side": "regressed",
    "accuracy": "regressed",
    "tokens": "not significant",
    "gsm8k": "regressed",
    "ifeval": "regressed",
}
_CHECK_WORD = {
    "target": "target beats",
    "side": "side effect accuracy",
    "accuracy": "accuracy non-inferior",
    "tokens": "tokens lower",
    "gsm8k": "suite gsm8k",
    "ifeval": "suite ifeval",
}
SCREEN_FAILS = [
    c for r in range(1, 4) for c in itertools.combinations(SCREENS, r)
]  # every non-empty subset
LATER_CASES = list(itertools.product((True, False), repeat=len(LATER)))


@pytest.mark.parametrize("fails", SCREEN_FAILS)
def test_truth_table_screened_out(fails: tuple[str, ...]) -> None:
    r = gate(GateInputs(screens(fails), 0.02, "pilot"))
    assert r.outcome == "screened out"
    assert [c.check for c in r.reasons] == [f"screen {n}" for n in fails]
    assert all(c.stage == 1 for c in r.checks)
    assert r.mde is None
    assert r.advice == "keep the original weights"
    assert "MDE" not in format_gate(r)


@pytest.mark.parametrize("bits", LATER_CASES)
def test_truth_table_later_stages(bits: tuple[bool, ...]) -> None:
    ok = dict(zip(LATER, bits, strict=True))
    r = gate(
        full(
            controls=controls(ok["target"], ok["side"]),
            test=scored(ok["accuracy"], ok["tokens"]),
            suites={"gsm8k": suite(ok["gsm8k"]), "ifeval": suite(ok["ifeval"])},
        )
    )
    failed = [n for n in LATER if not ok[n]]
    outs = {_ON_FAIL[n] for n in failed}
    want = (
        "regressed"
        if "regressed" in outs
        else "not significant"
        if outs
        else "confirmed"
    )
    assert r.outcome == want
    assert len(r.reasons) == len(failed)
    for c, n in zip(r.reasons, failed, strict=True):
        assert _CHECK_WORD[n] in c.check
        assert not c.passed
    assert len(r.checks) == 3 + len(LATER)
    assert r.mde == _BASE.score_mde
    assert r.mde_split == "test"
    assert (r.advice is None) == (want == "confirmed")
    text = format_gate(r)
    assert f"outcome: {want}" in text
    assert "MDE (test): 0.0280" in text
    assert ("advice: keep the original weights" in text) == (want != "confirmed")


def test_outcomes() -> None:
    assert OUTCOMES == ("confirmed", "not significant", "regressed", "screened out")


# ---- later stages optional once an earlier one failed ---------------------------------


def test_stage_4_optional_after_control_fail() -> None:
    r = gate(full(controls=controls(target=False), test=None, floor=None, k_test=None,
                  suites={}))  # fmt: skip
    assert r.outcome == "not significant"
    assert r.mde_split == "confirm"
    assert "MDE (confirm)" in format_gate(r)


def test_stage_5_optional_after_test_fail() -> None:
    r = gate(full(test=scored(acc=False), suites={}))
    assert r.outcome == "regressed"


def test_side_effect_without_numbers_formats() -> None:
    r = gate(full(controls=controls(side=False)))
    assert "value=-  limit=-" in format_gate(r)


# ---- boundaries -----------------------------------------------------------------------


def test_ci_lower_equal_to_minus_margin_fails() -> None:
    t = dataclasses.replace(_BASE, score_ci95_boot=(-0.02, 0.01))
    r = gate(full(test=t))
    assert r.outcome == "regressed"
    (c,) = r.reasons
    assert c.value == -0.02
    assert c.limit == -0.02


def test_margin_cap() -> None:
    assert gate(full(margin=0.02)).margin == 0.02
    assert gate(full(margin=0.0)).outcome == "regressed"  # lo -0.01 not > 0
    for bad in (0.0201, -0.001, math.nan, None, True):
        with pytest.raises(ValueError, match="margin"):
            gate(full(margin=bad))
    with pytest.raises(ValueError, match=r"suite 'gsm8k'.*margin"):
        gate(full(suites={"gsm8k": (_BASE, 0.0201, "pilot")}))


def test_tie_with_control_quantile_not_significant() -> None:
    ctl = [dataclasses.replace(_BASE, tokens_diff=float(t)) for t in range(-10, 10)]
    edit = dataclasses.replace(_BASE, tokens_diff=-9.0)  # equals the q95 of goodness
    acc = {"accuracy": (edit, ctl)}
    v = control_rule(
        edit, ctl, target="tokens_diff", direction="lower", side_effects=acc
    )
    assert v.target_quantile == -9.0
    r = gate(full(controls=v))
    assert r.outcome == "not significant"
    (c,) = r.reasons
    assert c.value == 18  # strictly beats 18
    assert c.limit == 19  # needs 19
    v2 = control_rule(
        dataclasses.replace(edit, tokens_diff=-9.5),
        ctl,
        target="tokens_diff",
        direction="lower",
        side_effects=acc,
    )
    assert gate(full(controls=v2)).outcome == "confirmed"


def test_tokens_at_floor_limit_not_significant() -> None:
    t = dataclasses.replace(_BASE, tokens_diff=-(1.0 + 1.96 * 2.0))
    r = gate(full(test=t))
    assert r.outcome == "not significant"
    assert r.reasons[0].limit == pytest.approx(-4.92)


# ---- CI rule --------------------------------------------------------------------------


def test_ni_ci_rule() -> None:
    small = dataclasses.replace(_BASE, small_n=True, newcombe_ci95=(-0.05, 0.05))
    assert ni_ci(small) == ((-0.05, 0.05), "newcombe")
    assert ni_ci(dataclasses.replace(small, small_n=False))[1] == "bootstrap"
    assert ni_ci(dataclasses.replace(small, newcombe_ci95=None))[1] == "bootstrap"
    r = gate(full(test=dataclasses.replace(small, score_ci95_boot=(0.0, 0.0))))
    assert r.outcome == "regressed"  # Newcombe chosen over the narrow bootstrap
    assert "newcombe" in r.reasons[0].check


# ---- refusals -------------------------------------------------------------------------

NAN = (math.nan, math.nan)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"margin_source": " "}, "margin_source"),
        ({"k_test": 1}, "K=4"),
        ({"test": dataclasses.replace(_BASE, k=1)}, "K=4"),
        ({"test": dataclasses.replace(_BASE, score_ci95_boot=NAN)}, "NaN"),
        ({"test": dataclasses.replace(_BASE, tokens_ci95=NAN)}, "NaN"),
        ({"test": dataclasses.replace(_BASE, score_mde=math.nan)}, "NaN"),
        ({"confirm": dataclasses.replace(_BASE, score_mde=math.inf)}, "confirm MDE"),
        ({"suites": {"x": (dataclasses.replace(_BASE, score_ci95_boot=NAN), 0.02, "p")}}, "NaN"),
        ({"suites": {"x": (_BASE, 0.02, " ")}}, "suite 'x': margin source"),
        ({"floor": dataclasses.replace(_FLOOR, k=1)}, "floor k=1"),
        ({"screens": screens()[:2]}, "missing screens: kl"),
        ({"screens": screens()[1:]}, "missing screens: norm"),
        ({"controls": dataclasses.replace(controls(), q=0.9, n_controls=20)}, "q outside"),
        ({"controls": dataclasses.replace(controls(), side_effects={})}, "side effect"),
        ({"controls": dataclasses.replace(controls(), side_effects={"ifeval": True})},
         "'accuracy' side effect"),
        ({"floor": None}, "together"),
        ({"test": None}, "together"),
        ({"test": None, "floor": None, "k_test": None}, "stage 4 is missing"),
        ({"suites": {}}, "stage 5 is missing"),
        ({"confirm": None}, "required"),
        ({"controls": None}, "required"),
        ({"screens": ()}, "no pre-screens"),
        ({"screens": (*screens(), Screen("kl", True, 0, 1))}, "duplicate"),
        ({"screens": (*screens()[:2], Screen("kl", True, math.nan, 1))}, "finite"),
        ({"screens": (*screens()[:2], Screen("kl", 1, 0, 1))}, "bool"),  # type: ignore[arg-type]
        ({"controls": dataclasses.replace(controls(), passed=False)}, "inconsistent"),
        ({"controls": dataclasses.replace(controls(), n_controls=5)}, "need 20"),
        ({"controls": dataclasses.replace(controls(), target_quantile=math.nan)}, "NaN"),
        ({"controls": dataclasses.replace(controls(), q=1.0)}, "q outside"),
    ],
)  # fmt: skip
def test_refuses_inconsistent_inputs(kw: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        gate(full(**kw))


def test_non_token_target_needs_any_side_effect_only() -> None:
    c = dataclasses.replace(controls(), side_effects={"ifeval": True})
    assert gate(full(controls=c, control_target="diff")).outcome == "confirmed"


def test_suite_margins_recorded() -> None:
    r = gate(full())
    assert r.suite_margins == {
        "gsm8k": (0.02, "pilot ok"),
        "ifeval": (0.02, "pilot ok"),
    }
    assert "suite gsm8k margin: 0.0200 (pilot ok)" in format_gate(r)


def test_refuses_suites_without_test() -> None:
    with pytest.raises(ValueError, match="stage 5 given without stage 4"):
        gate(full(controls=controls(target=False), test=None, floor=None, k_test=None))


@pytest.mark.parametrize(
    "kw",
    [{"confirm": _BASE}, {"controls": controls()}, {"test": _BASE}, {"k_test": 4},
     {"suites": {"gsm8k": suite(True)}}],
)  # fmt: skip
def test_screened_out_refuses_later_stages(kw: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="no later stage"):
        gate(GateInputs(screens(("kl",)), 0.02, "pilot", **kw))  # type: ignore[arg-type]


def test_screened_out_still_needs_margin() -> None:
    with pytest.raises(ValueError, match="margin is missing"):
        gate(GateInputs(screens(("kl",)), None, "pilot"))


# ---- planted overfit (PDR 0.3 section 7.1) -------------------------------------------

N_ITEMS = 100
TOKENS_SD = 50.0


class _World:
    """Items with a fixed answer (accuracy is identical in every arm, so only the
    token claim is at stake) and per-item mean length. An arm with `shift=0` is a
    no-effect edit: its tokens are fresh draws from the base distribution.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.correct = {
            s: rng.random(N_ITEMS) < 0.6 for s in ("confirm", "test", "suite")
        }
        self.mu = {
            s: rng.uniform(200, 400, N_ITEMS) for s in ("confirm", "test", "suite")
        }

    def arm(
        self, split: str, name: str, k: int = 1, shift: float = 0.0
    ) -> list[Record]:
        out = []
        for i in range(N_ITEMS):
            toks = self.rng.normal(self.mu[split][i] + shift, TOKENS_SD, k)
            out += [
                Record(f"{split}{i}", s, name, "h", "", float(self.correct[split][i]),
                       max(1, round(float(t))), False)
                for s, t in enumerate(toks)
            ]  # fmt: skip
        return out


def _run_gate(w: _World, n_cand: int, shift: float) -> tuple[bool, str]:
    """Pick the lowest-token of `n_cand` candidates (each truly shifted by `shift`) on
    confirm, judge it against 20 controls on the same confirm data, then score it once
    on a fresh K=4 test split. Returns (controls passed, outcome).
    """
    n_boot = 200
    cbase = w.arm("confirm", "base")
    cands = [w.arm("confirm", "edit", shift=shift) for _ in range(n_cand)]
    best = min(cands, key=lambda c: sum(r.n_output_tokens for r in c))
    edit = compare(cbase, best, n_boot=n_boot)
    ctl = [compare(cbase, w.arm("confirm", f"c{j}"), n_boot=n_boot) for j in range(20)]
    v = control_rule(
        edit, ctl, target="tokens_diff", direction="lower",
        side_effects={"accuracy": (edit, ctl)},  # diff is the accuracy side
    )  # fmt: skip
    tbase = w.arm("test", "base", k=4)
    floor = floor_from_records(tbase, w.arm("test", "base2", k=4), {"model": "sim"})
    test = compare(tbase, w.arm("test", "edit", k=4, shift=shift), n_boot=n_boot)
    sbase = w.arm("suite", "base")
    suite_r = compare(
        sbase, [dataclasses.replace(r, arm="edit") for r in sbase], n_boot=1
    )
    r = gate(
        GateInputs(
            screens(), 0.02, "sim", confirm=edit, controls=v, test=test, floor=floor,
            k_test=4, suites={"suite": (suite_r, 0.02, "sim pilot")},
        )
    )  # fmt: skip
    return v.target_beats_quantile, r.outcome


def test_planted_overfit_null_search_confirms_at_most_nominal() -> None:
    """Best of N=50 no-effect candidates picked on confirm, the same data the controls
    use. Selection beats the controls almost always, so stage 3 gives no protection here:
    only the once-scored test split (one-sided token floor, alpha 2.5%) does. Over 100
    seeded reps: controls pass >= 80%, confirmed <= 2.5% + 3 sigma.
    """
    rng = np.random.default_rng(20261123)
    reps = 100
    ctl_pass = confirmed = 0
    for _ in range(reps):
        passed, outcome = _run_gate(_World(rng), n_cand=50, shift=0.0)
        ctl_pass += passed
        confirmed += outcome == "confirmed"
    alpha = 0.025
    assert ctl_pass / reps >= 0.8  # selection on confirm defeats the control rule
    assert confirmed / reps <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / reps)


def test_planted_real_effect_is_confirmed() -> None:
    """Same harness, one candidate with a real 40-token cut: not vacuously strict."""
    rng = np.random.default_rng(7)
    outcomes = [_run_gate(_World(rng), n_cand=1, shift=-40.0)[1] for _ in range(10)]
    assert outcomes.count("confirmed") >= 9
