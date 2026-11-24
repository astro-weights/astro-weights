"""The 0.3 gate: one verdict from pre-screens, controls, the test split and suites.

Constraint style (PDR 0.3 section 4.6), no blended score. Stages, in order:

1. Pre-screens (update norm, perplexity, KL; all three required), given as `Screen`
   pass/fail results. Any
   fail gives outcome "screened out": no verdict, and no later stage may be given (the
   test split is never scored, its seal unused).
2. Selection on confirm is the agent loop's business and never reaches the gate.
3. Random controls on confirm (`ControlVerdict` from `control_rule`): the edit not above
   the controls' q-quantile (q >= 0.95) on the target gives "not significant"; any side
   effect outside the controls' range gives "regressed". At least one side effect is
   required, and "accuracy" when the target is `tokens_diff`.
4. Final test, scored once with K = 4 samples per item. Accuracy is non-inferior when
   the lower end of the two-sided 95% CI of the score difference is strictly above
   `-margin`, else "regressed". Tokens must be lower beyond the token floor
   (`tokens_beyond_floor`), else "not significant".
5. Regression suites: each suite's accuracy non-inferior within its own margin, else
   "regressed".

CI rule for non-inferiority (audit C3, C13): Newcombe's paired interval
(`newcombe_ci95`) when scores are binary per item and `small_n` (fewer than 100 items),
else the percentile paired bootstrap (`score_ci95_boot`). The Wald interval is used only
for the MDE.

A stage is required when every earlier stage passed; after a failure later stages may
still be given and are then judged too. Precedence: regressed beats not significant
beats confirmed. Every outcome but confirmed carries the advice to keep the original
weights; it is advice, not a verdict. Pure functions, no I/O. Never runs a model.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from astro_weights.eval.controls import ControlVerdict, min_controls
from astro_weights.eval.floor import FloorResult, tokens_beyond_floor
from astro_weights.eval.paired import PairedResult
from astro_weights.eval.verdict import VERDICTS

SCREENED_OUT = "screened out"
OUTCOMES = (*VERDICTS, SCREENED_OUT)
MAX_MARGIN = 0.02  # 2 pp cap on every non-inferiority margin (PDR 0.3 section 8)
K_TEST = 4  # samples per item on the test split
MIN_Q = 0.95  # control quantile the target must beat (PDR 0.3 section 4.6)
REQUIRED_SCREENS = frozenset({"norm", "perplexity", "kl"})
_Z = 1.96
ADVICE = "keep the original weights"


@dataclass(frozen=True)
class Screen:
    """One pre-screen result: `passed` as judged by the guard that made it, plus the
    measured `value` and the `limit` it was held to (both finite).
    """

    name: str
    passed: bool
    value: float
    limit: float


@dataclass(frozen=True)
class Check:
    """One check the gate made. `value` and `limit` are None when the source result
    carries no number for it (control side effects). `on_fail` is the outcome a failure
    leads to.
    """

    stage: int
    check: str
    passed: bool
    value: float | None
    limit: float | None
    on_fail: str


@dataclass(frozen=True)
class GateInputs:
    """Everything the gate judges.

    `screens` are stage 1. `confirm` is the edit's paired result on the confirm split and
    `controls` the control rule on it (stage 3). `test` and `floor` are stage 4, `k_test`
    the samples per item scored there (must be 4, and the floor's `k` too). `suites` maps
    a regression suite name to its paired result, margin and margin source (stage 5).
    `control_target` is the `PairedResult` field the control rule judged. `margin` is the test benchmark's
    non-inferiority margin in score units (0.02 = 2 pp), fixed from a pilot before the
    test was scored; `margin_source` says where it came from. Margins must lie in
    [0, 0.02].
    """

    screens: tuple[Screen, ...]
    margin: float | None
    margin_source: str
    confirm: PairedResult | None = None
    controls: ControlVerdict | None = None
    test: PairedResult | None = None
    floor: FloorResult | None = None
    k_test: int | None = None
    suites: Mapping[str, tuple[PairedResult, float, str]] = field(default_factory=dict)
    control_target: str = "tokens_diff"


@dataclass(frozen=True)
class GateResult:
    """Outcome of `gate`.

    `outcome` is one of `OUTCOMES`. `checks` lists every check made, in stage order, and
    `reasons` the failed ones. `mde` is the test split's score MDE, or the confirm split's
    when the test was not scored (`mde_split` says which); None when screened out.
    `advice` is None only when confirmed. `suite_margins` maps each judged suite to its
    margin and margin source.
    """

    outcome: str
    checks: tuple[Check, ...]
    reasons: tuple[Check, ...]
    mde: float | None
    mde_split: str | None
    margin: float
    margin_source: str
    advice: str | None
    suite_margins: Mapping[str, tuple[float, str]] = field(default_factory=dict)


def _finite(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def _check_margin(margin: object, what: str) -> float:
    if margin is None:
        raise ValueError(f"{what}: margin is missing; fix it from a pilot first")
    if not _finite(margin) or not 0 <= margin <= MAX_MARGIN:  # type: ignore[operator]
        raise ValueError(f"{what}: margin must be a number in [0, {MAX_MARGIN}]")
    return float(margin)  # type: ignore[arg-type]


def ni_ci(result: PairedResult) -> tuple[tuple[float, float], str]:
    """The 95% CI of the score difference used for non-inferiority, and its method.

    Newcombe when scores are binary per item and `small_n`, else the paired bootstrap.
    ValueError when that interval is NaN or infinite.
    """
    if result.small_n and result.newcombe_ci95 is not None:
        ci, method = result.newcombe_ci95, "newcombe"
    else:
        ci, method = result.score_ci95_boot, "bootstrap"
    if not all(_finite(v) for v in ci):
        raise ValueError(f"{method} score CI is NaN or infinite")
    return ci, method


def _ni(stage: int, name: str, r: PairedResult, margin: float) -> Check:
    (lo, _), method = ni_ci(r)
    check = f"{name} non-inferior ({method} CI lower > -margin)"
    return Check(stage, check, lo > -margin, lo, -margin, "regressed")


def _stage1(screens: tuple[Screen, ...]) -> list[Check]:
    if not screens:
        raise ValueError("stage 1: no pre-screens given")
    names = [s.name for s in screens]
    if len(set(names)) != len(names):
        raise ValueError("stage 1: duplicate screen names")
    if missing := sorted(REQUIRED_SCREENS - set(names)):
        raise ValueError(f"stage 1: missing screens: {', '.join(missing)}")
    for s in screens:
        if not isinstance(s.passed, bool) or not (
            _finite(s.value) and _finite(s.limit)
        ):
            raise ValueError(
                f"stage 1: screen {s.name!r} needs a bool and finite numbers"
            )
    return [
        Check(1, f"screen {s.name}", s.passed, s.value, s.limit, SCREENED_OUT)
        for s in screens
    ]


def _stage3(c: ControlVerdict, target: str) -> list[Check]:
    if not (_finite(c.target_quantile) and MIN_Q <= c.q < 1):
        raise ValueError(f"stage 3: control quantile is NaN or q outside [{MIN_Q}, 1)")
    if not c.side_effects:
        raise ValueError("stage 3: controls must be scored on at least one side effect")
    if target == "tokens_diff" and "accuracy" not in c.side_effects:
        raise ValueError("stage 3: a token target needs an 'accuracy' side effect")
    if c.n_controls < min_controls(c.q):
        raise ValueError(f"stage 3: {c.n_controls} controls, need {min_controls(c.q)}")
    if c.passed != (c.target_beats_quantile and not c.outside):
        raise ValueError("stage 3: control verdict is inconsistent")
    # beats the quantile <=> strictly beats at least ceil(q * n) controls
    need = math.ceil(c.q * c.n_controls - 1e-9)
    out = [
        Check(
            3,
            f"target beats control q{c.q:g} (controls beaten >= limit)",
            c.target_beats_quantile,
            float(c.target_rank),
            float(need),
            "not significant",
        )
    ]
    out += [
        Check(3, f"side effect {n} inside control range", ok, None, None, "regressed")
        for n, ok in c.side_effects.items()
    ]
    return out


def _stage4(inp: GateInputs, margin: float) -> list[Check]:
    test, floor = inp.test, inp.floor
    if test is None or floor is None:
        raise ValueError("stage 4: test and floor must be given together")
    if inp.k_test != K_TEST or test.k != K_TEST or floor.k != K_TEST:
        raise ValueError(
            f"stage 4: test and floor must be scored with K={K_TEST} "
            f"(k_test={inp.k_test}, result k={test.k}, floor k={floor.k})"
        )
    if not all(
        _finite(v) for v in (test.tokens_diff, *test.tokens_ci95, test.score_mde)
    ):
        raise ValueError("stage 4: test token statistics or MDE are NaN or infinite")
    token_limit = -(abs(floor.tokens_delta) + _Z * floor.tokens_se)
    return [
        _ni(4, "accuracy", test, margin),
        Check(
            4,
            "tokens lower beyond floor",
            tokens_beyond_floor(test, floor),
            test.tokens_diff,
            token_limit,
            "not significant",
        ),
    ]


def _suite_margins(
    suites: Mapping[str, tuple[PairedResult, float, str]],
) -> dict[str, tuple[float, str]]:
    out = {}
    for name, (_, m, src) in suites.items():
        what = f"stage 5 suite {name!r}"
        if not isinstance(src, str) or not src.strip():
            raise ValueError(
                f"{what}: margin source must say where the margin came from"
            )
        out[name] = (_check_margin(m, what), src)
    return out


def _outcome(checks: list[Check]) -> str:
    failed = {c.on_fail for c in checks if not c.passed}
    for outcome in (SCREENED_OUT, "regressed", "not significant"):  # precedence
        if outcome in failed:
            return outcome
    return "confirmed"


def gate(inputs: GateInputs) -> GateResult:
    """Judge `inputs` (see the module docstring for the rules).

    ValueError on inconsistent inputs: a margin missing or outside [0, 0.02], an empty
    `margin_source`, NaN in a used number, a stage missing although every earlier stage
    passed, any stage given after a failed pre-screen, a required screen missing,
    control q below 0.95 or no side effect scored, `k_test`, the test's or the floor's K
    not 4, an empty suite margin source, or a confirmed outcome without regression
    suites.
    """
    margin = _check_margin(inputs.margin, "test")
    if not isinstance(inputs.margin_source, str) or not inputs.margin_source.strip():
        raise ValueError("margin_source must say where the margin came from")
    checks = _stage1(inputs.screens)
    has4 = not (inputs.test is None and inputs.floor is None and inputs.k_test is None)
    if not all(c.passed for c in checks):
        if inputs.confirm or inputs.controls or has4 or inputs.suites:
            raise ValueError("screened out: no later stage may be given")
        return _result(checks, None, None, margin, inputs.margin_source)

    if inputs.confirm is None or inputs.controls is None:
        raise ValueError("stage 3: confirm result and control verdict are required")
    if not _finite(inputs.confirm.score_mde):
        raise ValueError("stage 3: confirm MDE is NaN or infinite")
    suite_margins = _suite_margins(inputs.suites)
    checks += _stage3(inputs.controls, inputs.control_target)
    if has4:
        checks += _stage4(inputs, margin)
    elif all(c.passed for c in checks):
        raise ValueError("stage 4 is missing although every earlier stage passed")
    if inputs.suites:
        if not has4:
            raise ValueError("stage 5 given without stage 4")
        checks += [
            _ni(5, f"suite {n}", r, suite_margins[n][0])
            for n, (r, _, _) in inputs.suites.items()
        ]
    elif all(c.passed for c in checks):
        raise ValueError("stage 5 is missing although every earlier stage passed")
    mde, split = (
        (inputs.test.score_mde, "test")
        if inputs.test is not None
        else (inputs.confirm.score_mde, "confirm")
    )
    return _result(checks, mde, split, margin, inputs.margin_source, suite_margins)


def _result(
    checks: list[Check],
    mde: float | None,
    split: str | None,
    margin: float,
    source: str,
    suite_margins: Mapping[str, tuple[float, str]] | None = None,
) -> GateResult:
    outcome = _outcome(checks)
    return GateResult(
        outcome=outcome,
        checks=tuple(checks),
        reasons=tuple(c for c in checks if not c.passed),
        mde=mde,
        mde_split=split,
        margin=margin,
        margin_source=source,
        advice=None if outcome == "confirmed" else ADVICE,
        suite_margins=dict(suite_margins or {}),
    )


def _num(v: float | None) -> str:
    return "-" if v is None else f"{v:+.4f}"


def format_gate(result: GateResult) -> str:
    """Plain-text gate report: outcome, every check, MDE, margin and advice."""
    lines = [f"outcome: {result.outcome}"]
    lines += [
        f"  stage {c.stage}  {'PASS' if c.passed else 'FAIL'}  {c.check}"
        f"  value={_num(c.value)}  limit={_num(c.limit)}"
        for c in result.checks
    ]
    if result.mde is not None:
        lines.append(f"MDE ({result.mde_split}): {result.mde:.4f}")
    lines.append(f"margin: {result.margin:.4f} ({result.margin_source})")
    lines += [
        f"suite {n} margin: {m:.4f} ({src})"
        for n, (m, src) in result.suite_margins.items()
    ]
    if result.advice:
        lines.append(f"advice: {result.advice}")
    return "\n".join(lines) + "\n"
