import numpy as np
import pytest

from astro_weights.edits import (
    Edit,
    LowRankDelta,
    check_norms,
    delta_norm,
    edit_norms,
)

DOWN = "model.layers.0.mlp.down_proj"
OPROJ = "model.layers.0.self_attn.o_proj"


def delta(module: str, scale: float = 1.0, r: int = 2) -> LowRankDelta:
    rng = np.random.default_rng(r)
    return LowRankDelta(
        module, scale * rng.standard_normal((6, r)), rng.standard_normal((r, 9))
    )


@pytest.mark.parametrize("r", range(1, 9))
def test_delta_norm_matches_dense(r: int) -> None:
    d = delta(DOWN, r=r)
    assert np.isclose(delta_norm(d), np.linalg.norm(d.B @ d.A), rtol=1e-5)


def make() -> Edit:
    return Edit((delta(DOWN), delta(OPROJ, 10.0)), "tiny")


def test_edit_norms_per_module() -> None:
    e = make()
    assert edit_norms(e) == {d.module: delta_norm(d) for d in e.deltas}


def test_global_range_inside_and_outside() -> None:
    e = make()
    n = edit_norms(e)
    ok = check_norms(e, (0.0, 2 * max(n.values())))
    assert ok.passed
    assert ok.outside == ()
    assert ok.norms == n
    bad = check_norms(e, (0.0, n[DOWN] * 2))
    assert not bad.passed
    assert bad.outside == (OPROJ,)


def test_bounds_inclusive() -> None:
    e = make()
    n = edit_norms(e)
    assert check_norms(e, (min(n.values()), max(n.values()))).passed


def test_per_suffix_and_exact_ranges() -> None:
    e = make()
    n = edit_norms(e)
    r = {"mlp.down_proj": (0.0, n[DOWN]), "self_attn.o_proj": (0.0, n[OPROJ] / 2)}
    assert check_norms(e, r).outside == (OPROJ,)
    # exact name beats suffix, longer suffix beats shorter
    r2 = {"o_proj": (0.0, 1e-9), OPROJ: (0.0, 1e9), "down_proj": (0.0, 1e9)}
    assert check_norms(e, r2).passed
    r3 = {"proj": (0.0, 1e9), "mlp.down_proj": (0.0, 1e-9), "o_proj": (0.0, 1e9)}
    assert check_norms(e, r3).outside == (DOWN,)


def test_unmatched_module_refused() -> None:
    with pytest.raises(ValueError, match="no norm range"):
        check_norms(make(), {"mlp.down_proj": (0.0, 1e9)})
    # a suffix must be a dotted tail, not a substring
    with pytest.raises(ValueError, match="no norm range"):
        check_norms(make(), {"down_proj": (0, 1), "proj": (0, 1), "j": (0, 1)})


@pytest.mark.parametrize(
    "bad",
    [
        (float("nan"), 1.0),
        (0.0, float("inf")),
        (-1.0, 1.0),
        (2.0, 1.0),
    ],
)
def test_bad_ranges_refused(bad: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="range"):
        check_norms(make(), bad)
    with pytest.raises(ValueError, match="range"):
        check_norms(make(), {"proj": bad})
