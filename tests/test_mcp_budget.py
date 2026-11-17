import json
import math
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from astro_weights.mcp.budget import (
    MAX_FAILURES,
    BudgetExceeded,
    Ledger,
    Limits,
    TooManyFailures,
)


def limits(
    forward_passes: int = 3, code_runs: int = 3, tokens: int = 100, dollars: float = 1.0
) -> Limits:
    return Limits(forward_passes, code_runs, tokens, dollars)


def ledger(tmp_path: Path, **kw: Any) -> Ledger:
    return Ledger(tmp_path / "ledger.jsonl", limits(**kw))


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_fresh_ledger_writes_nothing(tmp_path: Path) -> None:
    led = ledger(tmp_path)
    led.check("forward_passes")
    assert not led.path.exists()
    assert led.used() == {
        "forward_passes": 0,
        "code_runs": 0,
        "tokens": 0,
        "dollars": 0.0,
    }
    assert led.remaining() == {
        "forward_passes": 3,
        "code_runs": 3,
        "tokens": 100,
        "dollars": 1.0,
    }


@pytest.mark.parametrize("kind", ["forward_passes", "code_runs"])
def test_gated_limit_reachable_not_exceedable(tmp_path: Path, kind: Any) -> None:
    led = ledger(tmp_path)
    for i in range(3):
        led.check(kind)
        led.charge(kind, ref=f"r{i}")
    assert led.used()[kind] == 3
    assert led.remaining()[kind] == 0
    with pytest.raises(BudgetExceeded) as e:
        led.check(kind)
    assert (e.value.kind, e.value.used, e.value.limit, e.value.requested) == (
        kind,
        3,
        3,
        1,
    )
    with pytest.raises(BudgetExceeded):
        led.charge(kind, ref="over")
    assert led.used()[kind] == 3
    assert len(lines(led.path)) == 3


def test_amount_above_one_is_checked_whole(tmp_path: Path) -> None:
    led = ledger(tmp_path, tokens=10)
    led.check("tokens", 10)
    with pytest.raises(BudgetExceeded):
        led.check("tokens", 11)
    led.charge("tokens", 7, ref="t")
    led.check("tokens", 3)
    with pytest.raises(BudgetExceeded):
        led.check("tokens", 4)


@pytest.mark.parametrize("kind", ["forward_passes", "code_runs", "tokens", "dollars"])
def test_zero_limit_blocks(tmp_path: Path, kind: Any) -> None:
    led = Ledger(tmp_path / "ledger.jsonl", Limits(0, 0, 0, 0.0))
    with pytest.raises(BudgetExceeded):
        led.check(kind)


def test_zero_limit_gated_charge_writes_nothing(tmp_path: Path) -> None:
    led = Ledger(tmp_path / "ledger.jsonl", Limits(0, 0, 0, 0.0))
    with pytest.raises(BudgetExceeded):
        led.charge("forward_passes", ref="x")
    assert not led.path.exists()


def test_dollars_no_epsilon_leak(tmp_path: Path) -> None:
    led = ledger(tmp_path, dollars=0.3)
    led.charge("dollars", 0.1, ref="a")
    led.charge("dollars", 0.2, ref="b")
    # 0.1 + 0.2 > 0.3 in floating point: the cap holds, it does not round down
    with pytest.raises(BudgetExceeded):
        led.check("dollars", 1e-12)
    led2 = ledger(tmp_path / "x", dollars=1.0)
    led2.path.parent.mkdir()
    led2.check("dollars", 1.0)
    led2.charge("dollars", 1.0, ref="all")
    with pytest.raises(BudgetExceeded):
        led2.check("dollars", math.ulp(0.0))
    # sub-ulp charges still add up exactly: they cannot vanish into rounding
    # (a float total stays at 1.0: each 1e-16 is under half an ulp of 1.0)
    led3 = Ledger(tmp_path / "y.jsonl", limits(dollars=math.nextafter(1.0, 2.0)))
    led3.charge("dollars", 1.0, ref="big")
    for i in range(3):
        led3.charge("dollars", 1e-16, ref=f"tiny{i}")
    with pytest.raises(BudgetExceeded):
        led3.check("dollars", 1e-16)
    with pytest.raises(BudgetExceeded):
        Ledger(led3.path, led3.limits).check("dollars", 1e-16)


def test_dollars_actual_cost_recorded_past_limit(tmp_path: Path) -> None:
    led = ledger(tmp_path, dollars=1.0)
    led.check("dollars", 0.5)
    led.charge("dollars", 0.9, ref="turn1")
    led.charge("dollars", 0.4, ref="turn2")  # money already spent: never dropped
    assert led.used()["dollars"] == pytest.approx(1.3)
    assert led.remaining()["dollars"] == 0
    with pytest.raises(BudgetExceeded):
        led.check("dollars", 0.01)
    assert Ledger(led.path, limits()).used()["dollars"] == led.used()["dollars"]


def test_tokens_recorded_past_limit(tmp_path: Path) -> None:
    led = ledger(tmp_path, tokens=10)
    led.charge("tokens", 25, ref="turn")
    assert led.used()["tokens"] == 25
    assert led.remaining()["tokens"] == 0


def test_refusal_never_spends(tmp_path: Path) -> None:
    led = ledger(tmp_path, forward_passes=1)
    for i in range(10):
        led.refuse("forward_passes", "bad request", ref=f"r{i}")
    assert led.used()["forward_passes"] == 0
    assert led.to_json()["streaks"]["forward_passes"] == 0
    led.check("forward_passes")
    assert [x["op"] for x in lines(led.path)] == ["refuse"] * 10


def test_refuse_does_not_change_streak(tmp_path: Path) -> None:
    led = ledger(tmp_path)
    led.fail("code_runs", "boom", ref="a")
    led.refuse("code_runs", "no", ref="b")
    assert led.to_json()["streaks"]["code_runs"] == 1


def test_failure_is_free_and_streak_stops_kind(tmp_path: Path) -> None:
    led = ledger(tmp_path)
    assert [
        led.fail("forward_passes", "oom", ref=f"f{i}") for i in range(MAX_FAILURES)
    ] == [1, 2, 3]
    assert led.used()["forward_passes"] == 0
    with pytest.raises(TooManyFailures) as e:
        led.check("forward_passes")
    assert (e.value.kind, e.value.streak) == ("forward_passes", 3)
    led.check("code_runs")  # other kinds keep working
    led.reset_failures("forward_passes")
    led.check("forward_passes")
    assert led.to_json()["streaks"]["forward_passes"] == 0


def test_fail_then_success_resets_streak(tmp_path: Path) -> None:
    led = ledger(tmp_path)
    led.fail("forward_passes", "x", ref="a")
    led.fail("forward_passes", "x", ref="b")
    led.charge("forward_passes", ref="c")
    assert led.fail("forward_passes", "x", ref="d") == 1
    led.check("forward_passes")


def test_budget_checked_before_streak(tmp_path: Path) -> None:
    led = ledger(tmp_path, forward_passes=0)
    for i in range(3):
        led.fail("forward_passes", "x", ref=str(i))
    with pytest.raises(BudgetExceeded):
        led.check("forward_passes")


def test_reload_gives_same_state(tmp_path: Path) -> None:
    led = ledger(tmp_path)
    led.charge("forward_passes", ref="a")
    led.charge("dollars", 0.1, ref="b")
    led.charge("dollars", 0.2, ref="c")
    led.charge("tokens", 40, ref="d")
    led.fail("code_runs", "x", ref="e")
    led.fail("code_runs", "x", ref="f")
    led.fail("forward_passes", "x", ref="g")
    led.refuse("tokens", "no", ref="h")
    again = Ledger(led.path, limits())
    assert again.to_json() == led.to_json()
    assert again.used()["dollars"] == led.used()["dollars"]
    led.reset_failures("code_runs")
    assert Ledger(led.path, limits()).to_json()["streaks"]["code_runs"] == 0


def test_streak_survives_reload(tmp_path: Path) -> None:
    led = ledger(tmp_path)
    for i in range(3):
        led.fail("forward_passes", "x", ref=str(i))
    with pytest.raises(TooManyFailures):
        Ledger(led.path, limits()).check("forward_passes")


def test_rolled_back_state_file_cannot_refund(tmp_path: Path) -> None:
    led = ledger(tmp_path, forward_passes=1)
    led.charge("forward_passes", ref="p1")
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"forward_used": 1}))
    # what model-written code could do to any other state file
    state.write_text(json.dumps({"forward_used": 0}))
    state.unlink()
    fresh = Ledger(tmp_path / "ledger.jsonl", limits(forward_passes=1))
    assert fresh.remaining()["forward_passes"] == 0
    with pytest.raises(BudgetExceeded):
        fresh.check("forward_passes")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ledger.jsonl"]


def test_torn_final_line_ignored_and_cut(tmp_path: Path) -> None:
    led = ledger(tmp_path)
    led.charge("forward_passes", ref="a")
    with led.path.open("ab") as f:
        f.write(b'{"op":"charge","kind":"forw')
    again = Ledger(led.path, limits())
    assert again.used()["forward_passes"] == 1
    again.charge("forward_passes", ref="b")
    assert [x["ref"] for x in lines(led.path)] == ["a", "b"]
    assert Ledger(led.path, limits()).used()["forward_passes"] == 2


def test_torn_line_missing_only_newline_is_kept(tmp_path: Path) -> None:
    led = ledger(tmp_path)
    led.charge("dollars", 0.5, ref="a")
    with led.path.open("ab") as f:
        f.write(b'{"op":"charge","kind":"dollars","amount":0.25,"ref":"b"}')
    again = Ledger(led.path, limits())
    assert again.used()["dollars"] == 0.75  # never a refund
    again.charge("dollars", 0.25, ref="c")
    assert [x["ref"] for x in lines(led.path)] == ["a", "b", "c"]


def test_torn_line_valid_json_bad_entry_is_cut(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(
        b'{"op":"charge","kind":"tokens","amount":1,"ref":"a"}\n{"op":"x"}'
    )
    assert Ledger(path, limits()).used()["tokens"] == 1
    assert path.read_bytes().endswith(b"}\n")


@pytest.mark.parametrize(
    "bad",
    [
        b"not json",
        b"[1]",
        b'{"op":"spend","kind":"tokens"}',
        b'{"op":"charge","kind":"gpus","amount":1}',
        b'{"op":"charge","kind":"dollars","amount":NaN}',
        b'{"op":"charge","kind":"dollars","amount":-1}',
        b'{"op":"charge","kind":"tokens","amount":1.5}',
        b'{"op":"charge","kind":"tokens"}',
        b'{"op":[1],"kind":"tokens"}',
        b'{"op":"fail","kind":{"a":1}}',
        b"",
    ],
)
def test_corrupt_complete_line_raises(tmp_path: Path, bad: bytes) -> None:
    path = tmp_path / "ledger.jsonl"
    good = b'{"op":"charge","kind":"tokens","amount":1,"ref":"a"}\n'
    path.write_bytes(good + bad + b"\n" + good)
    before = path.read_bytes()
    with pytest.raises(ValueError, match=r"ledger\.jsonl:2"):
        Ledger(path, limits())
    assert path.read_bytes() == before


def test_race_exactly_limit_succeed(tmp_path: Path) -> None:
    n, threads = 5, 16
    led = ledger(tmp_path, forward_passes=n)
    barrier = threading.Barrier(threads)
    ok: list[int] = []
    over: list[int] = []

    def worker(i: int) -> None:
        barrier.wait()
        try:
            led.charge("forward_passes", ref=f"t{i}")
            ok.append(i)
        except BudgetExceeded:
            over.append(i)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(ok) == n
    assert len(over) == threads - n
    assert len(lines(led.path)) == n
    assert Ledger(led.path, limits(forward_passes=n)).used()["forward_passes"] == n


def test_race_dollars_all_recorded(tmp_path: Path) -> None:
    led = ledger(tmp_path, dollars=1.0)
    barrier = threading.Barrier(16)

    def worker(i: int) -> None:
        barrier.wait()
        led.charge("dollars", 0.25, ref=f"t{i}")

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert led.used()["dollars"] == 4.0
    assert len(lines(led.path)) == 16


@pytest.mark.parametrize(
    "amount", [math.nan, math.inf, -math.inf, -1, 0, 0.0, True, False, "1", None]
)
@pytest.mark.parametrize("kind", ["forward_passes", "dollars"])
def test_bad_amount_rejected(tmp_path: Path, kind: Any, amount: Any) -> None:
    led = ledger(tmp_path)
    with pytest.raises(ValueError, match="amount"):
        led.check(kind, amount)
    with pytest.raises(ValueError, match="amount"):
        led.charge(kind, amount, ref="x")
    assert not led.path.exists()


@pytest.mark.parametrize("kind", ["forward_passes", "code_runs", "tokens"])
def test_counted_kinds_need_int(tmp_path: Path, kind: Any) -> None:
    led = ledger(tmp_path)
    with pytest.raises(ValueError, match="int"):
        led.charge(kind, 1.0, ref="x")


def test_unknown_kind(tmp_path: Path) -> None:
    led = ledger(tmp_path)
    for call in (
        lambda: led.check("gpus"),  # type: ignore[arg-type]
        lambda: led.charge("gpus", ref="x"),  # type: ignore[arg-type]
        lambda: led.refuse("gpus", "r", ref="x"),  # type: ignore[arg-type]
        lambda: led.fail("gpus", "e", ref="x"),  # type: ignore[arg-type]
        lambda: led.reset_failures("gpus"),  # type: ignore[arg-type]
    ):
        with pytest.raises(ValueError, match="unknown budget kind"):
            call()
    assert not led.path.exists()


def test_text_fields_capped_and_typed(tmp_path: Path) -> None:
    led = ledger(tmp_path)
    led.fail("tokens", "e" * 5000, ref="r" * 5000)
    led.refuse("tokens", "q" * 5000, ref="x")
    first, second = lines(led.path)
    assert len(first["error"]) == len(first["ref"]) == len(second["reason"]) == 2000
    with pytest.raises(ValueError, match="ref"):
        led.charge("tokens", ref=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reason"):
        led.refuse("tokens", None, ref="x")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad",
    [
        (-1, 0, 0, 0.0),
        (0, 0, 0, -0.5),
        (0, 0, 0, math.nan),
        (0, 0, 0, math.inf),
        (True, 0, 0, 0.0),
        (0, 0, 1.5, 0.0),
        (0, 0, 0, False),
    ],
)
def test_limits_validated(bad: tuple[Any, ...]) -> None:
    with pytest.raises(ValueError, match="limit"):
        Limits(*bad)


def test_ledger_needs_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Limits"):
        Ledger(tmp_path / "ledger.jsonl", {"forward_passes": 1})  # type: ignore[arg-type]


def test_lines_fsynced_with_time(tmp_path: Path) -> None:
    led = ledger(tmp_path)
    led.charge("code_runs", ref="c")
    led.reset_failures("code_runs")
    a, b = lines(led.path)
    assert a["op"] == "charge"
    assert a["amount"] == 1
    assert a["ref"] == "c"
    assert "time" in a
    assert b == {"op": "reset", "kind": "code_runs", "time": b["time"]}


def test_failed_write_leaves_no_line_and_no_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    led = ledger(tmp_path)
    led.charge("forward_passes", ref="a")

    def broken(fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("astro_weights.mcp.record.os.fsync", broken)
    with pytest.raises(OSError, match="disk full"):
        led.charge("forward_passes", ref="b")
    monkeypatch.undo()
    assert led.used()["forward_passes"] == 1
    assert [x["ref"] for x in lines(led.path)] == ["a"]


def test_to_json(tmp_path: Path) -> None:
    led = ledger(tmp_path)
    led.charge("tokens", 30, ref="t")
    led.fail("code_runs", "x", ref="c")
    assert led.to_json() == {
        "limits": {"forward_passes": 3, "code_runs": 3, "tokens": 100, "dollars": 1.0},
        "used": {"forward_passes": 0, "code_runs": 0, "tokens": 30, "dollars": 0.0},
        "remaining": {
            "forward_passes": 3,
            "code_runs": 3,
            "tokens": 70,
            "dollars": 1.0,
        },
        "streaks": {"forward_passes": 0, "code_runs": 1, "tokens": 0, "dollars": 0},
        "max_failures": 3,
    }
    json.dumps(led.to_json(), allow_nan=False)


def test_huge_int_amount_is_exact(tmp_path: Path) -> None:
    big = 10**400
    led = Ledger(tmp_path / "ledger.jsonl", Limits(big, 0, big, big))
    led.charge("forward_passes", big, ref="a")
    led.charge("dollars", big, ref="b")
    assert Ledger(led.path, led.limits).remaining()["forward_passes"] == 0
    with pytest.raises(BudgetExceeded):
        led.check("forward_passes")
    out = led.to_json()
    assert out["used"]["dollars"] == sys.float_info.max
    json.dumps(out, allow_nan=False)
