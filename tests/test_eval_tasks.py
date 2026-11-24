"""TaskRecord JSONL and compare_tasks / format_tasks."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from astro_weights.eval import (
    Record,
    TaskRecord,
    ToolCall,
    Turn,
    compare,
    compare_tasks,
    format_tasks,
    read_tasks,
    write_tasks,
)


def _task(
    arm: str, i: int, trial: int, passed: bool, toks: int, calls: int, extra: int
) -> TaskRecord:
    """A task with `extra` + 1 assistant turns; `calls` tool calls in the first one."""
    tc = tuple(ToolCall("f", f'{{"x": {n}}}') for n in range(calls))
    turns = [Turn("user", "go"), Turn("assistant", "a", tc, toks)]
    turns += [Turn("tool", "r")] * extra
    return TaskRecord.from_turns(f"t{i}", trial, arm, f"h{i}", turns, passed)


def _arm(arm: str, k: int, shift: int) -> list[TaskRecord]:
    return [
        _task(
            arm,
            i,
            t,
            (i + t) % 3 != 0 or (shift == 1 and i % 5 == 0),
            50 + i + 3 * t * shift,
            (i + t) % 4 + shift,
            i % 3 + shift * (i % 2),
        )
        for i in range(30)
        for t in range(k)
    ]


def test_round_trip_and_unicode(tmp_path: Path) -> None:
    recs = [
        _task("base", 0, 0, True, 12, 2, 1),
        TaskRecord.from_turns(
            "é", 1, "edit", "h", [Turn("assistant", "ü", (), 3)], False, "c"
        ),
    ]
    p = tmp_path / "t.jsonl"
    write_tasks(p, recs)
    assert read_tasks(p) == recs
    assert (recs[0].output_tokens, recs[0].tool_calls, recs[0].n_turns) == (12, 2, 3)


def test_totals_mismatch_refused(tmp_path: Path) -> None:
    good = _task("base", 0, 0, True, 12, 2, 1)
    with pytest.raises(ValueError, match="do not match"):
        TaskRecord("t", 0, "base", "h", good.turns, True, 13, 2, 3)
    obj = json.loads(json.dumps(good.__dict__, default=lambda o: o.__dict__))
    obj["n_turns"] = 99
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps(obj) + "\n")
    with pytest.raises(ValueError, match=r"line 1: .*do not match"):
        read_tasks(p)


def test_turn_tokens_only_on_assistant() -> None:
    with pytest.raises(ValueError, match="only assistant"):
        Turn("tool", "r", (), 5)


def _line(**kw: object) -> str:
    good = _task("base", 0, 0, True, 12, 1, 0)
    obj = json.loads(json.dumps(good.__dict__, default=lambda o: o.__dict__))
    obj.update(kw)
    return json.dumps({k: v for k, v in obj.items() if v is not ...})


@pytest.mark.parametrize(
    ("line", "msg"),
    [
        (_line(extra=1), "unknown keys: extra"),
        (_line(task_id=...), "missing keys: task_id"),
        (_line(passed=1), "wrong value type"),
        (_line(trial=True), "wrong value type"),
        (_line(cluster_id=3), "wrong value type"),
        (
            _line(
                turns=[
                    {
                        "role": "user",
                        "text": "x",
                        "tool_calls": [],
                        "output_tokens": 0,
                        "z": 1,
                    }
                ]
            ),
            "unknown keys: z",
        ),
        (
            _line(
                turns=[
                    {"role": "user", "text": 1, "tool_calls": [], "output_tokens": 0}
                ]
            ),
            "wrong value type",
        ),
        (
            _line(
                turns=[
                    {
                        "role": "user",
                        "text": "x",
                        "tool_calls": [{"name": "f"}],
                        "output_tokens": 0,
                    }
                ]
            ),
            "missing keys: arguments",
        ),
        (
            _line(
                turns=[
                    {
                        "role": "user",
                        "text": "x",
                        "tool_calls": [{"name": "f", "arguments": 1}],
                        "output_tokens": 0,
                    }
                ]
            ),
            "wrong value type",
        ),
        (_line(turns=[1]), "expected a JSON object"),
        ('{"output_tokens": NaN}', "NaN is not allowed"),
        ("[1]", "expected a JSON object"),
        ("{", "invalid JSON"),
        ("[" * 100000, "nested too deeply|invalid JSON"),  # 3.14 builds differ
    ],
)
def test_bad_lines_refused(tmp_path: Path, line: str, msg: str) -> None:
    p = tmp_path / "t.jsonl"
    p.write_text("\n" + line + "\n")
    with pytest.raises(ValueError, match=msg):
        read_tasks(p)


def _recs(arm: list[TaskRecord], what: str | None = None) -> list[Record]:
    return [
        Record(t.task_id, t.trial, t.arm, t.prompt_hash, "", float(t.passed),
               t.output_tokens if what is None else getattr(t, what), False)
        for t in arm
    ]  # fmt: skip


@pytest.mark.parametrize("k", [1, 4])
def test_matches_compare_and_hand_means(k: int) -> None:
    base, edit = _arm("base", k, 0), _arm("edit", k, 1)
    got = compare_tasks(base, edit, n_boot=300)
    want = compare(_recs(base), _recs(edit), n_boot=300)
    assert got.pass_rate == want
    assert got.tokens.diff == want.tokens_diff
    assert got.tokens.ci95 == want.tokens_ci95
    assert (got.n_tasks, got.k) == (30, k)
    for field, d in (("tool_calls", got.tool_calls), ("n_turns", got.turns)):
        # hand mean over trials per task, then over tasks
        def per_task(arm: list[TaskRecord], f: str = field) -> float:
            return (
                sum(
                    sum(int(getattr(t, f)) for t in arm if t.task_id == f"t{i}") / k
                    for i in range(30)
                )
                / 30
            )

        assert d.base == pytest.approx(per_task(base))
        assert d.edit == pytest.approx(per_task(edit))
        assert d.diff == pytest.approx(per_task(edit) - per_task(base))
        assert d.ci95[0] <= d.diff <= d.ci95[1]
    assert got.pass_rate.k == k
    assert got.pass_rate.diff != 0
    assert got.turns.diff != 0


def test_mismatch_refused_naming_tasks() -> None:
    base, edit = _arm("base", 2, 0), _arm("edit", 2, 1)
    with pytest.raises(ValueError, match=r"task set.*'t29'"):
        compare_tasks(base, [t for t in edit if t.task_id != "t29"])
    with pytest.raises(ValueError, match=r"trial set.*\('t3', 1\)"):
        compare_tasks(base, [t for t in edit if (t.task_id, t.trial) != ("t3", 1)])
    other = [t if t.task_id != "t5" else _hash(t) for t in edit]
    with pytest.raises(ValueError, match="prompt_hash differs"):
        compare_tasks(base, other)


def _hash(t: TaskRecord) -> TaskRecord:
    return replace(t, prompt_hash="other")


def test_cluster_id_carried() -> None:
    base, edit = _arm("base", 1, 0), _arm("edit", 1, 1)

    def cl(arm: list[TaskRecord]) -> list[TaskRecord]:
        return [replace(t, cluster_id=f"c{int(t.task_id[1:]) % 5}") for t in arm]

    assert compare_tasks(base, edit, n_boot=50).pass_rate.n_clusters is None
    assert compare_tasks(cl(base), cl(edit), n_boot=50).pass_rate.n_clusters == 5


def test_format_tasks() -> None:
    out = format_tasks(
        compare_tasks(_arm("base", 4, 0), _arm("edit", 4, 1), n_boot=100)
    )
    lines = out.splitlines()
    assert lines[0] == "n=30 tasks  K=4"
    assert [x.split(":")[0] for x in lines[1:5]] == [
        "pass rate",
        "tokens/task",
        "tool calls/task",
        "turns/task",
    ]
    assert "MDE=" in lines[1]
    assert "MDE=" in lines[2]
    assert "MDE=" not in lines[3] + lines[4]
    assert lines[-1].startswith("WARNING: only 30 tasks")
