"""Multi-turn task records and their paired comparison (numpy only; never runs a model)."""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from astro_weights.eval.paired import PairedResult, compare
from astro_weights.eval.records import Record


@dataclass(frozen=True)
class ToolCall:
    """One tool call: the tool's name and its arguments as a JSON string."""

    name: str
    arguments: str


@dataclass(frozen=True)
class Turn:
    """One message of a task transcript. `output_tokens` is 0 unless `role` is "assistant"."""

    role: str
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.output_tokens < 0 or (self.role != "assistant" and self.output_tokens):
            raise ValueError(
                f"turn role {self.role!r} has output_tokens={self.output_tokens}; "
                "only assistant turns count tokens"
            )


@dataclass(frozen=True)
class TaskRecord:
    """One trial of one multi-turn task in one arm.

    `passed` comes from the scorer. `output_tokens`, `tool_calls` and `n_turns` are totals over
    `turns`, stored so a file is greppable; the constructor refuses totals that do not match
    (`from_turns` computes them). `prompt_hash` hashes the task's initial messages and tool schema.
    """

    task_id: str
    trial: int
    arm: str
    prompt_hash: str
    turns: tuple[Turn, ...]
    passed: bool
    output_tokens: int
    tool_calls: int
    n_turns: int
    cluster_id: str | None = None

    def __post_init__(self) -> None:
        want = _totals(self.turns)
        got = (self.output_tokens, self.tool_calls, self.n_turns)
        if got != want:
            raise ValueError(
                f"task {self.task_id!r} trial {self.trial}: totals (output_tokens, tool_calls, "
                f"n_turns) = {got} do not match its turns {want}"
            )

    @classmethod
    def from_turns(
        cls,
        task_id: str,
        trial: int,
        arm: str,
        prompt_hash: str,
        turns: Sequence[Turn],
        passed: bool,
        cluster_id: str | None = None,
    ) -> "TaskRecord":
        """Build a record with the totals computed from `turns`."""
        ts = tuple(turns)
        out, calls, n = _totals(ts)
        return cls(
            task_id, trial, arm, prompt_hash, ts, passed, out, calls, n, cluster_id
        )


def _totals(turns: Sequence[Turn]) -> tuple[int, int, int]:
    return (
        sum(t.output_tokens for t in turns),
        sum(len(t.tool_calls) for t in turns),
        len(turns),
    )


def write_tasks(path: str | Path, records: Iterable[TaskRecord]) -> None:
    """Write one JSON object per line."""
    with Path(path).open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in records)


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _keys(
    obj: object, required: set[str], optional: set[str], what: str
) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"{what}: expected a JSON object")  # noqa: TRY004
    missing = sorted(required - obj.keys())
    if missing:
        raise ValueError(f"{what}: missing keys: {', '.join(missing)}")
    unknown = sorted(obj.keys() - required - optional)
    if unknown:
        raise ValueError(f"{what}: unknown keys: {', '.join(unknown)}")
    return obj


def _call(obj: object) -> ToolCall:
    o = _keys(obj, {"name", "arguments"}, set(), "tool call")
    if not (isinstance(o["name"], str) and isinstance(o["arguments"], str)):
        raise ValueError("wrong value type")  # noqa: TRY004
    return ToolCall(o["name"], o["arguments"])


def _turn(obj: object) -> Turn:
    o = _keys(obj, {"role", "text", "tool_calls", "output_tokens"}, set(), "turn")
    if not (
        isinstance(o["role"], str)
        and isinstance(o["text"], str)
        and isinstance(o["tool_calls"], list)
        and _is_int(o["output_tokens"])
    ):
        raise ValueError("wrong value type")
    return Turn(
        o["role"],
        o["text"],
        tuple(_call(c) for c in o["tool_calls"]),
        o["output_tokens"],
    )


def _refuse_constant(name: str) -> float:
    raise ValueError(f"{name} is not allowed")


def _parse(line: str) -> TaskRecord:
    try:
        obj = json.loads(line, parse_constant=_refuse_constant)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e.msg}") from e
    except RecursionError as e:
        raise ValueError("JSON nested too deeply") from e
    o = _keys(
        obj,
        {"task_id", "trial", "arm", "prompt_hash", "turns", "passed"}
        | {"output_tokens", "tool_calls", "n_turns"},
        {"cluster_id"},
        "task record",
    )
    cluster = o.get("cluster_id")
    ok = (
        all(isinstance(o[k], str) for k in ("task_id", "arm", "prompt_hash"))
        and all(
            _is_int(o[k]) for k in ("trial", "output_tokens", "tool_calls", "n_turns")
        )
        and isinstance(o["passed"], bool)
        and isinstance(o["turns"], list)
        and (cluster is None or isinstance(cluster, str))
    )
    if not ok:
        raise ValueError("wrong value type")
    o["turns"] = tuple(_turn(t) for t in o["turns"])
    return TaskRecord(**o)


def read_tasks(path: str | Path) -> list[TaskRecord]:
    """Read records written by `write_tasks`; blank lines are skipped."""
    out: list[TaskRecord] = []
    with Path(path).open(encoding="utf-8") as f:
        for num, line in enumerate(f, start=1):
            if line.strip():
                try:
                    out.append(_parse(line))
                except ValueError as e:
                    raise ValueError(f"line {num}: {e}") from e
    return out


@dataclass(frozen=True)
class TaskDiff:
    """Paired per-task difference (edit - base) of one count, averaged over trials per task."""

    diff: float
    ci95: tuple[float, float]
    base: float
    edit: float


@dataclass(frozen=True)
class TaskComparison:
    """Paired comparison of two arms over the same tasks.

    `pass_rate` is `compare()` on pass/fail records (`pass_rate.tokens_*` are the output tokens
    per task). `tokens`, `tool_calls` and `turns` are per-task paired mean differences with the
    token bootstrap CI of `compare()`. `k` is the most trials any task has.
    """

    pass_rate: PairedResult
    tokens: TaskDiff
    tool_calls: TaskDiff
    turns: TaskDiff
    n_tasks: int
    k: int


def _records(tasks: Sequence[TaskRecord], count: str | None) -> list[Record]:
    """Records for `compare`: score is `passed`; n_output_tokens is the named total (or tokens)."""
    return [
        Record(
            t.task_id,
            t.trial,
            t.arm,
            t.prompt_hash,
            "",
            float(t.passed),
            t.output_tokens if count is None else getattr(t, count),
            False,
            t.cluster_id,
        )
        for t in tasks
    ]


def _same_tasks(base: Sequence[TaskRecord], edit: Sequence[TaskRecord]) -> None:
    b = {(t.task_id, t.trial) for t in base}
    e = {(t.task_id, t.trial) for t in edit}
    if b == e:
        return
    tb, te = {i for i, _ in b}, {i for i, _ in e}
    x: Sequence[object]
    y: Sequence[object]
    if tb != te:
        what, x, y = "task set", sorted(tb - te), sorted(te - tb)
    else:
        what, x, y = "trial set", sorted(b - e), sorted(e - b)
    raise ValueError(
        f"arms have a different {what}: {len(x)} only in base ({x[:5]}), "
        f"{len(y)} only in edit ({y[:5]})"
    )


def _diff(r: PairedResult) -> TaskDiff:
    return TaskDiff(r.tokens_diff, r.tokens_ci95, r.tokens_base, r.tokens_edit)


def compare_tasks(
    base: Sequence[TaskRecord],
    edit: Sequence[TaskRecord],
    *,
    n_boot: int = 10000,
    seed: int = 0,
    strata: Mapping[str, str] | None = None,
) -> TaskComparison:
    """Paired comparison of two arms; trials are averaged per task before pairing.

    Pass rate and tokens come from one `compare()` call. Tool calls and turns per task reuse
    `compare()` too, on records whose `n_output_tokens` holds that count, so the CI method is
    the token bootstrap and the statistics are not forked. Arms must have the same tasks and
    trials, and the same `prompt_hash` per task.
    """
    _same_tasks(base, edit)
    main = compare(
        _records(base, None),
        _records(edit, None),
        n_boot=n_boot,
        seed=seed,
        strata=strata,
    )
    calls = compare(
        _records(base, "tool_calls"), _records(edit, "tool_calls"), n_boot, seed
    )
    turns = compare(_records(base, "n_turns"), _records(edit, "n_turns"), n_boot, seed)
    return TaskComparison(
        main, _diff(main), _diff(calls), _diff(turns), main.n_items, main.k
    )


def _row(name: str, d: TaskDiff, fmt: str) -> str:
    return (
        f"{name}: base={d.base:{fmt}}  edit={d.edit:{fmt}}  diff={d.diff:+{fmt}}  "
        f"bootstrap 95% CI [{d.ci95[0]:+{fmt}}, {d.ci95[1]:+{fmt}}]"
    )


def format_tasks(cmp: TaskComparison) -> str:
    """Rows for pass rate, tokens, tool calls and turns per task; MDE on pass and tokens."""
    p = cmp.pass_rate
    lines = [
        f"n={cmp.n_tasks} tasks  K={cmp.k}",
        (
            f"pass rate: base={p.mean_base:.4f}  edit={p.mean_edit:.4f}  diff={p.diff:+.4f}  "
            f"95% CI [{p.ci95[0]:+.4f}, {p.ci95[1]:+.4f}] ({p.ci95_method})  MDE={p.score_mde:.4f}"
        ),
        _row("tokens/task", cmp.tokens, ".1f") + f"  MDE={p.tokens_mde:.1f}",
        _row("tool calls/task", cmp.tool_calls, ".2f"),
        _row("turns/task", cmp.turns, ".2f"),
    ]
    if p.small_n:
        lines.append(
            f"WARNING: only {p.n_items} tasks (<100); intervals are unreliable"
        )
    return "\n".join(lines)
