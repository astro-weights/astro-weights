import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from astro_weights.mcp.budget import Ledger, Limits
from astro_weights.mcp.handles import HandleStore
from astro_weights.mcp.record import RunRecord

LIMITS = {"forward_passes": 3, "code_runs": 3, "tokens": 100, "dollars": 1.0}


def write_run(run_dir: Path, **over: Any) -> Path:
    cfg = {"seed": 7, "editor": "scripted", "model": None, "limits": LIMITS, **over}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(cfg))
    return run_dir


pytest.importorskip("mcp")

import itertools

import anyio  # type: ignore[import-not-found, unused-ignore]
from mcp import Client, StdioServerParameters
from mcp_fakes import fake_model_dir

from astro_weights.mcp import kernel as kernel_mod
from astro_weights.mcp import sandbox
from astro_weights.mcp.server import build_server, main


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def call(client: Client, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    result = await client.call_tool(tool, args)
    assert not result.is_error
    assert result.structured_content is not None
    return dict(result.structured_content)


async def poll(client: Client, handle: str, timeout: float = 10) -> dict[str, Any]:
    with anyio.fail_after(timeout):
        while True:
            out = await call(client, "status", {"handle": handle})
            if out["state"] in {"done", "failed"}:
                return out
            await anyio.sleep(0.01)


def echo(params: dict[str, Any]) -> dict[str, Any]:
    return {"echo": params}


@pytest.mark.anyio
async def test_exec_python_description_names_the_kernel_api(tmp_path: Path) -> None:
    # outside clients (Claude Code, Codex) see only tool descriptions, not agent.md
    from astro_weights.mcp.edit_store import MAX_REL_CHANGE

    async with Client(build_server(write_run(tmp_path))) as client:
        tools = await client.list_tools()
    text = next(t.description for t in tools.tools if t.name == "exec_python") or ""
    helpers = [
        "NAMES",
        "names()",
        "load(name)",
        "load_original(name)",
        "save_edit(name, w)",
        "drop_edit(name)",
        "edited()",
        "forward_result(handle, key)",
    ]
    for helper in helpers:
        assert helper in text
    assert f"above {MAX_REL_CHANGE}" in text


@pytest.mark.anyio
async def test_descriptions_state_the_export_rank_limit(tmp_path: Path) -> None:
    # a client that saves a dense edit only learns at finish that export fails
    from astro_weights.mcp.jobs import RunConfig

    async with Client(build_server(write_run(tmp_path))) as client:
        tools = await client.list_tools()
    for name in ("exec_python", "finish"):
        text = next(t.description for t in tools.tools if t.name == name) or ""
        assert f"rank {RunConfig.export_rank}" in text
        assert f"above {RunConfig.export_max_rel_residual}" in text
        assert "export.rank" in text
        assert "export.max_rel_residual" in text


@pytest.mark.anyio
async def test_lists_seven_tools(tmp_path: Path) -> None:
    async with Client(build_server(write_run(tmp_path))) as client:
        tools = await client.list_tools()
    assert sorted(t.name for t in tools.tools) == [
        "check_spend",
        "exec_python",
        "finish",
        "forward_pass",
        "report_refusal",
        "report_usage",
        "status",
    ]


@pytest.mark.anyio
async def test_exec_job_goes_to_done_and_charges_once(tmp_path: Path) -> None:
    server = build_server(write_run(tmp_path), {"exec": echo})
    async with Client(server) as client:
        started = await call(client, "exec_python", {"code": "1+1"})
        assert started["state"] == "queued"
        out = await poll(client, started["handle"])
    assert out["state"] == "done"
    assert out["result"] == {"echo": {"code": "1+1"}}
    assert out["budget"]["used"]["code_runs"] == 1


@pytest.mark.anyio
async def test_forward_job(tmp_path: Path) -> None:
    server = build_server(write_run(tmp_path), {"forward": echo})
    async with Client(server) as client:
        started = await call(client, "forward_pass", {"request": {"prompt": "hi"}})
        out = await poll(client, started["handle"])
    assert out["result"] == {"echo": {"request": {"prompt": "hi"}}}
    assert out["budget"]["used"]["forward_passes"] == 1


@pytest.mark.anyio
async def test_no_executors_means_not_available(tmp_path: Path) -> None:
    async with Client(build_server(write_run(tmp_path))) as client:
        a = await call(client, "exec_python", {"code": "1"})
        b = await call(client, "forward_pass", {"request": {}})
        s = await call(client, "status", {"handle": "exec-0123456789abcdef"})
    assert a == {"error": "exec_python is not available yet"}
    assert b == {"error": "forward_pass is not available yet"}
    assert s == {"error": "unknown handle"}
    events = [e["event"] for e in RunRecord.open(tmp_path / "record.jsonl").read()]
    assert events == ["start", "tool_unavailable", "tool_unavailable"]
    assert not (tmp_path / "ledger.jsonl").exists()


@pytest.mark.anyio
async def test_failing_executor(tmp_path: Path) -> None:
    def boom(params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("bad")

    async with Client(build_server(write_run(tmp_path), {"exec": boom})) as client:
        started = await call(client, "exec_python", {"code": "x"})
        out = await poll(client, started["handle"])
    assert out["state"] == "failed"
    assert out["budget"]["used"]["code_runs"] == 0
    assert out["budget"]["streaks"]["code_runs"] == 1


@pytest.mark.anyio
async def test_exhausted_budget_refusal(tmp_path: Path) -> None:
    run = write_run(tmp_path, limits={**LIMITS, "code_runs": 0})
    async with Client(build_server(run, {"exec": echo})) as client:
        out = await call(client, "exec_python", {"code": "x"})
    assert "budget exhausted" in out["error"]
    assert out["budget"]["used"]["code_runs"] == 0
    events = RunRecord.open(tmp_path / "record.jsonl").read()
    assert [e["event"] for e in events] == ["start", "refusal"]
    assert events[1]["category"] == "budget"


@pytest.mark.anyio
async def test_failure_streak_refusal(tmp_path: Path) -> None:
    def boom(params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("bad")

    async with Client(build_server(write_run(tmp_path), {"exec": boom})) as client:
        for _ in range(3):
            started = await call(client, "exec_python", {"code": "x"})
            await poll(client, started["handle"])
        out = await call(client, "exec_python", {"code": "x"})
    assert "in a row" in out["error"]
    events = RunRecord.open(tmp_path / "record.jsonl").read()
    assert events[-1]["category"] == "failure_streak"


@pytest.mark.anyio
async def test_concurrent_calls_honour_limit(tmp_path: Path) -> None:
    run = write_run(tmp_path, limits={**LIMITS, "code_runs": 3})
    results: list[dict[str, Any]] = []
    async with Client(build_server(run, {"exec": echo})) as client:

        async def one() -> None:
            results.append(await call(client, "exec_python", {"code": "x"}))

        async with anyio.create_task_group() as tg:
            for _ in range(12):
                tg.start_soon(one)
        handles = [r["handle"] for r in results if "handle" in r]
        outs = [await poll(client, h) for h in handles]
    assert sum(o["state"] == "done" for o in outs) == 3
    assert all(o["error"] == "budget exhausted" for o in outs if o["state"] == "failed")
    assert (
        Ledger(tmp_path / "ledger.jsonl", Limits(3, 3, 100, 1.0)).used()["code_runs"]
        == 3
    )


@pytest.mark.anyio
async def test_finish_writes_record_once(tmp_path: Path) -> None:
    async with Client(build_server(write_run(tmp_path))) as client:
        first = await call(client, "finish", {"summary": "done"})
        second = await call(client, "finish", {"summary": "again"})
    assert first["state"] == "finished"
    assert first["budget"]["limits"]["code_runs"] == 3
    assert second == {"error": "run already finished"}
    events = RunRecord.open(tmp_path / "record.jsonl").read()
    assert [e["event"] for e in events] == ["start", "summary", "finish"]
    assert events[0]["seed"] == 7
    assert events[0]["editor"] == "scripted"
    # a restarted server sees the finish too
    async with Client(build_server(tmp_path)) as client:
        assert await call(client, "finish", {"summary": "x"}) == {
            "error": "run already finished"
        }


@pytest.mark.anyio
async def test_restart_keeps_jobs_and_budget(tmp_path: Path) -> None:
    run = write_run(tmp_path)
    async with Client(build_server(run, {"exec": echo})) as client:
        started = await call(client, "exec_python", {"code": "x"})
        await poll(client, started["handle"])
    async with Client(build_server(run)) as client:
        out = await call(client, "status", {"handle": started["handle"]})
    assert out["state"] == "done"
    assert out["budget"]["used"]["code_runs"] == 1


@pytest.mark.anyio
async def test_work_after_finish_is_refused(tmp_path: Path) -> None:
    async with Client(build_server(write_run(tmp_path), {"exec": echo})) as client:
        await call(client, "finish", {"summary": "x"})
        out = await call(client, "exec_python", {"code": "1"})
        out2 = await call(client, "forward_pass", {"request": {}})
    assert out == out2 == {"error": "run already finished"}
    assert not (tmp_path / "ledger.jsonl").exists()
    assert RunRecord.open(tmp_path / "record.jsonl").read()[-1]["event"] == "finish"
    async with Client(build_server(tmp_path, {"exec": echo})) as client:  # restart
        again = await call(client, "exec_python", {"code": "1"})
    assert again == {"error": "run already finished"}


@pytest.mark.anyio
async def test_restart_fails_orphaned_jobs(tmp_path: Path) -> None:
    run = write_run(tmp_path)
    store = HandleStore(run)
    queued = store.submit("exec", {})
    running = store.submit("exec", {})
    store.start(running.handle)
    done = store.submit("exec", {})
    store.start(done.handle)
    store.finish(done.handle, {"ok": True})
    async with Client(build_server(run)) as client:
        for h in (queued.handle, running.handle):
            out = await call(client, "status", {"handle": h})
            assert (out["state"], out["error"]) == ("failed", "server restarted")
        assert (await call(client, "status", {"handle": done.handle}))[
            "state"
        ] == "done"
        assert out["budget"]["used"]["code_runs"] == 0


def test_restart_with_same_run_json_ok(tmp_path: Path) -> None:
    run = write_run(tmp_path)
    build_server(run)
    build_server(run)


@pytest.mark.parametrize(
    "over",
    [
        {"limits": {**LIMITS, "code_runs": 10**9}},
        {"limits": {**LIMITS, "dollars": 1e6}},
        {"seed": 8},
        {"editor": "other"},
        {"model": "m"},
    ],
)
def test_restart_with_changed_run_json_refused(
    tmp_path: Path, over: dict[str, Any]
) -> None:
    run = write_run(tmp_path)
    build_server(run)
    write_run(run, **over)
    with pytest.raises(ValueError, match="differs from the run record"):
        build_server(run)


def test_build_server_bad_run_json(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"run\.json"):
        build_server(tmp_path)


def test_main_bad_run_dir_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path)]) == 2
    assert "run.json" in capsys.readouterr().err


def test_main_runs_stdio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    from mcp.server.mcpserver import MCPServer

    monkeypatch.setattr(
        MCPServer, "run", lambda self, transport: seen.append(transport)
    )
    assert main([str(write_run(tmp_path))]) == 0
    assert seen == ["stdio"]


@pytest.mark.anyio
async def test_stdio_round_trip(tmp_path: Path) -> None:
    run = write_run(tmp_path)
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "astro_weights.mcp", str(run)]
    )
    with anyio.fail_after(30):
        async with Client(params) as client:
            assert client.protocol_version == "2026-07-28"
            tools = await client.list_tools()
            assert len(tools.tools) == 7
            out = await call(client, "exec_python", {"code": "1"})
            assert out == {"error": "exec_python is not available yet"}
            fin = await call(client, "finish", {"summary": "ok"})
    assert fin["state"] == "finished"
    assert RunRecord.open(run / "record.jsonl").read()[-1]["event"] == "finish"


def test_stdio_bad_run_dir_exits_2(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "astro_weights.mcp", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 2
    assert "run.json" in proc.stderr


# ---------------------------------------------------------------- kernel

N0 = "model.layers.0.mlp.down_proj.weight"
NEEDS_SANDBOX = "no working OS sandbox"


def kernel_run(run_dir: Path, **over: Any) -> Path:
    fake_model_dir(run_dir / "model")
    cfg = {"limits": {**LIMITS, "code_runs": 10}, **over}
    return write_run(run_dir, model_dir="model", kernel={"timeout_s": 20}, **cfg)


@pytest.fixture
def no_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_detect", lambda: (None, "disabled for this test"))


@pytest.fixture
def unsafe(request: pytest.FixtureRequest) -> bool:
    """True: kernel runs unsandboxed (sandbox disabled); False: real sandbox."""
    if sandbox.probe() is None:
        request.getfixturevalue("no_sandbox")
        return True
    return False


async def run_cell(client: Client, code: str) -> dict[str, Any]:
    started = await call(client, "exec_python", {"code": code})
    return await poll(client, started["handle"], timeout=60)


@pytest.mark.anyio
async def test_exec_python_runs_in_kernel(tmp_path: Path, unsafe: bool) -> None:
    run = kernel_run(tmp_path)
    server = build_server(run, allow_unsafe_kernel=unsafe)
    async with Client(server) as client:
        out = await run_cell(client, "x = 20\nx + 22")
        assert out["state"] == "done"
        assert out["result"] == {"ok": True, "output": "42\n", "restarted": False}
        assert out["budget"]["used"]["code_runs"] == 1
        bad = await run_cell(client, "raise ValueError('boom')")
        assert bad["state"] == "done"  # a failing cell is still a code run
        assert not bad["result"]["ok"]
        assert bad["budget"]["used"]["code_runs"] == 2
        assert bad["budget"]["streaks"]["code_runs"] == 0
        saved = await run_cell(
            client, f"W = load('{N0}'); W[0] += 0.3; save_edit('{N0}', W)"
        )
        assert saved["result"]["ok"], saved["result"]["output"]
        rejected = await run_cell(
            client, f"save_edit('{N0}', load_original('{N0}') * 2)"
        )
        assert "scalar multiple" in rejected["result"]["output"]
        dropped = await run_cell(client, f"drop_edit('{N0}'); edited()")
        assert dropped["result"]["output"] == "[]\n"
    events = RunRecord.open(run / "record.jsonl").read()
    names = [e["event"] for e in events]
    assert names[1:] == ["edit_saved", "edit_rejected", "edit_dropped"]
    assert events[1]["tensor"] == N0
    assert 0 < events[1]["rel_frobenius"] < 0.5
    assert "scalar multiple" in events[2]["error"]


@pytest.mark.anyio
async def test_timeout_is_a_code_run_and_restarts(tmp_path: Path, unsafe: bool) -> None:
    run = kernel_run(tmp_path)
    write_run(
        run,
        model_dir="model",
        kernel={"timeout_s": 1},
        limits={**LIMITS, "code_runs": 10},
    )
    async with Client(build_server(run, allow_unsafe_kernel=unsafe)) as client:
        slow = await run_cell(client, "import time; time.sleep(30)")
        assert slow["state"] == "done"
        assert "killed after 1 s" in slow["result"]["output"]
        nxt = await run_cell(client, "1")
        assert nxt["result"]["restarted"]
        assert nxt["budget"]["used"]["code_runs"] == 2
    names = [e["event"] for e in RunRecord.open(run / "record.jsonl").read()]
    assert names[1:] == ["kernel_restarted"]


@pytest.mark.anyio
async def test_sandbox_missing_refuses_free(tmp_path: Path, no_sandbox: None) -> None:
    run = kernel_run(tmp_path)
    async with Client(build_server(run)) as client:
        out = await call(client, "exec_python", {"code": "1"})
    assert "no OS sandbox: disabled for this test" in out["error"]
    assert "allow_unsafe" in out["error"]
    events = RunRecord.open(run / "record.jsonl").read()
    assert events[-1]["event"] == "sandbox_unavailable"
    assert events[-1]["tool"] == "exec_python"
    assert not (run / "ledger.jsonl").exists()


@pytest.mark.anyio
async def test_unsafe_flag_allows(tmp_path: Path, no_sandbox: None) -> None:
    run = kernel_run(tmp_path)
    async with Client(build_server(run, allow_unsafe_kernel=True)) as client:
        out = await run_cell(client, "print('hi')")
    assert out["result"]["output"] == "hi\n"
    events = RunRecord.open(run / "record.jsonl").read()
    assert [e["event"] for e in events] == ["start", "unsafe_kernel"]


@pytest.mark.anyio
async def test_given_exec_executor_wins(tmp_path: Path) -> None:
    run = kernel_run(tmp_path)
    async with Client(build_server(run, {"exec": echo})) as client:
        out = await run_cell(client, "1")
    assert out["result"] == {"echo": {"code": "1"}}
    assert not (run / "kernel").exists()


@pytest.mark.anyio
async def test_kernel_start_failure_is_a_failed_job(
    tmp_path: Path, unsafe: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: kernel_mod.Kernel) -> Any:
        raise kernel_mod.KernelError("kernel failed to start: numpy missing")

    monkeypatch.setattr(kernel_mod.Kernel, "_start", broken)
    run = kernel_run(tmp_path)
    async with Client(build_server(run, allow_unsafe_kernel=unsafe)) as client:
        out = await run_cell(client, "1")
    assert out["state"] == "failed"
    assert "numpy missing" in out["error"]
    assert out["budget"]["used"]["code_runs"] == 0
    assert out["budget"]["streaks"]["code_runs"] == 1


@pytest.mark.anyio
async def test_restart_reports_lost_vars_and_keeps_edits(
    tmp_path: Path, unsafe: bool
) -> None:
    run = kernel_run(tmp_path)
    async with Client(build_server(run, allow_unsafe_kernel=unsafe)) as client:
        await run_cell(client, f"W = load('{N0}'); W[0] += 0.3; save_edit('{N0}', W)")
    async with Client(build_server(run, allow_unsafe_kernel=unsafe)) as client:
        first = await run_cell(client, "edited(), 'W' in globals()")
        second = await run_cell(client, "1")
    assert first["result"] == {
        "ok": True,
        "output": f"(['{N0}'], False)\n",
        "restarted": True,
    }
    assert not second["result"]["restarted"]
    assert second["budget"]["used"]["code_runs"] == 3


@pytest.mark.anyio
async def test_concurrent_exec_runs_one_at_a_time(tmp_path: Path, unsafe: bool) -> None:
    run = kernel_run(tmp_path, limits={**LIMITS, "code_runs": 10})
    cell = (
        "import time\nt0 = time.monotonic(); time.sleep(0.2)\n"
        "spans = globals().get('spans', []) + [(t0, time.monotonic())]"
    )
    async with Client(build_server(run, allow_unsafe_kernel=unsafe)) as client:
        started: list[dict[str, Any]] = []

        async def one() -> None:
            started.append(await call(client, "exec_python", {"code": cell}))

        async with anyio.create_task_group() as tg:
            for _ in range(4):
                tg.start_soon(one)
        outs = [await poll(client, s["handle"], timeout=60) for s in started]
        spans = await run_cell(client, "import json; print(json.dumps(sorted(spans)))")
    assert [o["state"] for o in outs] == ["done"] * 4
    assert all(o["result"]["ok"] for o in outs)
    got = json.loads(spans["result"]["output"])
    assert len(got) == 4
    assert all(a[1] <= b[0] for a, b in itertools.pairwise(got))
    assert spans["budget"]["used"]["code_runs"] == 5


def test_bad_model_dir_refused(tmp_path: Path) -> None:
    run = write_run(tmp_path, model_dir="missing")
    with pytest.raises(ValueError, match="model"):
        build_server(run)
    assert main([str(run)]) == 2


def test_main_unsafe_kernel_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_sandbox: None
) -> None:
    from mcp.server.mcpserver import MCPServer

    monkeypatch.setattr(MCPServer, "run", lambda self, transport: None)
    run = kernel_run(tmp_path)
    assert main([str(run), "--unsafe-kernel"]) == 0
    events = RunRecord.open(run / "record.jsonl").read()
    assert [e["event"] for e in events] == ["start", "unsafe_kernel"]


def test_forward_needs_the_hf_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from astro_weights.mcp import forward

    monkeypatch.setattr(forward, "hf_available", lambda: False)
    run = kernel_run(tmp_path)

    async def go() -> dict[str, Any]:
        async with Client(build_server(run, {"exec": echo})) as client:
            return await call(client, "forward_pass", {"request": {}})

    out = anyio.run(go)
    assert out["error"].startswith("forward_pass is not available yet")
    assert "astro-weights[hf]" in out["error"]
    event = RunRecord.open(run / "record.jsonl").read()[-1]
    assert (event["event"], event["tool"]) == ("tool_unavailable", "forward_pass")
    assert "astro-weights[hf]" in event["error"]


def test_unreadable_forward_results_are_deleted(tmp_path: Path) -> None:
    from astro_weights.mcp import forward

    run = write_run(tmp_path)
    store = HandleStore(run)
    folders: dict[str, Path] = {}

    def job(state: str) -> str:
        out = {"done": "d", "running": "e", "queued": "a", "failed": "f"}[state] * 32
        j = store.submit("forward", {"request": {}, "out": out})
        folders[state] = run / forward.RESULTS / out
        folders[state].mkdir(parents=True)
        if state != "queued":
            store.start(j.handle)
        if state == "done":
            store.finish(j.handle, {"keys": []})
        if state == "failed":
            store.fail(j.handle, "budget exhausted")
        return j.handle

    for state in ("done", "running", "queued", "failed"):
        job(state)
    build_server(run)  # a restart deletes what nobody can read
    assert {s for s, d in folders.items() if d.exists()} == {"done"}
    # a job refused at charge time while the server runs: status deletes it
    server = build_server(run)
    later = store.submit("forward", {"request": {}, "out": "b" * 32})
    folder = run / forward.RESULTS / ("b" * 32)
    folder.mkdir()
    store.start(later.handle)
    store.fail(later.handle, "budget exhausted")

    async def go() -> dict[str, Any]:
        async with Client(server) as client:
            return await call(client, "status", {"handle": later.handle})

    assert anyio.run(go)["state"] == "failed"
    assert not folder.exists()
    assert folders["done"].exists()


def ledger_lines(run: Path) -> list[str]:
    path = run / "ledger.jsonl"
    return path.read_text().splitlines() if path.exists() else []


def events(run: Path) -> list[dict[str, Any]]:
    return RunRecord.open(run / "record.jsonl").read()


@pytest.mark.anyio
async def test_report_usage_records_and_reloads(tmp_path: Path) -> None:
    async with Client(build_server(write_run(tmp_path))) as client:
        out = await call(
            client,
            "report_usage",
            {"tokens": 30, "dollars": 0.25, "turn": 2, "model": "m"},
        )
        zero = await call(
            client, "report_usage", {"tokens": 0, "dollars": 0, "turn": 3, "model": "m"}
        )
    assert out["over"] == []
    assert out["budget"]["used"]["tokens"] == 30
    assert out["budget"]["used"]["dollars"] == 0.25
    assert zero["budget"]["used"] == out["budget"]["used"]
    assert len(ledger_lines(tmp_path)) == 2
    usage = [e for e in events(tmp_path) if e["event"] == "usage"]
    assert [(e["turn"], e["model"], e["tokens"], e["dollars"]) for e in usage] == [
        (2, "m", 30, 0.25),
        (3, "m", 0, 0),
    ]
    async with Client(build_server(tmp_path)) as client:
        again = await call(client, "check_spend", {"estimate": 0})
    assert again["budget"]["used"]["tokens"] == 30
    assert again["budget"]["used"]["dollars"] == 0.25


@pytest.mark.anyio
async def test_report_usage_past_limit_still_records(tmp_path: Path) -> None:
    async with Client(build_server(write_run(tmp_path))) as client:
        out = await call(
            client,
            "report_usage",
            {"tokens": 150, "dollars": 1.5, "turn": 0, "model": "m"},
        )
    assert out["over"] == ["tokens", "dollars"]
    assert out["budget"]["used"]["tokens"] == 150
    assert out["budget"]["remaining"]["dollars"] == 0


NOT_JSON: list[dict[str, Any]] = [
    {"dollars": float("nan")},
    {"dollars": float("inf")},
]

BAD_USAGE: list[dict[str, Any]] = [
    {"tokens": True},
    {"tokens": -1},
    {"tokens": 1.5},
    {"dollars": True},
    {"dollars": -0.01},
    {"dollars": "x"},
    {"dollars": 10**400},
    {"tokens": "x"},
    {"turn": "x"},
    {"model": 5},
    {"model": "  "},
    {"turn": -1},
    {"turn": True},
    {"model": ""},
    {"model": "x" * 201},
]


@pytest.mark.anyio
@pytest.mark.parametrize("bad", BAD_USAGE, ids=lambda b: str(b)[:30])
async def test_report_usage_bad_args(tmp_path: Path, bad: dict[str, Any]) -> None:
    good = {"tokens": 5, "dollars": 0.5, "turn": 1, "model": "m"}
    async with Client(build_server(write_run(tmp_path))) as client:
        out = await call(client, "report_usage", {**good, **bad})
    assert "error" in out
    assert ledger_lines(tmp_path) == []
    assert [e["event"] for e in events(tmp_path)] == ["start"]


@pytest.mark.anyio
@pytest.mark.parametrize("bad", NOT_JSON, ids=lambda b: str(b)[:30])
async def test_report_usage_nan_inf(tmp_path: Path, bad: dict[str, Any]) -> None:
    good = {"tokens": 5, "dollars": 0.5, "turn": 1, "model": "m"}
    async with Client(build_server(write_run(tmp_path))) as client:
        try:  # the client may refuse to send NaN or inf at all
            result = await client.call_tool("report_usage", {**good, **bad})
        except Exception:  # noqa: BLE001
            refused = True
        else:
            refused = result.is_error or "error" in (result.structured_content or {})
    assert refused
    assert ledger_lines(tmp_path) == []
    assert [e["event"] for e in events(tmp_path)] == ["start"]


@pytest.mark.anyio
async def test_check_spend_boundary_writes_nothing(tmp_path: Path) -> None:
    async with Client(build_server(write_run(tmp_path))) as client:
        await call(
            client,
            "report_usage",
            {"tokens": 1, "dollars": 0.4, "turn": 0, "model": "m"},
        )
        before = ledger_lines(tmp_path)
        n = len(events(tmp_path))
        fits = await call(client, "check_spend", {"estimate": 0.6})
        over = await call(client, "check_spend", {"estimate": 0.61})
        free = await call(client, "check_spend", {"estimate": 0})
        bad = [
            await call(client, "check_spend", {"estimate": x})
            for x in (-1, True, "x", 10**400)
        ]
    assert fits["ok"] is True
    assert fits["budget"]["used"]["dollars"] == 0.4
    assert over["ok"] is False
    assert "dollars budget exhausted" in over["error"]
    assert over["budget"]["used"]["dollars"] == 0.4
    assert free["ok"] is True
    for r in bad:
        assert "error" in r
        assert "ok" not in r
    assert ledger_lines(tmp_path) == before
    assert len(events(tmp_path)) == n


@pytest.mark.anyio
async def test_check_spend_failure_streak_not_ok(tmp_path: Path) -> None:
    run = write_run(tmp_path)
    ledger = Ledger(run / "ledger.jsonl", Limits(3, 3, 100, 1.0))
    for i in range(3):
        ledger.fail("dollars", "x", ref=str(i))
    async with Client(build_server(run)) as client:
        out = await call(client, "check_spend", {"estimate": 0.1})
    assert out["ok"] is False
    assert "failed 3 times" in out["error"]
    async with Client(build_server(run)) as client:
        zero = await call(client, "check_spend", {"estimate": 0})
    assert zero["ok"] is False
    assert "failed 3 times" in zero["error"]


@pytest.mark.anyio
async def test_check_spend_zero_ok_when_over_limit(tmp_path: Path) -> None:
    async with Client(build_server(write_run(tmp_path))) as client:
        await call(
            client,
            "report_usage",
            {"tokens": 1, "dollars": 6, "turn": 0, "model": "m"},
        )
        zero = await call(client, "check_spend", {"estimate": 0})
        some = await call(client, "check_spend", {"estimate": 0.01})
    assert zero["ok"] is True
    assert some["ok"] is False


@pytest.mark.anyio
async def test_report_refusal_records(tmp_path: Path) -> None:
    async with Client(build_server(write_run(tmp_path))) as client:
        out = await call(
            client,
            "report_refusal",
            {"category": "policy", "model": "m", "detail": "d" * 5000},
        )
        bare = await call(client, "report_refusal", {"category": "c", "model": "m"})
    assert out == bare == {"recorded": True}
    refusals = [e for e in events(tmp_path) if e["event"] == "refusal"]
    assert [(r["category"], r["editor"], r["model"]) for r in refusals] == [
        ("policy", "scripted", "m"),
        ("c", "scripted", "m"),
    ]
    assert len(refusals[0]["detail"]) == 2000
    assert refusals[1]["detail"] == ""


@pytest.mark.anyio
@pytest.mark.parametrize(
    "bad",
    [
        {"category": ""},
        {"category": "  "},
        {"model": ""},
        {"model": "x" * 201},
        {"model": "  "},
        {"category": 5},
        {"model": 5},
        {"detail": 3},
    ],
)
async def test_report_refusal_bad_args(tmp_path: Path, bad: dict[str, Any]) -> None:
    async with Client(build_server(write_run(tmp_path))) as client:
        out = await call(
            client, "report_refusal", {"category": "c", "model": "m", **bad}
        )
    assert "error" in out
    assert [e["event"] for e in events(tmp_path)] == ["start"]


@pytest.mark.anyio
async def test_after_finish_only_report_usage_records(tmp_path: Path) -> None:
    async with Client(build_server(write_run(tmp_path))) as client:
        await call(client, "finish", {"summary": "x"})
        n = len(events(tmp_path))
        usage = await call(
            client,
            "report_usage",
            {"tokens": 7, "dollars": 0.1, "turn": 9, "model": "m"},
        )
        spend = await call(client, "check_spend", {"estimate": 0.1})
        refusal = await call(client, "report_refusal", {"category": "c", "model": "m"})
    assert usage["budget"]["used"]["tokens"] == 7
    assert spend == refusal == {"error": "run already finished"}
    after = events(tmp_path)[n:]
    assert [e["event"] for e in after] == ["usage"]
    assert len(ledger_lines(tmp_path)) == 2
    finish = next(e for e in events(tmp_path) if e["event"] == "finish")
    assert finish["budget"]["used"]["tokens"] == 0
    assert usage["budget"]["used"]["tokens"] == 7


# ---------------------------------------------------------------- finish exports


def export_run(run_dir: Path, **export: Any) -> Path:
    fake_model_dir(run_dir / "model")
    over = {"export": export} if export else {}
    return write_run(run_dir, model_dir="model", **over)


def save_edit(run_dir: Path, full_rank: bool = False) -> None:
    import numpy as np

    from astro_weights.mcp.edit_store import EditStore

    store = EditStore(run_dir, run_dir / "model")
    w = store.original(N0)
    rng = np.random.default_rng(0)
    if full_rank:
        d = rng.standard_normal(w.shape)
    else:
        d = np.outer(rng.standard_normal(w.shape[0]), rng.standard_normal(w.shape[1]))
    store.save(
        N0, (w + d * (0.3 * np.linalg.norm(w) / np.linalg.norm(d))).astype(np.float32)
    )


NO_KERNEL = {"exec": echo, "forward": echo}


@pytest.mark.anyio
async def test_finish_exports_the_stored_edits(tmp_path: Path) -> None:
    from astro_weights.export import load_peft

    run = export_run(tmp_path, rank=2)
    save_edit(run)
    async with Client(build_server(run, NO_KERNEL)) as client:
        out = await call(client, "finish", {"summary": "done"})
    assert out["state"] == "finished"
    export = out["export"]
    assert export["adapter"] == str(run / "export" / "adapter")
    assert export["gguf"] is None
    assert export["rank"] == 2
    assert export["residuals"][N0] < 0.05
    edit = load_peft(export["adapter"])
    assert [d.module for d in edit.deltas] == [N0.removesuffix(".weight")]
    assert edit.base_model == str(run / "model")
    record = RunRecord.open(run / "record.jsonl").read()
    names = [e["event"] for e in record]
    assert names[-3:] == ["summary", "export", "finish"]
    assert record[-2]["adapter"] == export["adapter"]


@pytest.mark.anyio
async def test_export_names_the_base_model_from_its_config(tmp_path: Path) -> None:
    from astro_weights.export import load_peft

    run = export_run(tmp_path)
    (run / "model" / "config.json").write_text(
        json.dumps({"_name_or_path": "org/base"})
    )
    save_edit(run)
    async with Client(build_server(run, NO_KERNEL)) as client:
        out = await call(client, "finish", {"summary": "done"})
    assert load_peft(out["export"]["adapter"]).base_model == "org/base"


@pytest.mark.anyio
async def test_export_past_the_residual_cap_still_finishes(tmp_path: Path) -> None:
    run = export_run(tmp_path, rank=1, max_rel_residual=0.05)
    save_edit(run, full_rank=True)
    async with Client(build_server(run, NO_KERNEL)) as client:
        out = await call(client, "finish", {"summary": "done"})
        again = await call(client, "finish", {"summary": "again"})
    assert out["state"] == "finished"
    assert out["export"] is None
    assert "exceeds max_rel_residual" in out["export_error"]
    assert again == {"error": "run already finished"}
    assert not (run / "export" / "adapter").exists()
    record = RunRecord.open(run / "record.jsonl").read()
    assert [e["event"] for e in record][-3:] == ["summary", "export_failed", "finish"]
    assert "exceeds max_rel_residual" in record[-2]["error"]


@pytest.mark.anyio
async def test_finish_without_edits_exports_nothing(tmp_path: Path) -> None:
    run = export_run(tmp_path)
    async with Client(build_server(run, NO_KERNEL)) as client:
        out = await call(client, "finish", {"summary": "done"})
    assert out["export"] is None
    assert out["note"] == "no edits: nothing to export"
    assert not (run / "export").exists()
    names = [e["event"] for e in RunRecord.open(run / "record.jsonl").read()]
    assert names[-3:] == ["summary", "export_skipped", "finish"]


@pytest.mark.anyio
async def test_finish_without_model_dir_has_nothing_to_export(tmp_path: Path) -> None:
    async with Client(build_server(write_run(tmp_path))) as client:
        out = await call(client, "finish", {"summary": "done"})
    assert out["export"] is None
    assert out["note"] == "no model_dir: nothing to export"


@pytest.mark.parametrize(
    ("export", "match"),
    [
        ({"rank": 0}, "export.rank"),
        ({"rank": True}, "export.rank"),
        ({"max_rel_residual": -1}, "export.max_rel_residual"),
        ({"max_rel_residual": "x"}, "export.max_rel_residual"),
        ({"max_rel_residual": 10**400}, "export.max_rel_residual"),
        ({"base_gguf": ""}, "export.base_gguf"),
        ({"gguf_out_type": "q4"}, "export.gguf_out_type"),
        ({"other": 1}, "export must be an object"),
    ],
)
def test_bad_export_config_is_refused(
    tmp_path: Path, export: dict[str, Any], match: str
) -> None:
    from astro_weights.mcp.jobs import load_run_config

    with pytest.raises(ValueError, match=match):
        load_run_config(write_run(tmp_path, export=export))


def test_export_config_defaults_and_paths(tmp_path: Path) -> None:
    from astro_weights.mcp.jobs import load_run_config

    cfg = load_run_config(write_run(tmp_path))
    assert (cfg.export_rank, cfg.export_max_rel_residual) == (8, 0.05)
    assert (cfg.export_base_gguf, cfg.export_gguf_out_type) == (None, "q8_0")
    export = {
        "rank": 4,
        "max_rel_residual": 1,
        "base_gguf": "b.gguf",
        "gguf_out_type": "f16",
    }
    cfg = load_run_config(write_run(tmp_path, export=export))
    assert cfg.export_rank == 4
    assert cfg.export_max_rel_residual == 1.0
    assert cfg.export_base_gguf == tmp_path / "b.gguf"
    assert cfg.export_gguf_out_type == "f16"


@pytest.mark.anyio
async def test_finish_passes_the_gguf_options_to_the_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from astro_weights.mcp import server

    run = export_run(tmp_path, base_gguf="base.gguf", gguf_out_type="bf16", rank=3)
    save_edit(run)
    seen: list[dict[str, Any]] = []

    def fake_export(store: Any, out_dir: Path, **kw: Any) -> dict[str, Any]:
        seen.append({"out_dir": out_dir, **kw})
        return {
            "adapter": out_dir / "adapter",
            "gguf": out_dir / "model.gguf",
            "residuals": {N0: 0.01},
            "rank": 3,
        }

    monkeypatch.setattr(server, "export_edits", fake_export)
    async with Client(build_server(run, NO_KERNEL)) as client:
        out = await call(client, "finish", {"summary": "done"})
    assert out["export"]["gguf"] == str(run / "export" / "model.gguf")
    assert seen[0]["base_gguf"] == run / "base.gguf"
    assert seen[0]["gguf_out_type"] == "bf16"
    assert seen[0]["rank"] == 3
    assert seen[0]["out_dir"] == run / "export"


def test_start_event_records_library_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from astro_weights.mcp import server as server_mod

    monkeypatch.setattr(server_mod, "library_commit", lambda: "abc-dirty")
    run = write_run(tmp_path)
    build_server(run)
    start = RunRecord.open(run / "record.jsonl").read()[0]
    assert start["library_commit"] == "abc-dirty"


@pytest.mark.anyio
async def test_split_manifests_and_ids_never_reach_the_agent(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """Confirm/test ids, split manifests and the seal ledger stay outside the agent.

    No module under ``astro_weights.mcp`` imports the splits or seal code; no
    RunConfig field is about splits, and none holds an absolute path outside the
    run dir; the tool list names none of it; after a kernel session neither the run
    dir nor the kernel work dir holds a manifest, a probe id or a manifest hash; and
    the sandboxed kernel cannot open the manifest by absolute path. On Linux CI
    (bwrap installed) the sandbox is required, so that check is never skipped there.
    """
    import ast
    import dataclasses
    import os
    import re

    import numpy as np

    import astro_weights.mcp as mcp_pkg
    from astro_weights.edits import Edit, rank1
    from astro_weights.eval import Record
    from astro_weights.eval.seal import record_scoring, scoring_of, seal_salt
    from astro_weights.eval.splits import SplitManifest
    from astro_weights.mcp.jobs import RunConfig, load_run_config

    on_ci_linux = os.environ.get("GITHUB_ACTIONS") == "true" and sys.platform == "linux"
    if sandbox.probe() is None:
        assert not on_ci_linux, "CI installs bwrap: the sandboxed branch must run"
        request.getfixturevalue("no_sandbox")
        unsafe = True
    else:
        unsafe = False

    about = re.compile(r"split|manifest|dataset|seal|item|test_ids", re.IGNORECASE)
    fields = [f.name for f in dataclasses.fields(RunConfig)]
    assert not [f for f in fields if about.search(f)], fields

    banned = {"astro_weights.eval.splits", "astro_weights.eval.seal"}
    for py in Path(mcp_pkg.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found = {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                found = {mod, *(f"{mod}.{a.name}" for a in node.names)}
            else:
                continue
            assert not found & banned, f"{py} imports {found & banned}"

    secret = tmp_path / "eval-secret"
    secret.mkdir()
    manifest = SplitManifest.build(
        [f"leakprobe-{i}" for i in range(60)],
        dataset="synthetic",
        revision="r1",
        file_sha256="ab" * 32,
        seed=0,
    )
    manifest.save(secret / "splits.json")
    rng = np.random.default_rng(0)
    probe_edit = Edit(
        (rank1("m", rng.standard_normal(4), rng.standard_normal(3)),), "b"
    )
    test_records = [
        Record(i, s, "edit", "h-" + i, "o", 1.0, 1, False)
        for i in manifest.test
        for s in range(manifest.k)
    ]
    salt = seal_salt(secret / "seal")
    record_scoring(
        secret / "seal", scoring_of(probe_edit, manifest, test_records, salt)
    )
    probes = [
        *manifest.confirm,
        *manifest.test,
        manifest.manifest_sha256,
        str(secret),
        "splits.json",
        "seal.jsonl",
        "seal.salt",
        salt.hex(),
    ]

    def scan(run_dir: Path) -> None:
        for path in run_dir.rglob("*"):
            assert path.name not in {"splits.json", "seal.jsonl", "seal.salt"}
            if path.is_file():
                data = path.read_bytes()
                assert salt not in data, f"{path} holds the seal salt"
                leaked = [p for p in probes if p.encode() in data]
                assert not leaked, f"{path} holds {leaked[:3]}"

    run = kernel_run(tmp_path / "run")
    root = run.resolve()

    def inside(value: object, where: str) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                inside(v, f"{where}.{k}")
        elif isinstance(value, list | tuple):
            for i, v in enumerate(value):
                inside(v, f"{where}[{i}]")
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            inside(dataclasses.asdict(value), where)
        elif isinstance(value, str | Path) and Path(value).is_absolute():
            assert Path(value).resolve().is_relative_to(root), f"{where}={value}"

    inside(json.loads((run / "run.json").read_text(encoding="utf-8")), "run.json")
    cfg = load_run_config(run)
    inside({f: getattr(cfg, f) for f in fields}, "RunConfig")
    assert cfg.model_dir is not None
    assert cfg.model_dir.resolve().is_relative_to(root)

    async with Client(build_server(run, allow_unsafe_kernel=unsafe)) as client:
        tools = await client.list_tools()
        text = json.dumps([t.model_dump(mode="json") for t in tools.tools])
        assert "manifest" not in text.lower()
        listing = await run_cell(
            client,
            "import os\nprint(sorted(os.listdir('.')), sorted(os.environ))",
        )
        assert listing["result"]["ok"], listing["result"]["output"]
        assert not [p for p in probes if p in listing["result"]["output"]]
        scan(run)
        if not unsafe:  # this cell's code and traceback name the path: scan first
            opened = await run_cell(
                client, f"open({str(secret / 'splits.json')!r}).read()"
            )
            assert not opened["result"]["ok"]
            out = opened["result"]["output"]
            assert "PermissionError" in out or "FileNotFoundError" in out, out
