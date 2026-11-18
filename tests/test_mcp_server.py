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
async def test_lists_four_tools(tmp_path: Path) -> None:
    async with Client(build_server(write_run(tmp_path))) as client:
        tools = await client.list_tools()
    assert sorted(t.name for t in tools.tools) == [
        "exec_python",
        "finish",
        "forward_pass",
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
            assert len(tools.tools) == 4
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
