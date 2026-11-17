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

import anyio  # type: ignore[import-not-found, unused-ignore]
from mcp import Client, StdioServerParameters

from astro_weights.mcp.server import build_server, main


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def call(client: Client, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    result = await client.call_tool(tool, args)
    assert not result.is_error
    assert result.structured_content is not None
    return dict(result.structured_content)


async def poll(client: Client, handle: str) -> dict[str, Any]:
    with anyio.fail_after(10):
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
