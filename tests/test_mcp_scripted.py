import json
import threading
from pathlib import Path
from typing import Any

import pytest

from astro_weights.mcp.record import RunRecord

LIMITS = {"forward_passes": 3, "code_runs": 3, "tokens": 100, "dollars": 1.0}


def write_run(run_dir: Path, **over: Any) -> Path:
    cfg = {"seed": 7, "editor": "scripted", "model": None, "limits": LIMITS, **over}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(cfg))
    return run_dir


pytest.importorskip("mcp")

from mcp import Client

from astro_weights.mcp.editors.scripted import (
    DEFAULT_SCRIPT,
    ScriptFailed,
    Step,
    main,
    run_scripted,
)
from astro_weights.mcp.server import build_server


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def ok(params: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "params": params}


@pytest.mark.anyio
async def test_default_script_walks_every_step(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    def exec_(params: dict[str, Any]) -> dict[str, Any]:
        seen.append(params)
        return {"ok": True}

    server = build_server(write_run(tmp_path), {"exec": exec_, "forward": ok})
    async with Client(server) as client:
        results = await run_scripted(client, poll_s=0.01)
    assert len(results) == len(DEFAULT_SCRIPT)
    assert [r.get("state") for r in results] == [
        "queued",
        "done",
        "queued",
        "done",
        "queued",
        "done",
        "queued",
        "done",
        "finished",
    ]
    assert results[-1]["budget"]["used"] == {
        "forward_passes": 2,
        "code_runs": 2,
        "tokens": 0,
        "dollars": 0,
    }
    h_base, h_edit = results[2]["handle"], results[4]["handle"]
    assert h_base in seen[1]["code"]
    assert h_edit in seen[1]["code"]
    events = [e["event"] for e in RunRecord.open(tmp_path / "record.jsonl").read()]
    assert events == ["start", "summary", "finish"]


@pytest.mark.anyio
async def test_failed_job_names_its_step(tmp_path: Path) -> None:
    def boom(params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("bad")

    server = build_server(write_run(tmp_path), {"exec": boom, "forward": ok})
    async with Client(server) as client:
        with pytest.raises(ScriptFailed) as err:
            await run_scripted(client, poll_s=0.01)
    assert err.value.index == 1
    assert err.value.result["state"] == "failed"


@pytest.mark.anyio
async def test_cell_that_raised_fails_the_step(tmp_path: Path) -> None:
    def cell(params: dict[str, Any]) -> dict[str, Any]:
        return {"ok": False, "output": "Traceback"}

    server = build_server(write_run(tmp_path), {"exec": cell, "forward": ok})
    async with Client(server) as client:
        with pytest.raises(ScriptFailed) as err:
            await run_scripted(client, poll_s=0.01)
    assert err.value.index == 1


@pytest.mark.anyio
async def test_error_result_names_its_step(tmp_path: Path) -> None:
    run = write_run(tmp_path, limits={**LIMITS, "code_runs": 0})
    async with Client(build_server(run, {"exec": ok, "forward": ok})) as client:
        with pytest.raises(ScriptFailed) as err:
            await run_scripted(client, poll_s=0.01)
    assert err.value.index == 0
    assert "budget exhausted" in err.value.result["error"]


@pytest.mark.anyio
async def test_tool_error_is_a_failure(tmp_path: Path) -> None:
    async with Client(build_server(write_run(tmp_path))) as client:
        with pytest.raises(ScriptFailed) as err:
            await run_scripted(client, [Step("exec_python", {"nope": 1})])
    assert err.value.index == 0
    assert "error" in err.value.result


@pytest.mark.anyio
async def test_poll_timeout(tmp_path: Path) -> None:
    release = threading.Event()

    def slow(params: dict[str, Any]) -> dict[str, Any]:
        release.wait(10)
        return {"ok": True}

    server = build_server(write_run(tmp_path), {"exec": slow})
    script = [Step("exec_python", {"code": "x"}), Step()]
    try:
        async with Client(server) as client:
            with pytest.raises(ScriptFailed) as err:
                await run_scripted(client, script, poll_s=0.01, timeout_s=0.1)
    finally:
        release.set()
    assert err.value.index == 1
    assert err.value.result["error"] == "poll timed out"


@pytest.mark.anyio
async def test_poll_without_handle(tmp_path: Path) -> None:
    async with Client(build_server(write_run(tmp_path))) as client:
        with pytest.raises(ScriptFailed) as err:
            await run_scripted(client, [Step()])
    assert err.value.index == 0


def test_main_without_run_json_exits_2(tmp_path: Path) -> None:
    assert main([str(tmp_path)]) == 2


def test_main_bad_args_exit_2() -> None:
    with pytest.raises(SystemExit) as err:
        main([])
    assert err.value.code == 2


def test_main_reports_a_failed_step_over_stdio(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # no model_dir: the server has no exec tool, so step 0 is an error result
    assert main([str(write_run(tmp_path))]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["step"] == 0
    assert "not available" in out["result"]["error"]


def test_scripted_run_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from mcp_tiny_hf import save_tiny_model

    from astro_weights.mcp import sandbox
    from astro_weights.mcp.budget import Ledger, Limits

    model = save_tiny_model(tmp_path / "model")
    limits = {"forward_passes": 5, "code_runs": 5, "tokens": 0, "dollars": 0}
    run = write_run(
        tmp_path / "run",
        limits=limits,
        model_dir=str(model),
        kernel={"timeout_s": 60},
        forward={"device": "cpu", "max_new_tokens": 8},
    )
    flags = ["--unsafe-kernel"] if sandbox.probe() is None else []
    assert main([str(run), *flags]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["steps"] == len(DEFAULT_SCRIPT)
    assert len(printed["handles"]) == 4
    assert float(printed["outputs"][-1]) > 0  # max |a - b|: the edit is visible
    events = [e["event"] for e in RunRecord.open(run / "record.jsonl").read()]
    assert events[0] == "start"
    assert {"edit_saved", "summary", "finish"} <= set(events)
    assert events[-1] == "finish"
    used = Ledger(run / "ledger.jsonl", Limits(**limits)).used()
    assert used == {"forward_passes": 2, "code_runs": 2, "tokens": 0, "dollars": 0}
    final = printed["budget"]["used"]
    assert final["tokens"] == 0
    assert final["dollars"] == 0
