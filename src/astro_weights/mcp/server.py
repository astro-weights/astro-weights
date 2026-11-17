"""Stdio MCP server: four tools over the handle store, budget ledger and run record.

Needs the ``mcp`` extra. All state lives under the run directory (``run.json``,
``jobs/``, ``ledger.jsonl``, ``record.jsonl``, ``edits/``, ``kernel/``); one store,
one ledger and one record are built per server process and shared by every tool
call. When run.json names a ``model_dir``, ``exec_python`` runs code in a sandboxed
numpy kernel (one per server process) that saves edits through the edit store.
"""

import argparse
import sys
import threading
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from mcp.server.mcpserver import MCPServer
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"{__name__} needs the official MCP SDK: pip install 'astro-weights[mcp]'",
        name=e.name,
    ) from e

from astro_weights.mcp.budget import BudgetExceeded, Kind, Ledger, TooManyFailures
from astro_weights.mcp.edit_store import EditStore
from astro_weights.mcp.handles import HandleStore, UnknownHandle
from astro_weights.mcp.jobs import Executor, JobRunner, RunConfig, load_run_config
from astro_weights.mcp.kernel import Kernel
from astro_weights.mcp.record import RunRecord
from astro_weights.mcp.sandbox import SandboxUnavailable

__all__ = ["build_server", "main"]

_BUDGET_KIND: dict[str, Kind] = {"exec": "code_runs", "forward": "forward_passes"}
_SUMMARY_MAX = 2000


def _open_record(run_dir: Path, cfg: RunConfig) -> RunRecord:
    path = run_dir / "record.jsonl"
    config = {"limits": asdict(cfg.limits)}
    if path.exists() and path.stat().st_size > 0:
        record = RunRecord.open(path)
        first = record.read()[0]
        same = (
            first.get("seed") == cfg.seed
            and first.get("editor") == cfg.editor
            and first.get("model") == cfg.model
            and first.get("config") == config
        )
        if not same:
            raise ValueError("run.json differs from the run record")
        return record
    return RunRecord.start(
        path, seed=cfg.seed, config=config, editor=cfg.editor, model=cfg.model
    )


def _kernel_executor(
    run_dir: Path, cfg: RunConfig, record: RunRecord, allow_unsafe: bool
) -> Executor | str:
    """``exec`` executor running cells in a new Kernel, or why there is none."""
    assert cfg.model_dir is not None
    store = EditStore(run_dir, cfg.model_dir)
    try:
        kernel = Kernel(
            run_dir,
            store,
            allow_unsafe=allow_unsafe,
            timeout_s=cfg.kernel_timeout_s,
            mem_gib=cfg.kernel_mem_gib,
            on_event=lambda name, fields: record.event(name, **fields),
        )
    except SandboxUnavailable as exc:
        return str(exc)
    return lambda params: kernel.exec(params["code"])


def build_server(
    run_dir: Path,
    executors: Mapping[str, Executor] | None = None,
    *,
    allow_unsafe_kernel: bool = False,
) -> MCPServer[Any]:
    """Build the server for ``run_dir``. With no executors the work tools say so.

    With ``model_dir`` in run.json and no ``exec`` executor given, ``exec_python``
    runs in a sandboxed kernel. Without an OS sandbox it returns an error, unless
    ``allow_unsafe_kernel`` runs the kernel unsandboxed.
    """
    run_dir = Path(run_dir)
    cfg = load_run_config(run_dir)
    store = HandleStore(run_dir)
    ledger = Ledger(run_dir / "ledger.jsonl", cfg.limits)
    record = _open_record(run_dir, cfg)
    for job in store.list():  # a restart killed their threads; they can never finish
        if job.state in ("queued", "running"):
            store.fail(job.handle, "server restarted")
    available = dict(executors or {})
    unavailable: dict[str, str] = {}
    if allow_unsafe_kernel:
        record.event("unsafe_kernel")
    if cfg.model_dir is not None and "exec" not in available:
        made = _kernel_executor(run_dir, cfg, record, allow_unsafe_kernel)
        if isinstance(made, str):
            unavailable["exec"] = made
        else:
            available["exec"] = made
    runner = JobRunner(store, ledger, available, _BUDGET_KIND)
    finish_lock = threading.Lock()
    finished = any(e["event"] == "finish" for e in record.read())
    server: MCPServer[Any] = MCPServer("astro-weights")

    def submit(tool: str, kind: str, params: dict[str, Any]) -> dict[str, Any]:
        with finish_lock:
            if finished:
                return {"error": "run already finished"}
            if kind in unavailable:
                record.event("sandbox_unavailable", tool=tool, error=unavailable[kind])
                return {"error": unavailable[kind]}
            if kind not in available:
                record.event("tool_unavailable", tool=tool)
                return {"error": f"{tool} is not available yet"}
            return _submit_locked(kind, params)

    def _submit_locked(kind: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            job = runner.submit(kind, params)
        except (BudgetExceeded, TooManyFailures) as exc:
            category = "budget" if isinstance(exc, BudgetExceeded) else "failure_streak"
            record.refusal(
                category=category, editor=cfg.editor, model=cfg.model, detail=str(exc)
            )
            return {"error": str(exc), "budget": ledger.to_json()}
        return {"handle": job.handle, "state": job.state}

    @server.tool()  # type: ignore[untyped-decorator, unused-ignore]
    def exec_python(code: str) -> dict[str, Any]:
        """Start a Python code run; returns a handle to poll with status."""
        return submit("exec_python", "exec", {"code": code})

    @server.tool()  # type: ignore[untyped-decorator, unused-ignore]
    def forward_pass(request: dict[str, Any]) -> dict[str, Any]:
        """Start a model forward pass; returns a handle to poll with status."""
        return submit("forward_pass", "forward", {"request": request})

    @server.tool()  # type: ignore[untyped-decorator, unused-ignore]
    def status(handle: str) -> dict[str, Any]:
        """Return a job's state and result, plus the budget."""
        try:
            job = store.get(handle)
        except UnknownHandle:
            return {"error": "unknown handle"}
        return {**job.to_json(), "budget": ledger.to_json()}

    @server.tool()  # type: ignore[untyped-decorator, unused-ignore]
    def finish(summary: str) -> dict[str, Any]:
        """End the run and write the run record. Works once."""
        nonlocal finished
        with finish_lock:
            if finished:
                return {"error": "run already finished"}
            record.event("summary", text=summary[:_SUMMARY_MAX])
            budget = ledger.to_json()
            record.finish(budget=budget, outcome="finished")
            finished = True
        return {"state": "finished", "budget": budget}

    return server


def main(argv: list[str] | None = None) -> int:
    """Run the server on stdio for ``RUN_DIR``; exit code 2 if the run directory is bad."""
    parser = argparse.ArgumentParser(prog="python -m astro_weights.mcp")
    parser.add_argument("run_dir", metavar="RUN_DIR", type=Path, help="run directory")
    parser.add_argument(
        "--unsafe-kernel",
        action="store_true",
        help="run agent code without an OS sandbox when none works (never safe)",
    )
    args = parser.parse_args(argv)
    try:
        server = build_server(args.run_dir, allow_unsafe_kernel=args.unsafe_kernel)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    server.run("stdio")
    return 0
