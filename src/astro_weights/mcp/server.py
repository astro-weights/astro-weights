"""Stdio MCP server: seven tools over the handle store, budget ledger and run record.

Needs the ``mcp`` extra. All state lives under the run directory (``run.json``,
``jobs/``, ``ledger.jsonl``, ``record.jsonl``, ``edits/``, ``kernel/``); one store,
one ledger and one record are built per server process and shared by every tool
call. When run.json names a ``model_dir``, ``exec_python`` runs code in a sandboxed
numpy kernel (one per server process) that saves edits through the edit store, and
``forward_pass`` runs the model in this process (``hf`` extra) on base or edited
weights. A forward request is checked in the tool call; a bad one is refused for
free, with no job. Its arrays are read from the kernel with ``forward_result``.

``finish`` exports the stored edits (when run.json names a ``model_dir`` and there
are any) to ``RUN_DIR/export``: a PEFT adapter, and a spliced GGUF when
``export.base_gguf`` is set. A failed export does not stop the run from finishing;
the result and the record say why.

Three tools are for editors, not the model: ``report_usage`` (tokens and dollars
spent, always recorded), ``check_spend`` (free, no write) and ``report_refusal``.
The ledger (or the sum of the ``usage`` events) is the source of truth for spend: a
``report_usage`` after ``finish`` is still recorded, so ``finish.budget`` can leave
out that last turn.
"""

import argparse
import json
import math
import secrets
import sys
import threading
from collections.abc import Callable, Mapping
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

from astro_weights.mcp import forward
from astro_weights.mcp.budget import (
    MAX_FAILURES,
    BudgetExceeded,
    Kind,
    Ledger,
    TooManyFailures,
)
from astro_weights.mcp.edit_store import EditStore, export_edits
from astro_weights.mcp.handles import HandleStore, UnknownHandle
from astro_weights.mcp.jobs import Executor, JobRunner, RunConfig, load_run_config
from astro_weights.mcp.kernel import Kernel
from astro_weights.mcp.record import RunRecord, library_commit
from astro_weights.mcp.sandbox import SandboxUnavailable

__all__ = ["build_server", "main"]

_BUDGET_KIND: dict[str, Kind] = {"exec": "code_runs", "forward": "forward_passes"}
_SUMMARY_MAX = 2000
_MODEL_MAX = 200


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _money(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:  # an int too large for a float
        return False


def _model(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != "" and len(value) <= _MODEL_MAX


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
        path,
        seed=cfg.seed,
        config=config,
        editor=cfg.editor,
        model=cfg.model,
        library_commit=library_commit(),
    )


def _kernel_executor(
    run_dir: Path,
    cfg: RunConfig,
    edits: EditStore,
    jobs: HandleStore,
    record: RunRecord,
    allow_unsafe: bool,
) -> Executor | str:
    """``exec`` executor running cells in a new Kernel, or why there is none."""
    try:
        kernel = Kernel(
            run_dir,
            edits,
            allow_unsafe=allow_unsafe,
            allow_nested_userns=cfg.kernel_allow_nested_userns,
            timeout_s=cfg.kernel_timeout_s,
            mem_gib=cfg.kernel_mem_gib,
            on_event=lambda name, fields: record.event(name, **fields),
            jobs=jobs,
        )
    except SandboxUnavailable as exc:
        return str(exc)
    return lambda params: kernel.exec(params["code"])


Checker = Callable[[Any], dict[str, Any]]


def _forward(
    run_dir: Path, cfg: RunConfig, edits: EditStore
) -> tuple[Executor, Checker] | str:
    """``forward`` executor and its request check, or why there are none."""
    if not forward.hf_available():
        return f"forward_pass is not available yet: {forward.HF_HINT}"
    backend = forward.ForwardBackend(
        edits.model_dir, edits, device=cfg.forward_device, dtype=cfg.forward_dtype
    )
    cap = cfg.forward_max_new_tokens

    def check(request: Any) -> dict[str, Any]:
        return forward.check_request(request, backend, edits, max_new_tokens=cap)

    return forward.executor(backend, run_dir), check


def _base_model(model_dir: Path) -> str:
    """The base model's name from its config.json, else its directory."""
    try:
        name = json.loads((model_dir / "config.json").read_text("utf-8")).get(
            "_name_or_path"
        )
    except (OSError, ValueError, RecursionError, AttributeError):
        name = None
    return name if isinstance(name, str) and name.strip() else str(model_dir)


def _export(
    run_dir: Path, cfg: RunConfig, edits: EditStore | None, record: RunRecord
) -> dict[str, Any]:
    """Export the stored edits for ``finish``; the keys to add to its result."""
    if edits is None:
        return {"export": None, "note": "no model_dir: nothing to export"}
    try:
        if not edits.edited():
            record.event("export_skipped", reason="no edits")
            return {"export": None, "note": "no edits: nothing to export"}
        out = export_edits(
            edits,
            run_dir / "export",
            rank=cfg.export_rank,
            max_rel_residual=cfg.export_max_rel_residual,
            base_model=_base_model(edits.model_dir),
            base_gguf=cfg.export_base_gguf,
            gguf_out_type=cfg.export_gguf_out_type,
        )
    except Exception as exc:  # noqa: BLE001 - the run must still finish
        error = f"{type(exc).__name__}: {exc}"[:_SUMMARY_MAX]
        record.event("export_failed", error=error)
        return {"export": None, "export_error": error}
    summary = {
        "adapter": str(out["adapter"]),
        "gguf": None if out["gguf"] is None else str(out["gguf"]),
        "residuals": dict(out["residuals"]),
        "rank": out["rank"],
    }
    record.event("export", **summary)
    return {"export": summary}


def build_server(
    run_dir: Path,
    executors: Mapping[str, Executor] | None = None,
    *,
    allow_unsafe_kernel: bool = False,
) -> MCPServer[Any]:
    """Build the server for ``run_dir``. With no executors the work tools say so.

    With ``model_dir`` in run.json and no ``exec`` executor given, ``exec_python``
    runs in a sandboxed kernel. Without an OS sandbox it returns an error, unless
    ``allow_unsafe_kernel`` runs the kernel unsandboxed. With no ``forward``
    executor given, ``forward_pass`` runs the model when the ``hf`` extra is
    installed; a given ``forward`` executor gets the request unchecked.
    """
    run_dir = Path(run_dir)
    cfg = load_run_config(run_dir)
    store = HandleStore(run_dir)
    ledger = Ledger(run_dir / "ledger.jsonl", cfg.limits)
    record = _open_record(run_dir, cfg)
    for job in store.list():  # a restart killed their threads; they can never finish
        if job.state in ("queued", "running"):
            job = store.fail(job.handle, "server restarted")
        forward.discard_results(run_dir, job)
    available = dict(executors or {})
    unavailable: dict[str, tuple[str, str]] = {}  # kind -> (event, error)
    check: Checker | None = None
    if allow_unsafe_kernel:
        record.event("unsafe_kernel")
    edits: EditStore | None = None
    if cfg.model_dir is not None:
        # one store and its lock, shared by the kernel and the model
        edits = EditStore(run_dir, cfg.model_dir)
        if "exec" not in available:
            made = _kernel_executor(
                run_dir, cfg, edits, store, record, allow_unsafe_kernel
            )
            if isinstance(made, str):
                unavailable["exec"] = ("sandbox_unavailable", made)
            else:
                available["exec"] = made
        if "forward" not in available:
            fwd = _forward(run_dir, cfg, edits)
            if isinstance(fwd, str):
                unavailable["forward"] = ("tool_unavailable", fwd)
            else:
                available["forward"], check = fwd
    runner = JobRunner(store, ledger, available, _BUDGET_KIND)
    finish_lock = threading.Lock()
    finished = any(e["event"] == "finish" for e in record.read())
    server: MCPServer[Any] = MCPServer("astro-weights")

    def blocked(tool: str, kind: str) -> dict[str, Any] | None:
        """Why ``tool`` cannot run now, or None. Call with ``finish_lock`` held."""
        if finished:
            return {"error": "run already finished"}
        if kind in unavailable:
            event, error = unavailable[kind]
            record.event(event, tool=tool, error=error)
            return {"error": error}
        if kind not in available:
            record.event("tool_unavailable", tool=tool)
            return {"error": f"{tool} is not available yet"}
        return None

    def submit(tool: str, kind: str, params: dict[str, Any]) -> dict[str, Any]:
        with finish_lock:
            return blocked(tool, kind) or _submit_locked(kind, params)

    def refuse(category: str, error: str) -> dict[str, Any]:
        record.refusal(
            category=category, editor=cfg.editor, model=cfg.model, detail=error
        )
        return {"error": error, "budget": ledger.to_json()}

    def _submit_locked(kind: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            job = runner.submit(kind, params)
        except (BudgetExceeded, TooManyFailures) as exc:
            category = "budget" if isinstance(exc, BudgetExceeded) else "failure_streak"
            return refuse(category, str(exc))
        return {"handle": job.handle, "state": job.state}

    @server.tool()  # type: ignore[untyped-decorator, unused-ignore]
    def exec_python(code: str) -> dict[str, Any]:
        """Start a Python code run; returns a handle to poll with status.

        The kernel has numpy (``np``) and the standard library only; variables stay
        between runs and the last bare expression is printed. Helpers: ``NAMES`` and
        ``names()`` (editable weight names), ``load(name)`` (edited weight if saved,
        else base, float32), ``load_original(name)``, ``save_edit(name, w)`` (the full
        edited weight, same shape; refused when its relative Frobenius change from the
        base is above 0.5), ``drop_edit(name)``, ``edited()``, and
        ``forward_result(handle, key)`` (an array or list of texts from a finished
        forward_pass). Keep each change low rank: finish exports it at rank 8 unless
        run.json sets ``export.rank``, and fails the export when the change's relative
        residual at that rank is above 0.05 (``export.max_rel_residual``).
        """
        return submit("exec_python", "exec", {"code": code})

    @server.tool()  # type: ignore[untyped-decorator, unused-ignore]
    def forward_pass(request: dict[str, Any]) -> dict[str, Any]:
        """Start a model forward pass; returns a handle to poll with status.

        ``{"kind": "teacher_forced", "texts": [str], "return": ["hidden", "mlp",
        "attn", "logprobs"], "layers": [int], "last_tokens": int, "arm": "base" |
        "edited"}`` or ``{"kind": "generate", "prompts": [str], "max_new_tokens":
        int, "arm": "base" | "edited" | "both", "seed": int, "temperature": float}``.
        Read arrays and full texts in exec_python with ``forward_result(handle, key)``.
        """
        if check is None:
            return submit("forward_pass", "forward", {"request": request})
        with finish_lock:
            stop = blocked("forward_pass", "forward")
        if stop is not None:
            return stop
        try:
            clean = check(request)
        except forward.BadRequest as exc:
            return refuse("bad_request", f"bad request: {exc}"[:2000])
        except Exception as exc:  # noqa: BLE001 - a broken tokenizer must not crash the tool
            return refuse("forward_check", f"{type(exc).__name__}: {exc}"[:2000])
        params = {"request": clean, "out": secrets.token_hex(16)}
        return submit("forward_pass", "forward", params)

    @server.tool()  # type: ignore[untyped-decorator, unused-ignore]
    def status(handle: str) -> dict[str, Any]:
        """Return a job's state and result, plus the budget."""
        try:
            job = store.get(handle)
        except UnknownHandle:
            return {"error": "unknown handle"}
        if job.state == "failed":  # e.g. refused at charge time: results unreadable
            forward.discard_results(run_dir, job)
        return {**job.to_json(), "budget": ledger.to_json()}

    @server.tool()  # type: ignore[untyped-decorator, unused-ignore]
    def finish(summary: str) -> dict[str, Any]:
        """End the run, export the stored edits and write the run record. Works once.

        Each edit is cut to rank 8 (``export.rank`` in run.json) by truncated SVD. An
        edit whose relative residual at that rank is above 0.05
        (``export.max_rel_residual``) fails the export: the result then has
        ``export_error`` and no adapter.
        """
        nonlocal finished
        with finish_lock:
            if finished:
                return {"error": "run already finished"}
            record.event("summary", text=summary[:_SUMMARY_MAX])
            exported = _export(run_dir, cfg, edits, record)
            budget = ledger.to_json()
            record.finish(budget=budget, outcome="finished")
            finished = True
        return {"state": "finished", "budget": budget, **exported}

    @server.tool()  # type: ignore[untyped-decorator, unused-ignore]
    def report_usage(
        tokens: Any, dollars: Any, turn: Any, model: Any
    ) -> dict[str, Any]:
        """Record one model turn's usage; for editors, not meant for the model.

        ``tokens`` and ``turn`` are ints >= 0, ``dollars`` a finite number >= 0,
        ``model`` a non-empty string of at most 200 characters (checked here, not
        coerced). Always recorded, even past a limit and after finish; a usage
        recorded after finish is not in ``finish.budget``, so read spend from the
        ledger. ``over`` lists the kinds now past their limit.
        """
        if not (_count(tokens) and _money(dollars) and _count(turn) and _model(model)):
            return {"error": "bad usage report"}
        with finish_lock:
            if tokens > 0:
                ledger.charge("tokens", tokens, ref=f"turn {turn}")
            if dollars > 0:
                ledger.charge("dollars", dollars, ref=f"turn {turn}")
            record.event(
                "usage", turn=turn, model=model, tokens=tokens, dollars=dollars
            )
            budget = ledger.to_json()
        over = [k for k, used in budget["used"].items() if used > budget["limits"][k]]
        return {"budget": budget, "over": over}

    @server.tool()  # type: ignore[untyped-decorator, unused-ignore]
    def check_spend(estimate: Any) -> dict[str, Any]:
        """Check that an estimated dollar cost fits the budget; for editors, not meant for the model.

        Free: writes nothing. An estimate of 0 is ok even when dollars are already
        past the limit (a local fallback turn costs nothing), but not while the
        dollars failure streak blocks work.
        """
        if not _money(estimate):
            return {"error": "estimate must be a finite number >= 0"}
        with finish_lock:
            if finished:
                return {"error": "run already finished"}
            budget = ledger.to_json()
            try:
                if estimate > 0:  # check() takes amounts > 0
                    ledger.check("dollars", estimate)
                elif budget["streaks"]["dollars"] >= MAX_FAILURES:
                    raise TooManyFailures("dollars", budget["streaks"]["dollars"])
            except (BudgetExceeded, TooManyFailures) as exc:
                return {"ok": False, "error": str(exc), "budget": budget}
        return {"ok": True, "budget": budget}

    @server.tool()  # type: ignore[untyped-decorator, unused-ignore]
    def report_refusal(category: Any, model: Any, detail: Any = "") -> dict[str, Any]:
        """Record that a model refused; for editors, not meant for the model."""
        if not (
            isinstance(category, str) and _model(model) and isinstance(detail, str)
        ):
            return {"error": "bad refusal report"}
        with finish_lock:
            if finished:
                return {"error": "run already finished"}
            try:
                record.refusal(
                    category=category, editor=cfg.editor, model=model, detail=detail
                )
            except ValueError as exc:
                return {"error": str(exc)}
        return {"recorded": True}

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
