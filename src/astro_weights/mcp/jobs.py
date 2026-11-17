"""Run jobs in background threads against the handle store and the budget ledger.

``submit`` returns at once with a queued job. A worker thread runs the executor, then
either charges the budget and stores the result, or stores the failure. Refused
requests cost nothing. Standard library only.
"""

import contextlib
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astro_weights.mcp.budget import (
    BudgetExceeded,
    Kind,
    Ledger,
    Limits,
    TooManyFailures,
)
from astro_weights.mcp.handles import HandleStore, Job, _json_copy

__all__ = ["JobRunner", "RunConfig", "load_run_config"]

Executor = Callable[[dict[str, Any]], dict[str, Any]]
_LIMIT_KEYS = ("forward_passes", "code_runs", "tokens", "dollars")


@dataclass(frozen=True)
class RunConfig:
    """The contents of ``RUN_DIR/run.json``."""

    seed: int
    editor: str
    model: str | None
    limits: Limits


def load_run_config(run_dir: Path) -> RunConfig:
    """Read and validate ``run_dir/run.json``; ValueError with a clear message if bad."""
    path = Path(run_dir) / "run.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{path} not found") from None
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path} is not readable JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must hold a JSON object")  # noqa: TRY004
    seed, editor, model, limits = (
        raw.get(k) for k in ("seed", "editor", "model", "limits")
    )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"{path}: seed must be an int")  # noqa: TRY004
    if not isinstance(editor, str) or not editor.strip():
        raise ValueError(f"{path}: editor must be a non-empty string")
    if not (model is None or isinstance(model, str)):
        raise ValueError(f"{path}: model must be a string or null")
    if not isinstance(limits, dict) or set(limits) != set(_LIMIT_KEYS):
        raise ValueError(
            f"{path}: limits must be an object with keys {list(_LIMIT_KEYS)}"
        )
    try:
        parsed = Limits(**limits)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return RunConfig(seed, editor, model, parsed)


class JobRunner:
    """Submit jobs by kind; one thread per job. Share one runner per run directory."""

    def __init__(
        self,
        store: HandleStore,
        ledger: Ledger,
        executors: Mapping[str, Executor],
        budget_kind: Mapping[str, Kind],
    ) -> None:
        self._store = store
        self._ledger = ledger
        self._executors = dict(executors)
        self._budget_kind = dict(budget_kind)

    def submit(self, kind: str, params: Mapping[str, Any]) -> Job:
        """Queue a job and return it at once.

        BudgetExceeded or TooManyFailures is logged as a free refusal and re-raised.
        ValueError for an unknown kind or a missing executor, before anything is written.
        """
        if kind not in self._budget_kind or kind not in self._executors:
            raise ValueError(f"no executor for job kind {kind!r}")
        budget = self._budget_kind[kind]
        try:
            self._ledger.check(budget)
        except (BudgetExceeded, TooManyFailures) as exc:
            self._ledger.refuse(budget, str(exc), ref=kind)
            raise
        job = self._store.submit(kind, params)
        thread = threading.Thread(
            target=self._run, args=(job, self._executors[kind], budget), daemon=True
        )
        thread.start()
        return job

    def _run(self, job: Job, fn: Executor, budget: Kind) -> None:
        handle = job.handle
        try:
            self._store.start(handle)
            try:
                result = _json_copy(fn(dict(job.params)), "result")
            except Exception as exc:  # noqa: BLE001
                self._store.fail(handle, f"{type(exc).__name__}: {exc}")
                self._ledger.fail(budget, str(exc), ref=handle)
                return
            try:
                self._ledger.charge(budget, ref=handle)
            except BudgetExceeded:
                self._store.fail(handle, "budget exhausted")
                return
            self._store.finish(handle, result)
        except Exception as exc:  # noqa: BLE001 - disk trouble must not strand the job
            with contextlib.suppress(Exception):
                self._store.fail(handle, f"{type(exc).__name__}: {exc}")
