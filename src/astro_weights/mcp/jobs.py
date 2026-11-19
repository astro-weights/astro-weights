"""Run jobs in background threads against the handle store and the budget ledger.

``submit`` returns at once with a queued job. A worker thread runs the executor, then
either charges the budget and stores the result, or stores the failure. Refused
requests cost nothing. Standard library only.
"""

import contextlib
import json
import math
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
_DEVICES = ("auto", "cpu", "cuda", "mps")
_DTYPES = ("auto", "float32", "bfloat16", "float16")
_FORWARD_KEYS = ("device", "dtype", "max_new_tokens")
_EXPORT_KEYS = ("rank", "max_rel_residual", "base_gguf", "gguf_out_type")
_GGUF_OUT_TYPES = ("q8_0", "f16", "bf16")
MAX_NEW_TOKENS = 32768


@dataclass(frozen=True)
class RunConfig:
    """The contents of ``RUN_DIR/run.json``."""

    seed: int
    editor: str
    model: str | None
    limits: Limits
    model_dir: Path | None = None
    kernel_timeout_s: float = 600.0
    kernel_mem_gib: int = 16
    forward_device: str = "auto"
    forward_dtype: str = "auto"
    forward_max_new_tokens: int = 4096
    export_rank: int = 8
    export_max_rel_residual: float = 0.05
    export_base_gguf: Path | None = None
    export_gguf_out_type: str = "q8_0"
    kernel_allow_nested_userns: bool = False


def load_run_config(run_dir: Path) -> RunConfig:
    """Read and validate ``run_dir/run.json``; ValueError with a clear message if bad."""
    path = Path(run_dir) / "run.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{path} not found") from None
    except (OSError, ValueError, RecursionError) as exc:
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
    model_dir = raw.get("model_dir")
    if not (model_dir is None or (isinstance(model_dir, str) and model_dir)):
        raise ValueError(f"{path}: model_dir must be a non-empty string or absent")
    timeout_s, mem_gib, nested_userns = _kernel_config(path, raw.get("kernel", {}))
    device, dtype, max_new = _forward_config(path, raw.get("forward", {}))
    rank, residual, base_gguf, out_type = _export_config(path, raw.get("export", {}))
    return RunConfig(
        seed,
        editor,
        model,
        parsed,
        None if model_dir is None else Path(run_dir) / model_dir,
        timeout_s,
        mem_gib,
        device,
        dtype,
        max_new,
        rank,
        residual,
        None if base_gguf is None else Path(run_dir) / base_gguf,
        out_type,
        nested_userns,
    )


def _kernel_config(path: Path, kernel: Any) -> tuple[float, int, bool]:
    """(timeout_s, mem_gib, allow_nested_userns) from run.json's optional ``kernel``."""
    keys = ["timeout_s", "mem_gib", "allow_nested_userns"]
    if not isinstance(kernel, dict) or not set(kernel) <= set(keys):
        raise ValueError(f"{path}: kernel must be an object with keys from {keys}")
    timeout_s = kernel.get("timeout_s", 600.0)
    mem_gib = kernel.get("mem_gib", 16)
    number = isinstance(timeout_s, int | float) and not isinstance(timeout_s, bool)
    if not number or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"{path}: kernel.timeout_s must be a positive number")
    if not isinstance(mem_gib, int) or isinstance(mem_gib, bool) or mem_gib < 1:
        raise ValueError(f"{path}: kernel.mem_gib must be an int >= 1")
    nested = kernel.get("allow_nested_userns", False)
    if not isinstance(nested, bool):
        raise ValueError(f"{path}: kernel.allow_nested_userns must be true or false")  # noqa: TRY004
    return float(timeout_s), mem_gib, nested


def _forward_config(path: Path, forward: Any) -> tuple[str, str, int]:
    """(device, dtype, max_new_tokens) from the optional ``forward`` object of run.json."""
    if not isinstance(forward, dict) or not set(forward) <= set(_FORWARD_KEYS):
        raise ValueError(
            f"{path}: forward must be an object with keys from {list(_FORWARD_KEYS)}"
        )
    device = forward.get("device", "auto")
    dtype = forward.get("dtype", "auto")
    max_new = forward.get("max_new_tokens", 4096)
    if device not in _DEVICES:
        raise ValueError(f"{path}: forward.device must be one of {list(_DEVICES)}")
    if dtype not in _DTYPES:
        raise ValueError(f"{path}: forward.dtype must be one of {list(_DTYPES)}")
    whole = isinstance(max_new, int) and not isinstance(max_new, bool)
    if not whole or not 1 <= max_new <= MAX_NEW_TOKENS:
        raise ValueError(
            f"{path}: forward.max_new_tokens must be an int from 1 to {MAX_NEW_TOKENS}"
        )
    return str(device), str(dtype), max_new


def _export_config(path: Path, export: Any) -> tuple[int, float, str | None, str]:
    """(rank, max_rel_residual, base_gguf, gguf_out_type) from the optional ``export`` object."""
    if not isinstance(export, dict) or not set(export) <= set(_EXPORT_KEYS):
        raise ValueError(
            f"{path}: export must be an object with keys from {list(_EXPORT_KEYS)}"
        )
    rank = export.get("rank", 8)
    residual = export.get("max_rel_residual", 0.05)
    base_gguf = export.get("base_gguf")
    out_type = export.get("gguf_out_type", "q8_0")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError(f"{path}: export.rank must be an int >= 1")
    number = isinstance(residual, int | float) and not isinstance(residual, bool)
    try:
        finite = number and math.isfinite(residual)
    except OverflowError:  # an int too large for a float
        finite = False
    if not finite or residual < 0:
        raise ValueError(
            f"{path}: export.max_rel_residual must be a finite number >= 0"
        )
    if not (base_gguf is None or (isinstance(base_gguf, str) and base_gguf)):
        raise ValueError(
            f"{path}: export.base_gguf must be a non-empty string or absent"
        )
    if out_type not in _GGUF_OUT_TYPES:
        raise ValueError(
            f"{path}: export.gguf_out_type must be one of {list(_GGUF_OUT_TYPES)}"
        )
    return rank, float(residual), base_gguf, str(out_type)


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
            except BaseException as exc:  # noqa: BLE001 - exec'd code may sys.exit()
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
