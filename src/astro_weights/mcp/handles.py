"""Disk-backed job store: submit returns a handle, a worker moves it along, the agent polls.

All state lives in one JSON file per job, so a restarted server sees every old job.
Standard library only.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast, get_args

JobState = Literal["queued", "running", "done", "failed"]
HANDLE_RE = re.compile(r"[a-z_]{1,32}-[0-9a-f]{16}")
_KIND_RE = re.compile(r"[a-z_]{1,32}")
_ERROR_CAP = 2000
_STATES = frozenset(get_args(JobState))


class UnknownHandle(KeyError):
    """The handle is malformed or names no job."""


class BadTransition(ValueError):
    """The job is not in a state that allows this move."""


@dataclass(frozen=True)
class Job:
    """One job as stored on disk."""

    handle: str
    kind: str
    state: JobState
    params: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    created: float
    updated: float

    def to_json(self) -> dict[str, Any]:
        """Return the dict the status tool sends to the agent."""
        return {
            "handle": self.handle,
            "kind": self.kind,
            "state": self.state,
            "params": self.params,
            "result": self.result,
            "error": self.error,
            "created": self.created,
            "updated": self.updated,
        }


_Jobs = list[Job]


def _json_copy(value: Mapping[str, Any], what: str) -> dict[str, Any]:
    """Round-trip through JSON; raise ValueError if the value is not JSON-serialisable."""
    try:
        copy = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} is not JSON-serialisable: {exc}") from exc
    return cast("dict[str, Any]", copy)


def _check_handle(handle: str) -> None:
    if not isinstance(handle, str) or HANDLE_RE.fullmatch(handle) is None:
        raise UnknownHandle(handle)


class HandleStore:
    """Job files under ``root/jobs``; safe to share between threads of one process."""

    def __init__(self, root: Path) -> None:
        self._dir = Path(root) / "jobs"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def submit(self, kind: str, params: Mapping[str, Any]) -> Job:
        """Create a queued job and return it."""
        if not isinstance(kind, str) or _KIND_RE.fullmatch(kind) is None:
            raise ValueError(f"kind must match [a-z_]{{1,32}}, got {kind!r}")
        now = time.time()
        job = Job(
            handle=f"{kind}-{secrets.token_hex(8)}",
            kind=kind,
            state="queued",
            params=_json_copy(params, "params"),
            result=None,
            error=None,
            created=now,
            updated=now,
        )
        with self._lock:
            self._write(job)
        return job

    def get(self, handle: str) -> Job:
        """Return the job; UnknownHandle if malformed or absent, ValueError if the file is corrupt."""
        _check_handle(handle)
        with self._lock:
            return self._read(handle)

    def start(self, handle: str) -> Job:
        """Move queued -> running."""
        return self._move(handle, {"queued"}, "running")

    def finish(self, handle: str, result: Mapping[str, Any]) -> Job:
        """Move running -> done and store the result."""
        return self._move(
            handle, {"running"}, "done", result=_json_copy(result, "result")
        )

    def fail(self, handle: str, error: str) -> Job:
        """Move queued|running -> failed and store the (capped) error text."""
        return self._move(
            handle, {"queued", "running"}, "failed", error=error[:_ERROR_CAP]
        )

    def list(self, kind: str | None = None) -> _Jobs:
        """Return all jobs (optionally one kind), oldest first."""
        with self._lock:
            jobs = [
                self._read(p.stem)
                for p in self._dir.glob("*.json")
                if HANDLE_RE.fullmatch(p.stem)
            ]
        if kind is not None:
            jobs = [j for j in jobs if j.kind == kind]
        return sorted(jobs, key=lambda j: (j.created, j.handle))

    def _move(
        self,
        handle: str,
        allowed: set[str],
        state: JobState,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> Job:
        _check_handle(handle)
        with self._lock:
            old = self._read(handle)
            if old.state not in allowed:
                raise BadTransition(f"{handle}: cannot go from {old.state} to {state}")
            new = Job(
                handle=old.handle,
                kind=old.kind,
                state=state,
                params=old.params,
                result=result if result is not None else old.result,
                error=error if error is not None else old.error,
                created=old.created,
                updated=time.time(),
            )
            self._write(new)
        return new

    def _path(self, handle: str) -> Path:
        return self._dir / f"{handle}.json"

    def _read(self, handle: str) -> Job:
        try:
            text = self._path(handle).read_text(encoding="utf-8")
        except FileNotFoundError:
            raise UnknownHandle(handle) from None
        try:
            raw = json.loads(text)
            if raw["handle"] != handle or raw["state"] not in _STATES:
                raise ValueError("inconsistent fields")
            return Job(
                handle=raw["handle"],
                kind=str(raw["kind"]),
                state=raw["state"],
                params=dict(raw["params"]),
                result=None if raw["result"] is None else dict(raw["result"]),
                error=None if raw["error"] is None else str(raw["error"]),
                created=float(raw["created"]),
                updated=float(raw["updated"]),
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"corrupt job file for {handle}: {exc}") from exc

    def _write(self, job: Job) -> None:
        tmp = self._dir / f".{secrets.token_hex(8)}.tmp"
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(job.to_json(), fh, allow_nan=False)
                fh.flush()
                os.fsync(fh.fileno())
            tmp.replace(self._path(job.handle))
        finally:
            tmp.unlink(missing_ok=True)
