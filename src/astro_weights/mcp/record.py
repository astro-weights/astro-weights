"""Run record: an append-only JSONL file describing one agent run.

The first line is a ``start`` event (seed, config, library version and commit, editor,
model, Python, platform). Later lines are events, refusals and the ``finish`` line with
the budget used. Every line is written, flushed and fsynced before the call returns.

One writer per run directory: there is no cross-process lock.
"""

import json
import math
import os
import platform
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import Any, Self

__all__ = ["RunRecord"]

CATEGORY_MAX = 200
DETAIL_MAX = 2000
_RESERVED_EVENTS = frozenset({"start", "refusal", "finish"})
_RESERVED_FIELDS = frozenset({"event", "time"})


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name} is not JSON")


def _check_json(value: Any, where: str) -> None:
    """Raise ValueError unless ``value`` is plain JSON (str keys, finite numbers)."""
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where}: non-finite float {value!r}")
        return
    if isinstance(value, list | tuple):
        for i, item in enumerate(value):
            _check_json(item, f"{where}[{i}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{where}: key {key!r} is not a string")  # noqa: TRY004
            _check_json(item, f"{where}.{key}")
        return
    raise ValueError(f"{where}: {type(value).__name__} is not JSON")


def _dumps(obj: Mapping[str, Any]) -> bytes:
    _check_json(obj, "record")
    return (json.dumps(obj, allow_nan=False, separators=(",", ":")) + "\n").encode()


def _append(path: Path, obj: Mapping[str, Any]) -> None:
    """Append one JSON line; write, flush, fsync. A failed write is cut back off the file."""
    line = _dumps(obj)
    with path.open("ab") as f:
        start = f.tell()
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            try:
                f.truncate(start)
            except OSError:  # pragma: no cover - the disk is already failing
                pass
            raise


def _parse(line: bytes, where: str) -> dict[str, Any]:
    try:
        obj = json.loads(line, parse_constant=_reject_constant)
    except ValueError as e:
        raise ValueError(f"{where}: invalid JSON line: {e}") from None
    if not isinstance(obj, dict):
        raise ValueError(f"{where}: line is not a JSON object")  # noqa: TRY004
    return obj


def _split(path: Path) -> tuple[list[dict[str, Any]], bytes, int]:
    """Complete lines, the final bytes after the last newline, and their offset.

    A complete line that is not a JSON object raises ValueError.
    """
    data = path.read_bytes()
    cut = data.rfind(b"\n") + 1
    lines = data[:cut].split(b"\n")[:-1]
    entries = [_parse(line, f"{path}:{i + 1}") for i, line in enumerate(lines)]
    return entries, data[cut:], cut


def _load(path: Path, check: Callable[[dict[str, Any]], None]) -> list[dict[str, Any]]:
    """Read a JSONL file for appending, repairing a final line torn by a crash.

    A torn final line that still parses and passes ``check`` lost only its newline:
    it is kept and the newline added, so a recorded line is never dropped. Any other
    torn final line is cut off. Every complete line must pass ``check``.
    """
    entries, tail, cut = _split(path)
    for i, entry in enumerate(entries):
        try:
            check(entry)
        except ValueError as e:
            raise ValueError(f"{path}:{i + 1}: {e}") from None
    if not tail:
        return entries
    try:
        last = _parse(tail, f"{path}:{len(entries) + 1}")
        check(last)
    except ValueError:
        with path.open("r+b") as f:
            f.truncate(cut)
            f.flush()
            os.fsync(f.fileno())
        return entries
    with path.open("ab") as f:
        f.write(b"\n")
        f.flush()
        os.fsync(f.fileno())
    return [*entries, last]


def _library_version() -> str | None:
    try:
        return _dist_version("astro-weights")
    except (
        PackageNotFoundError
    ):  # pragma: no cover - only in an uninstalled source tree
        return None


def _text(value: Any, name: str, cap: int | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value if cap is None else value[:cap]


def _optional_str(value: Any, name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or None")


def _check_start(entry: dict[str, Any]) -> None:
    if entry.get("event") != "start":
        raise ValueError("first line is not a start event")


def _check_any(entry: dict[str, Any]) -> None:
    if not isinstance(entry.get("event"), str):
        raise ValueError('line has no "event" string')  # noqa: TRY004


class RunRecord:
    """Append-only run record at ``run_dir/"record.jsonl"``. One writer per file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @classmethod
    def start(
        cls,
        path: Path,
        *,
        seed: int,
        config: Mapping[str, Any],
        editor: str,
        model: str | None,
        library_commit: str | None = None,
    ) -> Self:
        """Create the record with its start line; FileExistsError if it already has content."""
        path = Path(path)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an int")  # noqa: TRY004
        if not isinstance(config, Mapping):
            raise ValueError("config must be a mapping")  # noqa: TRY004
        _text(editor, "editor")
        _optional_str(model, "model")
        _optional_str(library_commit, "library_commit")
        if path.exists() and path.stat().st_size > 0:
            raise FileExistsError(
                f"{path} already has a run record; use RunRecord.open to resume"
            )
        _append(
            path,
            {
                "event": "start",
                "seed": seed,
                "config": dict(config),
                "editor": editor,
                "model": model,
                "library_version": _library_version(),
                "library_commit": library_commit,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "time": _now(),
            },
        )
        return cls(path)

    @classmethod
    def open(cls, path: Path) -> Self:
        """Resume an existing record; ValueError unless its first line is a start event."""
        path = Path(path)
        entries = _load(path, _check_any)
        if not entries:
            raise ValueError(f"{path} has no start line")
        _check_start(entries[0])
        return cls(path)

    def event(self, name: str, **fields: Any) -> None:
        """Append one event. ``start``, ``refusal`` and ``finish`` have their own methods."""
        _text(name, "event name")
        if name in _RESERVED_EVENTS:
            raise ValueError(f"event name {name!r} is reserved")
        clash = _RESERVED_FIELDS & fields.keys()
        if clash:
            raise ValueError(f"field names {sorted(clash)} are reserved")
        _append(self.path, {"event": name, **fields, "time": _now()})

    def refusal(
        self, *, category: str, editor: str, model: str | None, detail: str = ""
    ) -> None:
        """Log a refused request. Category is free text, capped at 200 chars; detail at 2000."""
        category = _text(category, "category", CATEGORY_MAX)
        _text(editor, "editor")
        _optional_str(model, "model")
        if not isinstance(detail, str):
            raise ValueError("detail must be a string")  # noqa: TRY004
        _append(
            self.path,
            {
                "event": "refusal",
                "category": category,
                "editor": editor,
                "model": model,
                "detail": detail[:DETAIL_MAX],
                "time": _now(),
            },
        )

    def finish(self, *, budget: Mapping[str, Any], outcome: str) -> None:
        """Append the finish line with the budget used (``Ledger.to_json()``) and the outcome."""
        if not isinstance(budget, Mapping):
            raise ValueError("budget must be a mapping")  # noqa: TRY004
        _text(outcome, "outcome")
        _append(
            self.path,
            {
                "event": "finish",
                "budget": dict(budget),
                "outcome": outcome,
                "time": _now(),
            },
        )

    def read(self) -> list[dict[str, Any]]:
        """All complete lines, in order."""
        return _split(self.path)[0]
