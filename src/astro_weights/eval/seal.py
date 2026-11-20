"""Sealed test split: each edit is scored once per test manifest.

The ledger is ``DIR/seal.jsonl``, append-only: one JSON line per scoring, written,
flushed and fsynced before ``record_scoring`` returns. Earlier lines are never
rewritten; each line holds ``prev``, the sha256 of the line before it, so an edited
or deleted earlier line is reported on the next read. A last line torn by a crash
(no newline, not JSON) is cut off: the scoring that wrote it never returned, so its
result was never shown. Deleting the whole ledger, or its last lines, is not
detected; the ledger guards against honest re-runs, not against its owner.

The key is ``seal_key(edit)``: a sha256 of the edit's content (factors and base),
never of a directory name, run id or ``meta``.

Concurrency: ``record_scoring`` reads, checks and appends under an exclusive lock on
``DIR/seal.lock`` (``fcntl.flock`` on POSIX, ``msvcrt.locking`` on Windows). Of any
number of processes scoring one key on one manifest at once, exactly one gets
through; the others raise ``SealedError``. The OS drops the lock when its holder
dies, so a crash never leaves the ledger locked. The lock needs a local file
system (``flock`` over some network file systems does not lock across hosts).
"""

import hashlib
import json
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

if sys.platform == "win32":  # pragma: no cover - CI's Windows job runs this
    import msvcrt
else:
    import fcntl

__all__ = [
    "LEDGER",
    "SealedError",
    "check_unsealed",
    "read_seal",
    "record_scoring",
    "seal_key",
]

LEDGER = "seal.jsonl"
LOCK = "seal.lock"
_TAG = b"astro-weights seal v1\0"
_KEYS = frozenset(
    {"key", "manifest_sha256", "scoring", "unseal", "note", "time", "prev"}
)


class SealedError(ValueError):
    """This edit was already scored on this test manifest."""


class _Delta(Protocol):
    @property
    def module(self) -> str: ...
    @property
    def B(self) -> np.ndarray: ...
    @property
    def A(self) -> np.ndarray: ...


class _Edit(Protocol):
    @property
    def deltas(self) -> Sequence[_Delta]: ...
    @property
    def base_model(self) -> str: ...
    @property
    def base_sha256(self) -> Mapping[str, str]: ...


def _f32(x: np.ndarray, where: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError(f"{where}: factor must be a finite 2-D array")
    # + 0.0 turns -0.0 into 0.0; little-endian, C order
    return np.ascontiguousarray(a + np.float32(0.0), dtype="<f4")


def _components(d: _Delta) -> list[bytes]:
    """The delta's rank-1 terms ``B[:, i] A[i, :]`` as bytes, zero terms dropped, sorted."""
    b, a = _f32(d.B, d.module), _f32(d.A, d.module)
    if b.shape[1] != a.shape[0]:
        raise ValueError(f"{d.module}: rank mismatch, B {b.shape} vs A {a.shape}")
    terms = [
        b[:, i].tobytes() + a[i, :].tobytes()
        for i in range(a.shape[0])
        if b[:, i].any() and a[i, :].any()
    ]
    return sorted(terms, key=lambda t: hashlib.sha256(t).digest())


def _chunk(h: "hashlib._Hash", data: bytes) -> None:
    h.update(len(data).to_bytes(8, "little"))
    h.update(data)


def seal_key(edit: _Edit) -> str:
    """sha256 hex of an edit's content: what it changes, not what it is called.

    Hashed: each module name with its nonzero rank-1 terms as float32 (modules sorted
    by name, terms sorted by their own hash, ``-0.0`` as ``0.0``), and the base:
    ``base_sha256`` when non-empty, else ``base_model``. Not hashed: ``meta``, file
    or directory names, run ids. So the key is the same after renaming, reordering
    modules or rank terms, or zero-padding the rank; any change of a factor value,
    including a rescale ``B*c, A/c`` that keeps ``B @ A``, gives a new key.
    """
    h = hashlib.sha256(_TAG)
    base: object = dict(edit.base_sha256) or edit.base_model
    _chunk(h, json.dumps(base, sort_keys=True).encode())
    modules: dict[str, list[bytes]] = {}
    for d in edit.deltas:
        if d.module in modules:
            raise ValueError(f"duplicate module {d.module!r}")
        terms = _components(d)
        shape = (int(np.shape(d.B)[0]), int(np.shape(d.A)[1]))
        modules[d.module] = [json.dumps(shape).encode(), *terms] if terms else []
    for name in sorted(modules):
        if not modules[name]:
            continue  # an all-zero delta changes nothing
        _chunk(h, name.encode())
        _chunk(h, len(modules[name]).to_bytes(8, "little"))
        for part in modules[name]:
            _chunk(h, part)
    return h.hexdigest()


@contextmanager
def _locked(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / LOCK).open("a+b") as f:
        if sys.platform == "win32":  # pragma: no cover - CI's Windows job runs this
            f.seek(0)
            while True:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:  # LK_LOCK gives up after ~10 s; keep waiting
                    continue
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _check_entry(obj: object, where: str) -> dict[str, Any]:
    if not isinstance(obj, dict) or set(obj) != _KEYS:
        raise ValueError(f"{where}: not a seal line")
    ok = (
        all(isinstance(obj[k], str) for k in ("key", "manifest_sha256", "note", "time"))
        and isinstance(obj["prev"], str)
        and isinstance(obj["unseal"], bool)
        and isinstance(obj["scoring"], int)
        and not isinstance(obj["scoring"], bool)
    )
    if not ok:
        raise ValueError(f"{where}: wrong value type in seal line")
    return obj


def _read(path: Path, repair: bool) -> tuple[list[dict[str, Any]], str]:
    """Entries and the sha256 of the last line; a torn tail is skipped (cut if repair)."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return [], ""
    cut = data.rfind(b"\n") + 1
    lines = data[:cut].split(b"\n")[:-1]
    tail = data[cut:]
    if tail:
        try:
            json.loads(tail)
            whole = True
        except ValueError:
            whole = False
        if whole:  # lost only its newline: a recorded line is never dropped
            lines.append(tail)
            if repair:
                _write(path, b"\n")
        elif repair:
            with path.open("r+b") as f:
                f.truncate(cut)
                f.flush()
                os.fsync(f.fileno())
    entries: list[dict[str, Any]] = []
    prev = ""
    for n, line in enumerate(lines, 1):
        where = f"{path}:{n}"
        try:
            obj = json.loads(line)
        except ValueError:
            raise ValueError(f"{where}: invalid JSON line") from None
        entry = _check_entry(obj, where)
        if entry["prev"] != prev:
            raise ValueError(f"{where}: ledger chain broken; an earlier line changed")
        entries.append(entry)
        prev = hashlib.sha256(line).hexdigest()
    return entries, prev


def _write(path: Path, data: bytes) -> None:
    with path.open("ab") as f:
        start = f.tell()
        try:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:  # pragma: no cover - the disk is failing
            f.truncate(start)
            raise


def read_seal(directory: str | Path) -> list[dict[str, Any]]:
    """Every scoring line of ``directory/seal.jsonl``; empty when there is none.

    ValueError when a complete line is malformed or the ``prev`` chain is broken.
    """
    return _read(Path(directory) / LEDGER, repair=False)[0]


def _count(entries: list[dict[str, Any]], key: str, manifest_sha256: str) -> int:
    return sum(
        1
        for e in entries
        if e["key"] == key and e["manifest_sha256"] == manifest_sha256
    )


def _refusal(n: int, key: str, directory: Path) -> SealedError:
    return SealedError(
        f"edit {key[:12]} was already scored {n} time(s) on this test split "
        f"(ledger {directory / LEDGER}); pass --unseal (unseal=True) to score it "
        "again, which is recorded"
    )


def check_unsealed(directory: str | Path, key: str, manifest_sha256: str) -> None:
    """SealedError if ``key`` was already scored on the manifest ``manifest_sha256``.

    A read-only early check; only ``record_scoring`` decides, under the lock.
    """
    directory = Path(directory)
    n = _count(read_seal(directory), key, manifest_sha256)
    if n:
        raise _refusal(n, key, directory)


def record_scoring(
    directory: str | Path,
    key: str,
    manifest_sha256: str,
    unseal: bool = False,
    note: str = "",
) -> dict[str, Any]:
    """Record one scoring of ``key`` on ``manifest_sha256`` and return its line.

    Under the lock: SealedError if the key was scored on that manifest before and
    ``unseal`` is False; else one line is appended and fsynced. With ``unseal`` the
    line says ``"unseal": true``. Show a test result only after this returns.
    """
    directory = Path(directory)
    for name, value in (("key", key), ("manifest_sha256", manifest_sha256)):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{name} must be a sha256 hex string")
    path = directory / LEDGER
    with _locked(directory):
        entries, prev = _read(path, repair=True)
        n = _count(entries, key, manifest_sha256)
        if n and not unseal:
            raise _refusal(n, key, directory)
        entry = {
            "key": key,
            "manifest_sha256": manifest_sha256,
            "scoring": n + 1,
            "unseal": bool(unseal),
            "note": str(note),
            "time": datetime.now(UTC).isoformat(),
            "prev": prev,
        }
        line = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
        _write(path, line + b"\n")
    return entry
