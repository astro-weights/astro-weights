"""Sealed test split: each edit is scored once on a test set.

The ledger is ``DIR/seal.jsonl``, append-only: one JSON line per scoring, written,
flushed and fsynced before ``record_scoring`` returns. Earlier lines are never
rewritten; each line holds ``prev``, the sha256 of the line before it, so an edited
or deleted earlier line is reported on the next read. A last line torn by a crash
(no newline, not JSON) is cut off: the scoring that wrote it never returned, so its
result was never shown. Deleting the whole ledger, or its last lines, is not
detected; the ledger guards against honest re-runs, not against its owner.

A line describes one ``Scoring``: the edit (``seal_key``, an exact content hash, and
``fingerprint``, a few random projections of each ``B @ A``), the test set
(``test_ids_sha256``, the sorted ``prompt_hashes`` scored, ``k`` seeds per item)
and the edit arm's records (``records_sha256``). ``record_scoring`` refuses, unless
``unseal``, when an earlier line

- has the same ``records_sha256`` (the same outputs, whatever adapter is named), or
- is the same edit (same ``seal_key``, or fingerprints within ``FINGERPRINT_TOL``
  relative distance) and has the same ``test_ids_sha256`` or shares any prompt hash.

Concurrency: ``record_scoring`` reads, checks and appends under an exclusive lock on
``DIR/seal.lock`` (``fcntl.flock`` on POSIX, ``msvcrt.locking`` on Windows). Of any
number of processes scoring one edit at once, exactly one gets through; the others
raise ``SealedError``. The OS drops the lock when its holder dies, so a crash never
leaves the ledger locked. The lock needs a local file system (``flock`` over some
network file systems does not lock across hosts).
"""

import base64
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from astro_weights.eval.records import Record
from astro_weights.eval.splits import SplitManifest

if sys.platform == "win32":  # pragma: no cover - CI's Windows job runs this
    import msvcrt
else:
    import fcntl

__all__ = [
    "FINGERPRINT_TOL",
    "LEDGER",
    "PROBES",
    "Scoring",
    "SealedError",
    "check_unsealed",
    "fingerprint",
    "read_seal",
    "record_scoring",
    "records_sha256",
    "scoring_of",
    "seal_key",
]

LEDGER = "seal.jsonl"
LOCK = "seal.lock"
PROBES = 4
FINGERPRINT_TOL = 1e-3
_TAG = b"astro-weights seal v2\0"
_SHA = re.compile(r"[0-9a-f]{64}")
_KEYS = frozenset(
    {
        "key",
        "fingerprint",
        "test_ids_sha256",
        "manifest_sha256",
        "records_sha256",
        "prompt_hashes",
        "k",
        "scoring",
        "unseal",
        "note",
        "time",
        "prev",
    }
)


class SealedError(ValueError):
    """This edit, or these records, were already scored on this test set."""


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


def _f32(x: np.ndarray, where: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError(f"{where}: factor must be a finite 2-D array")
    # + 0.0 turns -0.0 into 0.0; little-endian, C order
    return np.ascontiguousarray(a + np.float32(0.0), dtype="<f4")


def _factors(edit: _Edit) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each module's (B, A) as float32 with zero rank terms dropped; empty ones left out."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    seen: set[str] = set()
    for d in edit.deltas:
        if d.module in seen:
            raise ValueError(f"duplicate module {d.module!r}")
        seen.add(d.module)
        b, a = _f32(d.B, d.module), _f32(d.A, d.module)
        if b.shape[1] != a.shape[0]:
            raise ValueError(f"{d.module}: rank mismatch, B {b.shape} vs A {a.shape}")
        keep = [i for i in range(a.shape[0]) if b[:, i].any() and a[i, :].any()]
        if keep:  # an all-zero delta changes nothing
            out[d.module] = (b[:, keep], a[keep, :])
    return out


def _chunk(h: Any, data: bytes) -> None:
    h.update(len(data).to_bytes(8, "little"))
    h.update(data)


def seal_key(edit: _Edit) -> str:
    """sha256 hex of an edit's factors: an exact id for what it changes.

    Hashed: each module name, its shape and its nonzero rank-1 terms
    ``B[:, i] A[i, :]`` as float32 (modules sorted by name, terms sorted by their own
    hash, ``-0.0`` as ``0.0``). Not hashed: the base model, ``meta``, file or
    directory names, run ids. Renaming, reordering modules or rank terms and
    zero-padding the rank keep the key. A rescale ``B*c, A/c`` or a rotation gives a
    new key for the same ``B @ A``; ``fingerprint`` catches those.
    """
    h = hashlib.sha256(_TAG)
    for name, (b, a) in sorted(_factors(edit).items()):
        terms = sorted(
            (b[:, i].tobytes() + a[i, :].tobytes() for i in range(a.shape[0])),
            key=lambda t: hashlib.sha256(t).digest(),
        )
        _chunk(h, name.encode())
        _chunk(h, json.dumps([b.shape[0], a.shape[1]]).encode())
        _chunk(h, len(terms).to_bytes(8, "little"))
        for t in terms:
            _chunk(h, t)
    return h.hexdigest()


def _probes(module: str, m: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    digest = hashlib.sha256(f"{module}:{m}x{n}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return rng.standard_normal((m, PROBES)), rng.standard_normal((n, PROBES))


def fingerprint(edit: _Edit) -> dict[str, np.ndarray]:
    """Per module, ``U.T @ (B @ A) @ V`` as a PROBES x PROBES float64 array.

    ``U`` (out x PROBES) and ``V`` (in x PROBES) are Gaussian, seeded by the module
    name and shape, so the fingerprint depends on ``B @ A`` only: a rescale or a
    rotation of the factors moves it by float rounding (about 1e-6 relative).
    Costs O((out + in) * rank * PROBES); ``B @ A`` is never formed.
    """
    out: dict[str, np.ndarray] = {}
    for name, (b, a) in _factors(edit).items():
        u, v = _probes(name, b.shape[0], a.shape[1])
        out[name] = (u.T @ b.astype(np.float64)) @ (a.astype(np.float64) @ v)
    return out


def _distance(f: Mapping[str, np.ndarray], g: Mapping[str, np.ndarray]) -> float:
    """||f - g|| / max(||f||, ||g||) over all modules; a missing module counts as 0."""
    zero = np.zeros((PROBES, PROBES))
    diff = sum(float(np.sum((f.get(m, zero) - g.get(m, zero)) ** 2)) for m in {*f, *g})
    norm = max(math.sqrt(sum(float(np.sum(x**2)) for x in h.values())) for h in (f, g))
    return 0.0 if norm == 0.0 else math.sqrt(diff) / norm


def records_sha256(records: Iterable[Record]) -> str:
    """sha256 of the sorted (item_id, seed, prompt_hash, output) of ``records``."""
    rows = sorted((r.item_id, r.seed, r.prompt_hash, r.output) for r in records)
    return hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class Scoring:
    """What one test scoring covers: the edit, the test set and the records."""

    key: str
    fingerprint: Mapping[str, np.ndarray] = field(repr=False)
    test_ids_sha256: str
    manifest_sha256: str
    records_sha256: str
    prompt_hashes: tuple[str, ...] = field(repr=False)
    k: int

    def __post_init__(self) -> None:
        for name in ("key", "test_ids_sha256", "manifest_sha256", "records_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA.fullmatch(value):
                raise ValueError(f"{name} must be a sha256 hex string")
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise ValueError(f"k must be an int >= 1, got {self.k!r}")
        fp = {}
        for m, f in self.fingerprint.items():
            arr = np.asarray(f, dtype=np.float64)
            if arr.shape != (PROBES, PROBES) or not np.isfinite(arr).all():
                raise ValueError(
                    f"fingerprint of {m!r} must be finite {PROBES}x{PROBES}"
                )
            fp[m] = arr
        object.__setattr__(self, "fingerprint", fp)
        object.__setattr__(
            self, "prompt_hashes", tuple(sorted(set(self.prompt_hashes)))
        )


def scoring_of(
    edit: _Edit, manifest: SplitManifest, records: Sequence[Record]
) -> Scoring:
    """The ``Scoring`` of ``edit`` on ``manifest``'s test set, from the edit arm's records."""
    return Scoring(
        seal_key(edit),
        fingerprint(edit),
        manifest.test_ids_sha256,
        manifest.manifest_sha256,
        records_sha256(records),
        tuple(r.prompt_hash for r in records),
        len({r.seed for r in records}) or 1,
    )


def _encode(fp: Mapping[str, np.ndarray]) -> dict[str, str]:
    return {
        m: base64.b64encode(np.ascontiguousarray(f, dtype="<f8").tobytes()).decode()
        for m, f in sorted(fp.items())
    }


def _decode(fp: Mapping[str, str], where: str) -> dict[str, np.ndarray]:
    out = {}
    for m, text in fp.items():
        try:
            raw = base64.b64decode(text, validate=True)
        except ValueError:
            raise ValueError(f"{where}: bad fingerprint for {m!r}") from None
        if len(raw) != PROBES * PROBES * 8:
            raise ValueError(f"{where}: bad fingerprint for {m!r}")
        out[m] = np.frombuffer(raw, dtype="<f8").reshape(PROBES, PROBES)
    return out


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


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _check_entry(obj: object, where: str) -> dict[str, Any]:
    if not isinstance(obj, dict) or set(obj) != _KEYS:
        raise ValueError(f"{where}: not a seal line")
    text = ("key", "test_ids_sha256", "manifest_sha256", "records_sha256")
    ok = (
        all(isinstance(obj[k], str) for k in (*text, "note", "time", "prev"))
        and isinstance(obj["unseal"], bool)
        and _is_int(obj["scoring"])
        and _is_int(obj["k"])
        and isinstance(obj["prompt_hashes"], list)
        and all(isinstance(h, str) for h in obj["prompt_hashes"])
        and isinstance(obj["fingerprint"], dict)
        and all(isinstance(v, str) for v in obj["fingerprint"].values())
    )
    if not ok:
        raise ValueError(f"{where}: wrong value type in seal line")
    obj["fingerprint"] = _decode(obj["fingerprint"], where)
    return obj


def read_seal(directory: str | Path) -> list[dict[str, Any]]:
    """Every scoring line of ``directory/seal.jsonl``; empty when there is none.

    ``fingerprint`` comes back as ``{module: array}``. ValueError when a complete
    line is malformed or the ``prev`` chain is broken.
    """
    return _read(Path(directory) / LEDGER, repair=False)[0]


def _conflicts(entries: list[dict[str, Any]], s: Scoring) -> list[str]:
    """Why each earlier line seals ``s``; empty when none does."""
    prompts = set(s.prompt_hashes)
    out = []
    for n, e in enumerate(entries, 1):
        if e["records_sha256"] == s.records_sha256:
            out.append(f"line {n}: the same edit-arm records were scored")
            continue
        if e["key"] == s.key:
            same = "the same edit"
        elif _distance(e["fingerprint"], s.fingerprint) <= FINGERPRINT_TOL:
            same = f"an edit within {FINGERPRINT_TOL:g} relative of this one"
        else:
            continue
        if e["test_ids_sha256"] == s.test_ids_sha256:
            out.append(f"line {n}: {same} was scored on this test set")
        elif shared := len(prompts.intersection(e["prompt_hashes"])):
            out.append(f"line {n}: {same} was scored on {shared} of these prompts")
    return out


def _refusal(reasons: list[str], directory: Path) -> SealedError:
    shown = "; ".join(reasons[:3]) + ("; ..." if len(reasons) > 3 else "")
    return SealedError(
        f"already scored (ledger {directory / LEDGER}): {shown}. Pass --unseal "
        "(unseal=True) to score it again, which is recorded"
    )


def check_unsealed(directory: str | Path, scoring: Scoring) -> None:
    """SealedError if an earlier line seals ``scoring`` (see the module docstring).

    A read-only early check; only ``record_scoring`` decides, under the lock.
    """
    directory = Path(directory)
    if reasons := _conflicts(read_seal(directory), scoring):
        raise _refusal(reasons, directory)


def record_scoring(
    directory: str | Path,
    scoring: Scoring,
    unseal: bool = False,
    note: str = "",
) -> dict[str, Any]:
    """Record one scoring and return its line (fingerprint base64-encoded).

    Under the lock: SealedError if an earlier line seals ``scoring`` and ``unseal``
    is False; else one line is appended and fsynced. ``scoring`` in the line counts
    this one plus the earlier lines that seal it; with ``unseal`` the line says
    ``"unseal": true``. Show a test result only after this returns.
    """
    directory = Path(directory)
    if not isinstance(scoring, Scoring):
        raise TypeError("scoring must be a Scoring")
    path = directory / LEDGER
    with _locked(directory):
        entries, prev = _read(path, repair=True)
        reasons = _conflicts(entries, scoring)
        if reasons and not unseal:
            raise _refusal(reasons, directory)
        entry = {
            "key": scoring.key,
            "fingerprint": _encode(scoring.fingerprint),
            "test_ids_sha256": scoring.test_ids_sha256,
            "manifest_sha256": scoring.manifest_sha256,
            "records_sha256": scoring.records_sha256,
            "prompt_hashes": list(scoring.prompt_hashes),
            "k": scoring.k,
            "scoring": len(reasons) + 1,
            "unseal": bool(unseal),
            "note": str(note),
            "time": datetime.now(UTC).isoformat(),
            "prev": prev,
        }
        line = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
        _write(path, line + b"\n")
    return entry
