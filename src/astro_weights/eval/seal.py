"""Sealed test split: each edit is scored once on a test set.

The ledger is ``DIR/seal.jsonl``, append-only: one JSON line per scoring, written,
flushed and fsynced before ``record_scoring`` returns. Earlier lines are never
rewritten; each line holds ``prev``, the sha256 of the line before it, so an edited
or deleted earlier line is reported on the next read. A last line torn by a crash
(no newline, not JSON) is cut off: the scoring that wrote it never returned, so its
result was never shown. Deleting the whole ledger, or its last lines, is not
detected; the ledger guards against honest re-runs, not against its owner.

A line describes one ``Scoring``: the edit (``seal_key``, an exact content hash, and
``fingerprint``, random projections of each ``B @ A`` seeded by the ledger's secret
salt ``DIR/seal.salt``), the test set
(``test_ids_sha256``, the sorted ``prompt_hashes`` scored, ``k`` seeds per item)
and the edit arm's records (``records_sha256``). ``record_scoring`` refuses, unless
``unseal``, when an earlier line

- has the same ``records_sha256`` (the same outputs, whatever adapter is named), or
- is the same edit (same ``seal_key``, or fingerprints within ``FINGERPRINT_TOL``
  relative distance) and has the same ``test_ids_sha256`` or shares any prompt hash.

The records of the other arms shown with the result (``shown``: base and noise-floor
arms) are refused, even with ``unseal``, when they equal any line's ``records_sha256``:
a scored edit's records cannot come back as a base arm.

The salt is 32 random bytes made on first use, mode 0600, under the lock. A ledger
with lines and no salt is an error: a new salt would make every earlier fingerprint
incomparable. Keep the salt with the ledger and away from whatever makes the edits.

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
import secrets
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
    "seal_salt",
]

LEDGER = "seal.jsonl"
LOCK = "seal.lock"
SALT = "seal.salt"
SALT_BYTES = 32
PROBES = 8
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


def _probes(salt: bytes, module: str, m: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    h = hashlib.sha256(_TAG + b"probes\0")
    for part in (salt, module.encode(), f"{m}x{n}".encode()):
        _chunk(h, part)
    rng = np.random.default_rng(int.from_bytes(h.digest(), "little"))
    return rng.standard_normal((m, PROBES)), rng.standard_normal((n, PROBES))


def _check_salt(salt: bytes) -> bytes:
    if not isinstance(salt, bytes) or len(salt) != SALT_BYTES:
        raise ValueError(f"seal salt must be {SALT_BYTES} bytes")
    return salt


def fingerprint(edit: _Edit, salt: bytes) -> dict[str, np.ndarray]:
    """Per module, ``U.T @ (B @ A) @ V`` as a PROBES x PROBES float64 array.

    ``U`` (out x PROBES) and ``V`` (in x PROBES) are Gaussian, seeded by ``salt``
    (the ledger's, from ``seal_salt``), the module name and its shape. The
    fingerprint depends on ``B @ A`` only: a rescale or a rotation of the factors
    moves it by float rounding (about 1e-6 relative). Without the salt the probes
    are unknown, so a change cannot be aimed at them. Costs
    O((out + in) * rank * PROBES); ``B @ A`` is never formed.
    """
    _check_salt(salt)
    out: dict[str, np.ndarray] = {}
    for name, (b, a) in _factors(edit).items():
        u, v = _probes(salt, name, b.shape[0], a.shape[1])
        out[name] = (u.T @ b.astype(np.float64)) @ (a.astype(np.float64) @ v)
    return out


def _rel(x: float, nx: float, ny: float) -> float:
    norm = max(nx, ny)
    return 0.0 if norm == 0.0 else x / norm


def _distance(f: Mapping[str, np.ndarray], g: Mapping[str, np.ndarray]) -> float:
    """The larger of the global and the worst per-module relative distance.

    Global: ``||f - g|| / max(||f||, ||g||)`` over the union of modules, a missing
    module counting as 0, so adding a tiny module stays the same edit. Per module:
    the same ratio for each module in both, so a wholly different small module is a
    new edit even when its share of the norm is below ``FINGERPRINT_TOL``.
    """
    zero = np.zeros((PROBES, PROBES))

    def sq(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    diff = sum(sq(f.get(m, zero) - g.get(m, zero)) for m in {*f, *g})
    norms = [math.sqrt(sum(sq(x) for x in h.values())) for h in (f, g)]
    out = _rel(math.sqrt(diff), *norms)
    for m in f.keys() & g.keys():
        d = _rel(math.sqrt(sq(f[m] - g[m])), math.sqrt(sq(f[m])), math.sqrt(sq(g[m])))
        out = max(out, d)
    return out


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
    salt_sha256: str
    """sha256 of the salt ``fingerprint`` was made with; must be the ledger's."""

    def __post_init__(self) -> None:
        names = ("key", "test_ids_sha256", "manifest_sha256", "records_sha256")
        for name in (*names, "salt_sha256"):
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
    edit: _Edit, manifest: SplitManifest, records: Sequence[Record], salt: bytes
) -> Scoring:
    """The ``Scoring`` of ``edit`` on ``manifest``'s test set, from the edit arm's records.

    ``salt`` is the ledger's, from ``seal_salt``.
    """
    return Scoring(
        seal_key(edit),
        fingerprint(edit, salt),
        manifest.test_ids_sha256,
        manifest.manifest_sha256,
        records_sha256(records),
        tuple(r.prompt_hash for r in records),
        len({r.seed for r in records}) or 1,
        hashlib.sha256(salt).hexdigest(),
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


def _salt(directory: Path, entries: list[dict[str, Any]], create: bool) -> bytes | None:
    """The ledger's salt; made when absent, the ledger is empty and ``create``.

    Call under the lock when ``create``. ValueError when the ledger has lines and
    no salt, or the salt file is the wrong size.
    """
    path = directory / SALT
    try:
        return _check_salt(path.read_bytes())
    except FileNotFoundError:
        pass
    except ValueError:
        raise ValueError(f"{path}: not a seal salt ({SALT_BYTES} bytes)") from None
    if entries:
        raise ValueError(
            f"{path} is missing but {directory / LEDGER} has lines; restore the "
            "salt file (a new one would make every earlier fingerprint incomparable)"
        )
    if not create:
        return None
    salt = secrets.token_bytes(SALT_BYTES)
    tmp = directory / (SALT + ".tmp")
    tmp.unlink(missing_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(tmp, flags, 0o600), "wb") as f:
        tmp.chmod(0o600)  # whatever the umask
        f.write(salt)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)  # atomic: the salt is whole or absent
    if sys.platform != "win32":  # pragma: no branch - CI's Windows job runs the else
        dfd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    return salt


def seal_salt(directory: str | Path) -> bytes:
    """The salt of the ledger in ``directory``, made on first use (32 bytes, 0600).

    Pass it to ``scoring_of``. ValueError when the ledger has lines but the salt
    file is missing or malformed.
    """
    directory = Path(directory)
    with _locked(directory):
        salt = _salt(directory, _read(directory / LEDGER, repair=False)[0], True)
    assert salt is not None
    return salt


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


def _shown_conflicts(entries: list[dict[str, Any]], shown: Sequence[str]) -> list[str]:
    """Lines whose edit-arm records are among the other arms' ``shown`` records."""
    return [
        f"line {n}: a base or floor arm holds the records of an edit scored there "
        "(--unseal does not apply)"
        for n, e in enumerate(entries, 1)
        if e["records_sha256"] in shown
    ]


def _check_scoring(
    directory: Path, entries: list[dict[str, Any]], s: Scoring, create: bool
) -> None:
    salt = _salt(directory, entries, create)
    if salt is not None and hashlib.sha256(salt).hexdigest() != s.salt_sha256:
        raise ValueError(
            f"the scoring's fingerprint was not made with {directory / SALT}; "
            "build it with scoring_of(..., seal_salt(directory))"
        )


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


def check_unsealed(
    directory: str | Path, scoring: Scoring, shown: Iterable[str] = ()
) -> None:
    """SealedError if an earlier line seals ``scoring`` or ``shown`` (module docstring).

    ``shown``: ``records_sha256`` of the other arms shown with the result. A
    read-only early check; only ``record_scoring`` decides, under the lock.
    """
    directory = Path(directory)
    entries = read_seal(directory)
    _check_scoring(directory, entries, scoring, create=False)
    if hidden := _shown_conflicts(entries, tuple(shown)):
        raise SealedError("; ".join(hidden[:3]))
    if reasons := _conflicts(entries, scoring):
        raise _refusal(reasons, directory)


def record_scoring(
    directory: str | Path,
    scoring: Scoring,
    unseal: bool = False,
    note: str = "",
    shown: Iterable[str] = (),
) -> dict[str, Any]:
    """Record one scoring and return its line (fingerprint base64-encoded).

    Under the lock: SealedError if an earlier line seals ``scoring`` and ``unseal``
    is False, or, whatever ``unseal``, if a line's ``records_sha256`` is in
    ``shown`` (the other arms shown with the result); else one line is appended
    and fsynced. ValueError if ``scoring`` was not made with the ledger's salt.
    ``scoring`` in the line counts this one plus the earlier lines that seal it;
    with ``unseal`` the line says ``"unseal": true``. Show a test result only after
    this returns.
    """
    directory = Path(directory)
    if not isinstance(scoring, Scoring):
        raise TypeError("scoring must be a Scoring")
    path = directory / LEDGER
    others = tuple(shown)
    with _locked(directory):
        entries, prev = _read(path, repair=True)
        _check_scoring(directory, entries, scoring, create=True)
        if hidden := _shown_conflicts(entries, others):
            raise SealedError("; ".join(hidden[:3]))
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
