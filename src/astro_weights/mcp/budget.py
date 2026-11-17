"""Budget ledger: an append-only JSONL file that cannot be overspent or refunded.

The file (``run_dir/"ledger.jsonl"``) is the only source of truth. Each line is one of
``charge``, ``refuse``, ``fail`` or ``reset``, written, flushed and fsynced before the
call returns. ``used`` is the sum of the charge lines; nothing else is read, so rolling
back any other file cannot lower it.

Rules:

- Only successful work spends (``charge``). A refused request (``refuse``) is free.
- A backend failure (``fail``) is free but extends that kind's failure streak. After
  ``MAX_FAILURES`` failures in a row, ``check`` raises ``TooManyFailures`` until an
  operator calls ``reset_failures``. A charge resets the streak.
- ``forward_passes`` and ``code_runs`` are gated: ``charge`` raises ``BudgetExceeded``
  and writes nothing when the charge would pass the limit, so the server must withhold
  that result. Under the lock, at most ``limit`` charges ever succeed.
- ``tokens`` and ``dollars`` are reported after the money is spent, so ``charge``
  always records them, even past the limit; dropping them would be a refund. Gate
  these with ``check(kind, estimate)`` before each paid call.

Thread-safe within one process. One writer per run directory: there is no
cross-process lock.
"""

import math
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, get_args

from astro_weights.mcp.record import _append, _load

__all__ = [
    "MAX_FAILURES",
    "BudgetExceeded",
    "Kind",
    "Ledger",
    "Limits",
    "TooManyFailures",
]

Kind = Literal["forward_passes", "code_runs", "tokens", "dollars"]
KINDS: tuple[str, ...] = get_args(Kind)
MAX_FAILURES = 3
TEXT_MAX = 2000
_GATED = frozenset({"forward_passes", "code_runs"})
_COUNTED = frozenset({"forward_passes", "code_runs", "tokens"})
_FLOAT_MAX = Fraction(sys.float_info.max)
_OPS = (
    "charge",
    "refuse",
    "fail",
    "reset",
)  # a tuple: corrupt lines may hold unhashables


@dataclass(frozen=True)
class Limits:
    """Per-run caps. All >= 0; 0 allows none."""

    forward_passes: int
    code_runs: int
    tokens: int
    dollars: float

    def __post_init__(self) -> None:
        for kind in KINDS:
            value = getattr(self, kind)
            if kind in _COUNTED:
                ok = (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                )
            else:
                ok = (
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and not (isinstance(value, float) and not math.isfinite(value))
                    and value >= 0
                )
            if not ok:
                raise ValueError(
                    f"limit {kind} must be a finite number >= 0, got {value!r}"
                )


class BudgetExceeded(RuntimeError):
    """The request would take ``kind`` past its limit."""

    def __init__(self, kind: str, used: float, limit: float, requested: float) -> None:
        super().__init__(
            f"{kind} budget exhausted: used {used} of {limit}, requested {requested}"
        )
        self.kind, self.used, self.limit, self.requested = kind, used, limit, requested


class TooManyFailures(RuntimeError):
    """``kind`` failed ``MAX_FAILURES`` times in a row; an operator must reset it."""

    def __init__(self, kind: str, streak: int) -> None:
        super().__init__(
            f"{kind} failed {streak} times in a row; an operator must fix this and reset it"
        )
        self.kind, self.streak = kind, streak


def _kind(kind: Any) -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown budget kind {kind!r}; expected one of {list(KINDS)}")
    return str(kind)


def _amount(kind: str, amount: Any) -> float:
    """A finite amount > 0; whole numbers for every kind but dollars. Bools are rejected."""
    if isinstance(amount, bool) or not isinstance(amount, int | float):
        raise ValueError(f"{kind} amount must be a number, got {amount!r}")  # noqa: TRY004
    if kind in _COUNTED and not isinstance(amount, int):
        raise ValueError(f"{kind} amount must be an int, got {amount!r}")
    if (isinstance(amount, float) and not math.isfinite(amount)) or amount <= 0:
        raise ValueError(f"{kind} amount must be finite and > 0, got {amount!r}")
    return amount


def _out(kind: str, value: Fraction) -> float:
    """An exact total as reported: int for counted kinds, nearest float for dollars."""
    if kind in _COUNTED:
        return int(value)
    return float(min(value, _FLOAT_MAX))  # saturate rather than overflow to inf


def _floor_out(kind: str, value: Fraction) -> float:
    """Like ``_out`` but never above ``value`` (>= 0), so a reported remainder is usable."""
    out = _out(kind, value)
    if Fraction(out) > value:
        out = math.nextafter(out, 0.0)
    return out


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")  # noqa: TRY004
    return value[:TEXT_MAX]


def _check_entry(entry: dict[str, Any]) -> None:
    """Raise ValueError unless ``entry`` is a well-formed ledger line."""
    op = entry.get("op")
    if op not in _OPS:
        raise ValueError(f"unknown ledger op {op!r}")
    kind = _kind(entry.get("kind"))
    if op == "charge":
        _amount(kind, entry.get("amount"))


class Ledger:
    """Budget ledger backed by ``path`` (``run_dir/"ledger.jsonl"``); replays it on open."""

    def __init__(self, path: Path, limits: Limits) -> None:
        if not isinstance(limits, Limits):
            raise ValueError("limits must be a Limits")  # noqa: TRY004
        self.path = Path(path)
        self.limits = limits
        self._lock = threading.Lock()
        # exact sums: a float total would let sub-ulp charges vanish and slip past the cap
        self._used = dict.fromkeys(KINDS, Fraction(0))
        self._streak = dict.fromkeys(KINDS, 0)
        if self.path.exists():
            for entry in _load(self.path, _check_entry):
                self._apply(entry)

    def _apply(self, entry: dict[str, Any]) -> None:
        kind, op = entry["kind"], entry["op"]
        if op == "charge":
            self._used[kind] += Fraction(entry["amount"])
            self._streak[kind] = 0
        elif op == "fail":
            self._streak[kind] += 1
        elif op == "reset":
            self._streak[kind] = 0

    def _write(self, entry: dict[str, Any]) -> None:
        """Append first; update memory only once the line is on disk."""
        _append(self.path, {**entry, "time": datetime.now(UTC).isoformat()})
        self._apply(entry)

    def _limit(self, kind: str) -> Fraction:
        return Fraction(getattr(self.limits, kind))

    def _over(self, kind: str, amount: float) -> BudgetExceeded | None:
        used, limit = self._used[kind], self._limit(kind)
        if used + Fraction(amount) > limit:
            return BudgetExceeded(
                kind, _out(kind, used), getattr(self.limits, kind), amount
            )
        return None

    def check(self, kind: Kind, amount: float = 1) -> None:
        """Raise BudgetExceeded or TooManyFailures if ``amount`` of ``kind`` may not start. No write."""
        kind_ = _kind(kind)
        amount = _amount(kind_, amount)
        with self._lock:
            over = self._over(kind_, amount)
            if over is not None:
                raise over
            if self._streak[kind_] >= MAX_FAILURES:
                raise TooManyFailures(kind_, self._streak[kind_])

    def charge(self, kind: Kind, amount: float = 1, *, ref: str) -> None:
        """Record successful work and reset the failure streak.

        For ``forward_passes`` and ``code_runs`` the limit is checked under the same lock:
        past it, BudgetExceeded is raised and nothing is written. ``tokens`` and
        ``dollars`` are always recorded (the money is already spent).
        """
        kind_ = _kind(kind)
        amount = _amount(kind_, amount)
        ref = _text(ref, "ref")
        with self._lock:
            if kind_ in _GATED:
                over = self._over(kind_, amount)
                if over is not None:
                    raise over
            self._write({"op": "charge", "kind": kind_, "amount": amount, "ref": ref})

    def refuse(self, kind: Kind, reason: str, *, ref: str) -> None:
        """Log a refused request: free, streak unchanged."""
        entry = {"op": "refuse", "kind": _kind(kind), "reason": _text(reason, "reason")}
        entry["ref"] = _text(ref, "ref")
        with self._lock:
            self._write(entry)

    def fail(self, kind: Kind, error: str, *, ref: str) -> int:
        """Log a backend failure: free, extends the streak. Returns the new streak."""
        kind_ = _kind(kind)
        entry = {
            "op": "fail",
            "kind": kind_,
            "error": _text(error, "error"),
            "ref": _text(ref, "ref"),
        }
        with self._lock:
            self._write(entry)
            return self._streak[kind_]

    def reset_failures(self, kind: Kind) -> None:
        """Operator action: clear ``kind``'s failure streak."""
        kind_ = _kind(kind)
        with self._lock:
            self._write({"op": "reset", "kind": kind_})

    def used(self) -> dict[str, float]:
        """Amount charged so far, per kind."""
        with self._lock:
            return self._used_out()

    def _used_out(self) -> dict[str, float]:
        return {k: _out(k, v) for k, v in self._used.items()}

    def _remaining(self) -> dict[str, float]:
        return {
            k: _floor_out(k, max(self._limit(k) - self._used[k], Fraction(0)))
            for k in KINDS
        }

    def remaining(self) -> dict[str, float]:
        """Limit minus used, per kind; never below 0, dollars rounded toward zero."""
        with self._lock:
            return self._remaining()

    def to_json(self) -> dict[str, Any]:
        """Limits, used, remaining and failure streaks, for the status tool and run record."""
        with self._lock:
            return {
                "limits": asdict(self.limits),
                "used": self._used_out(),
                "remaining": self._remaining(),
                "streaks": dict(self._streak),
                "max_failures": MAX_FAILURES,
            }
