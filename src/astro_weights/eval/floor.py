"""The noise floor of one run configuration, stored under its `config_hash`.

The floor is two runs of the base arm under the same settings. A stored floor is
reused only for the exact `config_hash` it was made for. Never runs a model.
"""

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from astro_weights.eval.header import config_hash, library_version
from astro_weights.eval.paired import compare, pair
from astro_weights.eval.records import Record

_HASH = re.compile(r"[0-9a-f]{64}")


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_finite(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


@dataclass(frozen=True)
class FloorResult:
    """Run-to-run noise of the base arm under one config.

    `accuracy_delta` and `tokens_delta` are the mean of run 2 minus run 1 (per-item
    means over seeds) and `accuracy_se` and `tokens_se` their paired standard errors.
    `flip_rate` is the share of (item, seed) pairs whose output text differs between
    the runs. `k` is the number of seeds per item. `created` is a UTC ISO 8601 time
    and `library_version` the installed astro-weights version, or None from an
    uninstalled source tree. All numbers are finite; a deterministic server gives 0.
    """

    config_hash: str
    n_items: int
    k: int
    accuracy_delta: float
    accuracy_se: float
    flip_rate: float
    tokens_delta: float
    tokens_se: float
    created: str
    library_version: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.config_hash, str) or not _HASH.fullmatch(
            self.config_hash
        ):
            raise ValueError("config_hash must be 64 lowercase hex characters")
        for name in ("n_items", "k"):
            if not _is_int(getattr(self, name)) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("accuracy_delta", "tokens_delta"):
            if not _is_finite(getattr(self, name)):
                raise ValueError(f"{name} must be a finite number")
        for name in ("accuracy_se", "tokens_se", "flip_rate"):
            if not _is_finite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a finite number >= 0")
        if self.flip_rate > 1:
            raise ValueError("flip_rate must be at most 1")
        if not isinstance(self.created, str) or not self.created:
            raise ValueError("created must be a non-empty string")
        if self.library_version is not None and not isinstance(
            self.library_version, str
        ):
            raise ValueError("library_version must be a string or None")

    def to_json(self) -> str:
        """JSON text with sorted keys."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        return json.dumps(data, indent=1, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Self:
        """Parse `to_json` output. ValueError on bad JSON, a missing or unknown key,
        or a wrongly typed or non-finite value.
        """
        try:
            data = json.loads(text)
        except ValueError as e:
            raise ValueError(f"invalid floor JSON: {e}") from e
        if not isinstance(data, dict):
            raise ValueError("floor JSON must be an object")  # noqa: TRY004
        names = {f.name for f in fields(cls)}
        if unknown := sorted(set(data) - names):
            raise ValueError(f"unknown keys: {', '.join(unknown)}")
        if missing := sorted(names - set(data)):
            raise ValueError(f"missing keys: {', '.join(missing)}")
        return cls(**data)


def floor_from_records(
    run1: Sequence[Record], run2: Sequence[Record], header: Mapping[str, Any]
) -> FloorResult:
    """Floor from two runs of the base arm under the settings in `header`.

    The runs pair by (item_id, seed) and prompt hash as in `pair`. ValueError when they
    do not pair, or when fewer than 2 items (or clusters) leave a standard error
    undefined.
    """
    pairs = pair(run1, run2)
    if not pairs:
        raise ValueError("no records to compare")
    r = compare(run1, run2, n_boot=1)
    if not (math.isfinite(r.se) and math.isfinite(r.tokens_se)):
        raise ValueError("need at least 2 items (or clusters) for a standard error")
    return FloorResult(
        config_hash=config_hash(header),
        n_items=r.n_items,
        k=len({a.seed for a, _ in pairs}),
        accuracy_delta=r.diff,
        accuracy_se=r.se,
        flip_rate=sum(a.output != b.output for a, b in pairs) / len(pairs),
        tokens_delta=r.tokens_diff,
        tokens_se=r.tokens_se,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        library_version=library_version(),
    )


def _path(directory: str | Path, key: str) -> Path:
    if not isinstance(key, str) or not _HASH.fullmatch(key):
        raise ValueError("config_hash must be 64 lowercase hex characters")
    return Path(directory) / f"floor-{key}.json"


def save_floor(directory: str | Path, floor: FloorResult) -> Path:
    """Write `floor` to `DIRECTORY/floor-<config_hash>.json` and return the path."""
    path = _path(directory, floor.config_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(floor.to_json(), encoding="utf-8")
    return path


def load_floor(directory: str | Path, config_hash: str) -> FloorResult | None:
    """The stored floor for `config_hash`, or None when there is none.

    ValueError when the file is not a valid floor or its stored `config_hash` is not
    the one in its name. A floor made under another config is never returned.
    """
    path = _path(directory, config_hash)
    if not path.is_file():
        return None
    floor = FloorResult.from_json(path.read_text(encoding="utf-8"))
    if floor.config_hash != config_hash:
        raise ValueError(
            f"{path}: stored config_hash {floor.config_hash} does not match its name"
        )
    return floor
