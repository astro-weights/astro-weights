"""Seeded dev / confirm / test splits of a dataset's item ids, and their manifest.

An item's split is decided by ``sha256(f"{seed}:{item_id}")`` alone, so adding or
removing other items never moves it. The manifest pins the dataset (name, revision,
file sha256), the seed, the fractions and the ids of each split; its
``manifest_sha256`` is the sha256 of its canonical JSON and names the split set in a
seal ledger (``astro_weights.eval.seal``).
"""

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast, get_args

from astro_weights.eval.records import Record

__all__ = ["SPLITS", "Split", "SplitManifest", "assign_splits"]

Split = Literal["dev", "confirm", "test"]
SPLITS: tuple[Split, ...] = get_args(Split)
FORMAT = 1
_KEYS = frozenset(
    {
        "format",
        "dataset",
        "revision",
        "file_sha256",
        "seed",
        "fractions",
        "splits",
        "manifest_sha256",
    }
)
_SHA = re.compile(r"[0-9a-f]{64}")
_SHOW = 5


def _check_seed(seed: object) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an int, got {seed!r}")  # noqa: TRY004
    return seed


def _check_fractions(fractions: object) -> tuple[float, float, float]:
    if not isinstance(fractions, list | tuple) or len(fractions) != 3:
        raise ValueError("fractions must be three numbers (dev, confirm, test)")
    out: list[float] = []
    for f in fractions:
        if isinstance(f, bool) or not isinstance(f, int | float):
            raise ValueError(f"fraction {f!r} is not a number")  # noqa: TRY004
        if not (math.isfinite(f) and 0.0 <= f <= 1.0):
            raise ValueError(f"fraction {f!r} is not in [0, 1]")
        out.append(float(f))
    if abs(math.fsum(out) - 1.0) > 1e-9:
        raise ValueError(f"fractions {out} must sum to 1")
    return out[0], out[1], out[2]


def _unit(seed: int, item_id: str) -> float:
    """A uniform number in [0, 1) fixed by seed and item id only."""
    digest = hashlib.sha256(f"{seed}:{item_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def _split_of(u: float, fractions: tuple[float, float, float]) -> Split:
    if u < fractions[0]:
        return "dev"
    if u < fractions[0] + fractions[1]:
        return "confirm"
    return "test"


def _check_ids(item_ids: Iterable[object]) -> list[str]:
    ids = list(item_ids)
    for i in ids:
        if not isinstance(i, str) or not i:
            raise ValueError(f"item id {i!r} must be a non-empty string")
    dup = sorted(i for i, n in Counter(cast(list[str], ids)).items() if n > 1)
    if dup:
        raise ValueError(f"repeated item ids: {dup[:_SHOW]}")
    return cast(list[str], ids)


def assign_splits(
    item_ids: Iterable[str],
    seed: int,
    fractions: tuple[float, float, float] = (0.2, 0.3, 0.5),
) -> dict[str, Split]:
    """Map each item id to ``"dev"``, ``"confirm"`` or ``"test"``.

    The split of an item depends only on ``seed``, its id and ``fractions``: adding
    or removing items never moves another item. ValueError on a repeated or
    non-string id, a non-int seed, or fractions that are not three numbers in
    [0, 1] summing to 1.
    """
    seed = _check_seed(seed)
    fr = _check_fractions(fractions)
    return {i: _split_of(_unit(seed, i), fr) for i in _check_ids(item_ids)}


def _canonical(obj: Mapping[str, Any]) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class SplitManifest:
    """Which item ids of one pinned dataset file are in which split.

    ``file_sha256`` is the sha256 of the dataset file the ids index into. Ids are
    kept sorted. Construction checks that every id sits in the split that
    ``assign_splits`` gives it under ``seed`` and ``fractions``, so a hand-moved id
    is refused.
    """

    dataset: str
    revision: str
    file_sha256: str
    seed: int
    fractions: tuple[float, float, float]
    dev: tuple[str, ...]
    confirm: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.dataset, "dataset")
        _text(self.revision, "revision")
        if not isinstance(self.file_sha256, str) or not _SHA.fullmatch(
            self.file_sha256
        ):
            raise ValueError("file_sha256 must be 64 lowercase hex digits")
        _check_seed(self.seed)
        fr = _check_fractions(self.fractions)
        object.__setattr__(self, "fractions", fr)
        for name in SPLITS:
            value = getattr(self, name)
            if not isinstance(value, list | tuple):
                raise ValueError(f"{name} must be a list of item ids")  # noqa: TRY004
            object.__setattr__(self, name, tuple(sorted(value)))
        _check_ids([*self.dev, *self.confirm, *self.test])
        for name in SPLITS:
            wrong = [
                i
                for i in getattr(self, name)
                if _split_of(_unit(self.seed, i), fr) != name
            ]
            if wrong:
                raise ValueError(
                    f"{len(wrong)} id(s) listed under {name} belong to another split "
                    f"under seed {self.seed}: {wrong[:_SHOW]}"
                )

    @classmethod
    def build(
        cls,
        item_ids: Iterable[str],
        *,
        dataset: str,
        revision: str,
        file_sha256: str,
        seed: int,
        fractions: tuple[float, float, float] = (0.2, 0.3, 0.5),
    ) -> "SplitManifest":
        """Assign ``item_ids`` with ``assign_splits`` and pin the dataset file."""
        assigned = assign_splits(item_ids, seed, fractions)
        parts = {s: tuple(i for i, v in assigned.items() if v == s) for s in SPLITS}
        return cls(
            dataset,
            revision,
            file_sha256,
            seed,
            fractions,
            parts["dev"],
            parts["confirm"],
            parts["test"],
        )

    def _body(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "dataset": self.dataset,
            "revision": self.revision,
            "file_sha256": self.file_sha256,
            "seed": self.seed,
            "fractions": list(self.fractions),
            "splits": {s: list(getattr(self, s)) for s in SPLITS},
        }

    @property
    def manifest_sha256(self) -> str:
        """sha256 of the canonical JSON (sorted keys, no spaces, UTF-8)."""
        return hashlib.sha256(_canonical(self._body())).hexdigest()

    def ids(self, split: Split) -> frozenset[str]:
        """The item ids in ``split``; ValueError on an unknown split name."""
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        return frozenset(getattr(self, split))

    def check_records(
        self, records: Iterable[Record], split: Split, *, complete: bool = False
    ) -> None:
        """ValueError unless every record's item id is in ``split``.

        With ``complete``, also ValueError unless every id of ``split`` has a record.
        """
        allowed = self.ids(split)
        seen = {r.item_id for r in records}
        outside = sorted(seen - allowed)
        if outside:
            raise ValueError(
                f"{len(outside)} item id(s) are not in the {split} split of "
                f"{self.dataset}: {outside[:_SHOW]}"
            )
        missing = sorted(allowed - seen)
        if complete and missing:
            raise ValueError(
                f"{len(missing)} of {len(allowed)} {split} item id(s) have no "
                f"record: {missing[:_SHOW]}"
            )

    def save(self, path: str | Path) -> None:
        """Write the manifest as JSON, with its ``manifest_sha256``."""
        body = self._body() | {"manifest_sha256": self.manifest_sha256}
        Path(path).write_text(
            json.dumps(body, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "SplitManifest":
        """Read a manifest; ValueError on unknown or missing keys or a sha mismatch."""
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, RecursionError) as e:
            raise ValueError(f"{path}: not valid JSON: {e}") from None
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: must hold a JSON object")  # noqa: TRY004
        if set(raw) != _KEYS:
            extra, lack = sorted(set(raw) - _KEYS), sorted(_KEYS - set(raw))
            raise ValueError(f"{path}: unknown keys {extra}, missing keys {lack}")
        if raw["format"] != FORMAT or isinstance(raw["format"], bool):
            raise ValueError(f"{path}: format must be {FORMAT}")
        splits = raw["splits"]
        if not isinstance(splits, dict) or set(splits) != set(SPLITS):
            raise ValueError(f"{path}: splits must have exactly the keys {SPLITS}")
        try:
            manifest = cls(
                raw["dataset"],
                raw["revision"],
                raw["file_sha256"],
                raw["seed"],
                raw["fractions"],
                splits["dev"],
                splits["confirm"],
                splits["test"],
            )
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from None
        if raw["manifest_sha256"] != manifest.manifest_sha256:
            raise ValueError(f"{path}: manifest_sha256 does not match its contents")
        return manifest
