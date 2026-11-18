"""Edit types: a set of rank-r deltas on named linear modules."""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class LowRankDelta:
    """dW = B @ A for one linear module; B is [out, r], A is [r, in]."""

    module: str
    B: np.ndarray
    A: np.ndarray

    def __post_init__(self) -> None:
        if not self.module:
            raise ValueError("module name must be non-empty")
        with np.errstate(over="ignore", invalid="ignore"):
            b = np.ascontiguousarray(self.B, dtype=np.float32)
            a = np.ascontiguousarray(self.A, dtype=np.float32)
        if not (np.isfinite(b).all() and np.isfinite(a).all()):
            raise ValueError(f"{self.module}: B and A must be finite as float32")
        if b.ndim != 2 or a.ndim != 2:
            raise ValueError(
                f"{self.module}: B and A must be 2-D, got {b.ndim}-D and {a.ndim}-D"
            )
        if b.shape[1] != a.shape[0] or a.shape[0] < 1:
            raise ValueError(
                f"{self.module}: rank mismatch, B {b.shape} vs A {a.shape}"
            )
        object.__setattr__(self, "B", b)
        object.__setattr__(self, "A", a)

    @property
    def rank(self) -> int:
        return int(self.A.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.B.shape[0]), int(self.A.shape[1]))

    def dense(self) -> np.ndarray:
        """Return the dense `B @ A` as float32."""
        return np.asarray(self.B @ self.A, dtype=np.float32)

    def padded(self, rank: int) -> "LowRankDelta":
        """Return this delta at `rank` by appending zero columns/rows (dense unchanged)."""
        if rank < self.rank:
            raise ValueError(
                f"{self.module}: cannot pad rank {self.rank} down to {rank}"
            )
        if rank == self.rank:
            return self
        extra = rank - self.rank
        b = np.pad(self.B, ((0, 0), (0, extra)))
        a = np.pad(self.A, ((0, extra), (0, 0)))
        return LowRankDelta(self.module, b, a)


@dataclass(frozen=True)
class Edit:
    """A set of low-rank deltas against one base model."""

    deltas: tuple[LowRankDelta, ...]
    base_model: str
    base_sha256: dict[str, str] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.deltas:
            raise ValueError("an edit needs at least one delta")
        names = [d.module for d in self.deltas]
        if len(set(names)) != len(names):
            raise ValueError("duplicate module names in edit")
        ranks = {d.rank for d in self.deltas}
        if len(ranks) != 1:
            # llama.cpp's LoRA converter ignores per-module ranks
            raise ValueError(
                f"mixed ranks {sorted(ranks)}; all deltas must share one rank"
            )

    @classmethod
    def from_deltas(
        cls,
        deltas: Iterable[LowRankDelta],
        base_model: str,
        base_sha256: dict[str, str] | None = None,
        meta: dict[str, object] | None = None,
    ) -> "Edit":
        """Build an edit, zero-padding every delta to the largest rank."""
        items = tuple(deltas)
        if not items:
            raise ValueError("an edit needs at least one delta")
        rank = max(d.rank for d in items)
        return cls(
            tuple(d.padded(rank) for d in items),
            base_model,
            dict(base_sha256 or {}),
            dict(meta or {}),
        )

    @property
    def rank(self) -> int:
        return self.deltas[0].rank


def compose(*edits: Edit) -> Edit:
    """Combine edits on one base into one edit whose dense delta is their sum.

    A module in several edits gets concatenated factors (`hstack` of the B's,
    `vstack` of the A's), so `B @ A` is the exact sum and the rank adds up; the
    result is zero-padded to the largest rank. Edits must share `base_model`;
    recorded `base_sha256` entries must agree. `meta` lists each input's meta
    under `composed_from`.
    """
    if len(edits) < 2:
        raise ValueError("compose needs at least two edits")
    first = edits[0]
    for e in edits[1:]:
        if e.base_model != first.base_model:
            raise ValueError(
                f"different base models: {first.base_model!r} vs {e.base_model!r}"
            )
    prefix = _layer_prefix(edits)
    sha: dict[str, str] = {}
    for e in edits:
        for key, value in e.base_sha256.items():
            if sha.setdefault(_with_prefix(key, prefix), value) != value:
                raise ValueError(
                    f"{key}: base hashes differ; adapters built on different bases"
                )
    parts: dict[str, list[LowRankDelta]] = {}
    for e in edits:
        for d in e.deltas:
            d = LowRankDelta(_with_prefix(d.module, prefix), d.B, d.A)
            parts.setdefault(d.module, []).append(d)
    deltas = [_join(group) for group in parts.values()]
    return Edit.from_deltas(
        deltas,
        first.base_model,
        sha,
        {"composed_from": [dict(e.meta) for e in edits]},
    )


# the two HF layouts of a language model's layers; other names (vision towers…) stay
_LAYER = re.compile(r"^(model\.(?:language_model\.)?)(layers\.\d+\..*)$")


def _layer_prefix(edits: tuple[Edit, ...]) -> str | None:
    """`model.` or `model.language_model.` of the first module that has one, if any."""
    for e in edits:
        for d in e.deltas:
            if m := _LAYER.match(d.module):
                return m.group(1)
    return None


def _with_prefix(name: str, prefix: str | None) -> str:
    """`name` with its `model.` / `model.language_model.` layers prefix replaced by `prefix`.

    `model.layers.3.x` and `model.language_model.layers.3.x` name one layer in
    two HF layouts; composing renames all to the first edit's layout.
    """
    m = _LAYER.match(name)
    return name if m is None or prefix is None else prefix + m.group(2)


def _join(group: list[LowRankDelta]) -> LowRankDelta:
    """One delta equal to the sum of `group`, all on one module."""
    if len(group) == 1:
        return group[0]
    module = group[0].module
    for d in group[1:]:
        if d.shape != group[0].shape:
            raise ValueError(f"{module}: shape mismatch, {group[0].shape} vs {d.shape}")
    return LowRankDelta(
        module,
        np.hstack([d.B for d in group]),
        np.vstack([d.A for d in group]),
    )
