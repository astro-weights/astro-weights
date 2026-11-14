"""Edit types: a set of rank-r deltas on named linear modules."""

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
        b = np.ascontiguousarray(self.B, dtype=np.float32)
        a = np.ascontiguousarray(self.A, dtype=np.float32)
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
        return np.asarray(self.B @ self.A, dtype=np.float32)


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

    @property
    def rank(self) -> int:
        return self.deltas[0].rank
