"""Schema of `verdict.json`: what a gate run records beside its verdict.

Schema only. No command fills it yet. Verdicts are per backend: the base, the edit and
the build all ran on the backend named in the record, so `backend` and `build` are
required. Never runs a model.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Self

VERDICTS = ("confirmed", "not significant", "regressed")
SCHEMA_VERSION = 1


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


@dataclass(frozen=True)
class Verdict:
    """One verdict and the run record behind it.

    `verdict` is one of `VERDICTS`. `config_hash` is `config_hash(header)` of the
    arms' shared config; `split_manifest_sha256` pins the splits. `candidates_tried`
    and `confirm_queries` count the search that led here. `footprint` maps a
    measure name to its value for the edit after dtype rounding. `infra_errors` counts
    requests that failed for server reasons. `backend` and `build` name the serving
    engine ("llama.cpp", "vllm", "hf") and its version or build.
    """

    verdict: str
    backend: str
    build: str
    config_hash: str
    split_manifest_sha256: str
    candidates_tried: int
    confirm_queries: int
    infra_errors: int
    footprint: Mapping[str, float]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {', '.join(VERDICTS)}")
        for name in ("backend", "build", "config_hash", "split_manifest_sha256"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("candidates_tried", "confirm_queries", "infra_errors"):
            v = getattr(self, name)
            if not _is_int(v) or v < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if not isinstance(self.footprint, Mapping) or not all(
            isinstance(k, str)
            and isinstance(v, int | float)
            and not isinstance(v, bool)
            and math.isfinite(v)
            for k, v in self.footprint.items()
        ):
            raise ValueError("footprint must map names to finite numbers")

    def to_json(self) -> str:
        """JSON text with sorted keys."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["footprint"] = dict(self.footprint)
        return json.dumps(data, indent=1, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Self:
        """Parse `to_json` output. ValueError on bad JSON, a missing, unknown or
        wrongly typed key, or a value the schema rejects.
        """
        try:
            data = json.loads(text)
        except ValueError as e:
            raise ValueError(f"invalid verdict JSON: {e}") from e
        if not isinstance(data, dict):
            raise ValueError("verdict JSON must be an object")  # noqa: TRY004
        known = {f.name for f in fields(cls)}
        if unknown := sorted(set(data) - known):
            raise ValueError(f"unknown keys: {', '.join(unknown)}")
        required = known - {"schema_version"}
        if missing := sorted(required - set(data)):
            raise ValueError(f"missing keys: {', '.join(missing)}")
        return cls(**data)
