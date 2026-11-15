"""Per-item eval records and their JSONL form."""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Record:
    """One generation for one (item, seed) in one arm."""

    item_id: str
    seed: int
    arm: str
    prompt_hash: str
    output: str
    score: float
    n_output_tokens: int
    hit_cap: bool
    cluster_id: str | None = None


_REQUIRED = tuple(f.name for f in fields(Record) if f.name != "cluster_id")
_KNOWN = {f.name for f in fields(Record)}


def prompt_hash(text: str) -> str:
    """SHA-256 hex digest of the UTF-8 prompt."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: str | Path, records: Iterable[Record]) -> None:
    """Write one JSON object per line."""
    with Path(path).open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in records)


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _parse(line: str, num: int) -> Record:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"line {num}: invalid JSON: {e.msg}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"line {num}: expected a JSON object")  # noqa: TRY004
    missing = [k for k in _REQUIRED if k not in obj]
    if missing:
        raise ValueError(f"line {num}: missing keys: {', '.join(missing)}")
    unknown = sorted(set(obj) - _KNOWN)
    if unknown:
        raise ValueError(f"line {num}: unknown keys: {', '.join(unknown)}")
    cluster = obj.get("cluster_id")
    ok = (
        all(
            isinstance(obj[k], str) for k in ("item_id", "arm", "prompt_hash", "output")
        )
        and _is_int(obj["seed"])
        and _is_int(obj["n_output_tokens"])
        and isinstance(obj["score"], int | float)
        and not isinstance(obj["score"], bool)
        and isinstance(obj["hit_cap"], bool)
        and (cluster is None or isinstance(cluster, str))
    )
    if not ok:
        raise ValueError(f"line {num}: wrong value type")
    obj["score"] = float(obj["score"])
    return Record(**obj)


def read_jsonl(path: str | Path) -> list[Record]:
    """Read records written by `write_jsonl`; blank lines are skipped."""
    out: list[Record] = []
    with Path(path).open(encoding="utf-8") as f:
        for num, line in enumerate(f, start=1):
            if line.strip():
                out.append(_parse(line, num))
    return out
