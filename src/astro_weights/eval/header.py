"""Run headers: the settings that decide an arm's outputs, stored beside its records."""

import json
from collections.abc import Collection
from pathlib import Path
from typing import Any

_MISSING = "<missing>"


def header_path(records_path: str | Path) -> Path:
    """Header file for a records file: `X.jsonl` -> `X.header.json`."""
    return Path(records_path).with_suffix(".header.json")


def write_header(path: str | Path, header: dict[str, Any]) -> None:
    """Write `header` as JSON with sorted keys."""
    text = json.dumps(header, indent=1, sort_keys=True, ensure_ascii=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def read_header(path: str | Path) -> dict[str, Any]:
    """Read a header file; ValueError if it is not a JSON object."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")  # noqa: TRY004
    return data


def check_headers(
    base: dict[str, Any], edit: dict[str, Any], ignore: Collection[str] = ("device",)
) -> None:
    """Raise ValueError listing every key that differs or is missing on one side."""
    diffs = [
        f"{k}: base={base.get(k, _MISSING)!r} edit={edit.get(k, _MISSING)!r}"
        for k in sorted((base.keys() | edit.keys()) - set(ignore))
        if k not in base or k not in edit or base[k] != edit[k]
    ]
    if diffs:
        raise ValueError(
            "arms ran under different settings and are not comparable: "
            + "; ".join(diffs)
        )
