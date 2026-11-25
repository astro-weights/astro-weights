"""Refuse to resume a records file made under another config (shared by the runners)."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from astro_weights.eval.header import config_hash, header_path, read_header


def kept_header(
    path: Path,
    header: Mapping[str, Any] | None,
    check: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any] | None:
    """The header for a resumed `path`: the earlier run's if there is one, else `header`.

    An existing header file must pass `check` (when given) and, when `header` is
    given, have its `config_hash`, else `ValueError`. Nothing is written.
    """
    where = header_path(path)
    if not where.exists():
        return None if header is None else dict(header)
    old = read_header(where)
    if check is not None:
        check(old)
    if header is not None and config_hash(old) != config_hash(header):
        raise ValueError(
            f"{where}: the records were made under another config "
            "(config_hash differs); start a new file for this one"
        )
    return old
