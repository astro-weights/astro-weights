"""ARM.toml: the settings of one `astro-weights gen` run, one runner call (stdlib only).

Keys are the keyword arguments of the runner the `kind` names, with the same meaning.
Paths are relative to the directory of ARM.toml. Every key is checked and an unknown
key is an error, so a misspelt setting never falls back to a default. Reads the TOML
file (and, in `load_items`, the items and manifest it names); runs no model. See
docs/reference/cli.md.
"""

import json
import tomllib
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astro_weights.eval.splits import SPLITS, SplitManifest

__all__ = ["GRADERS", "KINDS", "ArmConfig", "load_arm_config", "load_items"]

KINDS = ("http", "suites", "bfcl")
GRADERS = ("exact", "math", "choice")

_COMMON = ("kind", "arm", "base_url", "model", "max_new_tokens")
_KEYS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # kind -> (required, optional)
    "http": (
        (*_COMMON, "out", "items", "grader"),
        (
            "seed",
            "api_key_env",
            "timeout",
            "extra_body",
            "concurrency",
            "serving_flags",
            "model_id",
            "system",
            "manifest",
            "split",
            "k",
        ),
    ),
    "suites": (
        (*_COMMON, "out_dir", "suites"),
        (
            "seed",
            "api_key_env",
            "extra_body",
            "concurrency",
            "limit",
            "cache_dir",
            "suite_paths",
            "system",
        ),
    ),
    "bfcl": (
        (*_COMMON, "out", "bfcl_root"),
        (
            "seed",
            "api_key_env",
            "timeout",
            "extra_body",
            "trials",
            "max_steps_per_turn",
            "serving_flags",
            "model_id",
        ),
    ),
}


@dataclass(frozen=True)
class ArmConfig:
    """One arm: `kind` says which runner; the fields are that runner's arguments.

    `out` is the records file (`http`, `bfcl`), `out_dir` the suites directory.
    `k` is the number of seeds of an `http` run (see `load_arm_config`).
    """

    kind: str
    arm: str
    base_url: str
    model: str
    max_new_tokens: int
    path: Path
    out: Path | None = None
    out_dir: Path | None = None
    items: Path | None = None
    grader: str | None = None
    manifest: Path | None = None
    split: str | None = None
    k: int = 1
    suites: tuple[str, ...] = ()
    limit: int | None = None
    cache_dir: Path | None = None
    suite_paths: Mapping[str, Path] = field(default_factory=dict)
    bfcl_root: Path | None = None
    trials: int = 1
    max_steps_per_turn: int = 20
    seed: int = 0
    api_key_env: str | None = None
    timeout: float = 120.0
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    concurrency: int = 1
    serving_flags: Mapping[str, str] | None = None
    model_id: str | None = None
    system: str | None = None


def _check_keys(
    raw: Mapping[str, Any], required: Collection[str], optional: Collection[str]
) -> None:
    if unknown := sorted(set(raw) - set(required) - set(optional)):
        raise ValueError(f"unknown keys {unknown}")
    if missing := sorted(set(required) - set(raw)):
        raise ValueError(f"missing keys {missing}")


def _str(raw: Mapping[str, Any], key: str) -> str:
    v = raw[key]
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{key}: must be a non-empty string")
    return v


def _opt_str(raw: Mapping[str, Any], key: str) -> str | None:
    return _str(raw, key) if key in raw else None


def _int(raw: Mapping[str, Any], key: str, default: int, low: int) -> int:
    v = raw.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int) or v < low:
        raise ValueError(f"{key}: must be an integer >= {low}")
    return v


def _timeout(raw: Mapping[str, Any]) -> float:
    v = raw.get("timeout", 120.0)
    if isinstance(v, bool) or not isinstance(v, int | float) or not v > 0:
        raise ValueError("timeout: must be a number > 0")
    return float(v)


def _table(raw: Mapping[str, Any], key: str) -> dict[str, Any]:
    v = raw.get(key, {})
    if not isinstance(v, dict):
        raise ValueError(f"{key}: must be a table")  # noqa: TRY004
    return v


def _str_table(raw: Mapping[str, Any], key: str) -> dict[str, str] | None:
    if key not in raw:
        return None
    t = _table(raw, key)
    if not all(isinstance(v, str) for v in t.values()):
        raise ValueError(f"{key}: values must be strings")
    return t


def _names(raw: Mapping[str, Any]) -> tuple[str, ...]:
    v = raw["suites"]
    if (
        not isinstance(v, list)
        or not v
        or not all(isinstance(n, str) and n for n in v)
        or len(set(v)) != len(v)
    ):
        raise ValueError("suites: must be a non-empty list of distinct names")
    return tuple(v)


def _fields(raw: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """The keys `kind` takes, as `ArmConfig` fields, paths joined to `root`."""
    out: dict[str, Any] = {}
    for key in ("out", "out_dir", "items", "manifest", "cache_dir", "bfcl_root"):
        if key in raw:
            out[key] = root / _str(raw, key)
    for key in ("api_key_env", "model_id", "system", "split"):
        if key in raw:
            out[key] = _str(raw, key)
    if "grader" in raw:
        out["grader"] = _str(raw, "grader")
    if "suites" in raw:
        out["suites"] = _names(raw)
    if "suite_paths" in raw:
        out["suite_paths"] = {
            k: root / v for k, v in (_str_table(raw, "suite_paths") or {}).items()
        }
    if "serving_flags" in raw:
        out["serving_flags"] = _str_table(raw, "serving_flags")
    if "extra_body" in raw:
        out["extra_body"] = _table(raw, "extra_body")
    if "limit" in raw:
        out["limit"] = _int(raw, "limit", 1, 1)
    for key, default, low in (
        ("seed", 0, 0),
        ("concurrency", 1, 1),
        ("trials", 1, 1),
        ("max_steps_per_turn", 20, 1),
    ):
        if key in raw:
            out[key] = _int(raw, key, default, low)
    if "timeout" in raw:
        out["timeout"] = _timeout(raw)
    return out


def load_arm_config(path: str | Path) -> ArmConfig:
    """Read ARM.toml. ValueError on bad TOML, an unknown, missing or mistyped key.

    `http`: `manifest` and `split` go together and keep only that split's items; for
    `split = "test"` `k` is the manifest's (an explicit `k` must equal it), else 1.
    """
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{path}: {e}") from None
    try:
        kind = raw.get("kind")
        if kind not in KINDS:
            raise ValueError(f"kind: must be one of {list(KINDS)}, got {kind!r}")
        required, optional = _KEYS[kind]
        _check_keys(raw, required, optional)
        f = _fields(raw, path.parent)
        if kind == "http":
            if f["grader"] not in GRADERS:
                raise ValueError(f"grader: must be one of {list(GRADERS)}")
            if ("manifest" in f) != ("split" in f):
                raise ValueError("manifest and split go together")
            if "split" in f and f["split"] not in SPLITS:
                raise ValueError(f"split: must be one of {list(SPLITS)}")
            f["k"] = _int(raw, "k", 1, 1)
        return ArmConfig(
            kind=kind,
            arm=_str(raw, "arm"),
            base_url=_str(raw, "base_url"),
            model=_str(raw, "model"),
            max_new_tokens=_int(raw, "max_new_tokens", 0, 1),
            path=path,
            **f,
        )
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None


def load_items(cfg: ArmConfig) -> tuple[list[dict[str, Any]], int]:
    """The items of an `http` arm and its `k`.

    `items` is JSONL of `{"item_id", "prompt", "gold"}`. With `manifest` and `split`
    only that split's ids are kept, in file order, and every id the split lists must
    be there. `k` is `cfg.k`; for `split = "test"` it must equal the manifest's `k`
    (left at 1 it becomes that `k`). ValueError when the file or the manifest
    does not fit.
    """
    if cfg.items is None:
        raise ValueError("items: this arm has none")
    items: list[dict[str, Any]] = []
    with cfg.items.open(encoding="utf-8") as f:
        for num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                items.append({k: obj[k] for k in ("item_id", "prompt", "gold")})
            except (json.JSONDecodeError, KeyError, TypeError):
                raise ValueError(
                    f"{cfg.items}:{num}: not a JSON object with item_id, prompt, gold"
                ) from None
    k = cfg.k
    if cfg.manifest is not None and cfg.split is not None:
        manifest = SplitManifest.load(cfg.manifest)
        ids = manifest.ids(cfg.split)  # type: ignore[arg-type]
        if missing := sorted(ids - {str(i["item_id"]) for i in items}):
            raise ValueError(
                f"{cfg.items}: {len(missing)} item(s) of split {cfg.split!r} are "
                f"missing, e.g. {missing[:3]}"
            )
        items = [i for i in items if str(i["item_id"]) in ids]
        if cfg.split == "test":
            if k not in (1, manifest.k):
                raise ValueError(f"k = {k}, but the manifest's test k is {manifest.k}")
            k = manifest.k
    return items, k
