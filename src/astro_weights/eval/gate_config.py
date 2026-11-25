"""GATE.toml: the files and settings one `astro-weights gate` run reads (stdlib only).

Paths are relative to the directory of GATE.toml. Every key is checked and an unknown
key is an error, so a misspelt setting never falls back to a default. Reads the TOML
file only; the files it names are read by the command. See docs/reference/cli.md.
"""

import math
import tomllib
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ConfirmConfig",
    "GateConfig",
    "SideFiles",
    "SuiteConfig",
    "TestFiles",
    "load_gate_config",
]

# `[arms]` key -> (name shown in the report, kind for `compare_arms`)
REPORT_ARMS = {
    "prompt": ("prompt", "prompt"),
    "scripted": ("scripted", "scripted"),
    "edit_prompt": ("edit+prompt", "edit"),
}
TARGETS = {"tokens_diff": "lower", "diff": "higher"}  # target -> better direction


@dataclass(frozen=True)
class SideFiles:
    """Records of one side effect other than "accuracy": the edit, the base, the controls."""

    base: Path
    edit: Path
    controls: Path


@dataclass(frozen=True)
class ConfirmConfig:
    """Stage 3 files. `controls` is a directory of control-arm `*.jsonl` files.

    `side_effects` names the side effects the controls' range is judged on; "accuracy"
    uses the confirm records above, any other name needs a `[confirm.side.NAME]` table.
    """

    base: Path
    edit: Path
    controls: Path
    q: float
    target: str
    side_effects: tuple[str, ...]
    side: Mapping[str, SideFiles] = field(default_factory=dict)

    @property
    def direction(self) -> str:
        return TARGETS[self.target]


@dataclass(frozen=True)
class TestFiles:
    """Stage 4 files: test records (K = 4), the seal ledger, the edit's adapter, floors."""

    base: Path
    edit: Path
    seal_dir: Path
    adapter: Path
    floor: Path


@dataclass(frozen=True)
class SuiteConfig:
    """One regression suite: records and its own non-inferiority margin and source."""

    base: Path
    edit: Path
    margin: float
    margin_source: str


@dataclass(frozen=True)
class GateConfig:
    """Everything `astro-weights gate` needs; see `load_gate_config`."""

    margin: float
    margin_source: str
    backend: str
    build: str
    screens: Path
    run_record: Path
    manifest: Path
    confirm: ConfirmConfig
    test: TestFiles
    suites: Mapping[str, SuiteConfig]
    arms: Mapping[str, Path] = field(default_factory=dict)
    footprint: Mapping[str, float] = field(default_factory=dict)
    boot: int = 10000
    seed: int = 0
    arm_keys: tuple[str, ...] = ()
    max_cap_hit: float | None = None


def _table(
    raw: object,
    where: str,
    required: Collection[str],
    optional: Collection[str] = (),
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: must be a table")  # noqa: TRY004
    if unknown := sorted(set(raw) - set(required) - set(optional)):
        raise ValueError(f"{where}: unknown keys {unknown}")
    if missing := sorted(set(required) - set(raw)):
        raise ValueError(f"{where}: missing keys {missing}")
    return raw


def _open_table(raw: object, where: str) -> dict[str, Any]:
    """A table whose keys are the user's own names (suites, side effects, footprint)."""
    return _table(raw, where, (), tuple(raw) if isinstance(raw, dict) else ())


def _str(t: Mapping[str, Any], key: str, where: str) -> str:
    v = t[key]
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{where}.{key}: must be a non-empty string")
    return v


def _num(t: Mapping[str, Any], key: str, where: str) -> float:
    v = t[key]
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
        raise ValueError(f"{where}.{key}: must be a finite number")
    return float(v)


def _int(t: Mapping[str, Any], key: str, where: str, low: int) -> int:
    v = t[key]
    if isinstance(v, bool) or not isinstance(v, int) or v < low:
        raise ValueError(f"{where}.{key}: must be an integer >= {low}")
    return v


def _path(t: Mapping[str, Any], key: str, where: str, root: Path) -> Path:
    return root / _str(t, key, where)


def _confirm(raw: object, root: Path) -> ConfirmConfig:
    t = _table(
        raw,
        "confirm",
        ("base", "edit", "controls", "target", "side_effects"),
        ("q", "side"),
    )
    target = _str(t, "target", "confirm")
    if target not in TARGETS:
        raise ValueError(f"confirm.target: must be one of {sorted(TARGETS)}")
    names = t["side_effects"]
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(n, str) and n for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("confirm.side_effects: must be a list of distinct names")
    q = _num({"q": t.get("q", 0.95)}, "q", "confirm")
    side_raw = t.get("side", {})
    side_t = _open_table(side_raw, "confirm.side")
    if bad := sorted(set(side_t) - set(names)):
        raise ValueError(f"confirm.side: {bad} are not in confirm.side_effects")
    side = {}
    for name, sub in side_t.items():
        w = f"confirm.side.{name}"
        s = _table(sub, w, ("base", "edit", "controls"))
        side[name] = SideFiles(
            _path(s, "base", w, root),
            _path(s, "edit", w, root),
            _path(s, "controls", w, root),
        )
    if missing := sorted(n for n in names if n != "accuracy" and n not in side):
        raise ValueError(
            f"confirm.side_effects {missing}: need a [confirm.side.NAME] table "
            "(only 'accuracy' uses the confirm records)"
        )
    return ConfirmConfig(
        _path(t, "base", "confirm", root),
        _path(t, "edit", "confirm", root),
        _path(t, "controls", "confirm", root),
        q,
        target,
        tuple(names),
        side,
    )


def _test(raw: object, root: Path) -> TestFiles:
    t = _table(raw, "test", ("base", "edit", "seal_dir", "adapter", "floor"))
    return TestFiles(
        *(
            _path(t, k, "test", root)
            for k in ("base", "edit", "seal_dir", "adapter", "floor")
        )
    )


def _suites(raw: object, root: Path) -> dict[str, SuiteConfig]:
    t = _open_table(raw, "suites")
    out = {}
    for name, sub in t.items():
        w = f"suites.{name}"
        s = _table(sub, w, ("base", "edit", "margin", "source"))
        out[name] = SuiteConfig(
            _path(s, "base", w, root),
            _path(s, "edit", w, root),
            _num(s, "margin", w),
            _str(s, "source", w),
        )
    return out


def load_gate_config(path: str | Path) -> GateConfig:
    """Read GATE.toml. ValueError on bad TOML, an unknown, missing or mistyped key.

    Margins are only checked for being numbers here; `gate` holds them to [0, 0.02].
    """
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{path}: {e}") from None
    root = path.parent
    try:
        t = _table(
            raw,
            "GATE.toml",
            (
                "margin",
                "margin_source",
                "backend",
                "build",
                "screens",
                "run_record",
                "manifest",
                "confirm",
                "test",
                "suites",
            ),
            ("arms", "footprint", "boot", "seed", "arm_keys", "max_cap_hit"),
        )
        arms_t = _table(t.get("arms", {}), "arms", (), REPORT_ARMS)
        foot_t = t.get("footprint", {})
        _open_table(foot_t, "footprint")
        keys = t.get("arm_keys", [])
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise ValueError("arm_keys: must be a list of strings")
        cap = None
        if "max_cap_hit" in t:
            cap = _num(t, "max_cap_hit", "GATE.toml")
            if not 0 <= cap <= 1:
                raise ValueError("GATE.toml.max_cap_hit: must lie in [0, 1]")
        return GateConfig(
            margin=_num(t, "margin", "GATE.toml"),
            margin_source=_str(t, "margin_source", "GATE.toml"),
            backend=_str(t, "backend", "GATE.toml"),
            build=_str(t, "build", "GATE.toml"),
            screens=_path(t, "screens", "GATE.toml", root),
            run_record=_path(t, "run_record", "GATE.toml", root),
            manifest=_path(t, "manifest", "GATE.toml", root),
            confirm=_confirm(t["confirm"], root),
            test=_test(t["test"], root),
            suites=_suites(t["suites"], root),
            arms={k: root / _str(arms_t, k, "arms") for k in arms_t},
            footprint={k: _num(foot_t, k, "footprint") for k in foot_t},
            boot=_int({"boot": t.get("boot", 10000)}, "boot", "GATE.toml", 1),
            seed=_int({"seed": t.get("seed", 0)}, "seed", "GATE.toml", 0),
            arm_keys=tuple(keys),
            max_cap_hit=cap,
        )
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
