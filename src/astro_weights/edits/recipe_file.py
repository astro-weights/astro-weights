"""Rebuild an Edit from a small JSON recipe plus the .npy arrays it names."""

import copy
import hashlib
import io
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from astro_weights.edits.core import Edit, LowRankDelta
from astro_weights.edits.recipes import (
    column_shift,
    masked_rank1,
    project_out,
    rank1,
    scale_column,
    scale_head,
    zero_head,
)

FORMAT = 1


@dataclass(frozen=True)
class _Spec:
    fn: Callable[..., LowRankDelta]
    scalars: dict[str, type]  # name -> int or float
    arrays: tuple[str, ...]
    needs_w0: bool


# Names match the keyword arguments of the recipe functions.
_RECIPES: dict[str, _Spec] = {
    "rank1": _Spec(rank1, {}, ("b", "a"), False),
    "masked_rank1": _Spec(masked_rank1, {"alpha": float}, ("u", "mask"), True),
    "column_shift": _Spec(column_shift, {"n_in": int}, ("shift", "idx", "coef"), False),
    "project_out": _Spec(project_out, {"strength": float}, ("v",), True),
    "scale_column": _Spec(scale_column, {"j": int, "s": float}, (), True),
    "scale_head": _Spec(
        scale_head, {"head": int, "n_heads": int, "s": float}, (), True
    ),
    "zero_head": _Spec(zero_head, {"head": int, "n_heads": int}, (), True),
}


def _check_names(
    where: str, kind: str, got: object, want: tuple[str, ...]
) -> dict[str, Any]:
    if not isinstance(got, dict):
        raise ValueError(f"{where}: {kind} must be an object")  # noqa: TRY004
    for name in got:
        if name not in want:
            raise ValueError(f"{where}: unknown {kind[:-1]} {name!r}")
    for name in want:
        if name not in got:
            raise ValueError(f"{where}: missing {kind[:-1]} {name!r}")
    return got


def _plain_name(name: str) -> bool:
    """True for a bare file name: no directories, no absolute path, no `.`/`..`."""
    return name not in ("", ".", "..") and Path(name).name == name and "\\" not in name


def _load_array(directory: Path, where: str, key: str, name: str) -> tuple[bytes, Any]:
    """Read one .npy file confined to `directory`; any failure is a ValueError."""
    path = directory / name
    label = f"{where}: array {key!r} ({name})"
    try:
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("escapes the recipe directory")
        data = path.read_bytes()
        arr = np.load(io.BytesIO(data), allow_pickle=False)
        if not isinstance(arr, np.ndarray):
            close = getattr(arr, "close", None)
            if close is not None:
                close()
            raise ValueError("not a .npy array")  # noqa: TRY004
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise ValueError(f"{label}: {exc}") from exc
    return data, arr


def _check_step(i: int, step: object) -> None:
    if not isinstance(step, dict):
        raise ValueError(f"step {i}: must be an object")  # noqa: TRY004
    name = step.get("recipe")
    if not isinstance(name, str) or name not in _RECIPES:
        raise ValueError(f"step {i}: unknown recipe {name!r}")
    spec = _RECIPES[name]
    where = f"step {i} ({name})"
    module = step.get("module")
    if not isinstance(module, str) or not module:
        raise ValueError(f"{where}: module must be a non-empty string")
    params = _check_names(where, "params", step.get("params", {}), tuple(spec.scalars))
    for key, value in params.items():
        ok = (
            isinstance(value, int)
            if spec.scalars[key] is int
            else isinstance(value, int | float)
        )
        if not ok or isinstance(value, bool):
            raise ValueError(
                f"{where}: param {key!r} must be a {spec.scalars[key].__name__}"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{where}: param {key!r} must be finite")
    arrays = _check_names(where, "arrays", step.get("arrays", {}), spec.arrays)
    for key, value in arrays.items():
        if not isinstance(value, str) or not _plain_name(value):
            raise ValueError(
                f"{where}: array {key!r} must be a plain file name in the recipe directory"
            )


def load_recipe(path: Path | str) -> dict[str, Any]:
    """Read and validate a recipe file; arrays are not loaded."""
    try:
        recipe = json.loads(Path(path).read_text())
    except (OSError, ValueError, RecursionError) as exc:
        raise ValueError(f"cannot read recipe {str(path)!r}: {exc}") from exc
    if not isinstance(recipe, dict):
        raise ValueError("recipe must be a JSON object")  # noqa: TRY004
    if recipe.get("format") != FORMAT:
        raise ValueError(f"unsupported recipe format: {recipe.get('format')!r}")
    base = recipe.get("base_model")
    if not isinstance(base, str) or not base:
        raise ValueError("base_model must be a non-empty string")
    steps = recipe.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("steps must be a non-empty list")
    for i, step in enumerate(steps):
        _check_step(i, step)
    return recipe


def save_recipe(path: Path | str, recipe: dict[str, Any]) -> None:
    """Write a recipe as JSON with sorted keys and a trailing newline."""
    Path(path).write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")


def build_edit(
    recipe: dict[str, Any],
    weights: Callable[[str], np.ndarray],
    directory: Path | str,
) -> Edit:
    """Build the Edit a recipe describes; `weights(module)` returns that module's W0."""
    directory = Path(directory)
    hashes: dict[str, str] = {}
    deltas: list[LowRankDelta] = []
    seen: set[str] = set()
    for i, step in enumerate(recipe["steps"]):
        _check_step(i, step)
        module = step["module"]
        if module in seen:
            raise ValueError(f"step {i}: module {module!r} already has a step")
        seen.add(module)
        spec = _RECIPES[step["recipe"]]
        kwargs: dict[str, Any] = dict(step.get("params", {}))
        for key, name in step.get("arrays", {}).items():
            data, kwargs[key] = _load_array(
                directory, f"step {i} ({step['recipe']})", key, name
            )
            hashes[name] = hashlib.sha256(data).hexdigest()
        if spec.needs_w0:
            kwargs["W0"] = weights(module)
        deltas.append(spec.fn(module, **kwargs))
    meta: dict[str, object] = {
        "recipe": {**copy.deepcopy(recipe), "arrays_sha256": hashes}
    }
    return Edit.from_deltas(deltas, recipe["base_model"], meta=meta)
