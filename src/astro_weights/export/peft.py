"""Save and load an Edit as a standard PEFT LoRA adapter plus a hash manifest."""

import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

import numpy as np
from safetensors import SafetensorError, safe_open
from safetensors.numpy import save_file

from astro_weights.edits import Edit, LowRankDelta

WEIGHTS = "adapter_model.safetensors"
CONFIG = "adapter_config.json"
MANIFEST = "astro_manifest.json"
PREFIX = "base_model.model."
SUFFIX_A = ".lora_A.weight"
SUFFIX_B = ".lora_B.weight"
FLOAT_DTYPES = {"F32", "F16"}


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def _sha256(arr: np.ndarray) -> str:
    """Hash of a C-contiguous float32 array (callers guarantee the layout)."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


def save_peft(edit: Edit, out_dir: Path | str) -> None:
    """Write edit as a PEFT LoRA adapter (scaling 1) with an astro manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    for d in edit.deltas:
        tensors[f"{PREFIX}{d.module}{SUFFIX_A}"] = d.A
        tensors[f"{PREFIX}{d.module}{SUFFIX_B}"] = d.B
    save_file(tensors, str(out / WEIGHTS))
    _write_json(
        out / CONFIG,
        {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": edit.base_model,
            "r": edit.rank,
            "lora_alpha": edit.rank,
            "lora_dropout": 0.0,
            "bias": "none",
            "fan_in_fan_out": False,
            "target_modules": sorted(d.module for d in edit.deltas),
        },
    )
    _write_json(
        out / MANIFEST,
        {
            "format": 1,
            "astro_weights_version": version("astro-weights"),
            "base_model": edit.base_model,
            "base_sha256": edit.base_sha256,
            "meta": edit.meta,
            "factors": {k: _sha256(v) for k, v in tensors.items()},
        },
    )


def _load_object(path: Path) -> dict[str, object]:
    """Parse a JSON file that must hold an object; ValueError otherwise."""
    try:
        obj = json.loads(path.read_text())
    except RecursionError:
        raise ValueError(f"{path.name}: JSON nested too deeply") from None
    except ValueError as e:
        raise ValueError(f"{path.name}: invalid JSON ({e})") from None
    if not isinstance(obj, dict):
        raise ValueError(f"{path.name}: expected a JSON object")  # noqa: TRY004
    return obj


def _scaling(cfg: dict[str, object]) -> np.float32:
    r, alpha = cfg["r"], cfg["lora_alpha"]  # KeyError when absent
    if not isinstance(r, int) or isinstance(r, bool) or r < 1:
        raise ValueError(
            f"unsupported adapter config: r must be an int >= 1, got {r!r}"
        )
    if not isinstance(alpha, int | float) or isinstance(alpha, bool):
        raise ValueError(  # noqa: TRY004
            f"unsupported adapter config: lora_alpha must be a number, got {alpha!r}"
        )
    try:
        scale = alpha / r
    except OverflowError:
        scale = math.inf
    if not math.isfinite(scale) or not math.isfinite(np.float32(scale)):
        raise ValueError(
            f"unsupported adapter config: lora_alpha {alpha!r} is not finite"
        )
    return np.float32(scale)


def _check_config(cfg: dict[str, object]) -> np.float32:
    """Reject configs this reader cannot represent; return the scaling alpha / r."""
    if cfg.get("peft_type") != "LORA":
        raise ValueError(f"unsupported peft_type: {cfg.get('peft_type')!r}")
    for field in ("use_dora", "use_rslora", "fan_in_fan_out"):
        if cfg.get(field):
            raise ValueError(f"unsupported adapter config: {field} is true")
    for field in ("rank_pattern", "alpha_pattern", "modules_to_save"):
        if cfg.get(field):
            raise ValueError(f"unsupported adapter config: {field} is set")
    if cfg.get("bias") not in (None, "none"):
        raise ValueError(f"unsupported adapter config: bias is {cfg['bias']!r}")
    base = cfg.get("base_model_name_or_path")
    if base is not None and not isinstance(base, str):
        raise ValueError("adapter config: base_model_name_or_path must be a string")
    targets = cfg.get("target_modules")
    if targets is not None and not isinstance(targets, str):
        names = targets if isinstance(targets, list) else [None]
        if not all(isinstance(t, str) for t in names):
            raise ValueError(
                "adapter config: target_modules must be a string or list of strings"
            )
    return _scaling(cfg)


def _check_targets(cfg: dict[str, object], modules: list[str]) -> None:
    """Every module with tensors must be named by a list-form target_modules."""
    targets = cfg.get("target_modules")
    if not isinstance(targets, list):
        return  # regex string or absent: PEFT's own matching, not checked here
    for m in modules:
        if not any(m == t or m.endswith("." + t) for t in targets):
            raise ValueError(f"module {m} is not covered by target_modules")


def _check_manifest(manifest: dict[str, object], path: Path) -> None:
    ok = isinstance(manifest.get("base_model"), str)
    factors = manifest.get("factors")
    ok = ok and isinstance(factors, dict)
    ok = ok and all(isinstance(h, str) for h in manifest["factors"].values())  # type: ignore[attr-defined]
    if not ok:
        raise ValueError(
            f"{path}: manifest needs a string base_model and a factors map"
        )
    sha = manifest.get("base_sha256", {})
    if not isinstance(sha, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in sha.items()
    ):
        raise ValueError(f"{path}: base_sha256 must map strings to strings")
    if not isinstance(manifest.get("meta", {}), dict):
        raise ValueError(f"{path}: meta must be an object")  # noqa: TRY004


def read_peft(directory: Path | str) -> tuple[Edit, list[str]]:
    """Parse an adapter; ValueError on structural problems, hash problems returned."""
    directory = Path(directory)
    cfg = _load_object(directory / CONFIG)
    scaling = _check_config(cfg)
    r = cfg["r"]

    with safe_open(str(directory / WEIGHTS), framework="numpy") as f:
        keys = list(f.keys())
        dtypes = {k: f.get_slice(k).get_dtype() for k in keys}
        bad = {k: t for k, t in dtypes.items() if t not in FLOAT_DTYPES}
        if bad:
            hint = " (BF16 cannot be read by numpy; re-save the adapter as float32)"
            raise ValueError(
                f"unsupported tensor dtypes {bad}"
                f"{hint if 'BF16' in bad.values() else ''}"
            )
        # contiguous float32 is what _sha256 and the manifest hashes assume
        raw = {k: np.ascontiguousarray(f.get_tensor(k), dtype=np.float32) for k in keys}

    a_of: dict[str, np.ndarray] = {}
    b_of: dict[str, np.ndarray] = {}
    for key, arr in raw.items():
        if not key.startswith(PREFIX):
            raise ValueError(f"unexpected tensor key: {key}")
        name = key.removeprefix(PREFIX)
        if name.endswith(SUFFIX_A):
            a_of[name.removesuffix(SUFFIX_A)] = arr
        elif name.endswith(SUFFIX_B):
            b_of[name.removesuffix(SUFFIX_B)] = arr
        else:
            raise ValueError(f"unexpected tensor key: {key}")
    missing = sorted(set(a_of) ^ set(b_of))
    if missing:
        raise ValueError(f"modules missing lora_A or lora_B: {missing}")
    _check_targets(cfg, sorted(a_of))

    deltas = tuple(LowRankDelta(m, b_of[m] * scaling, a_of[m]) for m in sorted(a_of))
    if any(d.rank != r for d in deltas):
        raise ValueError(f"tensor rank does not match config r={r}")

    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        return Edit(deltas, cfg.get("base_model_name_or_path") or ""), []  # type: ignore[arg-type]
    manifest = _load_object(manifest_path)
    _check_manifest(manifest, manifest_path)
    problems = [
        f"hash mismatch or missing factor: {k}"
        for k, h in sorted(manifest["factors"].items())  # type: ignore[attr-defined]
        if k not in raw or _sha256(raw[k]) != h
    ]
    edit = Edit(
        deltas,
        manifest["base_model"],  # type: ignore[arg-type]
        base_sha256=manifest.get("base_sha256", {}),  # type: ignore[arg-type]
        meta=manifest.get("meta", {}),  # type: ignore[arg-type]
    )
    return edit, problems


def verify_peft(directory: Path | str) -> list[str]:
    """Human-readable problems with an adapter directory; empty when fine."""
    try:
        _, problems = read_peft(directory)
    except (ValueError, KeyError, OSError, SafetensorError) as e:
        return [str(e)]
    return problems


def load_peft(directory: Path | str) -> Edit:
    """Load a PEFT LoRA adapter as an Edit with scaling folded into B."""
    edit, problems = read_peft(directory)
    if problems:
        raise ValueError("; ".join(problems))
    return edit
