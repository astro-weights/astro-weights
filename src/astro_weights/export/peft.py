"""Save and load an Edit as a standard PEFT LoRA adapter plus a hash manifest."""

import hashlib
import json
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


def _check_config(cfg: dict[str, object]) -> None:
    if cfg.get("peft_type") != "LORA":
        raise ValueError(f"unsupported peft_type: {cfg.get('peft_type')!r}")
    for field in ("use_dora", "use_rslora", "fan_in_fan_out"):
        if cfg.get(field):
            raise ValueError(f"unsupported adapter config: {field} is true")


def _check_manifest(manifest: object, path: Path) -> None:
    ok = (
        isinstance(manifest, dict)
        and isinstance(manifest.get("base_model"), str)
        and isinstance(manifest.get("factors"), dict)
        and all(isinstance(h, str) for h in manifest["factors"].values())
    )
    if not ok:
        raise ValueError(
            f"{path}: manifest needs a string base_model and a factors map"
        )


def read_peft(directory: Path | str) -> tuple[Edit, list[str]]:
    """Parse an adapter; ValueError on structural problems, hash problems returned."""
    directory = Path(directory)
    cfg = json.loads((directory / CONFIG).read_text())
    _check_config(cfg)
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scaling = np.float32(alpha / r)

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

    deltas = tuple(LowRankDelta(m, b_of[m] * scaling, a_of[m]) for m in sorted(a_of))
    if any(d.rank != r for d in deltas):
        raise ValueError(f"tensor rank does not match config r={r}")

    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        return Edit(deltas, cfg.get("base_model_name_or_path") or ""), []
    manifest = json.loads(manifest_path.read_text())
    _check_manifest(manifest, manifest_path)
    problems = [
        f"hash mismatch or missing factor: {k}"
        for k, h in sorted(manifest["factors"].items())
        if k not in raw or _sha256(raw[k]) != h
    ]
    edit = Edit(
        deltas,
        manifest["base_model"],
        base_sha256=manifest.get("base_sha256", {}),
        meta=manifest.get("meta", {}),
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
