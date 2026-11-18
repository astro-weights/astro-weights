"""Save and load merged (edited) weight tensors with a hash manifest."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import SafetensorError
from safetensors.numpy import load_file, save_file

from astro_weights.export.peft import _sha256, _write_json

TENSORS = "merged_tensors.safetensors"
MANIFEST = "merged_manifest.json"
SUFFIX = ".weight"
DTYPES = (np.dtype(np.float32), np.dtype(np.float16))


def save_merged_tensors(
    tensors: dict[str, np.ndarray],
    path: Path | str,
    base_model: str,
    edit_manifest_sha256: str | None = None,
) -> None:
    """Write merged weights keyed by module name; stored as `<module>.weight`.

    Accepts float32 or float16 arrays only. bf16 is not a numpy dtype: callers
    pass `tensor.float().numpy()`. `edit_manifest_sha256` links the adapter manifest.
    """
    arrays: dict[str, np.ndarray] = {}
    for module, arr in tensors.items():
        if arr.dtype not in DTYPES:
            raise ValueError(f"{module}: dtype {arr.dtype} unsupported, use float32/16")
        arrays[module + SUFFIX] = np.ascontiguousarray(arr)
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    save_file(arrays, str(out / TENSORS))
    _write_json(
        out / MANIFEST,
        {
            "format": 1,
            "base_model": base_model,
            "edit_manifest_sha256": edit_manifest_sha256,
            "tensors": {
                k: {
                    "sha256": _sha256(v),
                    "shape": list(v.shape),
                    "dtype": str(v.dtype),
                }
                for k, v in arrays.items()
            },
        },
    )


def _read_manifest(directory: Path) -> dict[str, dict[str, Any]]:
    """Parse the manifest's `tensors` table; any malformed shape is a ValueError."""
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    tensors = manifest.get("tensors") if isinstance(manifest, dict) else None
    if not isinstance(tensors, dict):
        raise ValueError("manifest must be an object with a `tensors` object")  # noqa: TRY004 (spec: ValueError)
    for key, entry in tensors.items():
        ok = (
            isinstance(entry, dict)
            and isinstance(entry.get("sha256"), str)
            and isinstance(entry.get("dtype"), str)
            and isinstance(entry.get("shape"), list)
            and all(
                isinstance(n, int) and not isinstance(n, bool) for n in entry["shape"]
            )
        )
        if not ok:
            raise ValueError(
                f"manifest entry {key!r} needs str sha256/dtype, int list shape"
            )
    return tensors


def load_merged_tensors(path: Path | str) -> dict[str, np.ndarray]:
    """Load merged weights keyed by module name, checking sha256, shape and dtype.

    The manifest sits next to the tensors, so a matching hash only proves the
    two files agree with each other. It is not a signature and does not show
    the weights are authentic.

    Raises `ValueError` on a missing file, unreadable weights, a malformed
    manifest, or any mismatch.
    """
    directory = Path(path)
    try:
        expected = _read_manifest(directory)
        arrays = load_file(str(directory / TENSORS))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SafetensorError) as e:
        raise ValueError(f"cannot read merged tensors in {directory}: {e}") from e
    bad = sorted(
        k
        for k in expected.keys() | arrays.keys()
        if k not in arrays
        or k not in expected
        or _sha256(np.ascontiguousarray(arrays[k])) != expected[k]["sha256"]
        or list(arrays[k].shape) != expected[k]["shape"]
        or str(arrays[k].dtype) != expected[k]["dtype"]
    )
    if bad:
        raise ValueError(
            f"hash mismatch (or wrong shape/dtype), missing or unexpected tensors: {bad}"
        )
    return {k.removesuffix(SUFFIX): v for k, v in arrays.items()}
