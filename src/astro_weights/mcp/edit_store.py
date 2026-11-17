"""Server-side store for weight edits, and export of the stored edits.

Agent code never writes edits itself. It sends a full edited weight, and
`EditStore.save` checks it against the base weight and writes it. Everything on
disk lives under `run_dir/edits`: one float32 safetensors file per edited tensor
and `manifest.json` with `{name: {sha256, rel_frobenius, shape}}`. Stored values
are already rounded to the base tensor's dtype. `validate` re-checks every
stored edit, so a changed file or manifest is an error, never a silent drop.
`export_edits` turns the stored edits into a PEFT adapter and, optionally, a
spliced GGUF.
"""

import hashlib
import json
import re
import secrets
import threading
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import SafetensorError, safe_open
from safetensors.numpy import load_file, save_file

from astro_weights.edits import Edit, LowRankDelta, from_dense
from astro_weights.export.base import (
    INDEX,
    SINGLE,
    _locate,
    _shard,
    read_base_weight,
    tensor_sha256,
)
from astro_weights.export.peft import save_peft

MAX_REL_CHANGE = 0.5
MAX_SCALAR_RESIDUAL = 1e-6
# bf16 rounding alone leaves a residual of ~1.6e-3 on a true scalar multiple
MAX_STORED_SCALAR_RESIDUAL = 3e-3

_NAME = re.compile(r"[A-Za-z0-9_.]{1,256}")
_EDITABLE = re.compile(r"model\.(?:language_model\.)?layers\.\d+\..+\.weight")
_EXCLUDED = ("norm", "embed", "lm_head", "mtp", "nextn")
_FLOAT_KINDS = (np.float16, np.float32, np.float64)


def round_bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 values to the nearest bfloat16 (ties to even), as float32."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    rounded = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & (
        np.uint32(0xFFFF0000)
    )
    return np.where(np.isnan(x), x, rounded.view(np.float32)).astype(np.float32)


def _round_to(dtype: str, x: np.ndarray) -> np.ndarray:
    """Round float32 values to the stored dtype of a base tensor (F32 is a no-op)."""
    if dtype == "BF16":
        return round_bf16(x)
    if dtype == "F16":
        with np.errstate(over="ignore"):
            return x.astype(np.float16).astype(np.float32)
    return x


def scalar_fit(w: np.ndarray, w0: np.ndarray) -> tuple[float, float]:
    """(s, relative residual) of the least-squares fit w ~ s * w0, in float64."""
    a = np.asarray(w, dtype=np.float64).ravel()
    b = np.asarray(w0, dtype=np.float64).ravel()
    s = float(a @ b) / max(float(b @ b), 1e-300)
    res = float(np.linalg.norm(a - s * b)) / max(float(np.linalg.norm(a)), 1e-300)
    return s, res


def rel_frobenius(w: np.ndarray, w0: np.ndarray) -> float:
    """|w - w0|_F / |w0|_F in float64."""
    a = np.asarray(w, dtype=np.float64)
    b = np.asarray(w0, dtype=np.float64)
    return float(np.linalg.norm(a - b)) / max(float(np.linalg.norm(b)), 1e-300)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tmp(path: Path) -> Path:
    return path.with_name(f"{path.name}.{secrets.token_hex(4)}.tmp")


def _candidate_keys(model_dir: Path) -> list[str]:
    index = model_dir / INDEX
    if index.exists():
        try:
            weight_map = json.loads(index.read_text())["weight_map"]
            return [k for k in weight_map if isinstance(k, str)]
        except (OSError, ValueError, RecursionError, KeyError, TypeError) as e:
            raise ValueError(f"{index}: invalid index file: {e!r}") from e
    try:
        with safe_open(model_dir / SINGLE, framework="numpy") as f:
            return list(f.keys())
    except (OSError, SafetensorError) as e:
        raise ValueError(f"{model_dir / SINGLE}: cannot read header: {e}") from e


def _editable_name(key: str) -> bool:
    low = key.lower()
    return (
        _NAME.fullmatch(key) is not None
        and _EDITABLE.fullmatch(key) is not None
        and not any(x in low for x in _EXCLUDED)
    )


class EditStore:
    """Validated dense edits of a model's linear weights, kept under `run_dir/edits`."""

    def __init__(self, run_dir: Path, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self.dir = Path(run_dir) / "edits"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dir / "manifest.json"
        self._lock = threading.RLock()
        info: dict[str, tuple[str, tuple[int, int]]] = {}
        for key in sorted(_candidate_keys(self.model_dir)):
            if not _editable_name(key):
                continue
            dtype, shape, _, _ = _locate(_shard(self.model_dir, key), key)
            if len(shape) == 2:
                info[key] = (dtype, (shape[0], shape[1]))
        self._info = info
        self.names: tuple[str, ...] = tuple(sorted(info))

    # reading

    def _known(self, name: object) -> str:
        if not isinstance(name, str) or name not in self._info:
            raise ValueError(f"{str(name)[:80]!r} is not an editable tensor")
        return name

    def shape(self, name: str) -> tuple[int, int]:
        """Shape of the base tensor, from the cached safetensors header."""
        return self._info[self._known(name)][1]

    def original(self, name: str) -> np.ndarray:
        """The base weight as float32."""
        return read_base_weight(self.model_dir, self._known(name))

    def _path(self, name: str) -> Path:
        return self.dir / f"{self._known(name)}.safetensors"

    def _manifest(self) -> dict[str, dict[str, Any]]:
        if not self.manifest_path.exists():
            return {}
        try:
            data = json.loads(self.manifest_path.read_text())
        except (OSError, ValueError, RecursionError) as e:
            raise ValueError(f"{self.manifest_path}: unreadable manifest: {e}") from e
        ok = isinstance(data, dict) and all(
            isinstance(k, str)
            and isinstance(v, dict)
            and isinstance(v.get("sha256"), str)
            for k, v in data.items()
        )
        if not ok:
            raise ValueError(f"{self.manifest_path}: malformed manifest")
        return data  # type: ignore[no-any-return]

    def _write_manifest(self, manifest: dict[str, dict[str, Any]]) -> None:
        tmp = _tmp(self.manifest_path)
        try:
            tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
            tmp.replace(self.manifest_path)
        finally:
            tmp.unlink(missing_ok=True)

    def _read_edit(self, name: str, rec: dict[str, Any]) -> np.ndarray:
        """Stored edit as float32; sha, key, dtype and shape are checked."""
        path = self._path(name)
        try:
            sha = _sha256_file(path)
        except OSError as e:
            raise ValueError(f"{name}: edit file unreadable: {e}") from e
        if sha != rec["sha256"]:
            raise ValueError(f"{name}: edit file sha256 does not match the manifest")
        try:
            tensors = load_file(str(path))
        except (OSError, SafetensorError, ValueError) as e:
            raise ValueError(f"{name}: edit file is corrupt: {e}") from e
        w = tensors.get(name)
        if set(tensors) != {name} or w is None or w.dtype != np.float32:
            raise ValueError(f"{name}: edit file must hold one float32 tensor {name!r}")
        if w.shape != self.shape(name):
            raise ValueError(
                f"{name}: stored shape {w.shape} != base {self.shape(name)}"
            )
        return w

    def load(self, name: str) -> np.ndarray:
        """The stored edit if there is one, else the base weight; float32."""
        self._known(name)
        with self._lock:
            rec = self._manifest().get(name)
            return self.original(name) if rec is None else self._read_edit(name, rec)

    def edited(self) -> list[str]:
        """Names with a stored edit, sorted."""
        with self._lock:
            return sorted(self._manifest())

    # rules

    def _check_stored(self, name: str, w: np.ndarray, w0: np.ndarray) -> float:
        """Rules on the rounded tensor. Returns rel_frobenius."""
        if not np.isfinite(w).all():
            raise ValueError(
                f"{name}: non-finite values after rounding to the base dtype"
            )
        rel = rel_frobenius(w, w0)
        if rel == 0:
            raise ValueError(
                f"{name}: edit is below resolution, the stored tensor equals the original"
            )
        s, res = scalar_fit(w, w0)
        # |s - 1| > res keeps genuine small edits, whose fit stays at s ~ 1
        if res < MAX_STORED_SCALAR_RESIDUAL and abs(s - 1) > res:
            raise ValueError(
                f"{name}: edit is a scalar multiple of the original "
                f"(s={s:.4f}, relative residual {res:.2e})"
            )
        if rel > MAX_REL_CHANGE:
            raise ValueError(
                f"{name}: relative Frobenius change {rel:.4f} is above the cap "
                f"{MAX_REL_CHANGE}"
            )
        return rel

    def save(self, name: str, w: np.ndarray) -> dict[str, Any]:
        """Check and store a full edited weight; ValueError and nothing written on any fault."""
        self._known(name)
        dtype, shape = self._info[name]
        if not isinstance(w, np.ndarray) or w.dtype.type not in _FLOAT_KINDS:
            raise ValueError(f"{name}: edit must be a float16/32/64 numpy array")
        if w.shape != shape:
            raise ValueError(f"{name}: shape {w.shape} does not match original {shape}")
        if not np.isfinite(w).all():
            raise ValueError(f"{name}: non-finite values")
        w0 = self.original(name)
        # before rounding: rounding noise would hide a scalar multiple from a 1e-6 limit
        _, res = scalar_fit(w, w0)
        if res < MAX_SCALAR_RESIDUAL:
            raise ValueError(
                f"{name}: edit is a scalar multiple of the original "
                f"(relative residual {res:.2e})"
            )
        with np.errstate(over="ignore"):
            narrow = w.astype(np.float32)
        stored = _round_to(dtype, narrow)
        rel = self._check_stored(name, stored, w0)
        with self._lock:
            path = self._path(name)
            tmp = _tmp(path)
            try:
                save_file({name: np.ascontiguousarray(stored)}, str(tmp))
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)
            manifest = self._manifest()
            manifest[name] = {
                "sha256": _sha256_file(path),
                "rel_frobenius": rel,
                "shape": list(shape),
            }
            self._write_manifest(manifest)
        return {"name": name, "rel_frobenius": rel}

    def drop(self, name: str) -> None:
        """Remove a stored edit; ValueError if there is none."""
        self._known(name)
        with self._lock:
            manifest = self._manifest()
            if name not in manifest:
                raise ValueError(f"{name!r} has no edit")
            del manifest[name]
            self._write_manifest(manifest)
            self._path(name).unlink(missing_ok=True)

    def validate(self) -> dict[str, dict[str, Any]]:
        """Re-check every stored edit (file sha, then the save rules); ValueError on any fault."""
        with self._lock:
            manifest = self._manifest()
            out: dict[str, dict[str, Any]] = {}
            for name, rec in manifest.items():
                if name not in self._info:
                    raise ValueError(
                        f"manifest lists non-editable tensor {name[:80]!r}"
                    )
                w = self._read_edit(name, rec)
                if not np.array_equal(_round_to(self._info[name][0], w), w):
                    raise ValueError(f"{name}: stored values are not in the base dtype")
                rel = self._check_stored(name, w, self.original(name))
                out[name] = {**rec, "rel_frobenius": rel}
            return out


def export_edits(
    store: EditStore,
    out_dir: Path,
    *,
    rank: int,
    max_rel_residual: float,
    base_model: str,
    base_gguf: Path | None = None,
    gguf_out_type: str = "q8_0",
) -> dict[str, Any]:
    """Write the stored edits as a PEFT adapter, and a spliced GGUF if `base_gguf` is given.

    Each edit becomes dW = stored - original, cut to `rank` (smaller where a
    tensor has fewer dimensions) by truncated SVD; the cut must keep the
    relative residual within `max_rel_residual`. Files: `out_dir/adapter` and
    `out_dir/model.gguf`. A module the GGUF export cannot map raises before
    anything is written. If the splice fails after the adapter is written, the
    adapter stays.
    """
    store.validate()
    names = store.edited()
    if not names:
        raise ValueError("no edits to export")
    deltas: list[LowRankDelta] = []
    residuals: dict[str, float] = {}
    shas: dict[str, str] = {}
    originals: dict[str, np.ndarray] = {}
    for name in names:
        w0 = store.original(name)
        originals[name] = w0
        module = name.removesuffix(".weight")
        r = min(rank, *store.shape(name))
        delta, rel = from_dense(module, store.load(name) - w0, r, max_rel_residual)
        deltas.append(delta)
        residuals[name] = rel
        shas[name] = tensor_sha256(_shard(store.model_dir, name), name)
    edit = Edit.from_deltas(
        deltas,
        base_model,
        shas,
        {"rank_requested": rank, "residuals": residuals},
    )
    gguf_names: dict[str, str] = {}
    if base_gguf is not None:
        from astro_weights.export.gguf import gguf_tensor_name

        gguf_names = {d.module: gguf_tensor_name(d.module) for d in edit.deltas}
    out = Path(out_dir)
    adapter = out / "adapter"
    save_peft(edit, adapter, overwrite=True)
    gguf_path: Path | None = None
    if base_gguf is not None:
        from astro_weights.export.gguf import splice_gguf, verify_splice

        merged = {
            gguf_names[d.module]: originals[d.module + ".weight"] + d.dense()
            for d in edit.deltas
        }
        gguf_path = out / "model.gguf"
        splice_gguf(base_gguf, gguf_path, merged, gguf_out_type)
        problems = verify_splice(base_gguf, gguf_path, list(merged))
        if problems:
            gguf_path.unlink(missing_ok=True)
            raise ValueError(f"GGUF splice failed verification: {problems}")
    return {
        "adapter": adapter,
        "gguf": gguf_path,
        "residuals": residuals,
        "rank": edit.rank,
    }
