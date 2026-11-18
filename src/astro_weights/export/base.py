"""Read one tensor of a base model from safetensors files, without torch."""

import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

MAX_HEADER = 100_000_000
INDEX = "model.safetensors.index.json"
SINGLE = "model.safetensors"
ITEMSIZE = {"F32": 4, "F16": 2, "BF16": 2}


def _fail(file: Path, key: str, why: str) -> ValueError:
    return ValueError(f"{file}: tensor {key!r}: {why}")


def _locate(file: Path, key: str) -> tuple[str, tuple[int, ...], int, int]:
    """Return (dtype, shape, absolute offset, byte count) of one tensor, validated."""
    try:
        size = file.stat().st_size
        with file.open("rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                raise _fail(file, key, "truncated file")
            (n,) = struct.unpack("<Q", raw)
            if n > MAX_HEADER:
                raise _fail(file, key, f"header length {n} too large")
            blob = f.read(n)
    except OSError as e:
        raise _fail(file, key, f"cannot read file: {e}") from e
    if len(blob) < n:
        raise _fail(file, key, "truncated header")
    try:
        header = json.loads(blob)
    except (ValueError, RecursionError) as e:
        raise _fail(file, key, "invalid header JSON") from e
    entry = (
        header.get(key) if isinstance(header, dict) and key != "__metadata__" else None
    )
    if not isinstance(entry, dict):
        raise _fail(file, key, "key not found")
    dtype, shape, offsets = (
        entry.get("dtype"),
        entry.get("shape"),
        entry.get("data_offsets"),
    )
    if not isinstance(dtype, str) or dtype not in ITEMSIZE:
        raise _fail(file, key, f"unsupported dtype {dtype!r}")
    ints = isinstance(shape, list) and all(type(d) is int and d >= 0 for d in shape)
    if not (ints and isinstance(shape, list)):
        raise _fail(file, key, "invalid shape")
    if not (
        isinstance(offsets, list)
        and len(offsets) == 2
        and all(type(o) is int for o in offsets)
        and 0 <= offsets[0] <= offsets[1]
    ):
        raise _fail(file, key, "invalid data_offsets")
    start, end = offsets
    if end - start != math.prod(shape) * ITEMSIZE[dtype]:
        raise _fail(file, key, "byte count does not match shape")
    base = 8 + n
    if base + end > size:
        raise _fail(file, key, "truncated file")
    return dtype, tuple(shape), base + start, end - start


def _read_raw(file: Path, key: str) -> tuple[str, tuple[int, ...], bytes]:
    dtype, shape, offset, count = _locate(file, key)
    try:
        with file.open("rb") as f:
            f.seek(offset)
            data = f.read(count)
    except OSError as e:
        raise _fail(file, key, f"cannot read file: {e}") from e
    if len(data) != count:
        raise _fail(file, key, "truncated file")
    return dtype, shape, data


def read_tensor(file: Path | str, key: str) -> np.ndarray:
    """Read one F32, F16 or BF16 tensor as float32; ValueError on any problem."""
    file = Path(file)
    dtype, shape, data = _read_raw(file, key)
    if dtype == "BF16":
        wide = np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16
        arr = wide.view(np.float32)
    else:
        arr = np.frombuffer(data, dtype="<f4" if dtype == "F32" else "<f2")
    return np.ascontiguousarray(arr.astype(np.float32).reshape(shape))


def tensor_sha256(file: Path | str, key: str) -> str:
    """sha256 of the tensor's raw stored bytes."""
    return hashlib.sha256(_read_raw(Path(file), key)[2]).hexdigest()


def _shard(model_dir: Path, key: str) -> Path:
    index = model_dir / INDEX
    if not index.exists():
        return model_dir / SINGLE
    try:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    except (OSError, ValueError, RecursionError, KeyError, TypeError) as e:
        raise ValueError(f"{index}: invalid index file: {e!r}") from e
    if not isinstance(weight_map, dict) or key not in weight_map:
        raise ValueError(f"{index}: key {key!r} not in weight_map")
    name = weight_map[key]
    if not isinstance(name, str):
        raise ValueError(f"{index}: bad shard name for {key!r}")  # noqa: TRY004
    if name in ("", ".", "..") or Path(name).name != name or "\\" in name:
        raise ValueError(f"{index}: unsafe shard name {name!r} for {key!r}")
    return model_dir / name


def read_base_weight(model_dir: Path | str, key: str) -> np.ndarray:
    """Read one base-model weight as float32, following the shard index if present."""
    return read_tensor(_shard(Path(model_dir), key), key)
