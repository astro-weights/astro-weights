import hashlib
import json
import struct
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from astro_weights.export.base import read_base_weight, read_tensor, tensor_sha256


def write_raw(path: Path, header: Mapping[str, object], body: bytes) -> None:
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + body)


def bf16_values() -> np.ndarray:
    # low 16 bits zero, so bf16 -> float32 is exact
    bits = np.array([0x3F800000, 0xC0490000, 0x00000000, 0x43480000], dtype=np.uint32)
    return bits.view(np.float32).reshape(2, 2)


def write_bf16(path: Path, key: str, arr: np.ndarray) -> bytes:
    raw = (arr.view(np.uint32) >> 16).astype("<u2").tobytes()
    meta = {"dtype": "BF16", "shape": list(arr.shape), "data_offsets": [0, len(raw)]}
    write_raw(path, {"__metadata__": {"format": "pt"}, key: meta}, raw)
    return raw


def test_f32_f16_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 4)).astype(np.float32)
    b = rng.normal(size=(2, 5)).astype(np.float16)
    save_file({"a": a, "b": b}, str(tmp_path / "f.safetensors"))
    ra = read_tensor(tmp_path / "f.safetensors", "a")
    rb = read_tensor(tmp_path / "f.safetensors", "b")
    assert ra.dtype == np.float32
    assert rb.dtype == np.float32
    np.testing.assert_array_equal(ra, a)
    np.testing.assert_array_equal(rb, b.astype(np.float32))


def test_bf16_roundtrip_and_sha(tmp_path: Path) -> None:
    f = tmp_path / "bf.safetensors"
    arr = bf16_values()
    raw = write_bf16(f, "w", arr)
    out = read_tensor(f, "w")
    assert out.dtype == np.float32
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, arr)
    assert tensor_sha256(f, "w") == hashlib.sha256(raw).hexdigest()


def test_sharded_and_single(tmp_path: Path) -> None:
    one = np.ones((2, 2), dtype=np.float32)
    two = np.full((3,), 2.0, dtype=np.float32)
    save_file({"x": one}, str(tmp_path / "s1.safetensors"))
    save_file({"y": two}, str(tmp_path / "s2.safetensors"))
    index = {"weight_map": {"x": "s1.safetensors", "y": "s2.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    np.testing.assert_array_equal(read_base_weight(tmp_path, "y"), two)
    np.testing.assert_array_equal(read_base_weight(tmp_path, "x"), one)
    with pytest.raises(ValueError, match="nope"):
        read_base_weight(tmp_path, "nope")


def test_single_file_layout(tmp_path: Path) -> None:
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    save_file({"w": arr}, str(tmp_path / "model.safetensors"))
    np.testing.assert_array_equal(read_base_weight(tmp_path, "w"), arr)


def test_missing_files_and_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"model\.safetensors"):
        read_base_weight(tmp_path, "w")
    f = tmp_path / "f.safetensors"
    save_file({"w": np.zeros(2, dtype=np.float32)}, str(f))
    with pytest.raises(ValueError, match="'zz'"):
        read_tensor(f, "zz")
    with pytest.raises(ValueError, match="__metadata__"):
        read_tensor(f, "__metadata__")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "gone.safetensors"}})
    )
    with pytest.raises(ValueError, match=r"gone\.safetensors"):
        read_base_weight(tmp_path, "w")


def test_bad_files(tmp_path: Path) -> None:
    f = tmp_path / "bad.safetensors"
    f.write_bytes(b"abc")
    with pytest.raises(ValueError, match="truncated"):
        read_tensor(f, "w")
    f.write_bytes(struct.pack("<Q", 100_000_001))
    with pytest.raises(ValueError, match="too large"):
        read_tensor(f, "w")
    f.write_bytes(struct.pack("<Q", 50) + b"{}")
    with pytest.raises(ValueError, match="truncated header"):
        read_tensor(f, "w")
    f.write_bytes(struct.pack("<Q", 3) + b"not")
    with pytest.raises(ValueError, match="JSON"):
        read_tensor(f, "w")


def test_bad_entries(tmp_path: Path) -> None:
    f = tmp_path / "e.safetensors"
    body = bytes(16)

    def entry(**kw: object) -> dict[str, object]:
        base: dict[str, object] = {
            "dtype": "F32",
            "shape": [4],
            "data_offsets": [0, 16],
        }
        return {"w": {**base, **kw}}

    cases = {
        "unsupported dtype": entry(dtype="I64", shape=[2], data_offsets=[0, 16]),
        "byte count": entry(shape=[3]),
        "invalid shape": entry(shape=[-4]),
        "invalid data_offsets": entry(data_offsets=[8, 0]),
        "truncated file": entry(shape=[8], data_offsets=[0, 32]),
    }
    for match, header in cases.items():
        write_raw(f, header, body)
        with pytest.raises(ValueError, match=match):
            read_tensor(f, "w")
        with pytest.raises(ValueError, match=match):
            tensor_sha256(f, "w")


def test_reads_only_needed_range(tmp_path: Path) -> None:
    f = tmp_path / "p.safetensors"
    good = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    header = {
        "good": {"dtype": "F32", "shape": [3], "data_offsets": [0, 12]},
        "huge": {
            "dtype": "F32",
            "shape": [10**6, 10**6],
            "data_offsets": [12, 12 + 4 * 10**12],
        },
    }
    write_raw(f, header, good.tobytes())
    np.testing.assert_array_equal(read_tensor(f, "good"), good)
    with pytest.raises(ValueError, match="truncated"):
        read_tensor(f, "huge")
