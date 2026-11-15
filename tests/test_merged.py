from pathlib import Path

import numpy as np
import pytest

from astro_weights.export import load_merged_tensors, save_merged_tensors


def tensors() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "model.layers.0.mlp.down_proj": rng.standard_normal((4, 6)).astype(np.float32),
        "model.layers.1.mlp.down_proj": rng.standard_normal((4, 6)).astype(np.float16),
    }


def test_round_trip_bit_exact(tmp_path: Path) -> None:
    src = tensors()
    save_merged_tensors(src, tmp_path, "tiny", edit_manifest_sha256="abc")
    got = load_merged_tensors(tmp_path)
    assert got.keys() == src.keys()
    for k, v in src.items():
        assert got[k].dtype == v.dtype
        assert got[k].tobytes() == v.tobytes()


def test_tampered_file_raises(tmp_path: Path) -> None:
    save_merged_tensors(tensors(), tmp_path, "tiny")
    path = tmp_path / "merged_tensors.safetensors"
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_merged_tensors(tmp_path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read"):
        load_merged_tensors(tmp_path)


def test_float64_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        save_merged_tensors({"m": np.zeros((2, 2))}, tmp_path, "tiny")


def _rewrite_file(tmp_path: Path, name: str, arr: np.ndarray) -> None:
    from safetensors.numpy import save_file

    save_file({name: arr}, str(tmp_path / "merged_tensors.safetensors"))


def test_reshaped_same_bytes_raises(tmp_path: Path) -> None:
    arr = np.arange(8, dtype=np.float32).reshape(2, 4)
    save_merged_tensors({"m": arr}, tmp_path, "tiny")
    _rewrite_file(tmp_path, "m.weight", arr.reshape(4, 2))
    with pytest.raises(ValueError, match="mismatch"):
        load_merged_tensors(tmp_path)


def test_reinterpreted_dtype_same_bytes_raises(tmp_path: Path) -> None:
    arr = np.arange(8, dtype=np.float32).reshape(2, 4)
    save_merged_tensors({"m": arr}, tmp_path, "tiny")
    _rewrite_file(tmp_path, "m.weight", arr.view(np.float16))
    with pytest.raises(ValueError, match="mismatch"):
        load_merged_tensors(tmp_path)


BAD_MANIFESTS = [
    "[]",
    "null",
    "{}",
    '{"tensors": []}',
    '{"tensors": {"m.weight": 3}}',
    '{"tensors": {"m.weight": {"shape": [2, 4], "dtype": "float32"}}}',
    '{"tensors": {"m.weight": {"sha256": 1, "shape": [2, 4], "dtype": "float32"}}}',
    '{"tensors": {"m.weight": {"sha256": "x", "shape": "2x4", "dtype": "float32"}}}',
    '{"tensors": {"m.weight": {"sha256": "x", "shape": [2, "4"], "dtype": "float32"}}}',
    '{"tensors": {"m.weight": {"sha256": "x", "shape": [2, 4]}}}',
]


@pytest.mark.parametrize("text", BAD_MANIFESTS)
def test_malformed_manifest_raises_value_error(tmp_path: Path, text: str) -> None:
    save_merged_tensors({"m": np.zeros((2, 4), dtype=np.float32)}, tmp_path, "tiny")
    (tmp_path / "merged_manifest.json").write_text(text)
    with pytest.raises(ValueError, match=r"manifest|cannot read"):
        load_merged_tensors(tmp_path)


def test_non_utf8_manifest_raises_value_error(tmp_path: Path) -> None:
    save_merged_tensors({"m": np.zeros((2, 4), dtype=np.float32)}, tmp_path, "tiny")
    (tmp_path / "merged_manifest.json").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ValueError, match="cannot read"):
        load_merged_tensors(tmp_path)
