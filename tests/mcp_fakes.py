"""Tiny fake HF model directory for MCP tests: numpy + safetensors only."""

import json
import struct
from pathlib import Path

import numpy as np

HIDDEN = 32  # a multiple of 32, so Q8_0 can quantize every weight
INTER = 64
KV = 16
VOCAB = 12
LAYERS = 2


def _bf16_bytes(a: np.ndarray) -> bytes:
    """float32 -> bf16 bytes by truncation (fake weights are made bf16-exact first)."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return bytes((u >> 16).astype("<u2").tobytes())


def _bf16_exact(a: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return (u & np.uint32(0xFFFF0000)).view(np.float32)


def _raw(tensors: dict[str, tuple[str, np.ndarray]], path: Path) -> None:
    """Write a safetensors file by hand (numpy has no bfloat16)."""
    header: dict[str, object] = {}
    blobs: list[bytes] = []
    pos = 0
    for name, (dtype, arr) in tensors.items():
        if dtype == "BF16":
            blob = _bf16_bytes(arr)
        elif dtype == "F16":
            blob = arr.astype("<f2").tobytes()
        else:
            blob = arr.astype("<f4").tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(arr.shape),
            "data_offsets": [pos, pos + len(blob)],
        }
        blobs.append(blob)
        pos += len(blob)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def fake_weights(layout: str = "model", seed: int = 0) -> dict[str, np.ndarray]:
    """name -> float32 array (bf16-exact). `layout` is the key prefix before `layers`."""
    rng = np.random.default_rng(seed)
    pre = "model" if layout == "model" else "model.language_model"

    def w(*shape: int) -> np.ndarray:
        return _bf16_exact(rng.standard_normal(shape).astype(np.float32))

    out: dict[str, np.ndarray] = {f"{pre}.embed_tokens.weight": w(VOCAB, HIDDEN)}
    for n in range(LAYERS):
        p = f"{pre}.layers.{n}"
        out[f"{p}.self_attn.q_proj.weight"] = w(HIDDEN, HIDDEN)
        out[f"{p}.self_attn.k_proj.weight"] = w(KV, HIDDEN)
        out[f"{p}.self_attn.o_proj.weight"] = w(HIDDEN, HIDDEN)
        out[f"{p}.mlp.down_proj.weight"] = w(HIDDEN, INTER)
        out[f"{p}.input_layernorm.weight"] = w(HIDDEN)
    out[f"{pre}.norm.weight"] = w(HIDDEN)
    out["lm_head.weight"] = w(VOCAB, HIDDEN)
    return out


def fake_model_dir(
    path: Path,
    *,
    dtype: str = "bf16",
    sharded: bool = False,
    layout: str = "model",
) -> Path:
    """Write a 2-layer model into `path` (created) and return it.

    dtype is "bf16", "f16" or "f32". sharded writes two shards plus an index.
    """
    path.mkdir(parents=True, exist_ok=True)
    st = {"bf16": "BF16", "f16": "F16", "f32": "F32"}[dtype]
    weights = fake_weights(layout)
    tensors = {k: (st, v) for k, v in weights.items()}
    if not sharded:
        _raw(tensors, path / "model.safetensors")
        return path
    names = sorted(tensors)
    shards = {"model-00001-of-00002.safetensors": names[::2]}
    shards["model-00002-of-00002.safetensors"] = names[1::2]
    weight_map: dict[str, str] = {}
    for shard, keys in shards.items():
        _raw({k: tensors[k] for k in keys}, path / shard)
        weight_map.update({k: shard for k in keys})
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    return path
