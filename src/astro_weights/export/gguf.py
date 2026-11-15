"""GGUF export: llama.cpp LoRA adapter files and tensor splice into a base GGUF.

Needs the optional `gguf` extra (`pip install astro-weights[gguf]`).

Orientation: GGUF stores dims reversed relative to numpy/torch. `GGUFReader`
`tensor.shape` is GGUF order (ne0 first); `tensor.data` is numpy order, i.e.
`reversed(tensor.shape)`. Example, Qwen3.5-9B-Q4_K_M `blk.15.ffn_down.weight`:
reader shape [12288, 4096] ([intermediate, hidden]) -> numpy (4096, 12288) =
[hidden, intermediate], the torch nn.Linear layout of HF `mlp.down_proj.weight`
([out, in]). So an HF [out, in] array is written and spliced as is, no transpose.
"""

import random
import re
import secrets
import struct
from collections.abc import Mapping
from pathlib import Path

import gguf
import numpy as np
from gguf import GGMLQuantizationType as Q
from gguf import GGUFValueType as VT
from gguf.quants import dequantize, quantize

from astro_weights.edits import Edit

SUFFIXES = {
    "mlp.down_proj": "ffn_down",
    "self_attn.o_proj": "attn_output",
}
SUFFIX_A = ".lora_a"
SUFFIX_B = ".lora_b"
OUT_TYPES = {"q8_0": Q.Q8_0, "f16": Q.F16, "bf16": Q.BF16}
_MODULE = re.compile(r"(?:^|\.)layers\.(\d+)\.(mlp\.down_proj|self_attn\.o_proj)$")
_BLK = re.compile(r"blk\.(\d+)\.")
_SAMPLE = 20
_ADAPTER_TYPES = (Q.F32, Q.F16)


def _open(path: Path | str) -> gguf.GGUFReader:
    """GGUFReader that raises ValueError naming the file on any read failure."""
    try:
        return gguf.GGUFReader(path)
    except (IndexError, ValueError, struct.error, OSError) as e:
        raise ValueError(f"cannot read GGUF file {str(path)!r}: {e}") from e


def _make_tmp(out: Path) -> Path:
    """Create a fresh, uniquely named temp file next to `out`; never reuses one."""
    tmp = out.with_name(f"{out.name}.{secrets.token_hex(4)}.tmp")
    tmp.open("xb").close()
    return tmp


def gguf_tensor_name(module: str) -> str:
    """Map an HF module name to its GGUF base tensor name."""
    # anchored at the end, so the last "layers.<N>" wins
    m = _MODULE.search(module)
    if m is None:
        raise ValueError(
            f"unsupported module {module!r}; expected 'layers.<N>.<suffix>' with "
            f"suffix one of {sorted(SUFFIXES)}"
        )
    return f"blk.{m.group(1)}.{SUFFIXES[m.group(2)]}.weight"


def save_lora_gguf(
    edit: Edit, path: Path | str, architecture: str, *, overwrite: bool = False
) -> None:
    """Write edit as a llama.cpp LoRA adapter GGUF (alpha = rank, so scale 1)."""
    out = Path(path)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass overwrite=True to replace it")
    names = {d.module: gguf_tensor_name(d.module) for d in edit.deltas}
    if len(set(names.values())) != len(names):
        raise ValueError("two modules map to the same GGUF tensor")
    tmp = _make_tmp(out)
    try:
        writer = gguf.GGUFWriter(tmp, architecture)
        writer.add_type(gguf.GGUFType.ADAPTER)
        writer.add_string("adapter.type", "lora")
        writer.add_float32("adapter.lora.alpha", float(edit.rank))
        for d in edit.deltas:
            base = names[d.module]
            writer.add_tensor(base + SUFFIX_A, d.A)
            writer.add_tensor(base + SUFFIX_B, d.B)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_lora_gguf(
    path: Path | str,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    """Read an adapter GGUF: ({base tensor name: (B, A)}, metadata).

    Only F32 and F16 tensors are accepted. B and A are the raw factors; the
    LoRA scale is not applied: delta = (meta["alpha"] / rank) * B @ A, with
    rank = A.shape[0].
    """
    reader = _open(path)
    meta: dict[str, object] = {}
    for key, name in (
        ("general.architecture", "architecture"),
        ("general.type", "type"),
        ("adapter.type", "adapter_type"),
    ):
        field = reader.get_field(key)
        if field is not None:
            meta[name] = str(field.contents())
    alpha = reader.get_field("adapter.lora.alpha")
    if alpha is not None:
        meta["alpha"] = float(alpha.contents())
    a_of: dict[str, np.ndarray] = {}
    b_of: dict[str, np.ndarray] = {}
    for t in reader.tensors:
        if t.tensor_type not in _ADAPTER_TYPES:
            raise ValueError(
                f"{t.name}: adapter tensor type {t.tensor_type.name} not supported; "
                "expected F32 or F16"
            )
        if t.name.endswith(SUFFIX_A):
            target, base = a_of, t.name.removesuffix(SUFFIX_A)
        elif t.name.endswith(SUFFIX_B):
            target, base = b_of, t.name.removesuffix(SUFFIX_B)
        else:
            raise ValueError(f"unexpected tensor in adapter: {t.name}")
        if base in target:
            raise ValueError(f"duplicate tensor in adapter: {t.name}")
        target[base] = np.array(t.data, dtype=np.float32)
    missing = sorted(set(a_of) ^ set(b_of))
    if missing:
        raise ValueError(f"tensors missing lora_a or lora_b: {missing}")
    for n, a in a_of.items():
        b = b_of[n]
        if b.ndim != 2 or a.ndim != 2 or b.shape[1] != a.shape[0]:
            raise ValueError(
                f"{n}: lora_b {b.shape} and lora_a {a.shape} are not [out, r] x [r, in]"
            )
    return {n: (b_of[n], a_of[n]) for n in sorted(a_of)}, meta


def _np_shape(t: gguf.ReaderTensor) -> tuple[int, ...]:
    """numpy-order shape of a ReaderTensor (reversed GGUF dims)."""
    return tuple(int(d) for d in reversed(t.shape.tolist()))


def _dequant(t: gguf.ReaderTensor) -> np.ndarray:
    return np.asarray(dequantize(t.data, t.tensor_type)).reshape(_np_shape(t))


def _copy_kv(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for key, f in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        vt = f.types[0]
        if vt == VT.ARRAY:
            writer.add_key_value(key, f.contents(), vt, f.types[-1])
        else:
            writer.add_key_value(key, f.contents(), vt)
    a = reader.get_field("general.alignment")
    if a is not None:
        writer.data_alignment = int(a.contents())


def _check_not_mtp(reader: gguf.GGUFReader, arch: str, names: list[str]) -> None:
    # block_count includes the trailing nextn/MTP block(s) (27B: 65 with nextn 1)
    count = reader.get_field(f"{arch}.block_count")
    if count is None:
        return
    nextn = reader.get_field(f"{arch}.nextn_predict_layers")
    limit = int(count.contents()) - (int(nextn.contents()) if nextn else 0)
    for n in names:
        m = _BLK.match(n)
        if m and int(m.group(1)) >= limit:
            raise ValueError(
                f"{n}: blk.{m.group(1)} >= block_count - nextn_predict_layers = "
                f"{limit} (MTP/nextn block, never touched)"
            )


def _as_float_array(name: str, value: object) -> np.ndarray:
    """Replacement as a finite floating-point ndarray, else ValueError."""
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name}: replacement is not an array: {e}") from e
    if arr.dtype.kind != "f":
        raise ValueError(f"{name}: replacement must be floating point, got {arr.dtype}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: replacement contains NaN/Inf")
    return arr


def splice_gguf(
    base: Path | str,
    out: Path | str,
    replace: Mapping[str, np.ndarray],
    out_type: str = "q8_0",
) -> None:
    """Write a copy of `base` with the named tensors replaced, stored as `out_type`.

    Other tensors are copied as raw bytes with their original type. Replacement
    arrays are HF [out, in] weights; a transposed array is accepted and
    transposed back.
    """
    if out_type not in OUT_TYPES:
        raise ValueError(
            f"out_type must be one of {sorted(OUT_TYPES)}, got {out_type!r}"
        )
    qt = OUT_TYPES[out_type]
    base_p, out_p = Path(base), Path(out)
    if out_p.exists() and out_p.samefile(base_p):
        raise ValueError("out must not be the same file as base")
    reader = _open(base_p)
    field = reader.get_field("general.architecture")
    if field is None:
        raise ValueError(f"{str(base_p)!r} has no general.architecture")
    arch = str(field.contents())
    missing = sorted(set(replace) - {t.name for t in reader.tensors})
    if missing:
        raise ValueError(f"tensors not in base: {missing}")
    _check_not_mtp(reader, arch, list(replace))
    arrays = {n: _as_float_array(n, v) for n, v in replace.items()}

    tmp = _make_tmp(out_p)  # write aside, replace after checks
    try:
        writer = gguf.GGUFWriter(tmp, arch)
        _copy_kv(reader, writer)
        for t in reader.tensors:
            if t.name not in replace:
                writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
                continue
            arr = arrays[t.name]
            want = _np_shape(t)
            if arr.shape != want:
                if arr.T.shape != want:
                    raise ValueError(
                        f"{t.name}: replacement shape {arr.shape} vs expected {want}"
                    )
                arr = arr.T
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            writer.add_tensor(t.name, quantize(arr, qt), raw_dtype=qt)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
        n_out = len(_open(tmp).tensors)
        if n_out != len(reader.tensors):
            raise ValueError(f"tensor count mismatch {len(reader.tensors)} vs {n_out}")
        tmp.replace(out_p)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _rel_fro(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))


def verify_splice(
    base: Path | str, out: Path | str, names: list[str], seed: int = 0
) -> list[str]:
    """Problems found comparing a spliced GGUF to its base; empty means ok."""
    try:
        rb, ro = _open(base), _open(out)
    except ValueError as e:
        return [str(e)]
    tb = {t.name: t for t in rb.tensors}
    to = {t.name: t for t in ro.tensors}
    errs: list[str] = []
    if len(rb.fields) != len(ro.fields):
        errs.append(f"KV field count {len(rb.fields)} != {len(ro.fields)}")
    for k, f in rb.fields.items():
        g = ro.fields.get(k)
        if g is None:
            errs.append(f"KV missing in out: {k}")
        elif not k.startswith("GGUF.") and repr(f.contents()) != repr(g.contents()):
            errs.append(f"KV value differs: {k}")
    if list(tb) != list(to):
        errs.append(f"tensor list differs: {len(tb)} vs {len(to)} (or order)")
    for n in names:
        if n not in tb or n not in to:
            errs.append(f"{n}: not in both files")
            continue
        a, b = _dequant(tb[n]), _dequant(to[n])
        if a.shape != b.shape:
            errs.append(f"{n}: shape {a.shape} vs {b.shape}")
            continue
        d = _rel_fro(a, b)
        if not np.isfinite(d):
            errs.append(f"{n}: non-finite values in out or base")
        elif d == 0:
            errs.append(f"{n}: listed as replaced but identical to base")
    others = [n for n in tb if n not in names and n in to]
    for n in others:
        ta, tn = tb[n], to[n]
        if (ta.tensor_type, ta.shape.tolist(), ta.n_bytes) != (
            tn.tensor_type,
            tn.shape.tolist(),
            tn.n_bytes,
        ):
            errs.append(f"{n}: type/shape/size differs but not listed as replaced")
    for n in random.Random(seed).sample(others, min(_SAMPLE, len(others))):
        if not np.array_equal(tb[n].data.view(np.uint8), to[n].data.view(np.uint8)):
            errs.append(f"{n}: raw bytes differ but not listed as replaced")
    return errs
