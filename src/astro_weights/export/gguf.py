"""GGUF export: llama.cpp LoRA adapter files and tensor splice into a base GGUF.

Needs the optional `gguf` extra (`pip install astro-weights[gguf]`).

Orientation: GGUF stores dims reversed relative to numpy/torch. `GGUFReader`
`tensor.shape` is GGUF order (ne0 first); `tensor.data` is numpy order, i.e.
`reversed(tensor.shape)`. Example, Qwen3.5-9B-Q4_K_M `blk.15.ffn_down.weight`:
reader shape [12288, 4096] ([intermediate, hidden]) -> numpy (4096, 12288) =
[hidden, intermediate], the torch nn.Linear layout of HF `mlp.down_proj.weight`
([out, in]). So an HF [out, in] array is written and spliced as is, no transpose.
"""

import mmap
import random
import re
import secrets
import stat
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


# Fixed sizes of GGUF scalar value types; STRING and ARRAY are variable.
_SCALAR = {
    VT.UINT8: 1,
    VT.INT8: 1,
    VT.BOOL: 1,
    VT.UINT16: 2,
    VT.INT16: 2,
    VT.UINT32: 4,
    VT.INT32: 4,
    VT.FLOAT32: 4,
    VT.UINT64: 8,
    VT.INT64: 8,
    VT.FLOAT64: 8,
}
_VERSIONS = (2, gguf.GGUF_VERSION)
_HEADER = 24  # magic u32, version u32, tensor count u64, kv count u64
# Smallest encodings, which bound the counts by the file size: a kv is a key
# string (u64 length) + u32 type + at least a 1-byte value; a tensor info is a
# name string (u64 length) + u32 n_dims + u32 type + u64 offset.
_MIN_KV = 8 + 4 + 1
_MIN_TENSOR = 8 + 4 + 4 + 8
_MAX_DEPTH = 8  # nested arrays
# Cap on kv array items summed over the whole file (nested arrays count every
# level). gguf.GGUFReader builds Python and numpy objects per item, about 1 KB
# each: a 4 MB uint8 array of 4M items took 15 s and 2.8 GB, so a crafted
# 40-50 MB file could take minutes and over 25 GB. 2**21 = 2,097,152 items is
# about 2 GB at worst. The largest real tokenizer metadata (a 262k vocab with
# tokens, scores, token_type and merges) is about 1.05M items, and a 152k-vocab
# 3B model has 455k, so real files stay under half the cap.
_MAX_ARRAY_ITEMS = 1 << 21


class _Walk:
    """Walks the kv/tensor-info layout with bounds checks; `e` is the endianness."""

    def __init__(self, buf: mmap.mmap, e: str) -> None:
        self.buf = buf
        self.e = e
        self.items = 0

    def take(self, off: int, n: int) -> int:
        """Offset after `n` bytes at `off`; ValueError if they run past the end."""
        if n < 0 or n > len(self.buf) - off:
            raise ValueError(f"data runs past end of file at offset {off}")
        return off + n

    def u32(self, off: int) -> int:
        self.take(off, 4)
        return int(struct.unpack_from(f"{self.e}I", self.buf, off)[0])

    def skip_string(self, off: int) -> int:
        self.take(off, 8)
        (n,) = struct.unpack_from(f"{self.e}Q", self.buf, off)
        return self.take(off + 8, n)

    def count_items(self, n: int) -> None:
        self.items += n
        if self.items > _MAX_ARRAY_ITEMS:
            raise ValueError(
                f"kv arrays hold more than {_MAX_ARRAY_ITEMS} items in total"
            )

    def skip_value(self, off: int, raw_type: int, depth: int) -> int:
        """Offset after one kv value, checking every length against the file size."""
        try:
            vt = VT(raw_type)
        except ValueError:
            raise ValueError(f"unsupported kv value type {raw_type}") from None
        if vt == VT.STRING:
            return self.skip_string(off)
        if vt != VT.ARRAY:
            return self.take(off, _SCALAR[vt])
        if depth >= _MAX_DEPTH:
            raise ValueError("kv arrays nested too deeply")
        end = self.take(off, 12)
        item, n = struct.unpack_from(f"{self.e}IQ", self.buf, off)
        try:
            size = _SCALAR.get(VT(item))
        except ValueError:
            raise ValueError(f"unsupported kv array item type {item}") from None
        self.count_items(n)
        if size is not None:
            return self.take(end, n * size)  # one bounds check, no per-item loop
        # string items take >= 8 bytes, nested arrays >= 12: n is bounded by the
        # file size (and the item cap) before looping, so a huge n cannot spin
        self.take(end, n * 8)
        for _ in range(n):
            end = self.skip_value(end, item, depth + 1)
        return end


def _check_layout(path: Path | str) -> None:
    """Reject a file gguf.GGUFReader would hang on or allocate for without bound.

    GGUFReader trusts every count and length; walk the same layout with bounds
    checks first: header, counts that cannot fit, lengths past the end of the
    file, and the total array item cap. Only regular files are opened, so a FIFO
    cannot block. Little- and big-endian files are both walked.
    """
    if not stat.S_ISREG(Path(path).stat().st_mode):
        raise ValueError("not a regular file")
    with Path(path).open("rb") as f:
        if f.seek(0, 2) < _HEADER:
            raise ValueError("file shorter than the GGUF header")
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
            (magic,) = struct.unpack_from("<I", buf)
            if magic != gguf.GGUF_MAGIC:
                raise ValueError("not a GGUF file (bad magic)")
            # the version is below 2**16, so a zero low half means big-endian
            e = ">" if struct.unpack_from("<I", buf, 4)[0] & 0xFFFF == 0 else "<"
            version, n_tensors, n_kv = struct.unpack_from(f"{e}IQQ", buf, 4)
            if version not in _VERSIONS:
                raise ValueError(f"unsupported GGUF version {version}")
            if n_kv * _MIN_KV + n_tensors * _MIN_TENSOR > len(buf) - _HEADER:
                raise ValueError(
                    f"header counts ({n_tensors} tensors, {n_kv} kv) cannot fit "
                    "in the file"
                )
            walk = _Walk(buf, e)
            off = _HEADER
            for _ in range(n_kv):
                off = walk.skip_string(off)
                off = walk.skip_value(off + 4, walk.u32(off), 0)
            for _ in range(n_tensors):
                off = walk.skip_string(off)
                off = walk.take(off + 4, walk.u32(off) * 8 + 12)  # dims, type, offset


def _open(path: Path | str) -> gguf.GGUFReader:
    """GGUFReader that raises ValueError naming the file on any read failure.

    Library errors on bad content (KeyError for a duplicate key, IndexError,
    struct.error, ...) are all turned into ValueError; so are OSErrors from
    opening the file. KeyboardInterrupt and SystemExit pass through.
    """
    try:
        _check_layout(path)
        return gguf.GGUFReader(path)
    except Exception as e:
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
