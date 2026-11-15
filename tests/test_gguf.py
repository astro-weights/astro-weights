import os
import struct
import time
from pathlib import Path

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from gguf import GGMLQuantizationType as Q
from gguf import GGUFWriter
from gguf.quants import dequantize, quantize

from astro_weights.edits import Edit, LowRankDelta
from astro_weights.export.gguf import (
    _open,
    gguf_tensor_name,
    read_lora_gguf,
    save_lora_gguf,
    splice_gguf,
    verify_splice,
)

N0 = "blk.0.ffn_down.weight"
rng = np.random.default_rng(0)
# numpy (32, 96) == GGUF dims [96, 32]; HF down_proj is [hidden=32, intermediate=96]
W0 = rng.standard_normal((32, 96)).astype(np.float32)
W1 = rng.standard_normal((32, 96)).astype(np.float16)
Q4K = rng.integers(0, 256, (4, 144), dtype=np.uint8)  # 4 Q4_K blocks of 256 elems
EMB = rng.standard_normal((50, 32)).astype(np.float32)
NEW = W0 + 0.5 * rng.standard_normal(W0.shape).astype(np.float32)


def _finish(w: GGUFWriter) -> None:
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("splice") / "base.gguf"
    w = gguf.GGUFWriter(path, "qwen35")
    w.add_block_count(3)
    w.add_string("general.name", "tiny")
    w.add_uint32("qwen35.nextn_predict_layers", 1)
    w.add_array("qwen35.rope.dimension_sections", [11, 11, 10, 0])
    w.add_array("tokenizer.ggml.tokens", ["a", "b", "c"])
    w.add_bool("tiny.flag", True)
    w.add_float32("tiny.eps", 1e-6)
    w.add_tensor(N0, quantize(W0, Q.Q8_0), raw_dtype=Q.Q8_0)
    w.add_tensor("blk.1.ffn_down.weight", W1)
    w.add_tensor("token_embd.weight", EMB)
    w.add_tensor("blk.1.ffn_gate.weight", Q4K, raw_dtype=Q.Q4_K)
    # MTP-like: index >= block_count - nextn
    w.add_tensor("blk.2.ffn_down.weight", W1.astype(np.float32))
    _finish(w)
    return path


def _edit() -> Edit:
    deltas = [
        LowRankDelta(
            "model.layers.3.mlp.down_proj",
            rng.standard_normal((8, 2)),
            rng.standard_normal((2, 6)),
        ),
        LowRankDelta(
            "model.language_model.layers.5.self_attn.o_proj",
            rng.standard_normal((4, 2)),
            rng.standard_normal((2, 4)),
        ),
    ]
    return Edit(tuple(deltas), "org/model")


# adapter GGUF


def test_tensor_name_mapping() -> None:
    assert gguf_tensor_name("model.layers.7.mlp.down_proj") == "blk.7.ffn_down.weight"
    assert (
        gguf_tensor_name("model.language_model.layers.12.self_attn.o_proj")
        == "blk.12.attn_output.weight"
    )
    # the layer index after the last "layers.<N>" is used
    assert (
        gguf_tensor_name("layers.1.x.layers.2.mlp.down_proj") == "blk.2.ffn_down.weight"
    )


@pytest.mark.parametrize(
    "module",
    [
        "model.layers.0.mlp.up_proj",
        "model.mlp.down_proj",
        "model.layers.x.mlp.down_proj",
    ],
)
def test_unsupported_module_raises(module: str) -> None:
    with pytest.raises(ValueError, match=r"mlp\.down_proj"):
        gguf_tensor_name(module)


def test_adapter_round_trip(tmp_path: Path) -> None:
    edit = _edit()
    path = tmp_path / "a.gguf"
    save_lora_gguf(edit, path, "qwen35")
    tensors, meta = read_lora_gguf(path)
    assert meta == {
        "architecture": "qwen35",
        "type": "adapter",
        "adapter_type": "lora",
        "alpha": float(edit.rank),
    }
    assert set(tensors) == {"blk.3.ffn_down.weight", "blk.5.attn_output.weight"}
    by_name = {gguf_tensor_name(d.module): d for d in edit.deltas}
    for name, (b, a) in tensors.items():
        d = by_name[name]
        assert a.shape == d.A.shape
        assert b.shape == d.B.shape
        assert a.dtype == b.dtype == np.float32
        np.testing.assert_allclose(b @ a, d.dense(), rtol=1e-6)
    names = {t.name for t in gguf.GGUFReader(path).tensors}
    assert "blk.3.ffn_down.weight.lora_a" in names
    assert "blk.3.ffn_down.weight.lora_b" in names


def test_adapter_unsupported_module_writes_nothing(tmp_path: Path) -> None:
    edit = Edit(
        (LowRankDelta("model.layers.0.mlp.up_proj", np.ones((2, 1)), np.ones((1, 2))),),
        "m",
    )
    with pytest.raises(ValueError, match="unsupported module"):
        save_lora_gguf(edit, tmp_path / "a.gguf", "qwen35")
    assert list(tmp_path.iterdir()) == []


def test_adapter_overwrite_refused(tmp_path: Path) -> None:
    path = tmp_path / "a.gguf"
    path.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        save_lora_gguf(_edit(), path, "qwen35")
    assert path.read_bytes() == b"keep"
    save_lora_gguf(_edit(), path, "qwen35", overwrite=True)
    assert read_lora_gguf(path)[1]["type"] == "adapter"


def _adapter_file(
    path: Path,
    tensors: dict[str, tuple[np.ndarray, "Q | None"]],
) -> Path:
    w = GGUFWriter(path, "qwen35")
    w.add_type(gguf.GGUFType.ADAPTER)
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", 2.0)
    for n, (arr, dt) in tensors.items():
        w.add_tensor(n, arr, raw_dtype=dt)
    _finish(w)
    return path


_A = np.ones((2, 64), np.float32)
_B = np.ones((4, 2), np.float32)
_NAME = "blk.0.ffn_down.weight"


def test_read_alpha_scale_documented() -> None:
    doc = read_lora_gguf.__doc__ or ""
    assert "alpha" in doc
    assert "rank" in doc


def test_read_f16_adapter(tmp_path: Path) -> None:
    p = _adapter_file(
        tmp_path / "f16.gguf",
        {
            _NAME + ".lora_a": (_A.astype(np.float16), None),
            _NAME + ".lora_b": (_B.astype(np.float16), None),
        },
    )
    ((b, a),) = read_lora_gguf(p)[0].values()
    assert a.dtype == b.dtype == np.float32
    assert (b @ a == 2).all()


def test_read_quantized_adapter_refused(tmp_path: Path) -> None:
    p = _adapter_file(
        tmp_path / "q8.gguf",
        {
            _NAME + ".lora_a": (quantize(_A, Q.Q8_0), Q.Q8_0),
            _NAME + ".lora_b": (_B, None),
        },
    )
    with pytest.raises(ValueError, match="Q8_0"):
        read_lora_gguf(p)
    p = _adapter_file(
        tmp_path / "bf16.gguf",
        {
            _NAME + ".lora_a": (quantize(_A, Q.BF16), Q.BF16),
            _NAME + ".lora_b": (_B, None),
        },
    )
    with pytest.raises(ValueError, match="BF16"):
        read_lora_gguf(p)


def test_read_rank_mismatch_refused(tmp_path: Path) -> None:
    p = _adapter_file(
        tmp_path / "rank.gguf",
        {
            _NAME + ".lora_a": (_A, None),  # rank 2
            _NAME + ".lora_b": (np.ones((4, 3), np.float32), None),  # rank 3
        },
    )
    with pytest.raises(ValueError, match="lora_b"):
        read_lora_gguf(p)


def test_read_not_2d_refused(tmp_path: Path) -> None:
    p = _adapter_file(
        tmp_path / "1d.gguf",
        {
            _NAME + ".lora_a": (np.ones(64, np.float32), None),
            _NAME + ".lora_b": (_B, None),
        },
    )
    with pytest.raises(ValueError, match="lora_b"):
        read_lora_gguf(p)


def test_read_duplicate_tensor_refused(tmp_path: Path) -> None:
    p = _adapter_file(
        tmp_path / "dup.gguf",
        {
            _NAME + ".lora_a": (_A, None),
            _NAME + ".lora_x": (_A, None),
            _NAME + ".lora_b": (_B, None),
        },
    )
    # same-length rename, so the file now holds two ".lora_a" tensors
    p.write_bytes(p.read_bytes().replace(b".lora_x", b".lora_a"))
    with pytest.raises(ValueError, match="duplicate"):
        read_lora_gguf(p)


def _broken_files(tmp_path: Path, good: Path) -> dict[str, Path]:
    raw = good.read_bytes()
    cases = {
        "empty": b"",
        "garbage": b"x" * 100,
        "trunc_header": raw[:20],
        "trunc_data": raw[:-100],
        "bad_version": raw[:4] + b"\xff\xff\xff\x7f" + raw[8:],
    }
    tmp_path.mkdir(exist_ok=True)
    out = {}
    for name, data in cases.items():
        out[name] = tmp_path / f"{name}.gguf"
        out[name].write_bytes(data)
    out["missing"] = tmp_path / "missing.gguf"
    return out


def test_read_lora_broken_files_raise_value_error(tmp_path: Path) -> None:
    good = tmp_path / "good.gguf"
    save_lora_gguf(_edit(), good, "qwen35")
    for p in _broken_files(tmp_path, good).values():
        with pytest.raises(ValueError, match="cannot read GGUF"):
            read_lora_gguf(p)


def test_save_tmp_file_not_deleted(tmp_path: Path) -> None:
    out = tmp_path / "a.gguf"
    precious = tmp_path / "a.gguf.tmp"
    precious.write_bytes(b"precious")
    bad = Edit(
        (LowRankDelta("model.layers.0.mlp.up_proj", np.ones((2, 1)), np.ones((1, 2))),),
        "m",
    )
    with pytest.raises(ValueError, match="unsupported module"):
        save_lora_gguf(bad, out, "qwen35")
    save_lora_gguf(_edit(), out, "qwen35")
    assert precious.read_bytes() == b"precious"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.gguf", "a.gguf.tmp"]


def test_save_failure_cleans_only_own_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "a.gguf"
    precious = tmp_path / "a.gguf.tmp"
    precious.write_bytes(b"precious")

    def boom(self: object, **kw: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(gguf.GGUFWriter, "write_tensors_to_file", boom)
    with pytest.raises(OSError, match="disk full"):
        save_lora_gguf(_edit(), out, "qwen35")
    assert [p.name for p in tmp_path.iterdir()] == ["a.gguf.tmp"]
    assert precious.read_bytes() == b"precious"


# splice


def check_out(base: Path, out: Path, new: np.ndarray) -> None:
    rb, ro = gguf.GGUFReader(base), gguf.GGUFReader(out)
    assert [t.name for t in rb.tensors] == [t.name for t in ro.tensors]
    assert len(rb.fields) == len(ro.fields)
    for k, f in rb.fields.items():
        if not k.startswith("GGUF."):
            assert ro.fields[k].contents() == f.contents(), k
    tb, to = {t.name: t for t in rb.tensors}, {t.name: t for t in ro.tensors}
    for n in (
        "blk.1.ffn_down.weight",
        "token_embd.weight",
        "blk.1.ffn_gate.weight",
        "blk.2.ffn_down.weight",
    ):
        assert to[n].tensor_type == tb[n].tensor_type
        assert np.array_equal(to[n].data.view(np.uint8), tb[n].data.view(np.uint8))
    t = to[N0]
    assert t.tensor_type == Q.Q8_0
    assert t.shape.tolist() == [96, 32]
    got = dequantize(t.data, t.tensor_type)
    assert np.linalg.norm(got - new) / np.linalg.norm(new) < 1e-2
    assert np.linalg.norm(got - W0) / np.linalg.norm(W0) > 0.1


def test_splice_replaces_only_named_tensor(base: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.gguf"
    splice_gguf(base, out, {N0: NEW})
    check_out(base, out, NEW)
    assert verify_splice(base, out, [N0]) == []


def test_transposed_replacement_is_transposed(base: Path, tmp_path: Path) -> None:
    out = tmp_path / "t.gguf"
    splice_gguf(base, out, {N0: np.ascontiguousarray(NEW.T)})
    check_out(base, out, NEW)


def test_out_type_f16(base: Path, tmp_path: Path) -> None:
    out = tmp_path / "f16.gguf"
    splice_gguf(base, out, {N0: NEW}, out_type="f16")
    t = {t.name: t for t in gguf.GGUFReader(out).tensors}[N0]
    assert t.tensor_type == Q.F16
    with pytest.raises(ValueError, match="out_type"):
        splice_gguf(base, tmp_path / "x.gguf", {N0: NEW}, out_type="q4_k")


def test_bad_shape_rejected(base: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="replacement shape"):
        splice_gguf(base, tmp_path / "x.gguf", {N0: np.zeros((10, 10), np.float32)})
    assert list(tmp_path.iterdir()) == []


def test_missing_tensor_rejected(base: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not in base"):
        splice_gguf(base, tmp_path / "x.gguf", {"blk.0.nope.weight": NEW})


def test_out_same_as_base_refused(base: Path, tmp_path: Path) -> None:
    copy = tmp_path / "same.gguf"
    copy.write_bytes(base.read_bytes())
    link = tmp_path / "link.gguf"
    link.symlink_to(copy)
    for o in (copy, link):
        with pytest.raises(ValueError, match="same file"):
            splice_gguf(copy, o, {N0: NEW})
    assert copy.read_bytes() == base.read_bytes()


def test_failed_write_leaves_no_out(
    base: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "fail.gguf"
    out.write_bytes(b"keep")

    def boom(self: object, **kw: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(gguf.GGUFWriter, "write_tensors_to_file", boom)
    with pytest.raises(OSError, match="disk full"):
        splice_gguf(base, out, {N0: NEW})
    assert out.read_bytes() == b"keep"
    assert not (tmp_path / "fail.gguf.tmp").exists()
    out.unlink()
    with pytest.raises(OSError, match="disk full"):
        splice_gguf(base, out, {N0: NEW})
    assert not out.exists()
    assert not (tmp_path / "fail.gguf.tmp").exists()


def test_nan_replacement_refused(base: Path, tmp_path: Path) -> None:
    for bad_value in (np.nan, np.inf):
        bad = NEW.copy()
        bad[3, 3] = bad_value
        with pytest.raises(ValueError, match="NaN/Inf"):
            splice_gguf(base, tmp_path / "nan.gguf", {N0: bad})
    assert not (tmp_path / "nan.gguf").exists()


def test_mtp_block_refused(base: Path, tmp_path: Path) -> None:
    mtp = W1.astype(np.float32)
    with pytest.raises(ValueError, match="nextn_predict_layers"):
        splice_gguf(base, tmp_path / "mtp.gguf", {"blk.2.ffn_down.weight": mtp})
    assert not (tmp_path / "mtp.gguf").exists()
    # the last real block is accepted
    splice_gguf(base, tmp_path / "ok1.gguf", {"blk.1.ffn_down.weight": mtp})
    assert (tmp_path / "ok1.gguf").exists()


def test_splice_tmp_file_not_deleted(
    base: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "o.gguf"
    precious = tmp_path / "o.gguf.tmp"
    precious.write_bytes(b"precious")
    with pytest.raises(ValueError, match="replacement shape"):
        splice_gguf(base, out, {N0: np.zeros((10, 10), np.float32)})
    assert precious.read_bytes() == b"precious"

    def boom(self: object, **kw: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(gguf.GGUFWriter, "write_tensors_to_file", boom)
    with pytest.raises(OSError, match="disk full"):
        splice_gguf(base, out, {N0: NEW})
    assert [p.name for p in tmp_path.iterdir()] == ["o.gguf.tmp"]
    assert precious.read_bytes() == b"precious"


def test_splice_broken_base_raises_value_error(base: Path, tmp_path: Path) -> None:
    for p in _broken_files(tmp_path / "x", base).values():
        with pytest.raises(ValueError, match="cannot read GGUF"):
            splice_gguf(p, tmp_path / "o.gguf", {N0: NEW})
    assert not (tmp_path / "o.gguf").exists()


def test_splice_base_without_architecture(tmp_path: Path) -> None:
    p = tmp_path / "noarch.gguf"
    w = GGUFWriter(p, "qwen35")
    w.add_tensor("a.weight", np.ones((2, 32), np.float32))
    _finish(w)
    # same-length rename: the key is no longer general.architecture
    p.write_bytes(
        p.read_bytes().replace(b"general.architecture", b"general.archXtecture")
    )
    with pytest.raises(ValueError, match=r"no general\.architecture"):
        splice_gguf(p, tmp_path / "o.gguf", {"a.weight": np.ones((2, 32), np.float32)})


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((32, 96), np.int32),
        np.full((32, 96), "a"),
        [["a"] * 96] * 32,
    ],
)
def test_non_float_replacement_rejected(
    base: Path, tmp_path: Path, bad: object
) -> None:
    with pytest.raises(ValueError, match=r"floating point|not an array"):
        splice_gguf(base, tmp_path / "x.gguf", {N0: bad})  # type: ignore[dict-item]
    assert list(tmp_path.iterdir()) == []


def test_list_replacement_accepted(base: Path, tmp_path: Path) -> None:
    out = tmp_path / "l.gguf"
    splice_gguf(base, out, {N0: NEW.tolist()})
    check_out(base, out, NEW)


# verify_splice


def test_verify_unreadable_files_reported(base: Path, tmp_path: Path) -> None:
    for name, p in _broken_files(tmp_path / "v", base).items():
        for problems in (verify_splice(base, p, [N0]), verify_splice(p, base, [N0])):
            assert len(problems) == 1, name
            assert "cannot read GGUF" in problems[0], name


def rewrite(
    src: Path, dst: Path, override: dict[str, tuple[np.ndarray, "Q | None"]]
) -> None:
    """Copy src GGUF to dst, replacing tensors by name with (array, raw_dtype|None)."""
    r = gguf.GGUFReader(src)
    w = gguf.GGUFWriter(dst, str(r.get_field("general.architecture").contents()))
    for key, f in r.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        vt = f.types[0]
        if vt == gguf.GGUFValueType.ARRAY:
            w.add_key_value(key, f.contents(), vt, f.types[-1])
        else:
            w.add_key_value(key, f.contents(), vt)
    for t in r.tensors:
        if t.name in override:
            arr, dt = override[t.name]
            w.add_tensor(t.name, arr, raw_dtype=dt)
        else:
            w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
    _finish(w)


def test_verify_wrong_names(base: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.gguf"
    splice_gguf(base, out, {N0: NEW})
    # identical tensor listed as replaced
    assert any(
        "identical" in p for p in verify_splice(base, out, ["token_embd.weight"])
    )
    # replaced tensor not listed
    assert any(
        "not listed" in p for p in verify_splice(base, out, ["blk.1.ffn_down.weight"])
    )
    assert any("not in both" in p for p in verify_splice(base, out, ["nope"]))


def test_verify_corrupt_copied_byte(base: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.gguf"
    splice_gguf(base, out, {N0: NEW})
    bad = tmp_path / "bad.gguf"
    b = bytearray(out.read_bytes())
    t = {t.name: t for t in gguf.GGUFReader(out).tensors}["token_embd.weight"]
    b[t.data_offset + 5] ^= 0xFF
    bad.write_bytes(b)
    assert any("raw bytes differ" in p for p in verify_splice(base, bad, [N0]))


def test_verify_kv_differs(base: Path, tmp_path: Path) -> None:
    out = tmp_path / "kv.gguf"
    r = gguf.GGUFReader(base)
    w = gguf.GGUFWriter(out, "qwen35")
    w.add_string("general.name", "other")
    for t in r.tensors:
        w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
    _finish(w)
    problems = verify_splice(base, out, [])
    assert any("KV" in p for p in problems)


def test_verify_nan_in_out_fails(base: Path, tmp_path: Path) -> None:
    out = tmp_path / "nanout.gguf"
    bad = NEW.copy()
    bad[0, 0] = np.nan
    rewrite(base, out, {N0: (bad, None)})
    assert any("non-finite" in p for p in verify_splice(base, out, [N0]))


def test_verify_unlisted_type_change_fails_beyond_sample(tmp_path: Path) -> None:
    b, o = tmp_path / "b.gguf", tmp_path / "o.gguf"
    w = gguf.GGUFWriter(b, "qwen35")
    for i in range(40):
        w.add_tensor(f"x.{i}", np.full((2, 32), i, np.float32))
    _finish(w)
    # x.5 changes F32 -> Q8_0; the type check must catch it, not the 20-tensor byte sample
    rewrite(
        b,
        o,
        {
            "x.5": (quantize(np.full((2, 32), 5, np.float32), Q.Q8_0), Q.Q8_0),
            "x.6": (np.full((2, 32), 7, np.float32), None),
        },
    )
    problems = verify_splice(b, o, ["x.6"])
    assert any("x.5: type/shape/size differs" in p for p in problems)


# malformed files: ValueError, quickly


def _str(b: bytes) -> bytes:
    return struct.pack("<Q", len(b)) + b


def _kv(key: bytes, vtype: int, value: bytes) -> bytes:
    return _str(key) + struct.pack("<I", vtype) + value


def _gguf(n_tensors: int, n_kv: int, body: bytes = b"", version: int = 3) -> bytes:
    return b"GGUF" + struct.pack("<IQQ", version, n_tensors, n_kv) + body


_U8, _STR, _ARR = 0, 8, 9
_MALFORMED = {
    "bad_magic": b"GGUX" + _gguf(0, 0)[4:],
    "version_0": _gguf(0, 0, version=0),
    "version_huge": _gguf(0, 0, version=2**31),
    "header_cut": _gguf(0, 0)[:23],
    "absurd_tensor_count": _gguf(2**63, 0),
    "absurd_kv_count": _gguf(0, 2**64 - 1),
    "counts_exceed_file": _gguf(1000, 0, b"x" * 100),
    "kv_cut_in_key": _gguf(0, 1, struct.pack("<Q", 50) + b"short"),
    "kv_huge_key": _gguf(0, 1, struct.pack("<Q", 2**64 - 1) + b"k" * 20),
    "kv_unknown_type": _gguf(0, 1, _kv(b"k", 99, b"\0" * 8)),
    "kv_huge_string": _gguf(0, 1, _kv(b"k", _STR, struct.pack("<Q", 2**62))),
    "array_huge_u8": _gguf(
        0, 1, _kv(b"k", _ARR, struct.pack("<IQ", _U8, 2**40) + b"\0" * 16)
    ),
    "array_huge_u8_small": _gguf(
        0, 1, _kv(b"k", _ARR, struct.pack("<IQ", _U8, 2**25) + b"\0" * 16)
    ),
    "array_overflow_len": _gguf(
        0, 1, _kv(b"k", _ARR, struct.pack("<IQ", 12, 2**64 - 1) + b"\0" * 16)
    ),
    "array_huge_strings": _gguf(
        0, 1, _kv(b"k", _ARR, struct.pack("<IQ", _STR, 2**40) + b"\0" * 64)
    ),
    "array_unknown_item": _gguf(0, 1, _kv(b"k", _ARR, struct.pack("<IQ", 77, 1))),
    # bytes really present: 4 MB of uint8 cost GGUFReader 15 s and 2.8 GB
    "array_4m_u8": _gguf(
        0, 1, _kv(b"k", _ARR, struct.pack("<IQ", _U8, 4_000_000) + bytes(4_000_000))
    ),
    "array_2m_empty_strings": _gguf(
        0,
        1,
        _kv(b"k", _ARR, struct.pack("<IQ", _STR, 2_100_000)) + bytes(8 * 2_100_000),
    ),
    # the cap is on the total over all kvs, not per array
    "arrays_total_over_cap": _gguf(
        0,
        3,
        b"".join(
            _kv(b"k%d" % i, _ARR, struct.pack("<IQ", _U8, 1_000_000) + bytes(1_000_000))
            for i in range(3)
        ),
    ),
    "array_nested_deep": _gguf(
        0, 1, _kv(b"k", _ARR, struct.pack("<IQ", _ARR, 1) * 20 + b"\0" * 64)
    ),
    "tensor_cut": _gguf(1, 0, _str(b"w") + struct.pack("<I", 2)),
    "tensor_huge_ndims": _gguf(
        1, 0, _str(b"w") + struct.pack("<I", 2**32 - 1) + b"\0" * 64
    ),
    "tensor_huge_name": _gguf(1, 0, struct.pack("<Q", 2**63) + b"\0" * 64),
}


@pytest.mark.parametrize("name", sorted(_MALFORMED))
def test_malformed_gguf_raises_value_error_fast(tmp_path: Path, name: str) -> None:
    p = tmp_path / "bad.gguf"
    p.write_bytes(_MALFORMED[name])
    t = time.monotonic()
    with pytest.raises(ValueError, match="cannot read GGUF"):
        read_lora_gguf(p)
    with pytest.raises(ValueError, match="cannot read GGUF"):
        splice_gguf(p, tmp_path / "o.gguf", {N0: NEW})
    assert time.monotonic() - t < 1


def test_duplicate_kv_key_raises_value_error(tmp_path: Path) -> None:
    good = tmp_path / "good.gguf"
    save_lora_gguf(_edit(), good, "qwen35")
    # same-length rename: the file now holds two "general.type" keys
    dup = tmp_path / "dup.gguf"
    dup.write_bytes(good.read_bytes().replace(b"adapter.type", b"general.type"))
    with pytest.raises(ValueError, match=r"cannot read GGUF.*Duplicate"):
        read_lora_gguf(dup)
    assert verify_splice(dup, good, [])[0].startswith("cannot read GGUF")


@pytest.mark.parametrize("exc", [KeyError("k"), MemoryError(), OverflowError()])
def test_open_wraps_library_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    good = tmp_path / "good.gguf"
    save_lora_gguf(_edit(), good, "qwen35")

    def boom(*_: object) -> None:
        raise exc

    monkeypatch.setattr(gguf, "GGUFReader", boom)
    with pytest.raises(ValueError, match="cannot read GGUF"):
        _open(good)


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
def test_open_does_not_swallow_interrupts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: type[BaseException]
) -> None:
    good = tmp_path / "good.gguf"
    save_lora_gguf(_edit(), good, "qwen35")

    def boom(*_: object) -> None:
        raise exc

    monkeypatch.setattr(gguf, "GGUFReader", boom)
    with pytest.raises(exc):
        _open(good)


def test_valid_files_open(base: Path, tmp_path: Path) -> None:
    """Adapter, base (kv arrays, quantized tensors) and spliced files all pass."""
    adapter = tmp_path / "a.gguf"
    save_lora_gguf(_edit(), adapter, "qwen35")
    out = tmp_path / "s.gguf"
    splice_gguf(base, out, {N0: NEW})
    assert len(_open(adapter).tensors) == 4
    assert len(_open(base).tensors) == 5
    assert len(_open(out).tensors) == 5
    assert set(read_lora_gguf(adapter)[0]) == {
        "blk.3.ffn_down.weight",
        "blk.5.attn_output.weight",
    }


def test_array_cap_allows_real_size_tokenizer(tmp_path: Path) -> None:
    """A 152k-string array (a real vocab) plus scalar arrays still opens."""
    n = 152_000
    toks = _kv(b"tokenizer.ggml.tokens", _ARR, struct.pack("<IQ", _STR, n))
    toks += b"".join(_str(b"t%d" % i) for i in range(n))
    types = _kv(b"tokenizer.ggml.token_type", _ARR, struct.pack("<IQ", 5, n))
    types += bytes(4 * n)
    p = tmp_path / "vocab.gguf"
    p.write_bytes(_gguf(0, 2, toks + types))
    t = time.monotonic()
    reader = _open(p)
    assert reader.get_field("tokenizer.ggml.token_type") is not None
    assert time.monotonic() - t < 10


def test_unsupported_kv_type_is_named(tmp_path: Path) -> None:
    p = tmp_path / "t.gguf"
    p.write_bytes(_gguf(0, 1, _kv(b"k", 99, bytes(8))))
    with pytest.raises(ValueError, match="unsupported kv value type 99"):
        _open(p)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no mkfifo")
def test_fifo_is_refused_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe.gguf"
    os.mkfifo(fifo)
    t = time.monotonic()
    with pytest.raises(ValueError, match="not a regular file"):
        read_lora_gguf(fifo)
    with pytest.raises(ValueError, match="not a regular file"):
        splice_gguf(fifo, tmp_path / "o.gguf", {N0: NEW})
    assert time.monotonic() - t < 1


def test_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a regular file"):
        _open(tmp_path)


def test_big_endian_file_still_opens(tmp_path: Path) -> None:
    p = tmp_path / "be.gguf"
    w = GGUFWriter(p, "qwen35", endianess=gguf.GGUFEndian.BIG)
    w.add_array("k", ["a", "b"])
    w.add_tensor("x", np.zeros((2, 2), dtype=np.float32))
    _finish(w)
    assert len(_open(p).tensors) == 1
    # and the walk is endian-aware: a big-endian absurd kv count is rejected
    bad = bytearray(p.read_bytes())
    bad[16:24] = struct.pack(">Q", 2**60)
    p.write_bytes(bytes(bad))
    with pytest.raises(ValueError, match="cannot fit"):
        _open(p)
