from pathlib import Path

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from gguf import GGMLQuantizationType as Q
from gguf import GGUFWriter
from gguf.quants import dequantize, quantize

from astro_weights.edits import Edit, LowRankDelta
from astro_weights.export.gguf import (
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


# verify_splice


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
