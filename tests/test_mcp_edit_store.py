import json
import threading
from pathlib import Path
from typing import Any

import mcp_fakes
import numpy as np
import pytest
from mcp_fakes import fake_model_dir, fake_weights

from astro_weights.export.base import read_tensor, tensor_sha256
from astro_weights.export.peft import load_peft, save_peft
from astro_weights.mcp.edit_store import (
    MAX_REL_CHANGE,
    EditStore,
    _round_bf16,
    _sha256_file,
    export_edits,
)

Q0 = "model.layers.0.self_attn.q_proj.weight"
O0 = "model.layers.0.self_attn.o_proj.weight"
O1 = "model.layers.1.self_attn.o_proj.weight"
K0 = "model.layers.0.self_attn.k_proj.weight"
D0 = "model.layers.0.mlp.down_proj.weight"
EDITABLE = {
    f"model.layers.{n}.{s}.weight"
    for n in (0, 1)
    for s in (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "mlp.down_proj",
    )
}


@pytest.fixture
def model(tmp_path: Path) -> Path:
    return fake_model_dir(tmp_path / "model")


@pytest.fixture
def store(tmp_path: Path, model: Path) -> EditStore:
    return EditStore(tmp_path / "run", model)


def edit_of(
    store: EditStore, name: str, rel: float = 0.05, seed: int = 1
) -> np.ndarray:
    """A genuine small edit: base plus rank-1 change of relative size `rel`."""
    w0 = store.original(name)
    rng = np.random.default_rng(seed)
    d = np.outer(rng.standard_normal(w0.shape[0]), rng.standard_normal(w0.shape[1]))
    d *= rel * np.linalg.norm(w0) / np.linalg.norm(d)
    out: np.ndarray = (w0 + d).astype(np.float32)
    return out


def edit_file(store: EditStore, name: str) -> Path:
    """Where the manifest says the stored edit of `name` lives."""
    sha = json.loads(store.manifest_path.read_text())[name]["sha256"]
    return store.dir / f"{sha}.safetensors"


def files(store: EditStore) -> list[str]:
    return sorted(p.name for p in store.dir.iterdir())


# names


@pytest.mark.parametrize("layout", ["model", "language_model"])
@pytest.mark.parametrize("sharded", [False, True])
def test_names(tmp_path: Path, layout: str, sharded: bool) -> None:
    d = fake_model_dir(tmp_path / "m", sharded=sharded, layout=layout)
    store = EditStore(tmp_path / "run", d)
    pre = "model" if layout == "model" else "model.language_model"
    want = {k.replace("model.", pre + ".", 1) for k in EDITABLE}
    assert set(store.names) == want
    assert list(store.names) == sorted(store.names)
    assert store.shape(next(iter(store.names))) in {(32, 32), (16, 32), (32, 64)}


def test_names_exclude_norm_embed_head_and_mtp(tmp_path: Path) -> None:
    d = fake_model_dir(tmp_path / "m")
    store = EditStore(tmp_path / "run", d)
    for bad in ("norm", "embed", "lm_head"):
        assert not [n for n in store.names if bad in n]
    weights = fake_weights()
    weights["model.layers.0.mtp.proj.weight"] = np.ones((32, 32), np.float32)
    weights["model.layers.0.nextn.proj.weight"] = np.ones((32, 32), np.float32)
    weights["model.layers.0.conv.weight"] = np.ones(32, np.float32)  # 1-D
    weights["model.layers.0.self_attn.q_proj.bias"] = np.ones(32, np.float32)
    mcp_fakes._raw(
        {k: ("F32", v) for k, v in weights.items()},
        tmp_path / "m" / "model.safetensors",
    )
    again = EditStore(tmp_path / "run2", d)
    assert set(again.names) == EDITABLE


def test_shape_is_from_header(store: EditStore) -> None:
    assert store.shape(D0) == (32, 64)
    assert store.shape(K0) == (16, 32)
    with pytest.raises(ValueError, match="not an editable"):
        store.shape("model.embed_tokens.weight")


# round trip


def test_save_load_drop_edited(store: EditStore) -> None:
    w = edit_of(store, O0)
    assert np.array_equal(store.load(O0), store.original(O0))
    res = store.save(O0, w)
    assert res["name"] == O0
    assert 0.03 < res["rel_frobenius"] < 0.07
    assert store.edited() == [O0]
    got = store.load(O0)
    assert got.dtype == np.float32
    assert not np.array_equal(got, store.original(O0))
    assert np.allclose(got, w, atol=0.02 * np.abs(w).max())
    assert np.array_equal(_round_bf16(got), got)  # stored values are bf16-exact
    manifest = json.loads(store.manifest_path.read_text())
    assert manifest[O0]["shape"] == [32, 32]
    assert set(manifest[O0]) == {"sha256", "rel_frobenius", "shape"}
    store.drop(O0)
    assert store.edited() == []
    assert files(store) == ["manifest.json"]
    assert np.array_equal(store.load(O0), store.original(O0))
    with pytest.raises(ValueError, match="no edit"):
        store.drop(O0)


def test_new_store_sees_old_edits(tmp_path: Path, model: Path) -> None:
    a = EditStore(tmp_path / "run", model)
    a.save(Q0, edit_of(a, Q0))
    b = EditStore(tmp_path / "run", model)
    assert b.edited() == [Q0]
    assert np.array_equal(b.load(Q0), a.load(Q0))
    assert b.validate()[Q0]["rel_frobenius"] > 0


@pytest.mark.parametrize("dtype", ["bf16", "f16", "f32"])
@pytest.mark.parametrize("sharded", [False, True])
def test_dtypes_and_shards(tmp_path: Path, dtype: str, sharded: bool) -> None:
    d = fake_model_dir(tmp_path / "m", dtype=dtype, sharded=sharded)
    s = EditStore(tmp_path / "run", d)
    w = edit_of(s, D0)
    s.save(D0, w)
    assert s.validate()[D0]["shape"] == [32, 64]
    if dtype == "f32":  # f32 stays as given
        assert np.array_equal(s.load(D0), w)
    f16 = w.astype(np.float16)  # accepted float dtypes
    s.save(O1, edit_of(s, O1).astype(np.float16))
    s.save(Q0, edit_of(s, Q0).astype(np.float64))
    assert f16.dtype == np.float16


# rejects


def assert_clean(store: EditStore) -> None:
    assert store.edited() == []
    assert files(store) == []


@pytest.mark.parametrize(
    "name",
    [
        "../x",
        "model.layers.0.self_attn.q_proj.weight\n",
        "a\x00b",
        "",
        "x" * 300,
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.0.input_layernorm.weight",
        "nope",
        "/etc/passwd",
    ],
)
def test_bad_names(store: EditStore, name: str) -> None:
    with pytest.raises(ValueError, match="not an editable"):
        store.save(name, np.zeros((32, 32), np.float32))
    assert_clean(store)


def test_non_str_name(store: EditStore) -> None:
    with pytest.raises(ValueError, match="not an editable"):
        store.save(5, np.zeros((32, 32), np.float32))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "make",
    [
        lambda: np.ones((32, 32), np.int32),
        lambda: np.ones((32, 32), bool),
        lambda: np.ones((32, 32), object),
        lambda: np.ones((32, 32), np.complex64),
        lambda: [[1.0] * 32] * 32,
    ],
)
def test_non_float(store: EditStore, make: Any) -> None:
    with pytest.raises(ValueError, match="float16/32/64 numpy array"):
        store.save(Q0, make())
    assert_clean(store)


def test_wrong_shape_without_large_alloc(store: EditStore) -> None:
    big = np.broadcast_to(np.float32(1), (50_000, 50_000))  # 10 GB if materialized
    with pytest.raises(ValueError, match="does not match original"):
        store.save(Q0, big)
    with pytest.raises(ValueError, match="does not match original"):
        store.save(Q0, np.zeros((32,), np.float32))
    assert_clean(store)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite(store: EditStore, bad: float) -> None:
    w = edit_of(store, Q0)
    w[1, 1] = bad
    with pytest.raises(ValueError, match="non-finite"):
        store.save(Q0, w)
    assert_clean(store)


def test_overflow_after_rounding(store: EditStore) -> None:
    w = edit_of(store, Q0)
    w[0, 0] = np.finfo(np.float32).max  # finite, but rounds to inf in bf16
    with pytest.raises(ValueError, match="non-finite values after rounding"):
        store.save(Q0, w)
    assert_clean(store)


def test_overflow_f64_to_f32(store: EditStore) -> None:
    w = edit_of(store, Q0).astype(np.float64)
    w[0, 0] = 1e300
    with pytest.raises(ValueError, match="non-finite values after rounding"):
        store.save(Q0, w)
    assert_clean(store)


@pytest.mark.parametrize("s", [2.0, 0.5, -1.0, 1.0])
def test_exact_scalar_multiple(store: EditStore, s: float) -> None:
    with pytest.raises(ValueError, match="scalar multiple"):
        store.save(Q0, (store.original(Q0) * s).astype(np.float64))
    assert_clean(store)


def test_scalar_multiple_hidden_under_bf16_noise(store: EditStore) -> None:
    w0 = store.original(Q0)
    noise = np.random.default_rng(3).standard_normal(w0.shape).astype(np.float32)
    w = 1.1 * w0 * (1 + 1e-4 * noise)  # passes the 1e-6 pre-check
    with pytest.raises(ValueError, match=r"scalar multiple.*s=1\.0\d+"):
        store.save(Q0, w)
    assert_clean(store)


def test_below_resolution(store: EditStore) -> None:
    w0 = store.original(Q0)
    noise = np.random.default_rng(3).standard_normal(w0.shape).astype(np.float32)
    with pytest.raises(ValueError, match="below resolution"):
        store.save(Q0, w0 * (1 + 1e-4 * noise))
    assert_clean(store)


def test_too_large_change(store: EditStore) -> None:
    w = edit_of(store, Q0, rel=MAX_REL_CHANGE + 0.3)
    with pytest.raises(ValueError, match="above the cap"):
        store.save(Q0, w)
    assert_clean(store)
    store.save(Q0, edit_of(store, Q0, rel=MAX_REL_CHANGE - 0.1))


def test_failed_save_keeps_old_edit(store: EditStore) -> None:
    store.save(Q0, edit_of(store, Q0))
    before = edit_file(store, Q0).read_bytes()
    with pytest.raises(ValueError, match="scalar multiple"):
        store.save(Q0, store.original(Q0) * 2)
    assert edit_file(store, Q0).read_bytes() == before
    assert files(store) == sorted([edit_file(store, Q0).name, "manifest.json"])


def test_f16_overflow(tmp_path: Path) -> None:
    s = EditStore(tmp_path / "run", fake_model_dir(tmp_path / "m", dtype="f16"))
    w = edit_of(s, Q0)
    w[0, 0] = 1e6
    with pytest.raises(ValueError, match="after rounding"):
        s.save(Q0, w)


# validate


def test_validate_empty(store: EditStore) -> None:
    assert store.validate() == {}


def test_tamper_edit_file(store: EditStore) -> None:
    store.save(Q0, edit_of(store, Q0))
    path = edit_file(store, Q0)
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0x55
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="sha256"):
        store.validate()
    with pytest.raises(ValueError, match="sha256"):
        store.load(Q0)
    assert store.edited() == [Q0]  # not silently dropped


def test_missing_edit_file(store: EditStore) -> None:
    store.save(Q0, edit_of(store, Q0))
    edit_file(store, Q0).unlink()
    with pytest.raises(ValueError, match="unreadable"):
        store.validate()


def _rewrite(store: EditStore, name: str, tensors: dict[str, np.ndarray]) -> None:
    """Replace an edit file and fix the manifest sha, as a forger with disk access would."""
    from safetensors.numpy import save_file

    m = json.loads(store.manifest_path.read_text())
    old = store.dir / f"{m[name]['sha256']}.safetensors"
    tmp = store.dir / "forge.tmp"
    save_file(tensors, str(tmp))
    m[name]["sha256"] = _sha256_file(tmp)
    tmp.replace(store.dir / f"{m[name]['sha256']}.safetensors")
    old.unlink(missing_ok=True)
    store.manifest_path.write_text(json.dumps(m))


def test_forged_file_with_fixed_sha_fails_rules(store: EditStore) -> None:
    store.save(Q0, edit_of(store, Q0))
    _rewrite(store, Q0, {Q0: store.original(Q0) * np.float32(2)})
    with pytest.raises(ValueError, match="scalar multiple"):
        store.validate()
    _rewrite(store, Q0, {Q0: store.original(Q0)})
    with pytest.raises(ValueError, match="below resolution"):
        store.validate()
    _rewrite(store, Q0, {Q0: edit_of(store, Q0, rel=0.9)})
    with pytest.raises(ValueError, match=r"above the cap|not in the base dtype"):
        store.validate()
    w = edit_of(store, Q0)
    w[0, 0] = np.nan
    _rewrite(store, Q0, {Q0: w})
    with pytest.raises(ValueError, match=r"not in the base dtype|non-finite"):
        store.validate()


def test_forged_file_wrong_contents(store: EditStore) -> None:
    store.save(Q0, edit_of(store, Q0))
    good = edit_of(store, Q0)
    _rewrite(store, Q0, {"other": good})
    with pytest.raises(ValueError, match="one float32 tensor"):
        store.validate()
    _rewrite(store, Q0, {Q0: good.astype(np.float64)})
    with pytest.raises(ValueError, match="one float32 tensor"):
        store.validate()
    _rewrite(store, Q0, {Q0: good[:4]})
    with pytest.raises(ValueError, match="stored shape"):
        store.validate()
    _rewrite(store, Q0, {Q0: good})  # unrounded float32 values
    with pytest.raises(ValueError, match="base dtype"):
        store.validate()
    m = json.loads(store.manifest_path.read_text())
    junk = store.dir / "junk"
    junk.write_bytes(b"junk")
    m[Q0]["sha256"] = _sha256_file(junk)
    junk.replace(store.dir / f"{m[Q0]['sha256']}.safetensors")
    store.manifest_path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="corrupt"):
        store.validate()


def test_tamper_manifest(store: EditStore) -> None:
    store.save(Q0, edit_of(store, Q0))
    good = store.manifest_path.read_text()
    m = json.loads(good)
    # names a non-editable tensor
    store.manifest_path.write_text(
        json.dumps({**m, "model.embed_tokens.weight": m[Q0]})
    )
    with pytest.raises(ValueError, match="non-editable"):
        store.validate()
    # wrong sha
    store.manifest_path.write_text(json.dumps({Q0: {**m[Q0], "sha256": "0" * 64}}))
    with pytest.raises(ValueError, match="unreadable"):
        store.validate()
    assert len(files(store)) == 2  # the real edit file was not deleted as an orphan
    # a sha that is not 64 hex digits never becomes a path
    for bad_sha in ("../x", "A" * 64, "0" * 63):
        store.manifest_path.write_text(json.dumps({Q0: {**m[Q0], "sha256": bad_sha}}))
        with pytest.raises(ValueError, match="malformed"):
            store.validate()
    for bad in ("not json", "[]", '{"a": 1}', json.dumps({Q0: {"rel_frobenius": 1}})):
        store.manifest_path.write_text(bad)
        with pytest.raises(ValueError, match="manifest"):
            store.validate()
        with pytest.raises(ValueError, match="manifest"):
            store.edited()
    store.manifest_path.write_text(good)
    assert Q0 in store.validate()


# bf16 rounding


def test_round_bf16_known_values() -> None:
    # (float32 bit pattern in, bf16 bit pattern out as float32 bits)
    cases = {
        0x3F800000: 0x3F800000,  # 1.0
        0xC0200000: 0xC0200000,  # -2.5
        0x00000000: 0x00000000,
        0x80000000: 0x80000000,  # -0.0
        0x3F808000: 0x3F800000,  # tie, lower neighbour is even: down
        0x3F818000: 0x3F820000,  # tie, upper neighbour is even: up
        0x3F808001: 0x3F810000,  # just above a tie: up
        0x3F807FFF: 0x3F800000,  # just below a tie: down
        0x40490FDB: 0x40490000,  # pi
        0xC0490FDB: 0xC0490000,  # -pi
        0x00018000: 0x00020000,  # subnormal tie, upper is even
        0x7F7FFFFF: 0x7F800000,  # max float32 rounds to inf
        0xFF7FFFFF: 0xFF800000,
    }
    x = np.array(list(cases), dtype=np.uint32).view(np.float32)
    got = _round_bf16(x).view(np.uint32)
    assert [hex(int(g)) for g in got] == [hex(v) for v in cases.values()]


def test_round_bf16_nan_inf_passthrough() -> None:
    out = _round_bf16(np.array([np.inf, -np.inf, np.nan, -np.nan], np.float32))
    assert out[0] == np.inf
    assert out[1] == -np.inf
    assert np.isnan(out[2])
    assert np.isnan(out[3])


def test_round_bf16_matches_truncation_on_exact_values() -> None:
    w = fake_weights()[Q0]
    assert np.array_equal(_round_bf16(w), w)
    x = np.random.default_rng(0).standard_normal(10_000).astype(np.float32)
    r = _round_bf16(x)
    assert (r.view(np.uint32) & 0xFFFF).max() == 0
    assert np.all(np.abs(r - x) <= np.abs(x) * 2.0**-8)  # at most half an ulp


# threads


def test_threads_save_different_names(store: EditStore) -> None:
    errors: list[BaseException] = []
    names = list(store.names) * 2  # 16 names
    assert len(names) == 16
    weights = {n: edit_of(store, n) for n in store.names}

    def run(name: str) -> None:
        try:
            store.save(name, weights[name])
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    # 16 threads, two per name: same-name races must also leave a valid store
    threads = [threading.Thread(target=run, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert store.edited() == sorted(store.names)
    assert set(json.loads(store.manifest_path.read_text())) == set(store.names)
    assert len(store.validate()) == len(store.names)
    assert not [p for p in store.dir.iterdir() if p.name.endswith(".tmp")]


# export


def edit_some(store: EditStore, names: tuple[str, ...]) -> None:
    for i, n in enumerate(names):
        store.save(n, edit_of(store, n, seed=10 + i))


def test_export_adapter(store: EditStore, tmp_path: Path) -> None:
    edit_some(store, (O0, D0))
    out = tmp_path / "out"
    res = export_edits(store, out, rank=4, max_rel_residual=0.9, base_model="org/tiny")
    assert res["gguf"] is None
    assert res["rank"] == 4
    assert res["adapter"] == out / "adapter"
    edit = load_peft(res["adapter"])
    assert edit.base_model == "org/tiny"
    assert {d.module for d in edit.deltas} == {
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.down_proj",
    }
    assert edit.base_sha256[O0] == tensor_sha256(
        store.model_dir / "model.safetensors", O0
    )
    assert set(res["residuals"]) == {O0, D0}
    for d in edit.deltas:
        name = d.module + ".weight"
        dw = store.load(name) - store.original(name)
        got = np.linalg.norm(dw - d.dense()) / np.linalg.norm(dw)
        assert got == pytest.approx(res["residuals"][name], abs=1e-4)
        assert got < 0.9


def test_export_sharded_full_rank_is_exact(tmp_path: Path) -> None:
    s = EditStore(tmp_path / "run", fake_model_dir(tmp_path / "m", sharded=True))
    edit_some(s, (O0,))
    res = export_edits(
        s, tmp_path / "o", rank=32, max_rel_residual=1e-4, base_model="m"
    )
    assert res["residuals"][O0] < 1e-4


def test_export_residual_cap_raises(store: EditStore, tmp_path: Path) -> None:
    store.save(
        O0,
        store.original(O0)
        + np.random.default_rng(0).standard_normal((32, 32)).astype(np.float32) * 0.2,
    )
    with pytest.raises(ValueError, match="max_rel_residual"):
        export_edits(
            store, tmp_path / "o", rank=1, max_rel_residual=0.1, base_model="m"
        )
    assert not (tmp_path / "o").exists()


def test_export_mixed_ranks_padded(store: EditStore, tmp_path: Path) -> None:
    edit_some(store, (K0, O0))  # k_proj is (16, 32): rank cut to 16, o_proj keeps 20
    res = export_edits(
        store, tmp_path / "o", rank=20, max_rel_residual=1.0, base_model="m"
    )
    assert res["rank"] == 20
    edit = load_peft(res["adapter"])
    assert {d.rank for d in edit.deltas} == {20}
    k = next(d for d in edit.deltas if d.module.endswith("k_proj"))
    assert np.linalg.matrix_rank(k.dense()) <= 16
    dw = store.load(K0) - store.original(K0)
    assert np.linalg.norm(dw - k.dense()) / np.linalg.norm(dw) < 1e-4


def test_export_no_edits(store: EditStore, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no edits"):
        export_edits(
            store, tmp_path / "o", rank=2, max_rel_residual=1.0, base_model="m"
        )


def test_export_validates_first(store: EditStore, tmp_path: Path) -> None:
    store.save(O0, edit_of(store, O0))
    path = edit_file(store, O0)
    path.write_bytes(path.read_bytes()[:-1] + b"\x01")
    with pytest.raises(ValueError, match="sha256"):
        export_edits(
            store, tmp_path / "o", rank=2, max_rel_residual=1.0, base_model="m"
        )
    assert not (tmp_path / "o").exists()


def test_export_twice_overwrites(store: EditStore, tmp_path: Path) -> None:
    edit_some(store, (O0,))
    kw: dict[str, Any] = {"rank": 2, "max_rel_residual": 1.0, "base_model": "m"}
    export_edits(store, tmp_path / "o", **kw)
    export_edits(store, tmp_path / "o", **kw)


# export with GGUF


def tiny_gguf(path: Path, store: EditStore) -> Path:
    """GGUF of the fake model: blk.N.attn_output / ffn_down hold the base weights."""
    gguf = pytest.importorskip("gguf", reason="needs the gguf extra")
    from gguf import GGMLQuantizationType as Q
    from gguf.quants import quantize

    w = gguf.GGUFWriter(path, "qwen35")
    w.add_block_count(3)
    w.add_uint32("qwen35.nextn_predict_layers", 0)
    for n in (0, 1):
        for hf, g in (
            ("self_attn.o_proj", "attn_output"),
            ("mlp.down_proj", "ffn_down"),
        ):
            base = store.original(f"model.layers.{n}.{hf}.weight")
            w.add_tensor(
                f"blk.{n}.{g}.weight", quantize(base, Q.Q8_0), raw_dtype=Q.Q8_0
            )
    w.add_tensor("token_embd.weight", np.ones((12, 8), np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    return path


def test_export_with_gguf(store: EditStore, tmp_path: Path) -> None:
    base = tiny_gguf(tmp_path / "base.gguf", store)
    edit_some(store, (O0, D0))
    res = export_edits(
        store,
        tmp_path / "o",
        rank=32,
        max_rel_residual=1e-3,
        base_model="m",
        base_gguf=base,
    )
    assert res["gguf"] == tmp_path / "o" / "model.gguf"
    from gguf import GGUFReader
    from gguf.quants import dequantize

    tensors = {t.name: t for t in GGUFReader(res["gguf"]).tensors}
    edit = load_peft(res["adapter"])
    for d in edit.deltas:
        name = d.module + ".weight"
        merged = store.original(name) + d.dense()
        g = {"o_proj": "attn_output", "down_proj": "ffn_down"}[
            d.module.rsplit(".", 1)[1]
        ]
        t = tensors[f"blk.0.{g}.weight"]
        got = dequantize(t.data, t.tensor_type)
        assert np.linalg.norm(got - merged) / np.linalg.norm(merged) < 2e-2


def test_export_gguf_tensor_missing_from_base_keeps_adapter_only(
    store: EditStore, tmp_path: Path
) -> None:
    """q_proj maps to attn_q, which the tiny base lacks: splice fails after the adapter."""
    base = tiny_gguf(tmp_path / "base.gguf", store)
    edit_some(store, (O0, Q0))
    with pytest.raises(ValueError, match="not in base"):
        export_edits(
            store,
            tmp_path / "o",
            rank=4,
            max_rel_residual=1.0,
            base_model="m",
            base_gguf=base,
        )
    assert (tmp_path / "o" / "adapter").is_dir()
    assert not (tmp_path / "o" / "model.gguf").exists()


def test_export_gguf_failure_leaves_no_gguf(store: EditStore, tmp_path: Path) -> None:
    base = tiny_gguf(tmp_path / "base.gguf", store)
    edit_some(store, (O0,))
    with pytest.raises(ValueError, match="out_type"):
        export_edits(
            store,
            tmp_path / "o",
            rank=4,
            max_rel_residual=1.0,
            base_model="m",
            base_gguf=base,
            gguf_out_type="q4_k",
        )
    assert not (tmp_path / "o" / "model.gguf").exists()


def test_export_gguf_verify_problem_removes_file(
    store: EditStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tiny_gguf(tmp_path / "base.gguf", store)
    from astro_weights.export import gguf as g

    edit_some(store, (O0,))
    monkeypatch.setattr(g, "verify_splice", lambda *a, **k: ["boom"])
    with pytest.raises(ValueError, match="boom"):
        export_edits(
            store,
            tmp_path / "o",
            rank=4,
            max_rel_residual=1.0,
            base_model="m",
            base_gguf=base,
        )
    assert not (tmp_path / "o" / "model.gguf").exists()


def test_read_tensor_sanity(model: Path) -> None:
    assert read_tensor(model / "model.safetensors", Q0).shape == (32, 32)


def test_broken_model_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="cannot read header"):
        EditStore(tmp_path / "r1", empty)
    (empty / "model.safetensors.index.json").write_text("{nope")
    with pytest.raises(ValueError, match="invalid index"):
        EditStore(tmp_path / "r2", empty)


# review fixes


def test_save_is_atomic_when_manifest_write_fails(
    store: EditStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.save(O0, edit_of(store, O0, seed=1))
    before = store.load(O0)
    snapshot = {p.name: p.read_bytes() for p in store.dir.iterdir()}

    def boom(manifest: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store, "_write_manifest", boom)
    with pytest.raises(OSError, match="disk full"):
        store.save(O0, edit_of(store, O0, seed=2))  # re-save of an existing edit
    with pytest.raises(OSError, match="disk full"):
        store.save(Q0, edit_of(store, Q0))  # a new name
    with pytest.raises(OSError, match="disk full"):
        store.save(O0, edit_of(store, O0, seed=1))  # same content: its file is kept
    assert {p.name: p.read_bytes() for p in store.dir.iterdir()} == snapshot
    assert np.array_equal(store.load(O0), before)
    assert store.edited() == [O0]
    assert O0 in store.validate()


def test_save_with_bad_manifest_touches_nothing(store: EditStore) -> None:
    store.manifest_path.write_text("not json")
    with pytest.raises(ValueError, match="manifest"):
        store.save(Q0, edit_of(store, Q0))
    assert files(store) == ["manifest.json"]


def test_resave_replaces_file_and_unlinks_old(store: EditStore) -> None:
    store.save(O0, edit_of(store, O0, seed=1))
    first = edit_file(store, O0)
    store.save(O0, edit_of(store, O0, seed=2))
    assert edit_file(store, O0) != first
    assert not first.exists()
    assert files(store) == sorted([edit_file(store, O0).name, "manifest.json"])
    store.save(O0, edit_of(store, O0, seed=2))  # same content: same file
    assert len(files(store)) == 2


def test_validate_removes_orphans_only(store: EditStore) -> None:
    store.save(O0, edit_of(store, O0))
    orphan = store.dir / f"{'a' * 64}.safetensors"
    orphan.write_bytes(b"x")
    (store.dir / "half.tmp").write_bytes(b"x")
    (store.dir / "notes.txt").write_text("keep")
    store.validate()
    assert not orphan.exists()
    assert not (store.dir / "half.tmp").exists()
    assert (store.dir / "notes.txt").exists()
    assert edit_file(store, O0).exists()


def test_edit_file_names_are_content_hashes(store: EditStore) -> None:
    store.save(O0, edit_of(store, O0))
    path = edit_file(store, O0)
    assert path.stem == _sha256_file(path)


def test_public_surface() -> None:
    from astro_weights.mcp import edit_store

    public = {n for n in vars(edit_store) if not n.startswith("_")}
    assert {"MAX_REL_CHANGE", "EditStore", "export_edits"} <= public
    for gone in ("round_bf16", "scalar_fit", "rel_frobenius", "MAX_SCALAR_RESIDUAL"):
        assert gone not in public


def test_export_holds_no_originals_without_gguf(
    store: EditStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gc
    import weakref

    edit_some(store, (O0, D0, Q0))
    alive: list[weakref.ref[np.ndarray]] = []
    real = store.original

    def tracked(name: str) -> np.ndarray:
        arr = real(name)
        alive.append(weakref.ref(arr))
        return arr

    seen: list[int] = []
    real_save = save_peft

    def spy(*a: Any, **k: Any) -> None:
        gc.collect()
        seen.append(sum(r() is not None for r in alive))
        real_save(*a, **k)

    monkeypatch.setattr(store, "original", tracked)
    monkeypatch.setattr("astro_weights.mcp.edit_store.save_peft", spy)
    export_edits(store, tmp_path / "o", rank=4, max_rel_residual=1.0, base_model="m")
    assert alive
    assert seen == [0]


def test_export_is_one_snapshot(store: EditStore, tmp_path: Path) -> None:
    """A drop from another thread waits until the export has read every edit."""
    edit_some(store, (O0, D0))
    started = threading.Event()
    outcome: dict[str, Any] = {}

    def dropper() -> None:
        started.set()
        store.drop(O0)
        outcome["done"] = True

    thread = threading.Thread(target=dropper)
    real_load = store.load
    loads: list[str] = []

    def load(name: str) -> np.ndarray:
        if not loads:
            thread.start()
            started.wait()
            thread.join(0.3)
            outcome["blocked"] = thread.is_alive()
        loads.append(name)
        return real_load(name)

    store.load = load  # type: ignore[method-assign]
    res = export_edits(
        store, tmp_path / "o", rank=4, max_rel_residual=1.0, base_model="m"
    )
    thread.join(5)
    assert outcome == {"blocked": True, "done": True}
    assert set(res["residuals"]) == {O0, D0}
    assert store.edited() == [D0]


def test_reexport_without_gguf_removes_stale_gguf(
    store: EditStore, tmp_path: Path
) -> None:
    base = tiny_gguf(tmp_path / "base.gguf", store)
    edit_some(store, (O0,))
    out = tmp_path / "o"
    kw: dict[str, Any] = {"rank": 8, "max_rel_residual": 1.0, "base_model": "m"}
    assert export_edits(store, out, base_gguf=base, **kw)["gguf"] == out / "model.gguf"
    assert (out / "model.gguf").exists()
    store.save(D0, edit_of(store, D0))
    res = export_edits(store, out, **kw)
    assert res["gguf"] is None
    assert not (out / "model.gguf").exists()


def test_reexport_splice_failure_removes_stale_gguf(
    store: EditStore, tmp_path: Path
) -> None:
    base = tiny_gguf(tmp_path / "base.gguf", store)
    edit_some(store, (O0,))
    out = tmp_path / "o"
    kw: dict[str, Any] = {"rank": 8, "max_rel_residual": 1.0, "base_model": "m"}
    export_edits(store, out, base_gguf=base, **kw)
    with pytest.raises(ValueError, match="out_type"):
        export_edits(store, out, base_gguf=base, gguf_out_type="q4_k", **kw)
    assert not (out / "model.gguf").exists()
