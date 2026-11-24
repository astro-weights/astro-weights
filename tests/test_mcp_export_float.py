"""Export takes dW from the agent's unrounded float edit, not the bf16-rounded one."""

import json
from pathlib import Path

import numpy as np
import pytest
from mcp_fakes import fake_model_dir
from safetensors.numpy import save_file

from astro_weights.edits import from_dense
from astro_weights.export.peft import load_peft
from astro_weights.mcp.edit_store import (
    EditStore,
    _round_bf16,
    _sha256_file,
    export_edits,
)

O0 = "model.layers.0.self_attn.o_proj.weight"
Q0 = "model.layers.0.self_attn.q_proj.weight"
RANK = 4


def small_edit(store: EditStore, name: str, rel: float = 0.005) -> np.ndarray:
    """Agent-style edit: base plus a rank-RANK float32 change of relative norm `rel`."""
    w0 = store.original(name)
    rng = np.random.default_rng(7)
    d = rng.standard_normal((w0.shape[0], RANK)) @ rng.standard_normal(
        (RANK, w0.shape[1])
    )
    d *= rel * np.linalg.norm(w0) / np.linalg.norm(d)
    out: np.ndarray = (w0 + d).astype(np.float32)
    return out


def export(store: EditStore, out: Path) -> dict[str, object]:
    return export_edits(store, out, rank=RANK, max_rel_residual=1.0, base_model="m")


def manifest(store: EditStore) -> dict[str, dict[str, object]]:
    data: dict[str, dict[str, object]] = json.loads(store.manifest_path.read_text())
    return data


def to_old_format(store: EditStore) -> None:
    """Rewrite the manifest as a store from before float edits were kept."""
    m = manifest(store)
    for rec in m.values():
        rec.pop("float_sha256")
    store.manifest_path.write_text(json.dumps(m))
    store.validate()  # the float files are now orphans and go


def rounded_residual(store: EditStore, name: str) -> float:
    dw = store.load(name).astype(np.float64) - store.original(name)
    return from_dense(name, dw, RANK)[1]


@pytest.fixture
def bf16(tmp_path: Path) -> EditStore:
    return EditStore(tmp_path / "run", fake_model_dir(tmp_path / "model"))


def test_bf16_export_uses_float_edit(bf16: EditStore, tmp_path: Path) -> None:
    bf16.save(O0, small_edit(bf16, O0))
    res = export(bf16, tmp_path / "o")
    assert res["delta_source"] == {O0: "float"}
    assert res["residuals"][O0] < 1e-3  # type: ignore[index]
    # the same edit through the rounded weight is mostly bf16 noise
    assert rounded_residual(bf16, O0) > 0.05
    edit = load_peft(res["adapter"])  # type: ignore[arg-type]
    assert edit.meta["delta_source"] == {O0: "float"}
    assert edit.meta["replay_max_abs"] == res["replay_max_abs"]


def test_adapter_replays_the_stored_weight(bf16: EditStore, tmp_path: Path) -> None:
    bf16.save(O0, small_edit(bf16, O0))
    res = export(bf16, tmp_path / "o")
    (d,) = load_peft(res["adapter"]).deltas  # type: ignore[arg-type]
    stored = bf16.load(O0)
    replayed = _round_bf16(bf16.original(O0) + d.dense())
    ulp = 2.0 ** (np.floor(np.log2(np.abs(stored).max())) - 7)  # bf16: 8-bit mantissa
    diff = float(np.abs(replayed - stored).max())
    assert diff <= ulp
    assert res["replay_max_abs"] == {O0: diff}


def test_old_format_store_exports_rounded(bf16: EditStore, tmp_path: Path) -> None:
    bf16.save(O0, small_edit(bf16, O0))
    to_old_format(bf16)
    assert bf16.float_edit(O0) is None
    assert len(list(bf16.dir.glob("*.safetensors"))) == 1
    res = export(bf16, tmp_path / "o")
    assert res["delta_source"] == {O0: "rounded"}
    assert res["residuals"][O0] == pytest.approx(rounded_residual(bf16, O0))  # type: ignore[index]
    assert load_peft(res["adapter"]).meta["delta_source"] == {O0: "rounded"}  # type: ignore[arg-type]


def test_mixed_store_records_each_source(bf16: EditStore, tmp_path: Path) -> None:
    bf16.save(O0, small_edit(bf16, O0))
    to_old_format(bf16)
    bf16.save(Q0, small_edit(bf16, Q0))  # saved by the new code: keeps its float edit
    assert export(bf16, tmp_path / "o")["delta_source"] == {O0: "rounded", Q0: "float"}


def test_f32_base_one_file_same_export(tmp_path: Path) -> None:
    store = EditStore(tmp_path / "run", fake_model_dir(tmp_path / "m", dtype="f32"))
    w = small_edit(store, O0)
    store.save(O0, w)
    rec = manifest(store)[O0]
    assert rec["float_sha256"] == rec["sha256"]
    assert len(list(store.dir.glob("*.safetensors"))) == 1
    assert np.array_equal(store.float_edit(O0), store.load(O0))  # type: ignore[arg-type]
    assert np.array_equal(store.load(O0), w)
    res = export(store, tmp_path / "o")
    assert res["delta_source"] == {O0: "float"}
    assert res["residuals"][O0] == pytest.approx(rounded_residual(store, O0))  # type: ignore[index]
    store.drop(O0)
    assert list(store.dir.glob("*.safetensors")) == []


def test_tampered_float_file_refused(bf16: EditStore) -> None:
    bf16.save(O0, small_edit(bf16, O0))
    m = manifest(bf16)
    path = bf16.dir / f"{m[O0]['float_sha256']}.safetensors"
    # changed bytes, manifest untouched
    good = path.read_bytes()
    path.write_bytes(good[:-4] + b"\0\0\0\0")
    with pytest.raises(ValueError, match="sha256 does not match"):
        bf16.validate()
    path.write_bytes(good)
    # a forger who also fixes the sha: the float edit must round to the stored one
    forged = bf16.dir / "forged.tmp"
    save_file({O0: small_edit(bf16, O0, rel=0.05)}, str(forged))
    m[O0]["float_sha256"] = sha = _sha256_file(forged)
    forged.replace(bf16.dir / f"{sha}.safetensors")
    bf16.manifest_path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="does not round to the stored edit"):
        bf16.validate()
    with pytest.raises(ValueError, match="does not round"):
        export(bf16, bf16.dir.parent / "o")
    # a float sha that is not 64 hex digits never becomes a path
    for bad in ("../x", "A" * 64, 3):
        m[O0]["float_sha256"] = bad
        bf16.manifest_path.write_text(json.dumps(m))
        with pytest.raises(ValueError, match="malformed"):
            bf16.validate()


def test_drop_and_resave_remove_float_files(bf16: EditStore) -> None:
    bf16.save(O0, small_edit(bf16, O0))
    first = set(bf16.dir.glob("*.safetensors"))
    assert len(first) == 2
    bf16.save(O0, small_edit(bf16, O0, rel=0.01))
    now = set(bf16.dir.glob("*.safetensors"))
    assert len(now) == 2
    assert not first & now
    bf16.drop(O0)
    assert list(bf16.dir.glob("*.safetensors")) == []


def test_float_edit_needs_an_edit(bf16: EditStore) -> None:
    with pytest.raises(ValueError, match="has no edit"):
        bf16.float_edit(O0)
