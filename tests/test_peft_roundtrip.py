import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from astro_weights import __version__
from astro_weights.edits import Edit, LowRankDelta
from astro_weights.export import load_peft, save_peft, verify_peft

MOD_A = "model.layers.1.mlp.down_proj"
MOD_B = "model.layers.2.mlp.down_proj"


def make_edit() -> Edit:
    rng = np.random.default_rng(0)
    deltas = tuple(
        LowRankDelta(m, rng.standard_normal((16, 1)), rng.standard_normal((1, 48)))
        for m in (MOD_B, MOD_A)
    )
    return Edit(
        deltas,
        "org/tiny",
        base_sha256={"w": "abc"},
        meta={"alpha": 0.5, "note": "x"},
    )


def by_module(edit: Edit) -> dict[str, LowRankDelta]:
    return {d.module: d for d in edit.deltas}


def write_adapter(
    path: Path,
    tensors: dict[str, np.ndarray],
    **cfg: object,
) -> None:
    config: dict[str, object] = {
        "peft_type": "LORA",
        "base_model_name_or_path": "org/shipped",
        "r": 1,
        "lora_alpha": 1,
        "fan_in_fan_out": False,
        "target_modules": ["down_proj"],
    }
    config.update(cfg)
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps(config))


def pair(module: str, r: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    return {
        f"base_model.model.{module}.lora_A.weight": rng.standard_normal((r, 48)).astype(
            np.float32
        ),
        f"base_model.model.{module}.lora_B.weight": rng.standard_normal((16, r)).astype(
            np.float32
        ),
    }


def test_roundtrip(tmp_path: Path) -> None:
    edit = make_edit()
    save_peft(edit, tmp_path / "out")
    loaded = load_peft(tmp_path / "out")
    assert set(by_module(loaded)) == {MOD_A, MOD_B}
    for name, d in by_module(edit).items():
        got = by_module(loaded)[name]
        assert np.array_equal(got.A, d.A)
        assert np.array_equal(got.B, d.B)
    assert loaded.rank == edit.rank == 1
    assert loaded.base_model == "org/tiny"
    assert loaded.base_sha256 == {"w": "abc"}
    assert loaded.meta == {"alpha": 0.5, "note": "x"}
    assert verify_peft(tmp_path / "out") == []


def test_written_layout(tmp_path: Path) -> None:
    save_peft(make_edit(), str(tmp_path))
    tensors = load_file(str(tmp_path / "adapter_model.safetensors"))
    assert sorted(tensors) == [
        f"base_model.model.{m}.lora_{ab}.weight" for m in (MOD_A, MOD_B) for ab in "AB"
    ]
    assert tensors[f"base_model.model.{MOD_A}.lora_A.weight"].shape == (1, 48)
    assert tensors[f"base_model.model.{MOD_A}.lora_B.weight"].shape == (16, 1)
    assert all(t.dtype == np.float32 for t in tensors.values())
    cfg_text = (tmp_path / "adapter_config.json").read_text()
    assert cfg_text.endswith("\n")
    assert json.loads(cfg_text) == {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "org/tiny",
        "r": 1,
        "lora_alpha": 1,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "target_modules": [MOD_A, MOD_B],
    }
    manifest = json.loads((tmp_path / "astro_manifest.json").read_text())
    assert manifest["format"] == 1
    assert manifest["astro_weights_version"] == __version__
    assert manifest["base_model"] == "org/tiny"
    assert sorted(manifest["factors"]) == sorted(tensors)


def test_alpha_scaling_folded_into_b(tmp_path: Path) -> None:
    tensors = pair(MOD_A, r=2)
    write_adapter(tmp_path, tensors, r=2, lora_alpha=6)
    got = by_module(load_peft(tmp_path))[MOD_A]
    b = tensors[f"base_model.model.{MOD_A}.lora_B.weight"]
    a = tensors[f"base_model.model.{MOD_A}.lora_A.weight"]
    assert np.allclose(got.B, b * 3)
    assert np.array_equal(got.A, a)


def test_shipped_layout_without_manifest(tmp_path: Path) -> None:
    m18 = "model.language_model.layers.18.mlp.down_proj"
    m20 = "model.language_model.layers.20.mlp.down_proj"
    write_adapter(tmp_path, {**pair(m18), **pair(m20)})
    edit = load_peft(tmp_path)
    assert {d.module for d in edit.deltas} == {m18, m20}
    assert edit.base_model == "org/shipped"
    assert edit.meta == {}
    assert edit.deltas[0].shape == (16, 48)


def test_f16_is_cast(tmp_path: Path) -> None:
    tensors = {k: v.astype(np.float16) for k, v in pair(MOD_A).items()}
    write_adapter(tmp_path, tensors)
    assert by_module(load_peft(tmp_path))[MOD_A].A.dtype == np.float32


def test_bf16_rejected(tmp_path: Path) -> None:
    write_adapter(tmp_path, pair(MOD_A))
    # numpy has no bfloat16, so hand-write a valid BF16 file: 2 bytes per element
    header: dict[str, object] = {}
    offset = 0
    for key, shape in (("lora_A", [1, 48]), ("lora_B", [16, 1])):
        n = 2 * shape[0] * shape[1]
        header[f"base_model.model.{MOD_A}.{key}.weight"] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [offset, offset + n],
        }
        offset += n
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    blob = len(head).to_bytes(8, "little") + head + bytes(offset)
    (tmp_path / "adapter_model.safetensors").write_bytes(blob)
    with pytest.raises(ValueError, match="BF16"):
        load_peft(tmp_path)
    assert "BF16" in verify_peft(tmp_path)[0]


def test_corrupt_weights_reported(tmp_path: Path) -> None:
    save_peft(make_edit(), tmp_path)
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(path.read_bytes()[:-10])
    problems = verify_peft(tmp_path)
    assert len(problems) == 1


def test_tamper_detected(tmp_path: Path) -> None:
    save_peft(make_edit(), tmp_path)
    path = tmp_path / "adapter_model.safetensors"
    tensors = load_file(str(path))
    key = f"base_model.model.{MOD_A}.lora_B.weight"
    tensors[key] = tensors[key] + np.float32(1.0)
    save_file(tensors, str(path))
    with pytest.raises(ValueError, match=r"lora_B"):
        load_peft(tmp_path)
    problems = verify_peft(tmp_path)
    assert len(problems) == 1
    assert key in problems[0]


@pytest.mark.parametrize("field", ["use_dora", "use_rslora", "fan_in_fan_out"])
def test_unsupported_config(tmp_path: Path, field: str) -> None:
    write_adapter(tmp_path, pair(MOD_A), **{field: True})
    with pytest.raises(ValueError, match=field):
        load_peft(tmp_path)
    assert field in verify_peft(tmp_path)[0]


def test_wrong_peft_type(tmp_path: Path) -> None:
    write_adapter(tmp_path, pair(MOD_A), peft_type="IA3")
    with pytest.raises(ValueError, match="peft_type"):
        load_peft(tmp_path)


def test_missing_b(tmp_path: Path) -> None:
    tensors = pair(MOD_A)
    del tensors[f"base_model.model.{MOD_A}.lora_B.weight"]
    write_adapter(tmp_path, tensors)
    with pytest.raises(ValueError, match=MOD_A):
        load_peft(tmp_path)


@pytest.mark.parametrize("manifest", [[], None, {"factors": 5}, {"base_model": "x"}])
def test_malformed_manifest_is_value_error(tmp_path: Path, manifest: object) -> None:
    save_peft(make_edit(), tmp_path)
    (tmp_path / "astro_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"astro_manifest\.json"):
        load_peft(tmp_path)
