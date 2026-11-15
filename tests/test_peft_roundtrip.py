import json
import os
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from astro_weights import __version__
from astro_weights.edits import Edit, LowRankDelta
from astro_weights.export import load_peft, save_peft, verify_peft
from astro_weights.export.peft import read_peft

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


def load_error(path: Path, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        load_peft(path)
    problems = verify_peft(path)
    assert len(problems) == 1
    assert field in problems[0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("r", 0),
        ("r", -1),
        ("r", "1"),
        ("r", 1.0),
        ("r", True),
        ("r", None),
        ("lora_alpha", "1"),
        ("lora_alpha", True),
        ("lora_alpha", None),
        ("lora_alpha", [1]),
        ("lora_alpha", 1e999),
        ("lora_alpha", 10**400),
        ("rank_pattern", {"down_proj": 2}),
        ("alpha_pattern", {"down_proj": 64}),
        ("modules_to_save", ["lm_head"]),
        ("bias", "all"),
        ("bias", "lora_only"),
        ("bias", True),
        ("target_modules", 5),
        ("target_modules", [5]),
        ("target_modules", ["up_proj"]),
        ("base_model_name_or_path", ["x"]),
    ],
)
def test_unrepresentable_config_rejected(
    tmp_path: Path, field: str, value: object
) -> None:
    write_adapter(tmp_path, pair(MOD_A), **{field: value})
    load_error(tmp_path, field)


def test_missing_r_or_alpha(tmp_path: Path) -> None:
    write_adapter(tmp_path, pair(MOD_A))
    cfg = json.loads((tmp_path / "adapter_config.json").read_text())
    del cfg["r"]
    (tmp_path / "adapter_config.json").write_text(json.dumps(cfg))
    assert "r" in verify_peft(tmp_path)[0]


def test_target_modules_error_names_module(tmp_path: Path) -> None:
    write_adapter(tmp_path, pair(MOD_A), target_modules=["q_proj"])
    with pytest.raises(ValueError, match=MOD_A):
        load_peft(tmp_path)


@pytest.mark.parametrize(
    "targets",
    [["down_proj"], [MOD_A], ["mlp.down_proj", "q_proj"], ".*down_proj", None],
)
def test_target_modules_accepted(tmp_path: Path, targets: object) -> None:
    write_adapter(tmp_path, pair(MOD_A), target_modules=targets)
    assert MOD_A in by_module(load_peft(tmp_path))


def test_target_modules_absent(tmp_path: Path) -> None:
    write_adapter(tmp_path, pair(MOD_A))
    cfg = json.loads((tmp_path / "adapter_config.json").read_text())
    del cfg["target_modules"]
    (tmp_path / "adapter_config.json").write_text(json.dumps(cfg))
    assert MOD_A in by_module(load_peft(tmp_path))


def test_target_modules_suffix_needs_dot(tmp_path: Path) -> None:
    write_adapter(tmp_path, pair(MOD_A), target_modules=["proj"])
    with pytest.raises(ValueError, match="target_modules"):
        load_peft(tmp_path)


@pytest.mark.parametrize("empty", [None, {}, []])
@pytest.mark.parametrize("field", ["rank_pattern", "alpha_pattern"])
def test_empty_patterns_accepted(tmp_path: Path, field: str, empty: object) -> None:
    write_adapter(tmp_path, pair(MOD_A), **{field: empty})
    assert MOD_A in by_module(load_peft(tmp_path))


@pytest.mark.parametrize("empty", [None, []])
def test_empty_modules_to_save_accepted(tmp_path: Path, empty: object) -> None:
    write_adapter(tmp_path, pair(MOD_A), modules_to_save=empty)
    assert MOD_A in by_module(load_peft(tmp_path))


@pytest.mark.parametrize("bias", [None, "none"])
def test_bias_none_forms_accepted(tmp_path: Path, bias: object) -> None:
    write_adapter(tmp_path, pair(MOD_A), bias=bias)
    assert MOD_A in by_module(load_peft(tmp_path))


def test_bias_absent_and_float_alpha_accepted(tmp_path: Path) -> None:
    write_adapter(tmp_path, pair(MOD_A), lora_alpha=0.5)
    assert np.allclose(
        by_module(load_peft(tmp_path))[MOD_A].B,
        pair(MOD_A)[f"base_model.model.{MOD_A}.lora_B.weight"] * 0.5,
    )


@pytest.mark.parametrize("text", ["[]", "5", "null", '"x"', "{", "", "[" * 100000])
@pytest.mark.parametrize("name", ["adapter_config.json", "astro_manifest.json"])
def test_non_object_json_is_value_error(tmp_path: Path, name: str, text: str) -> None:
    save_peft(make_edit(), tmp_path)
    (tmp_path / name).write_text(text)
    with pytest.raises(ValueError, match=name.replace(".", r"\.")):
        load_peft(tmp_path)
    assert name in verify_peft(tmp_path)[0]


@pytest.mark.parametrize(
    "patch",
    [
        {"base_sha256": []},
        {"base_sha256": {"w": 1}},
        {"base_sha256": 5},
        {"meta": []},
        {"meta": "x"},
    ],
)
def test_manifest_field_types(tmp_path: Path, patch: dict[str, object]) -> None:
    save_peft(make_edit(), tmp_path)
    manifest = json.loads((tmp_path / "astro_manifest.json").read_text())
    manifest.update(patch)
    (tmp_path / "astro_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=next(iter(patch))):
        load_peft(tmp_path)
    assert next(iter(patch)) in verify_peft(tmp_path)[0]


def test_verify_never_raises_on_junk(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_bytes(b"\xff\xfe\x00")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"junk")
    assert all(isinstance(p, str) for p in verify_peft(tmp_path))
    assert all(isinstance(p, str) for p in verify_peft(tmp_path / "missing"))


def test_verify_never_raises_on_truncation(tmp_path: Path) -> None:
    save_peft(make_edit(), tmp_path)
    good = (tmp_path / "adapter_model.safetensors").read_bytes()
    for n in range(0, len(good), 37):
        (tmp_path / "adapter_model.safetensors").write_bytes(good[:n])
        problems = verify_peft(tmp_path)
        assert problems
        assert all(isinstance(p, str) for p in problems)


def other_edit() -> Edit:
    rng = np.random.default_rng(1)
    deltas = tuple(
        LowRankDelta(m, rng.standard_normal((16, 1)), rng.standard_normal((1, 48)))
        for m in (MOD_A, MOD_B)
    )
    return Edit(deltas, "org/other")


def snapshot(path: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(path.iterdir())}


def test_save_creates_missing_parents_and_leaves_no_temp(tmp_path: Path) -> None:
    out = tmp_path / "a" / "b" / "out"
    save_peft(make_edit(), out)
    assert verify_peft(out) == []
    assert [p.name for p in out.parent.iterdir()] == ["out"]


def test_save_into_empty_dir(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    save_peft(make_edit(), out)
    assert verify_peft(out) == []
    assert [p.name for p in tmp_path.iterdir()] == ["out"]


def test_save_refuses_non_empty_dir(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError, match=str(tmp_path)):
        save_peft(make_edit(), tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["keep.txt"]


def test_save_refuses_existing_adapter(tmp_path: Path) -> None:
    save_peft(make_edit(), tmp_path)
    before = snapshot(tmp_path)
    with pytest.raises(FileExistsError):
        save_peft(other_edit(), tmp_path)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("overwrite", [False, True])
def test_save_out_is_file(tmp_path: Path, overwrite: bool) -> None:
    out = tmp_path / "f"
    out.write_text("x")
    with pytest.raises(NotADirectoryError):
        save_peft(make_edit(), out, overwrite=overwrite)
    assert out.read_text() == "x"
    assert [p.name for p in tmp_path.iterdir()] == ["f"]


def test_save_out_is_dangling_symlink(tmp_path: Path) -> None:
    out = tmp_path / "link"
    out.symlink_to(tmp_path / "nowhere")
    with pytest.raises(NotADirectoryError):
        save_peft(make_edit(), out, overwrite=True)
    assert not (tmp_path / "nowhere").exists()


def test_overwrite_replaces_only_adapter_files(tmp_path: Path) -> None:
    save_peft(make_edit(), tmp_path)
    (tmp_path / "README.md").write_text("card")
    save_peft(other_edit(), tmp_path, overwrite=True)
    assert (tmp_path / "README.md").read_text() == "card"
    assert load_peft(tmp_path).base_model == "org/other"
    assert verify_peft(tmp_path) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "astro_manifest.json",
    ]


def test_overwrite_into_non_adapter_dir(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("x")
    save_peft(make_edit(), tmp_path, overwrite=True)
    assert (tmp_path / "keep.txt").read_text() == "x"
    assert verify_peft(tmp_path) == []


def test_overwrite_directory_in_the_way(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").mkdir()
    with pytest.raises(IsADirectoryError):
        save_peft(make_edit(), tmp_path, overwrite=True)
    assert [p.name for p in tmp_path.iterdir()] == ["adapter_config.json"]


def test_failed_write_leaves_new_dir_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("astro_weights.export.peft.save_file", boom)
    out = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt):
        save_peft(make_edit(), out)
    assert list(tmp_path.iterdir()) == []


def test_failed_write_leaves_existing_adapter_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    save_peft(make_edit(), out)
    before = snapshot(out)

    def boom(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("astro_weights.export.peft._write_json", boom)
    with pytest.raises(KeyboardInterrupt):
        save_peft(other_edit(), out, overwrite=True)
    assert snapshot(out) == before
    assert [p.name for p in tmp_path.iterdir()] == ["out"]
    assert verify_peft(out) == []


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_failure_between_replaces_is_never_a_silent_mix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_at: int
) -> None:
    out = tmp_path / "out"
    old = make_edit()
    save_peft(old, out)
    real_replace = os.replace
    calls = 0

    def flaky(src: object, dst: object) -> None:
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError("injected")
        real_replace(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(OSError, match="injected"):
        save_peft(other_edit(), out, overwrite=True)
    monkeypatch.undo()
    problems = verify_peft(out)
    if fail_at == 1:  # nothing moved: the complete old adapter
        assert problems == []
        loaded = load_peft(out)
        assert loaded.base_model == old.base_model
        assert np.array_equal(by_module(loaded)[MOD_A].A, by_module(old)[MOD_A].A)
    else:  # new manifest, old weights: reported, not accepted
        assert problems
        with pytest.raises(ValueError, match="hash mismatch"):
            load_peft(out)
    assert [p.name for p in tmp_path.iterdir()] == ["out"]


def test_overwrite_does_not_write_through_symlinks(tmp_path: Path) -> None:
    out = tmp_path / "out"
    save_peft(make_edit(), out)
    outside = tmp_path / "outside"
    outside.mkdir()
    victims = {}
    for name in (
        "adapter_model.safetensors",
        "adapter_config.json",
        "astro_manifest.json",
    ):
        victim = outside / name
        victim.write_text("precious")
        (out / name).unlink()
        (out / name).symlink_to(victim)
        victims[name] = victim
    save_peft(other_edit(), out, overwrite=True)
    for name, victim in victims.items():
        assert victim.read_text() == "precious"
        assert not (out / name).is_symlink()
    assert load_peft(out).base_model == "org/other"


def test_save_into_symlinked_empty_dir(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    save_peft(make_edit(), link)
    assert link.is_symlink()
    assert verify_peft(real) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["link", "real"]


@pytest.mark.parametrize("overwrite", [False, True])
def test_save_into_cwd_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overwrite: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    save_peft(make_edit(), ".", overwrite=overwrite)
    assert verify_peft(tmp_path) == []
    assert Path.cwd().exists()


def test_save_into_cwd_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_peft(make_edit(), tmp_path)
    (tmp_path / "README.md").write_text("keep")
    monkeypatch.chdir(tmp_path)
    save_peft(other_edit(), ".", overwrite=True)
    assert load_peft(tmp_path).base_model == "org/other"
    assert (tmp_path / "README.md").read_text() == "keep"


def test_save_into_dotdot_path(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    save_peft(make_edit(), tmp_path / "a" / ".." / "out")
    assert verify_peft(tmp_path / "out") == []


def test_save_into_empty_dir_keeps_inode_and_mode(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    out.chmod(0o750)
    before = out.stat()
    save_peft(make_edit(), out)
    after = out.stat()
    assert (after.st_ino, after.st_mode) == (before.st_ino, before.st_mode)
    assert verify_peft(out) == []
    assert [p.name for p in tmp_path.iterdir()] == ["out"]


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs permission checks",
)
def test_save_into_existing_dir_with_read_only_parent(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    out = parent / "out"
    out.mkdir(parents=True)
    (out / "README.md").write_text("keep")
    parent.chmod(0o555)
    try:
        save_peft(make_edit(), out, overwrite=True)
        assert verify_peft(out) == []
        assert sorted(p.name for p in out.iterdir()) == sorted(
            [
                "README.md",
                "adapter_model.safetensors",
                "adapter_config.json",
                "astro_manifest.json",
            ]
        )
        assert [p.name for p in parent.iterdir()] == ["out"]
    finally:
        parent.chmod(0o755)


@pytest.mark.parametrize("field", ["r", "lora_alpha"])
def test_read_peft_config_missing_field_is_value_error(
    tmp_path: Path, field: str
) -> None:
    save_peft(make_edit(), tmp_path)
    cfg_path = tmp_path / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    del cfg[field]
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match=f"missing key .{field}."):
        read_peft(tmp_path)
    (problem,) = verify_peft(tmp_path)
    assert f"missing key {field!r}" in problem


def test_read_peft_corrupt_safetensors_is_value_error(tmp_path: Path) -> None:
    save_peft(make_edit(), tmp_path)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"\x00" * 4)
    with pytest.raises(ValueError, match="not valid safetensors") as exc:
        read_peft(tmp_path)
    assert exc.value.__cause__ is not None
    (problem,) = verify_peft(tmp_path)
    assert "not valid safetensors" in problem


def test_read_peft_missing_weights_is_os_error(tmp_path: Path) -> None:
    save_peft(make_edit(), tmp_path)
    (tmp_path / "adapter_model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        read_peft(tmp_path)
    assert len(verify_peft(tmp_path)) == 1
