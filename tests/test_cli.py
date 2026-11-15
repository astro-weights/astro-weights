import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from astro_weights import __version__
from astro_weights.cli import _norm_ba, main
from astro_weights.edits import Edit, LowRankDelta, project_out, rank1, save_recipe
from astro_weights.export import read_peft, save_peft

MODULE = "model.layers.1.mlp.down_proj"


@pytest.fixture
def adapter(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    delta = rank1(MODULE, rng.standard_normal(16), rng.standard_normal(48))
    save_peft(Edit((delta,), "org/tiny"), tmp_path)
    return tmp_path


def test_inspect_ok(adapter: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect", str(adapter)]) == 0
    out = capsys.readouterr().out
    assert "base model: org/tiny" in out
    assert "rank: 1" in out
    assert "modules: 1" in out
    assert MODULE in out
    assert "16x48" in out
    assert "manifest: ok" in out


def test_inspect_tampered(adapter: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = adapter / "adapter_model.safetensors"
    tensors = load_file(str(path))
    key = next(k for k in tensors if k.endswith("lora_A.weight"))
    tensors[key] = tensors[key] + 1.0
    save_file(tensors, str(path))
    assert main(["inspect", str(adapter)]) == 1
    assert "hash mismatch" in capsys.readouterr().err


def test_inspect_missing_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["inspect", str(tmp_path / "nope")]) == 2
    assert "error" in capsys.readouterr().err


def test_inspect_not_an_adapter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["inspect", str(tmp_path)]) == 2
    assert "error" in capsys.readouterr().err


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_norm_ba_matches_dense() -> None:
    rng = np.random.default_rng(2)
    d = LowRankDelta(MODULE, rng.standard_normal((16, 3)), rng.standard_normal((3, 48)))
    assert np.isclose(_norm_ba(d), np.linalg.norm(d.B @ d.A), rtol=1e-5)


def _records(path: Path, arm: str, hashes: list[str], scores: list[float]) -> None:
    from astro_weights.eval import Record, write_jsonl

    write_jsonl(
        path,
        [
            Record(f"i{k}", 0, arm, h, "out", s, 5, False)
            for k, (h, s) in enumerate(zip(hashes, scores, strict=True))
        ],
    )


def test_eval_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    hashes = [f"h{k}" for k in range(6)]
    _records(tmp_path / "b.jsonl", "base", hashes, [0, 1, 0, 1, 0, 1])
    _records(tmp_path / "e.jsonl", "edit", hashes, [1, 1, 0, 1, 1, 1])
    argv = ["eval", str(tmp_path / "b.jsonl"), str(tmp_path / "e.jsonl")]
    assert main([*argv, "--boot", "100", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "n=6 items" in out
    assert "score diff=" in out


def test_eval_hash_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _records(tmp_path / "b.jsonl", "base", ["a", "b"], [0, 1])
    _records(tmp_path / "e.jsonl", "edit", ["a", "X"], [1, 1])
    assert main(["eval", str(tmp_path / "b.jsonl"), str(tmp_path / "e.jsonl")]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_eval_unreadable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["eval", str(tmp_path / "no"), str(tmp_path / "no2")]) == 2
    assert "error:" in capsys.readouterr().err


def test_eval_bad_json_names_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _records(tmp_path / "b.jsonl", "base", ["a"], [0])
    (tmp_path / "e.jsonl").write_text("{not json\n")
    assert main(["eval", str(tmp_path / "b.jsonl"), str(tmp_path / "e.jsonl")]) == 2
    err = capsys.readouterr().err
    assert str(tmp_path / "e.jsonl") in err
    assert str(tmp_path / "b.jsonl") not in err


# build / export

BMOD = "model.layers.1.mlp.down_proj"
W0 = np.random.default_rng(5).standard_normal((16, 48)).astype(np.float32)


def _model(tmp_path: Path, key: str = BMOD + ".weight") -> Path:
    d = tmp_path / "model"
    d.mkdir(exist_ok=True)
    save_file({key: W0}, str(d / "model.safetensors"))
    return d


def _recipe(tmp_path: Path) -> Path:
    rng = np.random.default_rng(6)
    d = tmp_path / "rec"
    d.mkdir(exist_ok=True)
    np.save(d / "v.npy", rng.standard_normal(16).astype(np.float32))
    np.save(d / "b.npy", rng.standard_normal(16).astype(np.float32))
    np.save(d / "a.npy", rng.standard_normal(48).astype(np.float32))
    recipe = {
        "format": 1,
        "base_model": "org/tiny",
        "steps": [
            {
                "recipe": "project_out",
                "module": BMOD,
                "params": {"strength": 0.7},
                "arrays": {"v": "v.npy"},
            },
            {
                "recipe": "rank1",
                "module": "model.layers.2.mlp.down_proj",
                "arrays": {"b": "b.npy", "a": "a.npy"},
            },
        ],
    }
    path = d / "recipe.json"
    save_recipe(path, recipe)
    return path


def _build_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "build",
        str(_recipe(tmp_path)),
        "--base",
        str(_model(tmp_path)),
        "--out",
        str(tmp_path / "out"),
        *extra,
    ]


def _build_ok(tmp_path: Path) -> Path:
    # module 2 has no base weight, but rank1 does not need one
    assert main(_build_args(tmp_path)) == 0
    return tmp_path / "out"


def test_build_matches_recipe_functions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _build_ok(tmp_path)
    printed = capsys.readouterr().out
    assert BMOD in printed
    assert "16x48" in printed
    assert str(out) in printed
    assert main(["inspect", str(out)]) == 0
    edit, problems = read_peft(out)
    assert problems == []
    by_mod = {d.module: d for d in edit.deltas}
    rec = tmp_path / "rec"
    v = np.load(rec / "v.npy")
    b, a = np.load(rec / "b.npy"), np.load(rec / "a.npy")
    np.testing.assert_allclose(
        by_mod[BMOD].dense(), project_out(BMOD, W0, v, 0.7).dense(), atol=1e-6
    )
    np.testing.assert_allclose(
        by_mod["model.layers.2.mlp.down_proj"].dense(),
        rank1("m", b, a).dense(),
        atol=1e-6,
    )


def test_build_card(tmp_path: Path) -> None:
    assert main(_build_args(tmp_path, "--card", "apache-2.0")) == 0
    readme = (tmp_path / "out" / "README.md").read_text()
    assert 'base_model: "org/tiny"' in readme
    assert 'license: "apache-2.0"' in readme


def test_build_refuses_non_empty_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    assert main(_build_args(tmp_path)) == 2
    assert "--force" in capsys.readouterr().err
    assert not (out / "adapter_model.safetensors").exists()
    assert main(_build_args(tmp_path, "--force")) == 0
    assert (out / "keep.txt").exists()
    assert (out / "adapter_model.safetensors").exists()


def test_build_force_replaces_existing_adapter(tmp_path: Path) -> None:
    assert main(_build_args(tmp_path)) == 0
    out = tmp_path / "out"
    (out / "README.md").write_text("card")
    assert main(_build_args(tmp_path)) == 2  # non-empty, no --force
    assert main(_build_args(tmp_path, "--force")) == 0
    assert (out / "README.md").read_text() == "card"
    assert read_peft(out)[1] == []
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".")) == []


def test_build_force_onto_file_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    out.write_text("x")
    assert main(_build_args(tmp_path, "--force")) == 2
    assert "not a directory" in capsys.readouterr().err
    assert out.read_text() == "x"


def test_build_bad_recipe(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    recipe = tmp_path / "bad.json"
    recipe.write_text('{"format": 1, "base_model": "x", "steps": [{"recipe": "nope"}]}')
    argv = ["build", str(recipe), "--base", str(_model(tmp_path)), "--out"]
    assert main([*argv, str(tmp_path / "out")]) == 2
    assert "unknown recipe 'nope'" in capsys.readouterr().err
    assert main(["build", str(tmp_path / "none.json"), *argv[2:], "o2"]) == 2
    assert "cannot read recipe" in capsys.readouterr().err


def test_build_missing_base_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _build_args(tmp_path)
    save_file({"other.weight": W0}, str(tmp_path / "model" / "model.safetensors"))
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert f"{BMOD}.weight" in err
    assert "key not found" in err


def test_build_missing_array(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _build_args(tmp_path)
    (tmp_path / "rec" / "v.npy").unlink()
    assert main(argv) == 2
    assert "v.npy" in capsys.readouterr().err


def test_export_no_option(adapter: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["export", str(adapter)]) == 2
    assert "nothing to export" in capsys.readouterr().err


def test_export_card(adapter: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["export", str(adapter), "--card", "mit"]) == 0
    assert 'license: "mit"' in (adapter / "README.md").read_text()
    assert main(["export", str(adapter), "--card", "mit"]) == 2
    assert "README.md exists" in capsys.readouterr().err
    assert main(["export", str(adapter), "--card", "bsd", "--force"]) == 0
    assert 'license: "bsd"' in (adapter / "README.md").read_text()


def test_export_bad_adapter(
    adapter: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = adapter / "adapter_model.safetensors"
    tensors = load_file(str(path))
    key = next(k for k in tensors if k.endswith("lora_A.weight"))
    tensors[key] = tensors[key] + 1.0
    save_file(tensors, str(path))
    assert main(["export", str(adapter), "--card", "mit"]) == 2
    assert "hash mismatch" in capsys.readouterr().err
    assert not (adapter / "README.md").exists()
    assert main(["export", str(tmp_path / "nope"), "--card", "mit"]) == 2


def test_export_gguf_needs_arch(
    adapter: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["export", str(adapter), "--gguf", str(tmp_path / "a.gguf")]) == 2
    assert "--arch" in capsys.readouterr().err


def test_export_gguf_roundtrip(adapter: Path, tmp_path: Path) -> None:
    pytest.importorskip("gguf")
    from astro_weights.export.gguf import read_lora_gguf

    out = tmp_path / "a.gguf"
    argv = ["export", str(adapter), "--gguf", str(out), "--arch", "qwen35"]
    assert main(argv) == 0
    tensors, meta = read_lora_gguf(out)
    assert meta["architecture"] == "qwen35"
    edit, _ = read_peft(adapter)
    b, a = tensors["blk.1.ffn_down.weight"]
    np.testing.assert_allclose(b @ a, edit.deltas[0].dense(), atol=1e-6)
    assert main(argv) == 2  # exists
    assert main([*argv, "--force"]) == 0


def test_export_gguf_missing_package(
    adapter: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setitem(sys.modules, "astro_weights.export.gguf", None)
    argv = ["export", str(adapter), "--gguf", str(tmp_path / "a.gguf"), "--arch", "x"]
    assert main(argv) == 2
    assert "astro-weights[gguf]" in capsys.readouterr().err
    assert not (tmp_path / "a.gguf").exists()


# fix round 1


def test_export_gguf_refuses_adapter_files(
    adapter: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = (adapter / "adapter_model.safetensors").read_bytes()
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        target = adapter / ".." / adapter.name / name
        argv = ["export", str(adapter), "--gguf", str(target), "--arch", "qwen35"]
        assert main([*argv, "--force"]) == 2
        assert "adapter" in capsys.readouterr().err
    assert (adapter / "adapter_model.safetensors").read_bytes() == before


@pytest.mark.parametrize(
    "manifest", ["[]", "null", '{"factors": 5}', '{"base_model": 1}']
)
def test_bad_manifest_exits_2(
    adapter: Path, manifest: str, capsys: pytest.CaptureFixture[str]
) -> None:
    (adapter / "astro_manifest.json").write_text(manifest)
    for argv in (["inspect", str(adapter)], ["export", str(adapter), "--card", "mit"]):
        assert main(argv) == 2
        err = capsys.readouterr().err
        assert err.startswith("error:")
        assert "Traceback" not in err
        assert "astro_manifest.json" in err


def test_adapter_missing_key_names_dir(
    adapter: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (adapter / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    assert main(["export", str(adapter), "--card", "mit"]) == 2
    err = capsys.readouterr().err
    assert str(adapter) in err
    assert "missing key 'r'" in err


def test_build_deeply_nested_recipe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recipe = tmp_path / "deep.json"
    recipe.write_text("[" * 200_000)
    argv = ["build", str(recipe), "--base", str(_model(tmp_path)), "--out", "o"]
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "deep.json" in err


def test_build_recipe_error_names_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recipe = tmp_path / "bad.json"
    recipe.write_text('{"format": 1, "base_model": "x", "steps": [{"recipe": "nope"}]}')
    argv = ["build", str(recipe), "--base", str(_model(tmp_path)), "--out", "o"]
    assert main(argv) == 2
    assert "bad.json" in capsys.readouterr().err
    assert main(_build_args(tmp_path, "--card", " ")) == 2
    assert "recipe.json" in capsys.readouterr().err


def test_export_bad_license_writes_no_gguf(
    adapter: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("gguf")
    out = tmp_path / "a.gguf"
    argv = ["export", str(adapter), "--gguf", str(out), "--arch", "qwen35"]
    assert main([*argv, "--card", " "]) == 2
    assert "license" in capsys.readouterr().err
    assert not out.exists()
    assert not (adapter / "README.md").exists()


def test_export_blank_arch(
    adapter: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["export", str(adapter), "--gguf", str(tmp_path / "a.gguf"), "--arch", " "]
    assert main(argv) == 2
    assert "--arch" in capsys.readouterr().err
