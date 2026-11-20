import hashlib
import json
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


def test_export_gguf_roundtrip(
    adapter: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
    capsys.readouterr()
    assert main([*argv, "--force", "--json"]) == 0
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    assert json.loads(capsys.readouterr().out)["sha256"] == {str(out): sha}
    assert main([*argv, "--force"]) == 0
    assert capsys.readouterr().out.strip() == f"wrote {out}  sha256 {sha}"


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


# header check, floor, --json


def _arm_files(tmp_path: Path) -> tuple[Path, Path]:
    hashes = [f"h{k}" for k in range(6)]
    _records(tmp_path / "b.jsonl", "base", hashes, [0, 1, 0, 1, 0, 1])
    _records(tmp_path / "e.jsonl", "edit", hashes, [1, 1, 0, 1, 1, 1])
    return tmp_path / "b.jsonl", tmp_path / "e.jsonl"


def test_eval_header_mismatch_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from astro_weights.eval import write_header

    b, e = _arm_files(tmp_path)
    write_header(tmp_path / "b.header.json", {"seed": 0, "device": "cpu"})
    write_header(tmp_path / "e.header.json", {"seed": 1, "device": "cuda"})
    assert main(["eval", str(b), str(e), "--boot", "20"]) == 1
    cap = capsys.readouterr()
    assert "seed" in cap.err
    assert "device" not in cap.err
    assert cap.out == ""
    assert main(["eval", str(b), str(e), "--boot", "20", "--ignore-header"]) == 0


def test_eval_arm_key(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from astro_weights.eval import write_header

    b, e = _arm_files(tmp_path)
    write_header(tmp_path / "b.header.json", {"extra_body": {"lora": 0, "top_k": 1}})
    write_header(tmp_path / "e.header.json", {"extra_body": {"lora": 1, "top_k": 1}})
    assert main(["eval", str(b), str(e), "--boot", "20"]) == 1
    assert "extra_body" in capsys.readouterr().err
    assert main(["eval", str(b), str(e), "--boot", "20", "--arm-key", "lora"]) == 0
    args = [
        "eval",
        str(b),
        str(e),
        "--boot",
        "20",
        "--arm-key",
        "x",
        "--arm-key",
        "lora",
    ]
    assert main(args) == 0
    write_header(tmp_path / "e.header.json", {"extra_body": {"lora": 1, "top_k": 2}})
    assert main(["eval", str(b), str(e), "--boot", "20", "--arm-key", "lora"]) == 1


def test_eval_header_match_and_one_missing(tmp_path: Path) -> None:
    from astro_weights.eval import write_header

    b, e = _arm_files(tmp_path)
    write_header(tmp_path / "b.header.json", {"seed": 0})
    assert main(["eval", str(b), str(e), "--boot", "20"]) == 0
    write_header(tmp_path / "e.header.json", {"seed": 0})
    assert main(["eval", str(b), str(e), "--boot", "20"]) == 0


def test_eval_bad_header_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    b, e = _arm_files(tmp_path)
    (tmp_path / "b.header.json").write_text("{nope")
    (tmp_path / "e.header.json").write_text("{}")
    assert main(["eval", str(b), str(e)]) == 2
    assert "invalid JSON" in capsys.readouterr().err


def test_eval_floor(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    b, e = _arm_files(tmp_path)
    _records(tmp_path / "f1.jsonl", "base", [f"h{k}" for k in range(6)], [0, 1] * 3)
    _records(tmp_path / "f2.jsonl", "base", [f"h{k}" for k in range(6)], [0, 1] * 3)
    argv = ["eval", str(b), str(e), "--boot", "20", "--floor"]
    assert main([*argv, str(tmp_path / "f1.jsonl"), str(tmp_path / "f2.jsonl")]) == 0
    assert "noise floor (score)" in capsys.readouterr().out
    assert main([*argv, str(tmp_path / "f1.jsonl"), str(tmp_path / "nope")]) == 2
    assert "error:" in capsys.readouterr().err
    assert main([*argv, str(tmp_path / "f1.jsonl"), str(b)]) == 0
    _records(tmp_path / "f3.jsonl", "base", ["x"], [0])
    assert main([*argv, str(tmp_path / "f1.jsonl"), str(tmp_path / "f3.jsonl")]) == 2


def test_eval_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    b, e = _arm_files(tmp_path)
    _records(tmp_path / "f1.jsonl", "base", [f"h{k}" for k in range(6)], [0, 1] * 3)
    _records(tmp_path / "f2.jsonl", "base", [f"h{k}" for k in range(6)], [0, 1] * 3)
    assert main(["eval", str(b), str(e), "--boot", "20", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_items"] == 6
    assert out["ci95"] == pytest.approx([out["ci95"][0], out["ci95"][1]])
    assert isinstance(out["ci95"], list)
    assert out["floor"] is None
    argv = ["eval", str(b), str(e), "--boot", "20", "--json", "--floor"]
    assert main([*argv, str(tmp_path / "f1.jsonl"), str(tmp_path / "f2.jsonl")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out["floor"]["exceeds"]) == {"score", "tokens"}
    assert out["floor"]["result"]["n_items"] == 6


def test_eval_json_nonfinite_is_null(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    _records(tmp_path / "b.jsonl", "base", ["a"], [0])
    _records(tmp_path / "e.jsonl", "edit", ["a"], [1])
    assert (
        main(["eval", str(tmp_path / "b.jsonl"), str(tmp_path / "e.jsonl"), "--json"])
        == 0
    )
    out = json.loads(capsys.readouterr().out)  # strict parse: no NaN literals
    assert out["se"] is None
    assert out["score_mde"] is None


def test_eval_json_error_keeps_stdout_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["eval", str(tmp_path / "no"), str(tmp_path / "no2"), "--json"]) == 2
    cap = capsys.readouterr()
    assert cap.out == ""
    assert "error:" in cap.err


def test_inspect_json(adapter: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    assert main(["inspect", str(adapter), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["base_model"] == "org/tiny"
    assert out["rank"] == 1
    assert out["problems"] == []
    (m,) = out["modules"]
    assert m["module"] == MODULE
    assert m["shape"] == [16, 48]
    assert m["norm_BA"] > 0


def test_inspect_json_problems(
    adapter: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    path = adapter / "adapter_model.safetensors"
    tensors = load_file(str(path))
    key = next(k for k in tensors if k.endswith("lora_A.weight"))
    tensors[key] = tensors[key] + 1.0
    save_file(tensors, str(path))
    assert main(["inspect", str(adapter), "--json"]) == 1
    cap = capsys.readouterr()
    assert "hash mismatch" in json.loads(cap.out)["problems"][0]
    assert "hash mismatch" in cap.err


def test_inspect_json_error_stdout_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["inspect", str(tmp_path / "nope"), "--json"]) == 2
    assert capsys.readouterr().out == ""


def test_build_export_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    args = _build_args(tmp_path, "--card", "apache-2.0", "--json")
    assert main(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["written"][0].endswith("out")
    assert out["written"][1].endswith("README.md")
    adapter_dir = Path(out["written"][0])
    assert main(["export", str(adapter_dir), "--card", "mit", "--force", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [Path(p).name for p in out["written"]] == ["README.md"]


def test_epilog_documents_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "exit codes" in capsys.readouterr().out


def test_bare_command_prints_help_and_exits_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 2
    assert "usage: astro-weights" in capsys.readouterr().out


def test_python_dash_m_runs_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import runpy

    monkeypatch.setattr(sys, "argv", ["astro-weights", "--version"])
    with pytest.raises(SystemExit) as e:
        runpy.run_module("astro_weights", run_name="__main__")
    assert e.value.code == 0
    assert __version__ in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["astro-weights"])
    with pytest.raises(SystemExit) as e:
        runpy.run_module("astro_weights", run_name="__main__")
    assert e.value.code == 2


def test_every_subcommand_and_option_has_help(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import argparse

    class _Seen(Exception):
        def __init__(self, parser: argparse.ArgumentParser) -> None:
            self.parser = parser

    def grab(self: argparse.ArgumentParser, *a: object, **k: object) -> None:
        raise _Seen(self)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", grab)
    with pytest.raises(_Seen) as seen:
        main([])
    parser = seen.value.parser
    subs = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    assert len(subs) == 1
    assert {c.dest for c in subs[0]._choices_actions if c.help} == set(subs[0].choices)
    for sub in subs[0].choices.values():
        for action in sub._actions:
            assert action.help, action.dest


def test_eval_json_carries_cap_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    b, e = _arm_files(tmp_path)
    assert main(["eval", str(b), str(e), "--boot", "20", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["cap_hit_p"] == 1.0
    assert out["n_uncapped"] == 6
    assert out["uncapped_diff"] == pytest.approx(out["diff"])
    assert out["uncapped_tokens_diff"] == 0
    assert (out["loops_base"], out["loops_edit"]) == (0, 0)
    assert {"n_items", "diff", "ci95", "cap_hit_base", "floor"} <= set(out)


# mcp


def test_mcp_without_extra_exits_2_with_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("mcp", "mcp.server", "mcp.server.mcpserver"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, "astro_weights.mcp.server", raising=False)
    assert main(["mcp", str(tmp_path)]) == 2
    assert "pip install 'astro-weights[mcp]'" in capsys.readouterr().err


def test_help_works_without_extra(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("mcp", "mcp.server", "mcp.server.mcpserver"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    assert "mcp" in capsys.readouterr().out


def test_mcp_bad_run_dir_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("mcp")
    assert main(["mcp", str(tmp_path)]) == 2
    assert "run.json" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flags", "forwarded"), [([], []), (["--unsafe-kernel"], ["--unsafe-kernel"])]
)
def test_mcp_forwards_unsafe_kernel_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flags: list[str],
    forwarded: list[str],
) -> None:
    pytest.importorskip("mcp")
    from astro_weights.mcp import server

    seen: list[list[str] | None] = []

    def fake_main(argv: list[str] | None = None) -> int:
        seen.append(argv)
        return 0

    monkeypatch.setattr(server, "main", fake_main)
    assert main(["mcp", str(tmp_path), *flags]) == 0
    assert seen == [[str(tmp_path), *forwarded]]


# run

RUN_TOML = """\
[run]
dir = "run"
seed = 1

[limits]
forward_passes = 1
code_runs = 1
tokens = 10

[editor]
kind = "scripted"
"""


def _run_config(tmp_path: Path, text: str = RUN_TOML) -> Path:
    path = tmp_path / "run.toml"
    path.write_text(text)
    return path


def test_run_without_extra_exits_2_with_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "mcp", None)
    for name in list(sys.modules):
        if name.startswith("astro_weights.mcp.editors."):
            monkeypatch.delitem(sys.modules, name)
    assert main(["run", str(_run_config(tmp_path))]) == 2
    assert "pip install 'astro-weights[mcp]'" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_run_status_needs_no_extra(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "mcp", None)
    path = _run_config(tmp_path)
    assert main(["run", str(path), "--status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "not_started"
    assert main(["run", str(path), "--status"]) == 0
    assert "state: not_started" in capsys.readouterr().out


def test_run_bad_config_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _run_config(tmp_path, RUN_TOML.replace("seed = 1", "seed = 'x'"))
    assert main(["run", str(path)]) == 2
    out = capsys.readouterr()
    assert "[run] seed must be an int" in out.err
    assert out.out == ""
    assert main(["run", str(tmp_path / "missing.toml"), "--status", "--json"]) == 2
    out = capsys.readouterr()
    assert "not found" in out.err
    assert out.out == ""


def test_run_refuses_an_existing_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _run_config(tmp_path)
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "ledger.jsonl").write_text("")
    assert main(["run", str(path)]) == 2
    assert "pass --resume" in capsys.readouterr().err


# export --splice

SPLICE_MODULE = "model.layers.1.mlp.down_proj"
SPLICE_KEY = SPLICE_MODULE + ".weight"
SPLICE_NAME = "blk.1.ffn_down.weight"
SW0 = np.random.default_rng(11).standard_normal((16, 64)).astype(np.float32)
SB = np.random.default_rng(12).standard_normal((16, 2)).astype(np.float32)
SA = np.random.default_rng(13).standard_normal((2, 64)).astype(np.float32)


class Splice:
    """A tiny base model dir, base GGUF and adapter that line up."""

    def __init__(self, root: Path) -> None:
        gguf = pytest.importorskip("gguf")
        from gguf import GGMLQuantizationType as Q
        from gguf.quants import quantize

        rng = np.random.default_rng(14)
        self.model = root / "hf"
        self.model.mkdir()
        save_file({SPLICE_KEY: SW0}, str(self.model / "model.safetensors"))
        self.base = root / "base.gguf"
        w = gguf.GGUFWriter(self.base, "qwen35")
        w.add_block_count(3)
        w.add_string("general.name", "tiny")
        w.add_tensor(SPLICE_NAME, quantize(SW0, Q.Q8_0), raw_dtype=Q.Q8_0)
        w.add_tensor("token_embd.weight", rng.standard_normal((20, 16)).astype("f4"))
        w.add_tensor(
            "blk.1.ffn_gate.weight",
            rng.integers(0, 256, (4, 144), dtype=np.uint8),
            raw_dtype=Q.Q4_K,
        )
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file(progress=False)
        w.close()
        self.adapter = root / "adapter"
        self.save(self.adapter, SPLICE_MODULE, True)
        self.out = root / "merged.gguf"

    def save(self, directory: Path, module: str, sha: bool) -> None:
        from astro_weights.export import tensor_sha256

        hashes = {}
        if sha:
            hashes = {
                SPLICE_KEY: tensor_sha256(self.model / "model.safetensors", SPLICE_KEY)
            }
        delta = LowRankDelta(module, SB, SA)
        save_peft(
            Edit((delta,), "org/tiny", base_sha256=hashes), directory, overwrite=True
        )

    def argv(self, *extra: str) -> list[str]:
        return [
            "export",
            str(self.adapter),
            "--splice",
            str(self.base),
            "--base-model",
            str(self.model),
            "--merged",
            str(self.out),
            *extra,
        ]


@pytest.fixture
def sp(tmp_path: Path) -> Splice:
    return Splice(tmp_path)


def _tensors(path: Path) -> dict[str, tuple[str, bytes]]:
    import gguf

    return {
        t.name: (t.tensor_type.name, bytes(t.data.tobytes()))
        for t in gguf.GGUFReader(path).tensors
    }


def _dequant(path: Path, name: str) -> np.ndarray:
    import gguf
    from gguf.quants import dequantize

    t = next(t for t in gguf.GGUFReader(path).tensors if t.name == name)
    return np.asarray(dequantize(t.data, t.tensor_type)).reshape(
        tuple(int(d) for d in reversed(t.shape.tolist()))
    )


def _no_files(sp: Splice) -> None:
    assert not sp.out.exists()
    assert not list(sp.out.parent.glob("*.tmp"))


@pytest.mark.parametrize(
    ("out_type", "tol"), [("q8_0", 2e-2), ("f16", 2e-3), ("bf16", 2e-2)]
)
def test_splice_round_trip(
    sp: Splice,
    capsys: pytest.CaptureFixture[str],
    out_type: str,
    tol: float,
) -> None:
    assert main(sp.argv("--out-type", out_type, "--json")) == 0
    got = json.loads(capsys.readouterr().out)
    assert got == {
        "written": [str(sp.out)],
        "sha256": {str(sp.out): hashlib.sha256(sp.out.read_bytes()).hexdigest()},
        "replaced": [SPLICE_NAME],
        "out_type": out_type,
    }
    want = SW0 + SB @ SA
    have = _dequant(sp.out, SPLICE_NAME)
    assert np.linalg.norm(have - want) / np.linalg.norm(want) < tol
    old, new = _tensors(sp.base), _tensors(sp.out)
    assert list(old) == list(new)
    for name in old:
        if name != SPLICE_NAME:
            assert new[name] == old[name]
    assert new[SPLICE_NAME][0] == out_type.upper()
    assert not list(sp.out.parent.glob("*.tmp"))


def test_splice_default_is_q8_0_text(
    sp: Splice, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(sp.argv()) == 0
    assert f"wrote {sp.out}" in capsys.readouterr().out
    assert _tensors(sp.out)[SPLICE_NAME][0] == "Q8_0"


def test_splice_without_recorded_hash(sp: Splice) -> None:
    sp.save(sp.adapter, SPLICE_MODULE, False)
    assert main(sp.argv("--force")) == 0


def test_splice_wrong_base_hash(sp: Splice, capsys: pytest.CaptureFixture[str]) -> None:
    save_file({SPLICE_KEY: SW0 + 1.0}, str(sp.model / "model.safetensors"))
    assert main(sp.argv()) == 2
    err = capsys.readouterr().err
    assert SPLICE_MODULE in err
    assert "hash" in err
    _no_files(sp)


def test_splice_base_shape_mismatch(
    sp: Splice, capsys: pytest.CaptureFixture[str]
) -> None:
    sp.save(sp.adapter, SPLICE_MODULE, False)
    save_file({SPLICE_KEY: SW0[:, :32]}, str(sp.model / "model.safetensors"))
    assert main(sp.argv("--force")) == 2
    assert "shape" in capsys.readouterr().err
    _no_files(sp)


def test_splice_q8_0_row_length(sp: Splice, capsys: pytest.CaptureFixture[str]) -> None:
    w0 = np.ones((16, 50), np.float32)
    save_file({SPLICE_KEY: w0}, str(sp.model / "model.safetensors"))
    delta = LowRankDelta(SPLICE_MODULE, SB, SA[:, :50].copy())
    save_peft(Edit((delta,), "org/tiny"), sp.adapter, overwrite=True)
    assert main(sp.argv()) == 2
    assert "divisible by 32" in capsys.readouterr().err
    _no_files(sp)


def test_splice_unsupported_module(
    sp: Splice, capsys: pytest.CaptureFixture[str]
) -> None:
    sp.save(sp.adapter, "model.embed_tokens", False)
    assert main(sp.argv("--force")) == 2
    assert "unsupported module" in capsys.readouterr().err
    _no_files(sp)


def test_splice_two_modules_one_tensor(
    sp: Splice, capsys: pytest.CaptureFixture[str]
) -> None:
    other = "model.language_model.layers.1.mlp.down_proj"
    deltas = (LowRankDelta(SPLICE_MODULE, SB, SA), LowRankDelta(other, SB, SA))
    save_peft(Edit(deltas, "org/tiny"), sp.adapter, overwrite=True)
    assert main(sp.argv()) == 2
    assert "same GGUF tensor" in capsys.readouterr().err
    _no_files(sp)


def test_splice_base_tensor_not_in_gguf(
    sp: Splice, capsys: pytest.CaptureFixture[str]
) -> None:
    sp.save(sp.adapter, "model.layers.2.mlp.down_proj", False)
    save_file(
        {"model.layers.2.mlp.down_proj.weight": SW0},
        str(sp.model / "model.safetensors"),
    )
    assert main(sp.argv()) == 2
    assert "not in base" in capsys.readouterr().err
    _no_files(sp)


@pytest.mark.parametrize("drop", ["--base-model", "--merged"])
def test_splice_missing_flag(
    sp: Splice, capsys: pytest.CaptureFixture[str], drop: str
) -> None:
    argv = sp.argv()
    i = argv.index(drop)
    del argv[i : i + 2]
    assert main(argv) == 2
    assert "--splice needs" in capsys.readouterr().err
    _no_files(sp)


@pytest.mark.parametrize(
    "extra", [["--merged", "x.gguf"], ["--base-model", "d"], ["--out-type", "f16"]]
)
def test_splice_flags_need_splice(
    adapter: Path, capsys: pytest.CaptureFixture[str], extra: list[str]
) -> None:
    assert main(["export", str(adapter), "--card", "mit", *extra]) == 2
    assert "need --splice" in capsys.readouterr().err
    assert not (adapter / "README.md").exists()


def test_export_nothing_lists_splice(
    adapter: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["export", str(adapter)]) == 2
    assert "--splice" in capsys.readouterr().err


def test_splice_bad_out_type(sp: Splice) -> None:
    with pytest.raises(SystemExit) as e:
        main(sp.argv("--out-type", "q4_0"))
    assert e.value.code == 2


def test_splice_overwrite_refused_then_forced(
    sp: Splice, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(sp.argv()) == 0
    first = sp.out.read_bytes()
    assert main(sp.argv()) == 2
    assert "pass --force" in capsys.readouterr().err
    assert sp.out.read_bytes() == first
    assert main(sp.argv("--force", "--out-type", "f16")) == 0
    assert sp.out.read_bytes() != first
    assert not list(sp.out.parent.glob("*.tmp"))


def test_splice_out_is_base(sp: Splice, capsys: pytest.CaptureFixture[str]) -> None:
    before = sp.base.read_bytes()
    argv = sp.argv("--force")
    argv[argv.index("--merged") + 1] = str(sp.base)
    assert main(argv) == 2
    assert "--splice base file" in capsys.readouterr().err
    assert sp.base.read_bytes() == before
    argv.remove("--force")  # no misleading "pass --force" hint
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert "--splice base file" in err
    assert "pass --force" not in err
    assert sp.base.read_bytes() == before


def test_splice_gguf_same_as_merged(
    sp: Splice, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = sp.argv("--gguf", str(sp.out), "--arch", "qwen35", "--force")
    assert main(argv) == 2
    assert "both" in capsys.readouterr().err
    _no_files(sp)


def test_splice_gguf_same_as_base(
    sp: Splice, capsys: pytest.CaptureFixture[str]
) -> None:
    before = sp.base.read_bytes()
    argv = sp.argv("--gguf", str(sp.base), "--arch", "qwen35", "--force")
    assert main(argv) == 2
    assert "--splice base file" in capsys.readouterr().err
    assert sp.base.read_bytes() == before
    _no_files(sp)


def test_splice_merged_dir_missing(
    sp: Splice, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = sp.argv()
    argv[argv.index("--merged") + 1] = str(sp.out.parent / "nope" / "m.gguf")
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert "does not exist" in err
    assert str(sp.out.parent / "nope") in err


def test_splice_out_inside_adapter_dir(
    sp: Splice, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = sp.argv()
    argv[argv.index("--merged") + 1] = str(sp.adapter / "m.gguf")
    assert main(argv) == 2
    assert "inside the adapter" in capsys.readouterr().err
    assert not (sp.adapter / "m.gguf").exists()


def test_splice_base_not_a_file(sp: Splice, capsys: pytest.CaptureFixture[str]) -> None:
    sp.base.unlink()
    assert main(sp.argv()) == 2
    assert "not a file" in capsys.readouterr().err
    _no_files(sp)


def test_splice_unreadable_base(sp: Splice, capsys: pytest.CaptureFixture[str]) -> None:
    sp.base.write_bytes(b"not a gguf")
    assert main(sp.argv()) == 2
    assert "error" in capsys.readouterr().err
    _no_files(sp)


def test_splice_verify_failure_leaves_no_file(
    sp: Splice, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import astro_weights.export.gguf as g

    monkeypatch.setattr(g, "verify_splice", lambda *a, **k: ["boom"])
    assert main(sp.argv()) == 2
    assert "failed verification: boom" in capsys.readouterr().err
    _no_files(sp)


def test_splice_with_gguf_and_card(sp: Splice) -> None:
    lora = sp.out.parent / "lora.gguf"
    argv = sp.argv("--gguf", str(lora), "--arch", "qwen35", "--card", "mit")
    assert main(argv) == 0
    assert sp.out.exists()
    assert lora.exists()
    assert (sp.adapter / "README.md").exists()


def test_splice_failure_writes_no_card(sp: Splice) -> None:
    save_file({SPLICE_KEY: SW0 + 1.0}, str(sp.model / "model.safetensors"))
    assert main(sp.argv("--card", "mit")) == 2
    assert not (sp.adapter / "README.md").exists()


def test_splice_missing_gguf_package(
    sp: Splice,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setitem(sys.modules, "astro_weights.export.gguf", None)
    assert main(sp.argv()) == 2
    assert "astro-weights[gguf]" in capsys.readouterr().err
    _no_files(sp)


# compose


def _two_adapters(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(21)
    down = "model.layers.1.mlp.down_proj"
    q = "model.layers.1.self_attn.q_proj"

    def delta(m: str, r: int) -> LowRankDelta:
        return LowRankDelta(
            m, rng.standard_normal((8, r)), rng.standard_normal((r, 12))
        )

    a, b = tmp_path / "a", tmp_path / "b"
    save_peft(Edit((delta(down, 1),), "org/tiny"), a)
    save_peft(Edit((delta(down, 2), delta(q, 2)), "org/tiny"), b)
    return a, b


def test_compose_round_trip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = _two_adapters(tmp_path)
    out = tmp_path / "out"
    assert main(["compose", str(a), str(b), "-o", str(out)]) == 0
    assert f"wrote {out}" in capsys.readouterr().out
    got, problems = read_peft(out)
    assert problems == []
    ea, eb = read_peft(a)[0], read_peft(b)[0]
    want = (
        ea.deltas[0].dense() + next(d for d in eb.deltas if "down" in d.module).dense()
    )
    by = {d.module: d for d in got.deltas}
    np.testing.assert_allclose(
        by["model.layers.1.mlp.down_proj"].dense(), want, atol=1e-5
    )
    assert got.rank == 3
    assert main(["inspect", str(out)]) == 0


def test_compose_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = _two_adapters(tmp_path)
    out = tmp_path / "out"
    assert main(["compose", str(a), str(b), "-o", str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "written": str(out),
        "rank": 3,
        "modules": [
            "model.layers.1.mlp.down_proj",
            "model.layers.1.self_attn.q_proj",
        ],
        "sources": [str(a), str(b)],
    }


def test_compose_three_adapters(tmp_path: Path) -> None:
    a, b = _two_adapters(tmp_path)
    out = tmp_path / "out"
    assert main(["compose", str(a), str(b), str(a), "-o", str(out)]) == 0
    assert read_peft(out)[0].rank == 4


def test_compose_one_adapter_is_usage_error(tmp_path: Path) -> None:
    a, _ = _two_adapters(tmp_path)
    with pytest.raises(SystemExit) as e:
        main(["compose", str(a), "-o", str(tmp_path / "out")])
    assert e.value.code == 2


def test_compose_bad_input_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, _ = _two_adapters(tmp_path)
    out = tmp_path / "out"
    assert main(["compose", str(a), str(tmp_path / "nope"), "-o", str(out)]) == 2
    assert "nope" in capsys.readouterr().err
    assert not out.exists()


def test_compose_tampered_input_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _two_adapters(tmp_path)
    tensors = load_file(str(b / "adapter_model.safetensors"))
    key = next(k for k in tensors if k.endswith("lora_A.weight"))
    tensors[key] = tensors[key] + 1.0
    save_file(tensors, str(b / "adapter_model.safetensors"))
    assert main(["compose", str(a), str(b), "-o", str(tmp_path / "out")]) == 2
    assert "error:" in capsys.readouterr().err


def test_compose_different_bases_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _two_adapters(tmp_path)
    save_peft(Edit(read_peft(b)[0].deltas, "other/model"), b, overwrite=True)
    out = tmp_path / "out"
    assert main(["compose", str(a), str(b), "-o", str(out)]) == 2
    assert "different base models" in capsys.readouterr().err
    assert not out.exists()


def test_compose_overwrite_refused_then_forced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _two_adapters(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    assert main(["compose", str(a), str(b), "-o", str(out)]) == 2
    assert "--force" in capsys.readouterr().err
    assert not (out / "adapter_config.json").exists()
    assert main(["compose", str(a), str(b), "-o", str(out), "--force"]) == 0
    assert (out / "adapter_config.json").exists()
    assert (out / "keep.txt").exists()


@pytest.mark.parametrize("where", ["same", "inside"])
def test_compose_out_in_input_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], where: str
) -> None:
    a, b = _two_adapters(tmp_path)
    out = a if where == "same" else a / "sub"
    assert main(["compose", str(a), str(b), "-o", str(out), "--force"]) == 2
    assert "input adapter" in capsys.readouterr().err
    assert read_peft(a)[1] == []
    assert not (a / "sub").exists()


def test_compose_then_splice_down_and_up(tmp_path: Path) -> None:
    gguf = pytest.importorskip("gguf")
    from astro_weights.export import tensor_sha256

    rng = np.random.default_rng(31)
    down, up = "model.layers.1.mlp.down_proj", "model.layers.1.mlp.up_proj"
    w = {
        down: rng.standard_normal((8, 16)).astype(np.float32),
        up: rng.standard_normal((16, 8)).astype(np.float32),
    }
    model = tmp_path / "hf"
    model.mkdir()
    save_file(
        {k + ".weight": v for k, v in w.items()}, str(model / "model.safetensors")
    )
    hashes = {
        k + ".weight": tensor_sha256(model / "model.safetensors", k + ".weight")
        for k in w
    }
    base = tmp_path / "base.gguf"
    wr = gguf.GGUFWriter(base, "qwen35")
    wr.add_block_count(3)
    wr.add_tensor("blk.1.ffn_down.weight", w[down])
    wr.add_tensor("blk.1.ffn_up.weight", w[up])
    wr.add_tensor("token_embd.weight", rng.standard_normal((20, 8)).astype("f4"))
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_tensors_to_file(progress=False)
    wr.close()

    def delta(m: str, r: int) -> LowRankDelta:
        o, i = w[m].shape
        return LowRankDelta(m, rng.standard_normal((o, r)), rng.standard_normal((r, i)))

    ea = Edit((delta(down, 1),), "org/tiny", hashes)
    eb = Edit.from_deltas([delta(down, 2), delta(up, 1)], "org/tiny", hashes)
    save_peft(ea, tmp_path / "a")
    save_peft(eb, tmp_path / "b")
    comp = tmp_path / "comp"
    assert (
        main(["compose", str(tmp_path / "a"), str(tmp_path / "b"), "-o", str(comp)])
        == 0
    )
    merged = tmp_path / "merged.gguf"
    argv = ["export", str(comp), "--splice", str(base), "--base-model", str(model)]
    assert main([*argv, "--merged", str(merged), "--out-type", "f16", "--json"]) == 0
    have = {
        t.name: np.asarray(t.data, dtype=np.float32).reshape(w[down].shape)
        if "down" in t.name
        else np.asarray(t.data, dtype=np.float32).reshape(w[up].shape)
        for t in gguf.GGUFReader(merged).tensors
        if t.name.startswith("blk.1.")
    }
    want_down = w[down] + ea.deltas[0].dense() + eb.deltas[0].dense()
    want_up = w[up] + eb.deltas[1].dense()
    np.testing.assert_allclose(
        have["blk.1.ffn_down.weight"], want_down, rtol=2e-3, atol=2e-3
    )
    np.testing.assert_allclose(
        have["blk.1.ffn_up.weight"], want_up, rtol=2e-3, atol=2e-3
    )


def test_compose_base_model_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _two_adapters(tmp_path)
    save_peft(Edit(read_peft(b)[0].deltas, "hub/Model"), b, overwrite=True)
    out = tmp_path / "out"
    assert main(["compose", str(a), str(b), "-o", str(out)]) == 2
    assert "different base models" in capsys.readouterr().err
    argv = ["compose", str(a), str(b), "-o", str(out), "--base-model", "hub/Model"]
    assert main(argv) == 0
    assert read_peft(out)[0].base_model == "hub/Model"


def test_export_gguf_qk_needs_plain_arch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("gguf")
    rng = np.random.default_rng(41)
    q = LowRankDelta(
        "model.layers.1.self_attn.q_proj",
        rng.standard_normal((8, 1)),
        rng.standard_normal((1, 12)),
    )
    save_peft(Edit((q,), "org/tiny"), tmp_path / "a")
    out = tmp_path / "a.gguf"
    argv = ["export", str(tmp_path / "a"), "--gguf", str(out), "--arch"]
    assert main([*argv, "llama"]) == 2
    assert "q/k rows are reordered" in capsys.readouterr().err
    assert not out.exists()
    assert main([*argv, "qwen35"]) == 0
    assert out.exists()


def test_splice_qk_refused_on_llama_base(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gguf = pytest.importorskip("gguf")
    from astro_weights.export import tensor_sha256

    w0 = np.random.default_rng(42).standard_normal((8, 16)).astype(np.float32)
    model = tmp_path / "hf"
    model.mkdir()
    key = "model.layers.1.self_attn.q_proj.weight"
    save_file({key: w0}, str(model / "model.safetensors"))
    base = tmp_path / "base.gguf"
    wr = gguf.GGUFWriter(base, "llama")
    wr.add_block_count(3)
    wr.add_tensor("blk.1.attn_q.weight", w0)
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_tensors_to_file(progress=False)
    wr.close()
    d = LowRankDelta(key[: -len(".weight")], np.ones((8, 1)), np.ones((1, 16)))
    hashes = {key: tensor_sha256(model / "model.safetensors", key)}
    save_peft(Edit((d,), "org/tiny", hashes), tmp_path / "a")
    merged = tmp_path / "m.gguf"
    argv = ["export", str(tmp_path / "a"), "--splice", str(base)]
    argv += ["--base-model", str(model), "--merged", str(merged), "--out-type", "f16"]
    assert main(argv) == 2
    assert "q/k rows are reordered" in capsys.readouterr().err
    assert not merged.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_export_gguf_qk_llama_with_splice_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gguf = pytest.importorskip("gguf")
    from astro_weights.export import tensor_sha256

    w0 = np.random.default_rng(43).standard_normal((8, 16)).astype(np.float32)
    model = tmp_path / "hf"
    model.mkdir()
    key = "model.layers.1.self_attn.q_proj.weight"
    save_file({key: w0}, str(model / "model.safetensors"))
    base = tmp_path / "base.gguf"
    wr = gguf.GGUFWriter(base, "qwen35")
    wr.add_block_count(3)
    wr.add_tensor("blk.1.attn_q.weight", w0)
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_tensors_to_file(progress=False)
    wr.close()
    d = LowRankDelta(key[: -len(".weight")], np.ones((8, 1)), np.ones((1, 16)))
    hashes = {key: tensor_sha256(model / "model.safetensors", key)}
    save_peft(Edit((d,), "org/tiny", hashes), tmp_path / "a")
    merged, lora = tmp_path / "m.gguf", tmp_path / "l.gguf"
    argv = ["export", str(tmp_path / "a"), "--splice", str(base)]
    argv += ["--base-model", str(model), "--merged", str(merged)]
    argv += ["--out-type", "f16", "--gguf", str(lora), "--arch"]
    assert main([*argv, "llama"]) == 2
    assert "q/k rows are reordered" in capsys.readouterr().err
    assert not merged.exists()
    assert not lora.exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert main([*argv, "qwen35"]) == 0
    assert merged.exists()
    assert lora.exists()
