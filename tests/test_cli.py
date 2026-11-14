from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from astro_weights import __version__
from astro_weights.cli import _norm_ba, main
from astro_weights.edits import Edit, LowRankDelta, rank1
from astro_weights.export import save_peft

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
