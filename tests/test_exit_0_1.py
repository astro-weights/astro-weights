"""0.1.0 exit test on real 9B files, plus a tiny synthetic run of the same code.

The real tests need $ASTRO_REALDATA (see `_Real`) and skip without it. Numbers are
printed (`pytest -s`) and, if $ASTRO_REALDATA_OUT is set, merged into that JSON file.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import save_file

from astro_weights.edits import Edit, LowRankDelta
from astro_weights.edits.dense import from_dense
from astro_weights.edits.recipe_file import build_edit, load_recipe, save_recipe
from astro_weights.export import load_peft, save_peft, verify_peft
from astro_weights.export.base import read_base_weight, read_tensor

RECIPE = Path(__file__).parent / "realdata" / "s18k6s25.recipe.json"
S18K6S25_IDX = [5217, 8241, 11141, 3435, 9096, 7726]
S18K6S25_COEF = [
    0.6166451573371887,
    0.7347180843353271,
    0.7981113791465759,
    0.7968157529830933,
    0.6714792847633362,
    0.49340373277664185,
]
S18K6S25_REL_CHANGE = 0.17884531617164612


def _fro(x: np.ndarray) -> float:
    """Frobenius norm accumulated in float64 without a float64 copy."""
    v = x.reshape(-1)
    return float(np.sqrt(np.einsum("i,i->", v, v, dtype=np.float64)))


def _edited_file(directory: Path, module: str) -> Path:
    return directory / f"{module}.weight.safetensors"


def leg_a(
    recipe_src: Path,
    shift_src: Path,
    idx: list[int],
    coef: list[float],
    base_dir: Path,
    edited_dir: Path,
    work: Path,
    *,
    expect_rel_change: float,
    rel_err_tol: float,
    change_tol: float,
) -> dict[str, float]:
    """Rebuild one column_shift edit from a recipe and compare it with edited - base."""
    recipe = load_recipe(recipe_src)
    step = recipe["steps"][0]
    module, names = step["module"], step["arrays"]
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy(recipe_src, work / "recipe.json")
    shutil.copy(shift_src, work / names["shift"])
    np.save(work / names["idx"], np.asarray(idx, dtype=np.int64))
    np.save(work / names["coef"], np.asarray(coef, dtype=np.float64))

    def no_weights(_module: str) -> np.ndarray:
        raise AssertionError("column_shift needs no W0")

    edit = build_edit(load_recipe(work / "recipe.json"), no_weights, work)
    (delta,) = edit.deltas
    assert delta.module == module
    key = f"{module}.weight"
    base = read_base_weight(base_dir, key)
    d = read_tensor(_edited_file(edited_dir, module), key) - base
    dense = delta.dense()
    assert dense.shape == d.shape

    outside = np.ones(d.shape[1], dtype=bool)
    outside[idx] = False
    assert not d[:, outside].any(), "edited weight differs outside the edited columns"

    d_norm = _fro(d)
    rel_err = _fro(dense - d) / d_norm
    rel_change = d_norm / _fro(base)
    assert rel_err < rel_err_tol
    assert rel_change == pytest.approx(expect_rel_change, rel=change_tol)
    return {
        "rel_err": rel_err,
        "rel_change": rel_change,
        "expect_rel_change": expect_rel_change,
        "delta_norm": d_norm,
    }


def leg_b(
    base_dir: Path,
    edited_dir: Path,
    adapter_dir: Path,
    work: Path,
    *,
    expect_count: int,
    residual_tol: float = 2e-4,
    norm_tol: float = 1e-4,
    adapter_slack: float = 1e-4,
) -> dict[str, dict[str, float]]:
    """Refit each edited tensor at rank 1 and compare with the published adapter."""
    manifest = json.loads((adapter_dir / "lora_manifest.json").read_text())["tensors"]
    published = {d.module: d for d in load_peft(adapter_dir).deltas}
    files = sorted(edited_dir.glob("*.safetensors"))
    assert len(files) == expect_count
    deltas: list[LowRankDelta] = []
    numbers: dict[str, dict[str, float]] = {}
    for file in files:
        key = file.name.removesuffix(".safetensors")
        module = key.removesuffix(".weight")
        want = manifest[key]
        d = read_tensor(file, key) - read_base_weight(base_dir, key)
        delta, rel = from_dense(module, d, rank=1)
        d_norm = _fro(d)
        pub_err = _fro(delta.dense() - published[module].dense()) / d_norm
        numbers[key] = {
            "rel_residual": rel,
            "manifest_rel_residual": float(want["rel_residual"]),
            "delta_norm": d_norm,
            "manifest_delta_norm": float(want["delta_norm"]),
            "published_rel_diff": pub_err,
        }
        assert abs(rel - want["rel_residual"]) < residual_tol
        assert d_norm == pytest.approx(want["delta_norm"], rel=norm_tol)
        assert pub_err <= 2 * want["rel_residual"] + adapter_slack
        deltas.append(delta)
    edit = Edit.from_deltas(deltas, "realdata-exit-test")
    save_peft(edit, work / "rebuilt")
    assert verify_peft(work / "rebuilt") == []
    return numbers


def _record(name: str, numbers: dict[str, Any]) -> None:
    print(f"\n[exit-0.1] {name}: {json.dumps(numbers, indent=2, sort_keys=True)}")
    out = os.environ.get("ASTRO_REALDATA_OUT")
    if out:
        path = Path(out)
        data = json.loads(path.read_text()) if path.exists() else {}
        data[name] = numbers
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


@pytest.fixture
def realdata() -> Path:
    """Directory named by $ASTRO_REALDATA: base/, s18k6s25/, fable1/, adapter/."""
    root = os.environ.get("ASTRO_REALDATA")
    if not root or not Path(root).is_dir():
        pytest.skip("ASTRO_REALDATA not set to a directory")
    return Path(root)


@pytest.mark.realdata
def test_leg_a_recipe_real(realdata: Path, tmp_path: Path) -> None:
    numbers = leg_a(
        RECIPE,
        realdata / "s18k6s25" / "shift_l18.npy",
        S18K6S25_IDX,
        S18K6S25_COEF,
        realdata / "base",
        realdata / "s18k6s25",
        tmp_path,
        expect_rel_change=S18K6S25_REL_CHANGE,
        rel_err_tol=5e-3,
        change_tol=1e-3,
    )
    _record("leg_a", numbers)


@pytest.mark.realdata
def test_leg_b_dense_real(realdata: Path, tmp_path: Path) -> None:
    numbers = leg_b(
        realdata / "base",
        realdata / "fable1",
        realdata / "adapter",
        tmp_path,
        expect_count=5,
    )
    _record("leg_b", numbers)


OUT, IN = 8, 24
LAYERS = (1, 3, 5)


def _module(layer: int) -> str:
    return f"model.language_model.layers.{layer}.mlp.down_proj"


def _write_weight(directory: Path, filename: str, key: str, w: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: np.ascontiguousarray(w, dtype=np.float32)}, str(directory / filename)
    )


def test_exit_legs_synthetic(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    base_dir = tmp_path / "base"
    bases = {
        layer: rng.standard_normal((OUT, IN)).astype(np.float32) for layer in LAYERS
    }
    base_dir.mkdir()
    save_file(
        {f"{_module(layer)}.weight": w for layer, w in bases.items()},
        str(base_dir / "model.safetensors"),
    )

    # leg A: one column_shift edit on layer 1
    module = _module(1)
    idx, coef = [2, 7, 11, 20], [0.5, -0.25, 1.5, 0.75]
    shift = rng.standard_normal(OUT).astype(np.float32)
    a_dir = tmp_path / "s18k6s25"
    a_dir.mkdir()
    np.save(a_dir / "shift_l18.npy", shift)
    c = np.zeros(IN)
    c[idx] = coef
    dense = shift.astype(np.float64)[:, None] * c[None, :]
    key = f"{module}.weight"
    _write_weight(a_dir, f"{key}.safetensors", key, bases[1] + dense.astype(np.float32))
    recipe = load_recipe(RECIPE)
    recipe["steps"][0]["module"] = module
    recipe["steps"][0]["params"]["n_in"] = IN
    recipe_src = tmp_path / "synthetic.recipe.json"
    save_recipe(recipe_src, recipe)
    a = leg_a(
        recipe_src,
        a_dir / "shift_l18.npy",
        idx,
        coef,
        base_dir,
        a_dir,
        tmp_path / "work_a",
        expect_rel_change=float(np.linalg.norm(dense) / np.linalg.norm(bases[1])),
        rel_err_tol=1e-5,
        change_tol=1e-5,
    )
    assert a["rel_err"] < 1e-5

    # leg B: rank-1 plus small noise on three layers, published as a sign-flipped adapter
    b_dir = tmp_path / "fable1"
    adapter_dir = tmp_path / "adapter"
    tensors: dict[str, dict[str, float]] = {}
    published: list[LowRankDelta] = []
    for layer in LAYERS:
        mod, key = _module(layer), f"{_module(layer)}.weight"
        d = np.outer(rng.standard_normal(OUT), rng.standard_normal(IN))
        d += 0.05 * rng.standard_normal((OUT, IN))
        _write_weight(b_dir, f"{key}.safetensors", key, bases[layer] + d)
        # the saved weight is float32, so measure the delta the reader will see
        seen = read_tensor(b_dir / f"{key}.safetensors", key) - bases[layer]
        s = np.linalg.svd(seen.astype(np.float64), compute_uv=False)
        tensors[key] = {
            "rank": 1,
            "rel_residual": float(np.sqrt(np.sum(s[1:] ** 2) / np.sum(s**2))),
            "delta_norm": float(np.sqrt(np.sum(s**2))),
        }
        delta, _ = from_dense(mod, seen, rank=1)
        published.append(LowRankDelta(mod, -delta.B, -delta.A))
    save_peft(Edit.from_deltas(published, "synthetic"), adapter_dir)
    (adapter_dir / "lora_manifest.json").write_text(json.dumps({"tensors": tensors}))
    b = leg_b(
        base_dir,
        b_dir,
        adapter_dir,
        tmp_path / "work_b",
        expect_count=len(LAYERS),
        residual_tol=1e-5,
        norm_tol=1e-5,
        adapter_slack=1e-5,
    )
    assert set(b) == set(tensors)


def test_record_merges_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out.json"
    monkeypatch.setenv("ASTRO_REALDATA_OUT", str(out))
    _record("one", {"x": 1})
    _record("two", {"y": 2})
    assert json.loads(out.read_text()) == {"one": {"x": 1}, "two": {"y": 2}}
