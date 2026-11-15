import copy
import hashlib
import io
import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from astro_weights.edits import (
    column_shift,
    masked_rank1,
    project_out,
    rank1,
    recipe_file,
    scale_column,
    scale_head,
    top_k_mask,
    zero_head,
)
from astro_weights.edits.recipe_file import build_edit, load_recipe, save_recipe
from astro_weights.export import load_peft, save_peft

RNG = np.random.default_rng(0)
OUT, IN = 8, 12
W0 = RNG.standard_normal((OUT, IN)).astype(np.float32)
MOD = "model.layers.{}.mlp.down_proj"


def _u() -> np.ndarray:
    return RNG.standard_normal(OUT).astype(np.float32)


# name -> (params, arrays, direct call given module)
def _cases() -> dict[str, tuple[dict[str, Any], dict[str, np.ndarray], Any]]:
    b, a = _u(), RNG.standard_normal(IN).astype(np.float32)
    u, v = _u(), _u()
    mask = top_k_mask(RNG.standard_normal(IN), 4)
    shift = _u()
    idx = np.array([1, 5, 7])
    coef = np.array([0.5, -1.0, 2.0], dtype=np.float32)
    return {
        "rank1": ({}, {"b": b, "a": a}, lambda m: rank1(m, b, a)),
        "masked_rank1": (
            {"alpha": 0.8},
            {"u": u, "mask": mask},
            lambda m: masked_rank1(m, W0, u, mask, 0.8),
        ),
        "column_shift": (
            {"n_in": IN},
            {"shift": shift, "idx": idx, "coef": coef},
            lambda m: column_shift(m, shift, idx, coef, IN),
        ),
        "project_out": (
            {"strength": 0.9},
            {"v": v},
            lambda m: project_out(m, W0, v, 0.9),
        ),
        "scale_column": ({"j": 3, "s": 1.5}, {}, lambda m: scale_column(m, W0, 3, 1.5)),
        "scale_head": (
            {"head": 1, "n_heads": 4, "s": 0.5},
            {},
            lambda m: scale_head(m, W0, 1, 4, 0.5),
        ),
        "zero_head": (
            {"head": 2, "n_heads": 3},
            {},
            lambda m: zero_head(m, W0, 2, 3),
        ),
    }


def _write(tmp_path: Path, steps: list[dict[str, Any]], base: str = "org/tiny") -> Path:
    path = tmp_path / "recipe.json"
    save_recipe(path, {"format": 1, "base_model": base, "steps": steps})
    return path


def _step(
    name: str,
    module: str,
    params: dict[str, Any],
    arrays: dict[str, np.ndarray],
    directory: Path,
) -> dict[str, Any]:
    files = {}
    for key, arr in arrays.items():
        fname = f"{module}.{key}.npy"
        np.save(directory / fname, arr)
        files[key] = fname
    return {"recipe": name, "module": module, "params": params, "arrays": files}


@pytest.mark.parametrize("name", list(_cases()))
def test_each_recipe_matches_direct_call(tmp_path: Path, name: str) -> None:
    params, arrays, direct = _cases()[name]
    mod = MOD.format(1)
    path = _write(tmp_path, [_step(name, mod, params, arrays, tmp_path)])
    calls: list[str] = []

    def weights(module: str) -> np.ndarray:
        calls.append(module)
        return W0

    edit = build_edit(load_recipe(path), weights, tmp_path)
    np.testing.assert_allclose(edit.deltas[0].dense(), direct(mod).dense(), atol=1e-7)
    assert edit.base_model == "org/tiny"
    needs = name not in ("rank1", "column_shift")
    assert calls == ([mod] if needs else [])


def test_mixed_ranks_combine(tmp_path: Path) -> None:
    cases = _cases()
    steps = []
    for i, name in enumerate(("zero_head", "rank1")):
        params, arrays, _ = cases[name]
        steps.append(_step(name, MOD.format(i), params, arrays, tmp_path))
    edit = build_edit(load_recipe(_write(tmp_path, steps)), lambda m: W0, tmp_path)
    assert edit.rank == 4  # zero_head with n_heads=3 on 12 inputs
    assert {d.module for d in edit.deltas} == {MOD.format(0), MOD.format(1)}
    direct = cases["rank1"][2](MOD.format(1))
    by_mod = {d.module: d for d in edit.deltas}
    np.testing.assert_allclose(by_mod[MOD.format(1)].dense(), direct.dense(), atol=1e-7)


def test_meta_carries_recipe_and_hashes(tmp_path: Path) -> None:
    params, arrays, _ = _cases()["rank1"]
    step = _step("rank1", MOD.format(1), params, arrays, tmp_path)
    recipe = load_recipe(_write(tmp_path, [step]))
    edit = build_edit(recipe, lambda m: W0, tmp_path)
    got = edit.meta["recipe"]
    assert isinstance(got, dict)
    expected = {
        f: hashlib.sha256((tmp_path / f).read_bytes()).hexdigest()
        for f in step["arrays"].values()
    }
    assert got["arrays_sha256"] == expected
    assert got["steps"] == recipe["steps"]
    assert "arrays_sha256" not in recipe  # input recipe is not mutated


def test_meta_survives_peft_roundtrip(tmp_path: Path) -> None:
    params, arrays, _ = _cases()["scale_column"]
    step = _step("scale_column", MOD.format(1), params, arrays, tmp_path)
    edit = build_edit(load_recipe(_write(tmp_path, [step])), lambda m: W0, tmp_path)
    out = tmp_path / "adapter"
    save_peft(edit, out)
    assert load_peft(out).meta["recipe"] == edit.meta["recipe"]


def test_save_recipe_sorted_with_newline(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    save_recipe(path, {"format": 1, "base_model": "x", "steps": []})
    text = path.read_text()
    assert text.endswith("\n")
    assert text.index('"base_model"') < text.index('"format"') < text.index('"steps"')


def test_duplicate_module_rejected(tmp_path: Path) -> None:
    step = _step("zero_head", MOD.format(1), {"head": 0, "n_heads": 3}, {}, tmp_path)
    recipe = load_recipe(_write(tmp_path, [step, copy.deepcopy(step)]))
    with pytest.raises(ValueError, match=r"step 1.*already"):
        build_edit(recipe, lambda m: W0, tmp_path)


def test_pickled_array_refused(tmp_path: Path) -> None:
    np.save(tmp_path / "b.npy", np.array([object()], dtype=object), allow_pickle=True)
    np.save(tmp_path / "a.npy", np.ones(IN))
    step = {
        "recipe": "rank1",
        "module": MOD.format(1),
        "params": {},
        "arrays": {"b": "b.npy", "a": "a.npy"},
    }
    recipe = load_recipe(_write(tmp_path, [step]))
    with pytest.raises(ValueError, match="pickle"):
        build_edit(recipe, lambda m: W0, tmp_path)


def _good_step() -> dict[str, Any]:
    return {
        "recipe": "masked_rank1",
        "module": "m",
        "params": {"alpha": 1.0},
        "arrays": {"u": "u.npy", "mask": "mask.npy"},
    }


def _bad(step: dict[str, Any], **changes: Any) -> dict[str, Any]:
    out = copy.deepcopy(step)
    out.update(changes)
    return out


def _with(step: dict[str, Any], key: str, **changes: Any) -> dict[str, Any]:
    out = copy.deepcopy(step)
    out[key] = {**out[key], **changes}
    return out


BAD_STEPS: list[tuple[dict[str, Any], str]] = [
    (_bad(_good_step(), recipe="nope"), "step 1.*nope"),
    (_bad(_good_step(), module=""), "step 1.*module"),
    (_bad(_good_step(), module=3), "step 1.*module"),
    (_with(_good_step(), "params", extra=1), "step 1.*unknown param 'extra'"),
    (_bad(_good_step(), params={}), "step 1.*missing param 'alpha'"),
    (_with(_good_step(), "params", alpha="1"), "step 1.*alpha"),
    (_with(_good_step(), "params", alpha=True), "step 1.*alpha"),
    (_with(_good_step(), "arrays", extra="x.npy"), "step 1.*unknown array 'extra'"),
    (_bad(_good_step(), arrays={"u": "u.npy"}), "step 1.*missing array 'mask'"),
    (_with(_good_step(), "arrays", u=3), "step 1.*array 'u'"),
    (_bad(_good_step(), params=[]), "step 1.*params"),
    (
        {"recipe": "scale_column", "module": "m", "params": {"j": 1.5, "s": 1}},
        "step 1.*'j'",
    ),
    ("nope", "step 1"),  # type: ignore[list-item]
]


@pytest.mark.parametrize(("step", "match"), BAD_STEPS)
def test_bad_step_rejected(tmp_path: Path, step: dict[str, Any], match: str) -> None:
    path = _write(tmp_path, [_good_step(), step])
    with pytest.raises(ValueError, match=match):
        load_recipe(path)


@pytest.mark.parametrize(
    ("doc", "match"),
    [
        ([], "JSON object"),
        ({"format": 2, "base_model": "x", "steps": [_good_step()]}, "format"),
        ({"base_model": "x", "steps": [_good_step()]}, "format"),
        ({"format": 1, "base_model": "", "steps": [_good_step()]}, "base_model"),
        ({"format": 1, "base_model": 3, "steps": [_good_step()]}, "base_model"),
        ({"format": 1, "base_model": "x", "steps": []}, "steps"),
        ({"format": 1, "base_model": "x", "steps": "s"}, "steps"),
        ({"format": 1, "base_model": "x"}, "steps"),
    ],
)
def test_bad_top_level_rejected(tmp_path: Path, doc: object, match: str) -> None:
    path = tmp_path / "r.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match=match):
        load_recipe(path)


def _rank1_step(b: str = "b.npy", a: str = "a.npy") -> dict[str, Any]:
    return {
        "recipe": "rank1",
        "module": MOD.format(1),
        "params": {},
        "arrays": {"b": b, "a": a},
    }


@pytest.mark.parametrize(
    "name", ["../x.npy", "/abs/x.npy", "a/../../x.npy", "d/x.npy", ".", ".."]
)
def test_array_name_must_be_plain(tmp_path: Path, name: str) -> None:
    path = _write(tmp_path, [_rank1_step(b=name)])
    with pytest.raises(ValueError, match=r"step 0.*array 'b'"):
        load_recipe(path)


def test_array_traversal_not_read_by_build_edit(tmp_path: Path) -> None:
    root = tmp_path / "recipe"
    root.mkdir()
    np.save(tmp_path / "outside.npy", np.ones(OUT))
    np.save(root / "a.npy", np.ones(IN))
    recipe = {
        "format": 1,
        "base_model": "x",
        "steps": [_rank1_step(b="../outside.npy")],
    }
    with pytest.raises(ValueError, match="array 'b'"):
        build_edit(recipe, lambda m: W0, root)


def test_symlink_escape_rejected(tmp_path: Path) -> None:
    root = tmp_path / "recipe"
    root.mkdir()
    np.save(tmp_path / "outside.npy", np.ones(OUT))
    np.save(root / "a.npy", np.ones(IN))
    (root / "b.npy").symlink_to(tmp_path / "outside.npy")
    recipe = {"format": 1, "base_model": "x", "steps": [_rank1_step()]}
    with pytest.raises(ValueError, match="escapes"):
        build_edit(recipe, lambda m: W0, root)


def test_missing_array_is_value_error(tmp_path: Path) -> None:
    np.save(tmp_path / "a.npy", np.ones(IN))
    recipe = load_recipe(_write(tmp_path, [_rank1_step()]))
    with pytest.raises(ValueError, match=r"step 0 \(rank1\): array 'b' \(b.npy\)"):
        build_edit(recipe, lambda m: W0, tmp_path)


def test_npz_under_npy_name_is_value_error(tmp_path: Path) -> None:
    with (tmp_path / "b.npy").open("wb") as f:
        np.savez(f, x=np.ones(OUT))
    np.save(tmp_path / "a.npy", np.ones(IN))
    recipe = load_recipe(_write(tmp_path, [_rank1_step()]))
    with pytest.raises(ValueError, match=r"array 'b'"):
        build_edit(recipe, lambda m: W0, tmp_path)


def test_garbage_array_is_value_error(tmp_path: Path) -> None:
    (tmp_path / "b.npy").write_bytes(b"not numpy")
    np.save(tmp_path / "a.npy", np.ones(IN))
    recipe = load_recipe(_write(tmp_path, [_rank1_step()]))
    with pytest.raises(ValueError, match=r"array 'b'"):
        build_edit(recipe, lambda m: W0, tmp_path)


def test_missing_recipe_file_is_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read recipe"):
        load_recipe(tmp_path / "nope.json")


def test_invalid_json_names_path(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text("{oops")
    with pytest.raises(ValueError, match=r"cannot read recipe.*r\.json"):
        load_recipe(path)


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_non_finite_param_rejected(tmp_path: Path, bad: str) -> None:
    step = json.dumps(
        {
            "recipe": "masked_rank1",
            "module": "m",
            "params": {"alpha": "@@"},
            "arrays": {"u": "u.npy", "mask": "mask.npy"},
        }
    ).replace('"@@"', bad)
    path = tmp_path / "r.json"
    path.write_text(f'{{"format": 1, "base_model": "x", "steps": [{step}]}}')
    with pytest.raises(ValueError, match=r"step 0.*alpha.*finite"):
        load_recipe(path)


def test_deeply_nested_json_is_value_error(tmp_path: Path) -> None:
    path = tmp_path / "deep.json"
    path.write_text("[" * 200_000)
    with pytest.raises(ValueError, match=r"cannot read recipe.*deep\.json"):
        load_recipe(path)


# --- regressions found by tests/test_fuzz.py ---------------------------------


def _build_rank1(tmp_path: Path, **files: bytes) -> None:
    for name, raw in files.items():
        (tmp_path / name).write_bytes(raw)
    recipe = {"format": 1, "base_model": "x", "steps": [_rank1_step()]}
    build_edit(recipe, lambda _m: W0, tmp_path)


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def test_malformed_npy_header_is_value_error(tmp_path: Path) -> None:
    # numpy's header parser raises tokenize.TokenError on an unclosed bracket
    header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (4,"
    header += b" " * (-(10 + len(header) + 1) % 64) + b"\n"
    raw = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header
    good = _npy_bytes(np.ones(1, np.float32))
    with pytest.raises(ValueError, match=r"invalid \.npy header"):
        _build_rank1(tmp_path, **{"b.npy": raw, "a.npy": good})


def test_npy_header_claiming_huge_shape_is_value_error(tmp_path: Path) -> None:
    header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (1000000000000,), }"
    header += b" " * (-(10 + len(header) + 1) % 64) + b"\n"
    raw = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header
    good = _npy_bytes(np.ones(1, np.float32))
    with pytest.raises(ValueError, match=r"truncated \.npy data"):
        _build_rank1(tmp_path, **{"b.npy": raw, "a.npy": good})


@pytest.mark.parametrize(
    ("arr", "match"),
    [
        (np.array(1.0, dtype=np.float32), "1-D"),
        (np.ones((2, 2), dtype=np.float32), "1-D"),
        (np.ones(4, dtype=np.complex64), "unsupported dtype"),
        (np.array(["a", "b"]), "unsupported dtype"),
    ],
)
def test_npy_must_be_1d_numeric(tmp_path: Path, arr: np.ndarray, match: str) -> None:
    good = _npy_bytes(np.ones(1, np.float32))
    with pytest.raises(ValueError, match=match):
        _build_rank1(tmp_path, **{"b.npy": _npy_bytes(arr), "a.npy": good})


def test_npy_over_size_cap_rejected_unread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recipe_file, "MAX_NPY_BYTES", 200)
    small = _npy_bytes(np.ones(1, np.float32))
    assert len(small) <= 200
    big = _npy_bytes(np.ones(64, np.float32))
    assert len(big) > 200
    _build_rank1(tmp_path, **{"b.npy": small, "a.npy": small})  # at or under the cap
    with pytest.raises(ValueError, match="over the 200 limit"):
        _build_rank1(tmp_path, **{"b.npy": big, "a.npy": small})


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs os.mkfifo")
def test_fifo_npy_is_value_error_not_hang(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "b.npy")
    good = _npy_bytes(np.ones(1, np.float32))
    with pytest.raises(ValueError, match="not a regular file"):
        _build_rank1(tmp_path, **{"a.npy": good})


def test_npy_growing_past_cap_after_fstat_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read itself is bounded, not just the fstat size."""
    path = tmp_path / "x.npy"
    path.write_bytes(b"x" * 300)
    monkeypatch.setattr(recipe_file, "MAX_NPY_BYTES", 200)
    real = os.fstat
    monkeypatch.setattr(
        os,
        "fstat",
        lambda fd: SimpleNamespace(st_mode=real(fd).st_mode, st_size=0),
    )
    with pytest.raises(ValueError, match="over the 200 limit"):
        recipe_file._read_capped(path)


def test_huge_int_param_is_value_error(tmp_path: Path) -> None:
    step = {
        "recipe": "masked_rank1",
        "module": "m",
        "params": {"alpha": 10**400},
        "arrays": {"u": "u.npy", "mask": "mask.npy"},
    }
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"format": 1, "base_model": "x", "steps": [step]}))
    with pytest.raises(ValueError, match=r"alpha.*finite"):
        load_recipe(path)


def test_column_shift_n_in_capped(tmp_path: Path) -> None:
    step = {
        "recipe": "column_shift",
        "module": "m",
        "params": {"n_in": 10**12},
        "arrays": {"shift": "s.npy", "idx": "i.npy", "coef": "c.npy"},
    }
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"format": 1, "base_model": "x", "steps": [step]}))
    with pytest.raises(ValueError, match="n_in must be at most"):
        load_recipe(path)


def test_npy_with_crlf_and_ctrl_z_bytes_read_verbatim(tmp_path: Path) -> None:
    """Payload bytes 0d 0a 1a must survive the read (Windows text-mode guard)."""
    payload = np.frombuffer(bytes([0x0D, 0x0A, 0x1A, 0x3F]) * OUT, dtype="<f4")
    assert b"\r\n\x1a" in payload.tobytes()
    a_payload = np.frombuffer(bytes([0x0D, 0x0A, 0x1A, 0x3F]) * IN, dtype="<f4")
    raw_b, raw_a = _npy_bytes(payload), _npy_bytes(a_payload)
    for name, raw in (("b.npy", raw_b), ("a.npy", raw_a)):
        (tmp_path / name).write_bytes(raw)
    recipe = {"format": 1, "base_model": "x", "steps": [_rank1_step()]}
    edit = build_edit(recipe, lambda _m: W0, tmp_path)
    got = edit.meta["recipe"]
    assert isinstance(got, dict)
    assert got["arrays_sha256"] == {
        "b.npy": hashlib.sha256(raw_b).hexdigest(),
        "a.npy": hashlib.sha256(raw_a).hexdigest(),
    }
    assert recipe_file._read_capped(tmp_path / "b.npy") == raw_b


@pytest.mark.parametrize(
    ("name", "key", "bad", "match"),
    [
        (
            "column_shift",
            "idx",
            np.array([1, 5, 7], dtype=bool),
            "idx must be an integer",
        ),
        ("column_shift", "idx", np.array([1.0, 5.0, 7.0]), "idx must be an integer"),
        (
            "column_shift",
            "coef",
            np.array([1, 0, 1], dtype=bool),
            "coef must be a real",
        ),
        ("column_shift", "shift", np.ones(OUT, dtype=bool), "shift must be a real"),
        ("rank1", "b", np.ones(OUT, dtype=bool), "b must be a real"),
        ("project_out", "v", np.ones(OUT, dtype=bool), "v must be a real"),
    ],
)
def test_wrong_dtype_array_is_value_error(
    tmp_path: Path, name: str, key: str, bad: np.ndarray, match: str
) -> None:
    params, arrays, _ = _cases()[name]
    arrays = {**arrays, key: bad}
    step = _step(name, "m", params, arrays, tmp_path)
    recipe = load_recipe(_write(tmp_path, [step]))
    with pytest.raises(ValueError, match=match):
        build_edit(recipe, lambda m: W0, tmp_path)
