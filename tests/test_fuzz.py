"""Property tests: parsers raise ValueError on bad input, never anything else."""

import io
import json
import math
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from astro_weights.edits import Edit, LowRankDelta
from astro_weights.edits.recipe_file import build_edit, load_recipe
from astro_weights.eval import Record, read_jsonl, write_jsonl
from astro_weights.export.base import read_tensor, tensor_sha256
from astro_weights.export.peft import (
    CONFIG,
    MANIFEST,
    WEIGHTS,
    read_peft,
    save_peft,
    verify_peft,
)

FUZZ = settings(max_examples=300, deadline=None)


def only_value_error(parse: Callable[[Path], object], path: Path) -> None:
    """Call `parse(path)`; any exception other than ValueError fails the test."""
    try:
        parse(path)
    except ValueError:
        pass


# --- eval.read_jsonl -------------------------------------------------------

json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.integers(min_value=10**300, max_value=10**500)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda children: st.lists(children) | st.dictionaries(st.text(), children),
    max_leaves=10,
)

_FIELDS = (
    "item_id",
    "seed",
    "arm",
    "prompt_hash",
    "output",
    "score",
    "n_output_tokens",
    "hit_cap",
    "cluster_id",
)
# Objects with the right keys and arbitrary JSON values reach the type checks.
record_like = st.fixed_dictionaries({}, optional=dict.fromkeys(_FIELDS, json_values))

finite_floats = st.floats(allow_nan=False, allow_infinity=False)
records = st.builds(
    Record,
    item_id=st.text(),
    seed=st.integers(),
    arm=st.text(),
    prompt_hash=st.text(),
    output=st.text(),
    score=finite_floats,
    n_output_tokens=st.integers(),
    hit_cap=st.booleans(),
    cluster_id=st.none() | st.text(),
)


@FUZZ
@given(data=st.binary())
def test_read_jsonl_arbitrary_bytes(
    tmp_path_factory: pytest.TempPathFactory, data: bytes
) -> None:
    path = tmp_path_factory.mktemp("fuzz") / "x.jsonl"
    path.write_bytes(data)
    only_value_error(read_jsonl, path)


@FUZZ
@given(text=st.text())
def test_read_jsonl_arbitrary_text(
    tmp_path_factory: pytest.TempPathFactory, text: str
) -> None:
    path = tmp_path_factory.mktemp("fuzz") / "x.jsonl"
    path.write_text(text, encoding="utf-8")
    only_value_error(read_jsonl, path)


@FUZZ
@given(values=st.lists(json_values | record_like, max_size=5))
def test_read_jsonl_json_lines(
    tmp_path_factory: pytest.TempPathFactory, values: list[Any]
) -> None:
    path = tmp_path_factory.mktemp("fuzz") / "x.jsonl"
    path.write_text("".join(json.dumps(v) + "\n" for v in values), encoding="utf-8")
    only_value_error(read_jsonl, path)


@FUZZ
@given(depth=st.integers(0, 50_000), opener=st.sampled_from(["[", '{"a":']))
def test_read_jsonl_deep_nesting(
    tmp_path_factory: pytest.TempPathFactory, depth: int, opener: str
) -> None:
    path = tmp_path_factory.mktemp("fuzz") / "x.jsonl"
    path.write_text(opener * depth + "\n", encoding="utf-8")
    only_value_error(read_jsonl, path)


@FUZZ
@given(recs=st.lists(records, max_size=5))
def test_jsonl_round_trip(
    tmp_path_factory: pytest.TempPathFactory, recs: list[Record]
) -> None:
    path = tmp_path_factory.mktemp("fuzz") / "x.jsonl"
    write_jsonl(path, recs)
    back = read_jsonl(path)
    assert back == recs
    assert all(math.isfinite(r.score) for r in back)


# --- regressions found by the fuzz tests ------------------------------------

_LINE = {
    "item_id": "a",
    "seed": 1,
    "arm": "x",
    "prompt_hash": "h",
    "output": "o",
    "score": 1,
    "n_output_tokens": 1,
    "hit_cap": False,
}


@pytest.mark.parametrize(
    ("line", "match"),
    [
        (json.dumps({**_LINE, "score": 10**400}), "finite"),
        # Python 3.14 parses deep nesting without RecursionError; any ValueError is fine
        ("[" * 100_000 + "]" * 100_000, ""),
        ('{"a":' * 100_000, ""),
    ],
)
def test_read_jsonl_rejects_overflow_and_deep_nesting(
    tmp_path: Path, line: str, match: str
) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        read_jsonl(path)


# --- shared helpers for the file-format fuzzers -----------------------------


def mutate_json(obj: Any, data: st.DataObject) -> Any:
    """Copy of `obj` with one randomly chosen node replaced by an arbitrary JSON value."""
    paths: list[tuple[Any, ...]] = []

    def walk(node: Any, path: tuple[Any, ...]) -> None:
        paths.append(path)
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, (*path, k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, (*path, i))

    walk(obj, ())
    path = data.draw(st.sampled_from(paths))
    new = data.draw(json_values)
    if not path:
        return new
    out = json.loads(json.dumps(obj))
    node = out
    for p in path[:-1]:
        node = node[p]
    if data.draw(st.booleans()) and isinstance(node, dict):
        del node[path[-1]]
    else:
        node[path[-1]] = new
    return out


def mutate_bytes(raw: bytes, data: st.DataObject) -> bytes:
    """`raw` truncated and/or with a few bytes flipped."""
    out = bytearray(raw)
    for _ in range(data.draw(st.integers(0, 4))):
        if out:
            out[data.draw(st.integers(0, len(out) - 1))] = data.draw(
                st.integers(0, 255)
            )
    if data.draw(st.booleans()):
        del out[data.draw(st.integers(0, len(out))) :]
    return bytes(out)


# --- export.base: read_tensor / tensor_sha256 -------------------------------


def _st_bytes(header: object, body: bytes, n: int | None = None) -> bytes:
    blob = json.dumps(header).encode()
    return struct.pack("<Q", len(blob) if n is None else n) + blob + body


def _both_value_error(path: Path) -> None:
    only_value_error(lambda p: read_tensor(p, "w"), path)
    only_value_error(lambda p: tensor_sha256(p, "w"), path)


_VALID_ST = _st_bytes(
    {"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}, bytes(24)
)


@FUZZ
@given(raw=st.binary(max_size=300))
def test_read_tensor_arbitrary_bytes(
    tmp_path_factory: pytest.TempPathFactory, raw: bytes
) -> None:
    path = tmp_path_factory.mktemp("fuzz") / "x.safetensors"
    path.write_bytes(raw)
    _both_value_error(path)


@FUZZ
@given(data=st.data())
def test_read_tensor_mutated_bytes(
    tmp_path_factory: pytest.TempPathFactory, data: st.DataObject
) -> None:
    path = tmp_path_factory.mktemp("fuzz") / "x.safetensors"
    path.write_bytes(mutate_bytes(_VALID_ST, data))
    _both_value_error(path)


@FUZZ
@given(
    n=st.none() | st.integers(0, 2**64 - 1),
    entry=st.fixed_dictionaries(
        {},
        optional={
            "dtype": st.sampled_from(["F32", "F16", "BF16", "I8", "F64", ""])
            | json_values,
            "shape": st.lists(st.integers(-5, 2**70), max_size=4) | json_values,
            "data_offsets": st.lists(st.integers(-5, 2**70), max_size=3) | json_values,
        },
    ),
    body=st.binary(max_size=64),
    header=st.none() | json_values,
)
def test_read_tensor_mutated_header(
    tmp_path_factory: pytest.TempPathFactory,
    n: int | None,
    entry: dict[str, Any],
    body: bytes,
    header: Any,
) -> None:
    path = tmp_path_factory.mktemp("fuzz") / "x.safetensors"
    path.write_bytes(_st_bytes({"w": entry} if header is None else header, body, n))
    _both_value_error(path)


# --- edits.recipe_file: load_recipe / build_edit ----------------------------

_W0 = np.ones((4, 6), dtype=np.float32)
_RECIPES_VALID: dict[str, Any] = {
    "format": 1,
    "base_model": "org/m",
    "steps": [
        {
            "recipe": "rank1",
            "module": "model.layers.0.mlp.down_proj",
            "arrays": {"b": "b.npy", "a": "a.npy"},
        },
        {
            "recipe": "scale_column",
            "module": "model.layers.1.mlp.down_proj",
            "params": {"j": 2, "s": 0.5},
        },
        {
            "recipe": "masked_rank1",
            "module": "model.layers.2.mlp.down_proj",
            "params": {"alpha": 0.5},
            "arrays": {"u": "u.npy", "mask": "mask.npy"},
        },
        {
            "recipe": "column_shift",
            "module": "model.layers.3.mlp.down_proj",
            "params": {"n_in": 6},
            "arrays": {"shift": "u.npy", "idx": "idx.npy", "coef": "coef.npy"},
        },
        {
            "recipe": "project_out",
            "module": "model.layers.4.mlp.down_proj",
            "params": {"strength": 1.0},
            "arrays": {"v": "u.npy"},
        },
        {
            "recipe": "scale_head",
            "module": "model.layers.5.self_attn.o_proj",
            "params": {"head": 1, "n_heads": 2, "s": 0.5},
        },
        {
            "recipe": "zero_head",
            "module": "model.layers.6.self_attn.o_proj",
            "params": {"head": 0, "n_heads": 3},
        },
    ],
}


def _npy(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


_NPY_FILES = {
    "b.npy": _npy(np.ones(4, dtype=np.float32)),
    "a.npy": _npy(np.ones(6, dtype=np.float32)),
    "u.npy": _npy(np.ones(4, dtype=np.float32)),
    "mask.npy": _npy(np.array([1, 0, 1, 0, 1, 0], dtype=bool)),
    "idx.npy": _npy(np.array([1, 5], dtype=np.int64)),
    "coef.npy": _npy(np.array([0.5, -1.0], dtype=np.float32)),
}


def _weights(_module: str) -> np.ndarray:
    return _W0


def _load_and_build(path: Path) -> None:
    recipe = load_recipe(path)
    build_edit(recipe, _weights, path.parent)


def _recipe_dir(root: Path, files: dict[str, bytes]) -> Path:
    for name, raw in files.items():
        (root / name).write_bytes(raw)
    return root / "recipe.json"


@FUZZ
@given(raw=st.binary(max_size=300))
def test_load_recipe_arbitrary_bytes(
    tmp_path_factory: pytest.TempPathFactory, raw: bytes
) -> None:
    path = _recipe_dir(tmp_path_factory.mktemp("fuzz"), {"recipe.json": raw})
    only_value_error(_load_and_build, path)


@FUZZ
@given(value=json_values)
def test_load_recipe_arbitrary_json(
    tmp_path_factory: pytest.TempPathFactory, value: Any
) -> None:
    path = _recipe_dir(
        tmp_path_factory.mktemp("fuzz"), {"recipe.json": json.dumps(value).encode()}
    )
    only_value_error(_load_and_build, path)


@FUZZ
@given(data=st.data())
def test_build_edit_mutated_recipe(
    tmp_path_factory: pytest.TempPathFactory, data: st.DataObject
) -> None:
    recipe = mutate_json(_RECIPES_VALID, data)
    path = _recipe_dir(
        tmp_path_factory.mktemp("fuzz"),
        {"recipe.json": json.dumps(recipe).encode(), **_NPY_FILES},
    )
    only_value_error(_load_and_build, path)


@FUZZ
@given(data=st.data(), name=st.sampled_from(sorted(_NPY_FILES)))
def test_build_edit_mutated_npy(
    tmp_path_factory: pytest.TempPathFactory, data: st.DataObject, name: str
) -> None:
    raw = (
        data.draw(st.binary(max_size=300))
        if data.draw(st.booleans())
        else mutate_bytes(_NPY_FILES[name], data)
    )
    path = _recipe_dir(
        tmp_path_factory.mktemp("fuzz"),
        {
            "recipe.json": json.dumps(_RECIPES_VALID).encode(),
            **_NPY_FILES,
            name: raw,
        },
    )
    only_value_error(_load_and_build, path)


@FUZZ
@given(
    arr=hnp.arrays(
        dtype=hnp.scalar_dtypes(),
        shape=hnp.array_shapes(min_dims=0, max_dims=3, max_side=8),
    ),
    name=st.sampled_from(sorted(_NPY_FILES)),
)
def test_build_edit_odd_arrays(
    tmp_path_factory: pytest.TempPathFactory, arr: np.ndarray, name: str
) -> None:
    path = _recipe_dir(
        tmp_path_factory.mktemp("fuzz"),
        {
            "recipe.json": json.dumps(_RECIPES_VALID).encode(),
            **_NPY_FILES,
            name: _npy(arr),
        },
    )
    only_value_error(_load_and_build, path)


# --- export.peft: read_peft / verify_peft ------------------------------------


@pytest.fixture(scope="module")
def adapter_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, bytes]:
    """File name -> bytes of a small valid adapter directory."""
    rng = np.random.default_rng(0)
    deltas = tuple(
        LowRankDelta(
            f"model.layers.{i}.mlp.down_proj",
            rng.standard_normal((4, 2)),
            rng.standard_normal((2, 3)),
        )
        for i in range(2)
    )
    out = tmp_path_factory.mktemp("adapter")
    save_peft(Edit(deltas, "org/m", meta={"k": [1, 2]}), out)
    return {p.name: p.read_bytes() for p in out.iterdir()}


def _adapter_dir(root: Path, files: dict[str, bytes]) -> Path:
    for name, raw in files.items():
        (root / name).write_bytes(raw)
    return root


def _read_peft_value_or_os_error(path: Path) -> None:
    """read_peft may raise ValueError or OSError for bad content, nothing else."""
    try:
        read_peft(path)
    except (ValueError, OSError):
        pass


def _check_peft(path: Path) -> None:
    """verify_peft never raises and returns strings; read_peft is ValueError/OSError-only."""
    problems = verify_peft(path)
    assert all(isinstance(p, str) for p in problems)
    _read_peft_value_or_os_error(path)


def _fuzz_adapter(
    tmp_path_factory: pytest.TempPathFactory,
    files: dict[str, bytes],
    name: str,
    raw: bytes,
) -> None:
    path = _adapter_dir(tmp_path_factory.mktemp("fuzz"), {**files, name: raw})
    _check_peft(path)


@FUZZ
@given(data=st.data(), name=st.sampled_from([CONFIG, MANIFEST]))
def test_peft_mutated_json(
    tmp_path_factory: pytest.TempPathFactory,
    adapter_files: dict[str, bytes],
    data: st.DataObject,
    name: str,
) -> None:
    doc = mutate_json(json.loads(adapter_files[name]), data)
    _fuzz_adapter(tmp_path_factory, adapter_files, name, json.dumps(doc).encode())


@FUZZ
@given(raw=st.binary(max_size=300), name=st.sampled_from([CONFIG, MANIFEST, WEIGHTS]))
def test_peft_arbitrary_bytes(
    tmp_path_factory: pytest.TempPathFactory,
    adapter_files: dict[str, bytes],
    raw: bytes,
    name: str,
) -> None:
    _fuzz_adapter(tmp_path_factory, adapter_files, name, raw)


@FUZZ
@given(data=st.data(), name=st.sampled_from([CONFIG, MANIFEST, WEIGHTS]))
def test_peft_mutated_bytes(
    tmp_path_factory: pytest.TempPathFactory,
    adapter_files: dict[str, bytes],
    data: st.DataObject,
    name: str,
) -> None:
    raw = mutate_bytes(adapter_files[name], data)
    _fuzz_adapter(tmp_path_factory, adapter_files, name, raw)


@FUZZ
@given(data=st.data())
def test_peft_mutated_safetensors_header(
    tmp_path_factory: pytest.TempPathFactory,
    adapter_files: dict[str, bytes],
    data: st.DataObject,
) -> None:
    raw = adapter_files[WEIGHTS]
    (n,) = struct.unpack("<Q", raw[:8])
    header = mutate_json(json.loads(raw[8 : 8 + n]), data)
    _fuzz_adapter(
        tmp_path_factory, adapter_files, WEIGHTS, _st_bytes(header, raw[8 + n :])
    )


# --- export.gguf: read_lora_gguf ---------------------------------------------


@pytest.fixture(scope="module")
def adapter_gguf(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """Bytes of a small valid adapter GGUF."""
    pytest.importorskip("gguf")
    from astro_weights.export.gguf import save_lora_gguf

    rng = np.random.default_rng(1)
    delta = LowRankDelta(
        "model.layers.1.mlp.down_proj",
        rng.standard_normal((4, 2)),
        rng.standard_normal((2, 3)),
    )
    path = tmp_path_factory.mktemp("gguf") / "a.gguf"
    save_lora_gguf(Edit((delta,), "org/m"), path, "qwen35")
    return path.read_bytes()


def _read_lora(path: Path) -> None:
    from astro_weights.export.gguf import read_lora_gguf

    read_lora_gguf(path)


@FUZZ
@given(raw=st.binary(max_size=300))
def test_read_lora_gguf_arbitrary_bytes(
    tmp_path_factory: pytest.TempPathFactory, adapter_gguf: bytes, raw: bytes
) -> None:
    path = tmp_path_factory.mktemp("fuzz") / "x.gguf"
    path.write_bytes(b"GGUF" + raw)  # past the magic check, into the parser
    only_value_error(_read_lora, path)


@FUZZ
@given(data=st.data())
def test_read_lora_gguf_mutated(
    tmp_path_factory: pytest.TempPathFactory, adapter_gguf: bytes, data: st.DataObject
) -> None:
    path = tmp_path_factory.mktemp("fuzz") / "x.gguf"
    path.write_bytes(mutate_bytes(adapter_gguf, data))
    only_value_error(_read_lora, path)


@FUZZ
@given(
    n_tensors=st.none() | st.integers(0, 2**64 - 1) | st.sampled_from([2**31, 2**63]),
    n_kv=st.none() | st.integers(0, 2**64 - 1) | st.sampled_from([2**31, 2**63]),
    version=st.none() | st.integers(0, 2**32 - 1),
    patch=st.lists(
        st.tuples(st.integers(0, 400), st.binary(min_size=1, max_size=8)), max_size=3
    ),
)
def test_read_lora_gguf_mutated_header(
    tmp_path_factory: pytest.TempPathFactory,
    adapter_gguf: bytes,
    n_tensors: int | None,
    n_kv: int | None,
    version: int | None,
    patch: list[tuple[int, bytes]],
) -> None:
    out = bytearray(adapter_gguf)
    if version is not None:
        out[4:8] = struct.pack("<I", version)
    if n_tensors is not None:
        out[8:16] = struct.pack("<Q", n_tensors)
    if n_kv is not None:
        out[16:24] = struct.pack("<Q", n_kv)
    for at, raw in patch:  # kv strings, value types, tensor dims and offsets
        out[at : at + len(raw)] = raw
    path = tmp_path_factory.mktemp("fuzz") / "x.gguf"
    path.write_bytes(bytes(out))
    only_value_error(_read_lora, path)


@FUZZ
@given(
    item=st.integers(0, 14),
    length=st.integers(0, 2**64 - 1) | st.sampled_from([2**25, 2**32, 2**40, 2**63]),
    tail=st.binary(max_size=64),
)
def test_read_lora_gguf_array_kv(
    tmp_path_factory: pytest.TempPathFactory,
    adapter_gguf: bytes,  # skips without the gguf extra
    item: int,
    length: int,
    tail: bytes,
) -> None:
    """An array kv with any item type and length never hangs or leaks."""
    key = b"k"
    kv = struct.pack("<Q", len(key)) + key + struct.pack("<II", 9, item)
    kv += struct.pack("<Q", length) + tail
    path = tmp_path_factory.mktemp("fuzz") / "x.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + kv)
    only_value_error(_read_lora, path)
