"""Property tests: parsers raise ValueError on bad input, never anything else."""

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from astro_weights.eval import Record, read_jsonl, write_jsonl

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
        ("[" * 100_000 + "]" * 100_000, "nested too deeply"),
        ('{"a":' * 100_000, "nested too deeply"),
    ],
)
def test_read_jsonl_rejects_overflow_and_deep_nesting(
    tmp_path: Path, line: str, match: str
) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        read_jsonl(path)
