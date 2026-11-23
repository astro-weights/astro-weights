import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from astro_weights.eval import (
    FloorResult,
    Record,
    compare,
    config_hash,
    floor_from_records,
    load_floor,
    prompt_hash,
    read_jsonl,
    save_floor,
    tokens_beyond_floor,
)

HEADER: dict[str, Any] = {
    "model": "m",
    "temperature": 0.0,
    "max_new_tokens": 64,
    "k": 2,
}


def rec(i: int, seed: int, out: str, score: float = 1.0, tokens: int = 10) -> Record:
    return Record(
        f"i{i}", seed, "base", prompt_hash(f"p{i}"), out, score, tokens, False
    )


def runs(n: int = 8, k: int = 2) -> tuple[list[Record], list[Record]]:
    a = [rec(i, s, f"o{i}") for i in range(n) for s in range(k)]
    return a, list(a)


def floor() -> FloorResult:
    a, b = runs()
    return floor_from_records(a, b, HEADER)


def test_deterministic_runs_give_a_zero_floor() -> None:
    f = floor()
    assert f.config_hash == config_hash(HEADER)
    assert (f.n_items, f.k) == (8, 2)
    assert (f.accuracy_delta, f.accuracy_se, f.flip_rate) == (0.0, 0.0, 0.0)
    assert (f.tokens_delta, f.tokens_se) == (0.0, 0.0)
    assert not any(math.isnan(v) for v in (f.accuracy_se, f.tokens_se))
    assert FloorResult.from_json(f.to_json()) == f  # to_json rejects NaN


def test_flip_rate_counts_changed_outputs_per_pair() -> None:
    a, b = runs()
    b[0] = rec(0, 0, "changed")  # text differs, score and tokens do not
    b[5] = rec(2, 1, "x", score=0.0, tokens=14)
    f = floor_from_records(a, b, HEADER)
    assert f.flip_rate == 2 / 16
    assert f.accuracy_delta == pytest.approx(-0.5 / 8)  # item 2: mean 1 -> 0.5
    assert f.tokens_delta == pytest.approx(2 / 8)  # item 2: mean 10 -> 12
    assert f.accuracy_se > 0
    assert f.tokens_se > 0


def test_floor_does_not_depend_on_record_order() -> None:
    a, b = runs()
    b[3] = rec(1, 1, "x")
    assert floor_from_records(a, b[::-1], HEADER).flip_rate == 1 / 16
    assert floor_from_records(a[::-1], b, HEADER).flip_rate == 1 / 16


def test_unpaired_runs_raise() -> None:
    a, b = runs()
    with pytest.raises(ValueError, match="same"):
        floor_from_records(a, b[1:], HEADER)
    with pytest.raises(ValueError, match="no records"):
        floor_from_records([], [], HEADER)
    with pytest.raises(ValueError, match="at least 2"):
        floor_from_records(a[:1], b[:1], HEADER)


def test_roundtrip(tmp_path: Path) -> None:
    f = floor()
    path = save_floor(tmp_path / "new" / "dir", f)
    assert path.name == f"floor-{f.config_hash}.json"
    assert load_floor(tmp_path / "new" / "dir", f.config_hash) == f


def test_other_config_hash_gives_none(tmp_path: Path) -> None:
    f = floor()
    save_floor(tmp_path, f)
    other = config_hash({**HEADER, "max_new_tokens": 128})
    assert other != f.config_hash
    assert load_floor(tmp_path, other) is None
    assert load_floor(tmp_path / "missing", other) is None


def test_provenance_change_keeps_the_hash(tmp_path: Path) -> None:
    f = floor()
    save_floor(tmp_path, f)
    assert load_floor(tmp_path, config_hash({**HEADER, "run_id": "x"})) == f


def test_bad_hash_is_refused(tmp_path: Path) -> None:
    for bad in ("../x", "abc", "A" * 64, ""):
        with pytest.raises(ValueError, match="64 lowercase hex"):
            load_floor(tmp_path, bad)


def test_file_under_wrong_name_raises(tmp_path: Path) -> None:
    f = floor()
    other = config_hash({**HEADER, "max_new_tokens": 128})
    (tmp_path / f"floor-{other}.json").write_text(f.to_json())  # copied to another name
    with pytest.raises(ValueError, match="does not match its name"):
        load_floor(tmp_path, other)


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.pop("flip_rate"),
        lambda d: d.update(extra=1),
        lambda d: d.update(flip_rate=1.5),
        lambda d: d.update(flip_rate=-0.1),
        lambda d: d.update(accuracy_se="0"),
        lambda d: d.update(accuracy_se=float("nan")),
        lambda d: d.update(n_items=True),
        lambda d: d.update(k=0),
        lambda d: d.update(config_hash="zz"),
        lambda d: d.update(created=""),
        lambda d: d.update(library_version=3),
    ],
)
def test_tampered_file_raises(tmp_path: Path, edit: Any) -> None:
    f = floor()
    path = save_floor(tmp_path, f)
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))  # json.dumps writes NaN as a bare token
    with pytest.raises(ValueError):  # noqa: PT011
        load_floor(tmp_path, f.config_hash)


@pytest.mark.parametrize("text", ["{", "[]", "null"])
def test_garbage_file_raises(tmp_path: Path, text: str) -> None:
    f = floor()
    save_floor(tmp_path, f).write_text(text)
    with pytest.raises(ValueError, match="floor JSON"):
        load_floor(tmp_path, f.config_hash)


def test_old_records_and_headers_load(tmp_path: Path) -> None:
    # 0.1/0.2 records have no cluster_id; 0.1/0.2 headers have no k, concurrency or run id.
    lines = [
        json.dumps(
            {
                "item_id": f"i{i}",
                "seed": 0,
                "arm": "base",
                "prompt_hash": prompt_hash(f"p{i}"),
                "output": f"o{i % 3}",
                "score": float(i % 2),
                "n_output_tokens": 5 + i,
                "hit_cap": False,
            }
        )
        for i in range(6)
    ]
    p1, p2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    p1.write_text("\n".join(lines) + "\n")
    p2.write_text("\n".join(lines) + "\n")
    old = {"model": "m", "temperature": 0.0, "max_new_tokens": 64}
    f = floor_from_records(read_jsonl(p1), read_jsonl(p2), old)
    assert f.config_hash == config_hash({**old, "k": 1, "concurrency": 1})
    assert (f.k, f.n_items, f.flip_rate) == (1, 6, 0.0)


# --- tokens_beyond_floor --------------------------------------------------------------


def _tok(diff: float, ci: tuple[float, float]) -> Any:
    a, b = runs()
    return replace(compare(a, b, n_boot=1), tokens_diff=diff, tokens_ci95=ci)


def _floor(delta: float, se: float) -> FloorResult:
    return replace(floor(), tokens_delta=delta, tokens_se=se)


# floor 1.5 + 1.96 * 2 = 5.42: an edit must lower tokens by more than that
@pytest.mark.parametrize(
    ("diff", "ci", "want"),
    [
        (-6.0, (-9.0, -3.0), True),  # lower, beyond the floor, CI below 0
        (-5.0, (-9.0, -1.0), False),  # lower but inside the floor
        (-5.43, (-9.0, -1.0), True),  # just past it
        (-6.0, (-9.0, 0.0), False),  # CI reaches 0
        (-6.0, (-9.0, 2.0), False),  # CI crosses 0
        (6.0, (3.0, 9.0), False),  # more tokens
        (-6.0, (-9.0, math.nan), False),
        (math.nan, (-9.0, -3.0), False),
    ],
)
def test_tokens_beyond_floor(diff: float, ci: tuple[float, float], want: bool) -> None:
    assert tokens_beyond_floor(_tok(diff, ci), _floor(-1.5, 2.0)) is want


def test_tokens_beyond_floor_uses_the_floor_size_not_its_sign() -> None:
    r = _tok(-6.0, (-9.0, -3.0))
    assert tokens_beyond_floor(r, _floor(1.5, 2.0))  # |delta| is what counts
    assert not tokens_beyond_floor(r, _floor(0.0, 3.1))  # 1.96 * 3.1 = 6.076
