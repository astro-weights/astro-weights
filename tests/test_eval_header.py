from pathlib import Path

import pytest

from astro_weights.eval import check_headers, header_path, read_header, write_header


def test_header_path() -> None:
    assert header_path("runs/base.jsonl") == Path("runs/base.header.json")
    assert header_path(Path("a.b.jsonl")) == Path("a.b.header.json")
    assert header_path("noext") == Path("noext.header.json")


def test_roundtrip_sorted_keys(tmp_path: Path) -> None:
    h = {"seed": 0, "chat": True, "model": "M", "sha": None}
    p = tmp_path / "x.header.json"
    write_header(p, h)
    assert read_header(p) == h
    keys = [ln.split('"')[1] for ln in p.read_text().splitlines()[1:-1]]
    assert keys == sorted(h)


def test_read_bad_json(tmp_path: Path) -> None:
    p = tmp_path / "h.json"
    p.write_text("{nope")
    with pytest.raises(ValueError, match="invalid JSON"):
        read_header(p)


def test_read_not_object(tmp_path: Path) -> None:
    p = tmp_path / "h.json"
    p.write_text("[1]")
    with pytest.raises(ValueError, match="expected a JSON object"):
        read_header(p)


def test_check_equal_and_ignored() -> None:
    check_headers({"a": 1, "device": "cpu"}, {"a": 1, "device": "cuda:0"})


def test_check_lists_every_difference() -> None:
    with pytest.raises(ValueError, match="not comparable") as e:
        check_headers(
            {"a": 1, "b": 2, "only_base": 3, "same": 0},
            {"a": 9, "b": 2, "only_edit": 4, "same": 0},
        )
    msg = str(e.value)
    assert "a: base=1 edit=9" in msg
    assert "only_base: base=3 edit='<missing>'" in msg
    assert "only_edit: base='<missing>' edit=4" in msg
    assert "b:" not in msg
    assert "same" not in msg


def test_check_custom_ignore() -> None:
    check_headers({"a": 1, "seed": 0}, {"a": 1, "seed": 1}, ignore=("seed",))
    with pytest.raises(ValueError, match="device"):
        check_headers({"device": "a"}, {"device": "b"}, ignore=())
