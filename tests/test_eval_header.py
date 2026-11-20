from pathlib import Path

import pytest

from astro_weights.eval import (
    CONFIG_KEYS,
    PROVENANCE_KEYS,
    check_headers,
    config_hash,
    header_path,
    read_header,
    write_header,
)
from astro_weights.eval.header import run_identity


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
    check_headers({"a": 1, "device": "a"}, {"a": 1, "device": "b"}, ignore=())
    check_headers({"device": "a"}, {"device": "b"}, ignore=("device",))  # old default


def test_key_sets_disjoint() -> None:
    assert not CONFIG_KEYS & PROVENANCE_KEYS


def test_check_ignores_provenance_by_default() -> None:
    a = {"seed": 0, "run_id": "1", "gpu_model": "H200", "torch": "2.1"}
    b = {"seed": 0, "run_id": "2", "base_url": "http://x", "torch": "2.1"}
    check_headers(a, b)  # provenance missing on one side is not a difference
    with pytest.raises(ValueError, match="seed: base=0 edit=1"):
        check_headers(a, {**b, "seed": 1})


def test_check_unknown_key_is_config() -> None:
    with pytest.raises(ValueError, match="novel: base=1 edit=2"):
        check_headers({"novel": 1}, {"novel": 2})


def test_check_explicit_ignore_is_union_with_provenance() -> None:
    a, b = {"seed": 0, "run_id": "1"}, {"seed": 1, "run_id": "2"}
    with pytest.raises(ValueError, match="seed"):
        check_headers(a, b, ignore=("run_id",))
    check_headers(a, b, ignore=("seed",))  # run_id is provenance, still skipped


def test_backend_and_build_are_config() -> None:
    for key in ("backend", "llama_cpp_build", "vllm_version", "torch", "transformers"):
        assert key in CONFIG_KEYS
        with pytest.raises(ValueError, match=key):
            check_headers({key: "1"}, {key: "2"})
        assert config_hash({key: "1"}) != config_hash({key: "2"})
    for key in ("model_id", "model_name_or_path"):
        with pytest.raises(ValueError, match=key):
            check_headers({key: "a"}, {key: "b"})


def test_check_arm_keys() -> None:
    a = {"extra_body": {"top_k": 1, "lora": [{"id": 0, "scale": 0.0}]}}
    b = {"extra_body": {"top_k": 1, "lora": [{"id": 0, "scale": 1.0}]}}
    with pytest.raises(ValueError, match="extra_body"):
        check_headers(a, b)
    check_headers(a, b, arm_keys=("lora",))
    c = {"extra_body": {"top_k": 5, "lora": [{"id": 0, "scale": 1.0}]}}
    with pytest.raises(ValueError, match="extra_body"):
        check_headers(a, c, arm_keys=("lora",))  # only the declared key may differ
    check_headers({"extra_body": 1}, {"extra_body": 1}, arm_keys=("lora",))


def test_http_arms_on_two_names_compare() -> None:
    base = {
        "base_url": "http://a",
        "served_model": "base",
        "server_model": {"id": "base"},
        "seed": 0,
        "extra_body": {},
        "k": 1,
        "concurrency": 1,
    }
    edit = {**base, "base_url": "http://b", "served_model": "edit"}
    edit["server_model"] = {"id": "edit"}
    check_headers(base, edit)


def test_old_headers_still_compare() -> None:
    old = {"model": "M", "dtype": "float32", "device": "cpu", "seed": 0, "torch": "2"}
    new = {**old, "device": "cuda:0", "run_id": "r", "k": 1, "concurrency": 1}
    check_headers(old, old)
    check_headers(old, new)  # new CONFIG keys at their defaults are no difference
    # a new CONFIG key the old header lacks is a difference
    with pytest.raises(ValueError, match="backend: base='<missing>' edit='cpu'"):
        check_headers(old, {**new, "backend": "cpu"})
    # an old HTTP header has `model`, a new one does not
    with pytest.raises(ValueError, match="model: base='m' edit='<missing>'"):
        check_headers({"model": "m"}, {"served_model": "m"})
    with pytest.raises(ValueError, match="k: base=1 edit=4"):
        check_headers(old, {**new, "k": 4})
    assert config_hash(old) == config_hash(new)


def test_config_hash_properties() -> None:
    h = {"seed": 0, "dtype": "bf16", "sampling": {"top_k": 5, "top_p": 0.9}}
    reordered = {"sampling": {"top_p": 0.9, "top_k": 5}, "dtype": "bf16", "seed": 0}
    assert config_hash(h) == config_hash(reordered)
    assert len(config_hash(h)) == 64
    for key in PROVENANCE_KEYS:
        assert config_hash({**h, key: "x"}) == config_hash(h)
    assert config_hash({**h, "seed": 1}) != config_hash(h)
    assert config_hash({**h, "sampling": {"top_k": 6, "top_p": 0.9}}) != config_hash(h)
    assert config_hash({**h, "novel": 1}) != config_hash(h)  # unknown key is CONFIG
    assert config_hash({**h, "k": 2}) != config_hash(h)
    assert config_hash({**h, "k": 1}) == config_hash(h)  # default


def test_config_hash_whole_floats() -> None:
    a = {"sampling": {"temperature": 1, "top_p": 0.9}, "x": [1, {"y": 2}]}
    b = {"sampling": {"temperature": 1.0, "top_p": 0.9}, "x": [1.0, {"y": 2.0}]}
    assert config_hash(a) == config_hash(b)
    assert config_hash({"t": 1.5}) != config_hash({"t": 1})
    assert config_hash({"t": True}) != config_hash({"t": 1})


def test_config_hash_arm_keys() -> None:
    a = {"extra_body": {"lora": 0}}
    b = {"extra_body": {"lora": 1}}
    assert config_hash(a) != config_hash(b)
    assert config_hash(a, arm_keys=("lora",)) == config_hash(b, arm_keys=("lora",))


def test_run_identity() -> None:
    r = run_identity(None, 1, 1)
    assert len(str(r["run_id"])) == 32
    assert r["run_id"] != run_identity(None, 1, 1)["run_id"]
    assert run_identity("x", 3, 4) | {"astro_weights": 0, "library_commit": 0} == {
        **r,
        "run_id": "x",
        "k": 3,
        "concurrency": 4,
        "astro_weights": 0,
        "library_commit": 0,
    }
    assert set(r) <= (CONFIG_KEYS | PROVENANCE_KEYS)
    with pytest.raises(ValueError, match="at least 1"):
        run_identity(None, 1, 0)
