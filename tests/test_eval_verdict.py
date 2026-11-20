import json
from typing import Any

import pytest

from astro_weights.eval import Verdict
from astro_weights.eval.verdict import VERDICTS


def _v(**kw: Any) -> Verdict:
    base: dict[str, Any] = {
        "verdict": "confirmed",
        "backend": "llama.cpp",
        "build": "b4567-abc",
        "config_hash": "c" * 64,
        "split_manifest_sha256": "s" * 64,
        "candidates_tried": 12,
        "confirm_queries": 3,
        "infra_errors": 0,
        "footprint": {"rel_norm": 0.01},
    }
    return Verdict(**{**base, **kw})


def test_roundtrip() -> None:
    v = _v()
    text = v.to_json()
    assert Verdict.from_json(text) == v
    data = json.loads(text)
    assert list(data) == sorted(data)
    assert data["schema_version"] == 1
    assert _v(build=None).to_json() == _v(build=None).to_json()
    assert Verdict.from_json(_v(build=None).to_json()).build is None


@pytest.mark.parametrize("name", VERDICTS)
def test_every_verdict_value(name: str) -> None:
    assert _v(verdict=name).verdict == name


@pytest.mark.parametrize(
    ("kw", "msg"),
    [
        ({"verdict": "keep original"}, "verdict must be"),
        ({"backend": ""}, "backend"),
        ({"config_hash": 5}, "config_hash"),
        ({"split_manifest_sha256": ""}, "split_manifest_sha256"),
        ({"build": 3}, "build"),
        ({"candidates_tried": -1}, "candidates_tried"),
        ({"confirm_queries": True}, "confirm_queries"),
        ({"infra_errors": 1.5}, "infra_errors"),
        ({"schema_version": 2}, "schema_version"),
        ({"footprint": {"a": float("nan")}}, "footprint"),
        ({"footprint": {"a": "x"}}, "footprint"),
        ({"footprint": {"a": True}}, "footprint"),
        ({"footprint": {1: 1.0}}, "footprint"),
        ({"footprint": [1]}, "footprint"),
    ],
)
def test_rejects(kw: dict[str, Any], msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        _v(**kw)


def test_from_json_strict() -> None:
    data = json.loads(_v().to_json())
    with pytest.raises(ValueError, match="invalid verdict JSON"):
        Verdict.from_json("{nope")
    with pytest.raises(ValueError, match="must be an object"):
        Verdict.from_json("[]")
    with pytest.raises(ValueError, match="unknown keys: extra"):
        Verdict.from_json(json.dumps({**data, "extra": 1}))
    del data["backend"]
    with pytest.raises(ValueError, match="missing keys: backend"):
        Verdict.from_json(json.dumps(data))
    data["backend"] = "vllm"
    data["verdict"] = "nope"
    with pytest.raises(ValueError, match="verdict must be"):
        Verdict.from_json(json.dumps(data))
    del data["schema_version"]  # optional on input
    data["verdict"] = "regressed"
    assert Verdict.from_json(json.dumps(data)).verdict == "regressed"
