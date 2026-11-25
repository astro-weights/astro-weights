"""scripts/results_entry.py: gate run to results-page entry."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from astro_weights.eval.verdict import Verdict


def _script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "results_entry.py"
    spec = importlib.util.spec_from_file_location("results_entry", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


re = _script()


def _files(tmp_path: Path, outcome: str) -> tuple[Path, Path]:
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"outcome": outcome, "checks": [], "mde": 0.01}))
    verdict = tmp_path / "verdict.json"
    v = Verdict(
        "confirmed", "vllm", "0.1", "abc", "d" * 64, 3, 2, 0, {"rel_norm": 0.01}
    )
    verdict.write_text(v.to_json())
    return verdict, gate


def test_append_replaces_placeholder_and_same_name(tmp_path: Path) -> None:
    verdict, gate = _files(tmp_path, "confirmed")
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"placeholder": True, "runs": [{"name": "example"}]}))
    argv = [
        "--name",
        "a",
        "--base",
        "b",
        "--verdict",
        str(verdict),
        "--gate",
        str(gate),
    ]
    assert re.main([*argv, "--append", str(results)]) == 0
    assert re.main([*argv, "--append", str(results)]) == 0
    data = json.loads(results.read_text())
    assert "placeholder" not in data
    assert [r["name"] for r in data["runs"]] == ["a"]
    assert data["runs"][0]["verdict"]["verdict"] == "confirmed"
    assert data["runs"][0]["gate"]["mde"] == 0.01


def test_screened_out_has_gate_only(tmp_path: Path) -> None:
    _, gate = _files(tmp_path, "screened out")
    e = re.entry("a", "b", None, gate)
    assert "verdict" not in e
    assert e["gate"]["outcome"] == "screened out"


@pytest.mark.parametrize(
    ("outcome", "with_verdict", "match"),
    [
        ("not significant", True, "verdict.json says"),
        ("confirmed", False, "needs --verdict"),
        ("screened out", True, "drop --verdict"),
        ("maybe", True, "not 'astro-weights gate --json'"),
    ],
)
def test_bad_pairs_are_refused(
    tmp_path: Path, outcome: str, with_verdict: bool, match: str
) -> None:
    verdict, gate = _files(tmp_path, outcome)
    with pytest.raises(ValueError, match=match):
        re.entry("a", "b", verdict if with_verdict else None, gate)
