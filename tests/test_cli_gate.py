"""`astro-weights gate`: the whole gate from recorded files, on a synthetic fixture."""

import importlib.util
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from astro_weights.cli import main
from astro_weights.eval import Verdict
from astro_weights.eval.gate_config import load_gate_config


def _load() -> ModuleType:
    path = Path(__file__).parent / "data" / "gate" / "make_fixture.py"
    spec = importlib.util.spec_from_file_location("gate_make_fixture", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fixture = _load()
Build = Callable[..., Path]


@pytest.fixture
def build(tmp_path: Path) -> Build:
    def make(**knobs: Any) -> Path:
        return fixture.build(tmp_path / "run", **knobs)  # type: ignore[no-any-return]

    return make


def gate(
    cfg: Path, *flags: str, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str, str]:
    code = main(["gate", str(cfg), *flags])
    out, err = capsys.readouterr()
    return code, out, err


def ledger(cfg: Path) -> bytes:
    return (cfg.parent / "seal" / "seal.jsonl").read_bytes()


def verdict(cfg: Path) -> Verdict:
    return Verdict.from_json((cfg.parent / "verdict.json").read_text())


@pytest.mark.parametrize(
    ("knobs", "code", "outcome"),
    [
        ({}, 0, "confirmed"),
        ({"tokens_cut": False}, 1, "not significant"),
        ({"acc_drop": True}, 1, "regressed"),
        ({"suite_drop": True}, 1, "regressed"),
    ],
)
def test_scored_outcomes(
    build: Build,
    capsys: pytest.CaptureFixture[str],
    knobs: dict[str, Any],
    code: int,
    outcome: str,
) -> None:
    cfg = build(**knobs)
    got, out, _ = gate(cfg, capsys=capsys)
    assert got == code
    assert f"outcome: {outcome}" in out
    v = verdict(cfg)
    assert v.verdict == outcome
    assert (v.candidates_tried, v.confirm_queries, v.infra_errors) == (2, 3, 3)
    assert dict(v.footprint) == {"rank": 1.0}
    assert (v.backend, v.build) == ("hf", "torch-test")
    assert len(ledger(cfg).splitlines()) == 1  # scored once
    for needle in ("MDE (test)", "candidates tried: 2", "confirm queries: 3"):
        assert needle in out
    assert ("keep the original weights" in out) == (code != 0)


@pytest.mark.parametrize(
    ("knobs", "outcome"),
    [({"controls_beat": False}, "not significant"), ({"side_drop": True}, "regressed")],
)
def test_confirm_failure_never_scores_test(
    build: Build,
    capsys: pytest.CaptureFixture[str],
    knobs: dict[str, Any],
    outcome: str,
) -> None:
    cfg = build(**knobs)
    code, out, _ = gate(cfg, capsys=capsys)
    assert code == 1
    assert f"outcome: {outcome}" in out
    assert "MDE (confirm)" in out
    assert "test split" not in out
    assert not (cfg.parent / "seal").exists()
    assert verdict(cfg).verdict == outcome


def test_screened_out_leaves_the_seal_alone(
    build: Build, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = build()
    assert gate(cfg, capsys=capsys)[0] == 0
    before = ledger(cfg)
    salt = (cfg.parent / "seal" / "seal.salt").read_bytes()
    (cfg.parent / "verdict.json").unlink()
    fail = [
        {"name": n, "passed": n != "kl", "value": 3.0, "limit": 1.0}
        for n in ("norm", "perplexity", "kl")
    ]
    (cfg.parent / "screens.json").write_text(json.dumps(fail))
    code, out, _ = gate(cfg, capsys=capsys)
    assert code == 3
    assert "outcome: screened out" in out
    assert "keep the original weights" in out
    assert ledger(cfg) == before
    assert (cfg.parent / "seal" / "seal.salt").read_bytes() == salt
    assert not (cfg.parent / "verdict.json").exists()


def test_screened_out_without_a_sealed_dir(
    build: Build, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = build(screens_fail="perplexity")
    code, out, _ = gate(cfg, "--json", capsys=capsys)
    assert code == 3
    data = json.loads(out)
    assert data["outcome"] == "screened out"
    assert data["verdict_file"] is None
    assert data["arms"] == {"confirm": None, "test": None}
    assert not (cfg.parent / "seal").exists()


def test_second_scoring_is_refused_unless_unsealed(
    build: Build, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = build()
    assert gate(cfg, capsys=capsys)[0] == 0
    before, first = ledger(cfg), (cfg.parent / "verdict.json").read_bytes()
    code, out, err = gate(cfg, capsys=capsys)
    assert code == 2
    assert out == ""
    assert "already scored" in err
    assert ledger(cfg) == before
    code, _, err = gate(cfg, "--unseal", capsys=capsys)
    assert code == 0
    assert "--unseal" in err
    lines = [json.loads(x) for x in ledger(cfg).splitlines()]
    assert [x["unseal"] for x in lines] == [False, True]
    assert (cfg.parent / "verdict.json").read_bytes() == first  # no timestamps


@pytest.mark.parametrize("exported", ["not_best", "missing"])
def test_exported_not_best_is_refused(
    build: Build, capsys: pytest.CaptureFixture[str], exported: str
) -> None:
    cfg = build(exported=exported)
    code, out, err = gate(cfg, capsys=capsys)
    assert code == 2
    assert out == ""
    assert "exported_is_best" in err
    assert not (cfg.parent / "seal").exists()
    assert not (cfg.parent / "verdict.json").exists()


def test_missing_floor_is_refused(
    build: Build, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = build(floor=False)
    code, _, err = gate(cfg, capsys=capsys)
    assert code == 2
    assert "no floor for config_hash" in err
    assert not (cfg.parent / "seal").exists()


def test_floor_under_another_config_is_refused(
    build: Build, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = build()
    for header in cfg.parent.rglob("*.header.json"):  # every arm: a new config_hash
        h = json.loads(header.read_text())
        header.write_text(json.dumps({**h, "max_new_tokens": 512}))
    code, _, err = gate(cfg, capsys=capsys)
    assert code == 2
    assert "no floor" in err
    assert not (cfg.parent / "seal").exists()


def test_manifest_k_must_be_four(
    build: Build, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = build(manifest_k=3)
    code, _, err = gate(cfg, capsys=capsys)
    assert code == 2
    assert "K=4" in err
    assert not (cfg.parent / "seal").exists()


def test_same_files_same_verdict_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = fixture.build(tmp_path / "a")
    b = fixture.build(tmp_path / "b")
    assert gate(a, capsys=capsys)[0] == gate(b, capsys=capsys)[0] == 0
    assert (a.parent / "verdict.json").read_bytes() == (
        b.parent / "verdict.json"
    ).read_bytes()


def test_every_arm_is_reported(
    build: Build, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = build(arms=True)
    code, out, _ = gate(cfg, capsys=capsys)
    assert code == 0
    for name in ("edit", "scripted", "prompt", "edit+prompt", "controls: n=20"):
        assert name in out
    assert "tokens diff" in out
    assert "MDE" in out


def test_json_form(
    build: Build, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    cfg = build(arms=True)
    out_file = tmp_path / "elsewhere" / "v.json"
    out_file.parent.mkdir()
    code, out, _ = gate(cfg, "--json", "--out", str(out_file), capsys=capsys)
    assert code == 0
    data = json.loads(out)
    assert data["outcome"] == "confirmed"
    assert (data["candidates_tried"], data["confirm_queries"]) == (2, 3)
    assert data["verdict_file"] == str(out_file)
    assert data["mde_split"] == "test"
    names = [r["arm"] for r in data["arms"]["test"]["arms"]]
    assert names == ["edit", "prompt", "scripted", "edit+prompt"]
    assert data["arms"]["confirm"]["controls"]["n"] == 20
    assert Verdict.from_json(out_file.read_text()).verdict == "confirmed"
    assert not (cfg.parent / "verdict.json").exists()


@pytest.mark.parametrize(
    "path",
    ["confirm/controls/control-03.header.json", "confirm/edit.header.json",
     "test/edit.header.json", "gsm8k-edit.header.json"],
)  # fmt: skip
def test_header_disagreement_is_refused(
    build: Build, capsys: pytest.CaptureFixture[str], path: str
) -> None:
    cfg = build()
    header = cfg.parent / path
    h = json.loads(header.read_text())
    header.write_text(json.dumps({**h, "dtype": "bfloat16"}))
    code, _, err = gate(cfg, capsys=capsys)
    assert code == 2
    assert "different settings" in err
    assert not (cfg.parent / "seal").exists()


def test_missing_header_is_refused(
    build: Build, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = build()
    (cfg.parent / "test" / "edit.header.json").unlink()
    code, _, err = gate(cfg, capsys=capsys)
    assert code == 2
    assert "edit.header.json" in err
    assert not (cfg.parent / "seal").exists()


def test_confirm_and_test_may_differ_in_k_only(
    build: Build, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = build()
    header = cfg.parent / "confirm" / "base.header.json"
    assert json.loads(header.read_text())["k"] == 1
    header = cfg.parent / "test" / "base.header.json"
    assert json.loads(header.read_text())["k"] == 4
    assert gate(cfg, capsys=capsys)[0] == 0


def test_incomplete_test_records_are_refused(
    build: Build, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = build()
    edit = cfg.parent / "test" / "edit.jsonl"
    edit.write_text("".join(edit.read_text().splitlines(keepends=True)[:-4]))
    code, _, err = gate(cfg, capsys=capsys)
    assert code == 2
    assert "item set differs" in err
    assert not (cfg.parent / "seal").exists()


def test_incomplete_test_split_is_refused(
    build: Build, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = build()
    for name in ("base", "edit"):  # both arms short: the split itself is incomplete
        path = cfg.parent / "test" / f"{name}.jsonl"
        path.write_text("".join(path.read_text().splitlines(keepends=True)[:-4]))
    code, _, err = gate(cfg, capsys=capsys)
    assert code == 2
    assert "have no record" in err
    assert not (cfg.parent / "seal").exists()
    assert not (cfg.parent / "seal").exists()


def test_records_outside_the_confirm_split_are_refused(
    build: Build, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = build()
    shutil.copy(
        cfg.parent / "test" / "edit.jsonl", cfg.parent / "confirm" / "edit.jsonl"
    )
    code, _, err = gate(cfg, capsys=capsys)
    assert code == 2
    assert "not in the confirm split" in err


def test_too_few_controls_are_refused(
    build: Build, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = build()
    (cfg.parent / "confirm" / "controls" / "control-00.jsonl").unlink()
    code, _, err = gate(cfg, capsys=capsys)
    assert code == 2
    assert "19 controls" in err


def test_controls_must_be_a_directory(
    build: Build, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = build()
    shutil.rmtree(cfg.parent / "confirm" / "controls")
    code, _, err = gate(cfg, capsys=capsys)
    assert code == 2
    assert "not a directory" in err


def test_a_control_cannot_be_named_edit(
    build: Build, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = build()
    c = cfg.parent / "confirm" / "controls"
    shutil.copy(c / "control-00.jsonl", c / "edit.jsonl")
    code, _, err = gate(cfg, capsys=capsys)
    assert code == 2
    assert "edit.jsonl" in err


def test_a_base_arm_labelled_edit_is_refused(
    build: Build, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = build()
    base = cfg.parent / "test" / "base.jsonl"
    base.write_text(base.read_text().replace('"arm": "base"', '"arm": "edit"'))
    code, _, err = gate(cfg, capsys=capsys)
    assert code == 2
    assert "arm 'edit'" in err
    assert not (cfg.parent / "seal").exists()


def test_extra_side_effect_table(
    build: Build, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = build()
    cfg.write_text(
        cfg.read_text()
        .replace('side_effects = ["accuracy"]', 'side_effects = ["accuracy", "style"]')
        .replace(
            "[test]",
            '[confirm.side.style]\nbase = "confirm/base.jsonl"\n'
            'edit = "confirm/edit.jsonl"\ncontrols = "confirm/controls"\n\n[test]',
        )
    )
    code, out, _ = gate(cfg, capsys=capsys)
    assert code == 0
    assert "side effect style inside control range" in out


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda t: t.replace("margin = 0.02\n", "", 1), "missing keys"),
        (lambda t: "bogus = 1\n" + t, "unknown keys"),
        (lambda t: "this is = = toml", "GATE.toml"),
        (lambda t: t.replace('target = "tokens_diff"', 'target = "score"'), "target"),
        (lambda t: t.replace('["accuracy"]', '["style"]'), "confirm.side.NAME"),
        (lambda t: t.replace('["accuracy"]', "[]"), "side_effects"),
        (lambda t: t.replace('["accuracy"]', '["accuracy", "accuracy"]'), "distinct"),
        (lambda t: t.replace("boot = 300", "boot = 0"), "boot"),
        (lambda t: t.replace("boot = 300", 'boot = "x"'), "boot"),
        (lambda t: t.replace('backend = "hf"', 'backend = ""'), "backend"),
        (lambda t: t.replace("margin = 0.02\n", "margin = true\n", 1), "finite number"),
        (lambda t: t.replace("rank = 1", 'rank = "one"'), "footprint.rank"),
        (lambda t: t + "\n[arms]\nother = 'x'\n", "arms"),
        (lambda t: "arm_keys = 3\n" + t, "arm_keys"),
        (lambda t: "footprint = 3\n" + t.replace("[footprint]\nrank = 1\n", ""), "must be a table"),
        (lambda t: t.replace('target = "tokens_diff"', 'target = "tokens_diff"\nside = {a = 1}'), "confirm.side"),
        (lambda t: t.replace('target = "tokens_diff"', 'target = "tokens_diff"\nside = {style = {base = "x"}}'), "not in confirm.side_effects"),
    ],
)  # fmt: skip
def test_bad_config_is_refused(
    build: Build,
    capsys: pytest.CaptureFixture[str],
    edit: Callable[[str], str],
    message: str,
) -> None:
    cfg = build()
    cfg.write_text(edit(cfg.read_text()))
    code, out, err = gate(cfg, capsys=capsys)
    assert code == 2
    assert out == ""
    assert message in err
    assert not (cfg.parent / "seal").exists()


def test_side_table_keys_are_checked(build: Build) -> None:
    cfg = build()
    text = cfg.read_text().replace(
        'side_effects = ["accuracy"]',
        'side_effects = ["accuracy", "style"]\nside = {style = {base = "x"}}',
    )
    cfg.write_text(text)
    with pytest.raises(ValueError, match=r"confirm\.side\.style: missing keys"):
        load_gate_config(cfg)


def test_config_paths_are_relative_to_the_file(build: Build) -> None:
    cfg = build()
    c = load_gate_config(cfg)
    assert c.test.seal_dir == cfg.parent / "seal"
    assert c.confirm.direction == "lower"
    assert set(c.suites) == {"gsm8k", "ifeval"}
    assert c.arms == {}


@pytest.mark.parametrize(
    "bad",
    ["not json", "{}", '[{"name": "kl"}]', '[{"name": "kl", "passed": true, "value": 1, "limit": 1, "x": 1}]'],
)  # fmt: skip
def test_bad_screens_file(
    build: Build, capsys: pytest.CaptureFixture[str], bad: str
) -> None:
    cfg = build()
    (cfg.parent / "screens.json").write_text(bad)
    code, _, err = gate(cfg, capsys=capsys)
    assert code == 2
    assert "screens.json" in err


def test_missing_screen_is_refused(
    build: Build, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = build()
    (cfg.parent / "screens.json").write_text(
        json.dumps([{"name": "norm", "passed": True, "value": 1, "limit": 2}])
    )
    code, _, err = gate(cfg, capsys=capsys)
    assert code == 2
    assert "missing screens" in err
    assert not (cfg.parent / "seal").exists()


def test_bad_run_record_is_refused(
    build: Build, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = build()
    (cfg.parent / "record.jsonl").write_text("")
    code, _, err = gate(cfg, capsys=capsys)
    assert code == 2
    assert "record.jsonl" in err


def test_gate_is_in_the_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    out = capsys.readouterr().out
    assert "gate" in out
    assert "3 screened out" in out


def test_a_seal_refusal_at_write_time_shows_nothing(
    build: Build, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from astro_weights import cli
    from astro_weights.eval.seal import SealedError

    def refuse(*_: object, **__: object) -> None:
        raise SealedError("raced by another run")

    monkeypatch.setattr(cli, "record_scoring", refuse)
    cfg = build()
    code, out, err = gate(cfg, capsys=capsys)
    assert code == 2
    assert out == ""
    assert "raced" in err
    assert not (cfg.parent / "verdict.json").exists()


def test_a_bad_out_path_never_spends_the_seal(
    build: Build, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    cfg = build()
    code, out, _ = gate(
        cfg, "--out", str(tmp_path / "missing" / "v.json"), capsys=capsys
    )
    assert code == 2
    assert out == ""
    seal = cfg.parent / "seal" / "seal.jsonl"
    assert not seal.exists() or seal.read_bytes() == b""
    assert gate(cfg, capsys=capsys)[0] == 0  # the test split is still unspent


def test_a_report_arm_cannot_reshow_a_sealed_edit(
    build: Build, capsys: pytest.CaptureFixture[str]
) -> None:
    from dataclasses import replace

    from astro_weights.edits import Edit
    from astro_weights.eval import read_jsonl
    from astro_weights.eval.seal import record_scoring, scoring_of, seal_salt
    from astro_weights.eval.splits import SplitManifest
    from astro_weights.export import load_peft

    cfg = build(arms=True)
    edit = load_peft(cfg.parent / "adapter")
    other = Edit.from_deltas(
        [replace(d, B=d.B * 3) for d in edit.deltas], edit.base_model
    )
    conf = load_gate_config(cfg)
    manifest = SplitManifest.load(conf.manifest)
    scripted = read_jsonl(conf.arms["scripted"])
    seal_dir = cfg.parent / "seal"
    record_scoring(seal_dir, scoring_of(other, manifest, scripted, seal_salt(seal_dir)))
    before = ledger(cfg)
    code, out, err = gate(cfg, capsys=capsys)
    assert code == 2
    assert out == ""
    assert err
    assert ledger(cfg) == before
