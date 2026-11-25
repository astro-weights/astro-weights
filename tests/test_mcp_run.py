"""``astro-weights run CONFIG``: TOML config, start guard, resume, status, editors."""

import contextlib
import json
import math
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import pytest

from astro_weights.cli import main as cli
from astro_weights.mcp import forward, run
from astro_weights.mcp.budget import Ledger, Limits
from astro_weights.mcp.jobs import load_run_config
from astro_weights.mcp.record import RunRecord
from astro_weights.mcp.run import RunError, RunSpec, load_config

# the MCP stdio client binds sys.stderr as a default when first imported: import it
# now, not inside a test whose capsys has replaced sys.stderr (no fileno to pass on)
with contextlib.suppress(ModuleNotFoundError):
    import mcp.client.stdio  # noqa: F401

DEL = object()  # marks a key or table to drop from the default config
LIMITS = {"forward_passes": 3, "code_runs": 3, "tokens": 100}
HOSTED_MODEL = "claude-opus-5-5"
FAKE_KEY = "sk-ant-test-key-0123456789"


def _value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return "nan" if math.isnan(v) else ("inf" if math.isinf(v) else repr(v))
    if isinstance(v, int):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    return json.dumps(v)


def toml(cfg: dict[str, Any]) -> str:
    """A small TOML writer: top-level scalars first, then one table per dict."""
    lines = [f"{k} = {_value(v)}" for k, v in cfg.items() if not isinstance(v, dict)]
    for name, table in cfg.items():
        if isinstance(table, dict):
            lines.append(f"[{name}]")
            lines += [f"{k} = {_value(v)}" for k, v in table.items()]
    return "\n".join(lines) + "\n"


def write_config(where: Path, **tables: Any) -> Path:
    """Write a scripted-editor config to ``where/cfg/run.toml``; tables merge in."""
    cfg: dict[str, Any] = {
        "run": {"dir": "run", "seed": 7},
        "limits": dict(LIMITS),
        "editor": {"kind": "scripted"},
    }
    for name, table in tables.items():
        if table is DEL:
            cfg.pop(name, None)
        elif isinstance(table, dict):
            merged = {**cfg.get(name, {}), **table}
            cfg[name] = {k: v for k, v in merged.items() if v is not DEL}
        else:
            cfg[name] = table
    path = where / "cfg" / "run.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(toml(cfg))
    return path


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


class FakeEditor:
    """Stands in for an editor's main: records its argv and returns ``code``."""

    def __init__(self, code: int = 0) -> None:
        self.code = code
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> int:
        self.calls.append(list(argv))
        return self.code


@pytest.fixture
def editor(monkeypatch: pytest.MonkeyPatch) -> FakeEditor:
    fake = FakeEditor()
    monkeypatch.setattr(run, "_editor_main", lambda kind: fake)
    return fake


def started_run(spec: RunSpec, *, finish: bool = False) -> None:
    """A run directory as a server leaves it: run.json, record, ledger with spend."""
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    (spec.run_dir / "run.json").write_text(json.dumps(spec.run_json()))
    (spec.run_dir / "config.toml").write_text(spec.text)
    record = RunRecord.start(
        spec.run_dir / "record.jsonl",
        seed=spec.seed,
        config={"limits": spec.run_json()["limits"]},
        editor=spec.editor,
        model=spec.model,
    )
    ledger = Ledger(spec.run_dir / "ledger.jsonl", spec.limits)
    ledger.charge("code_runs", ref="j1")
    ledger.charge("tokens", 42, ref="turn 1")
    record.event("usage", turn=1, model="m", tokens=42, dollars=0.0)
    if finish:
        record.finish(budget=ledger.to_json(), outcome="finished")


# --- config ----------------------------------------------------------------------


def test_minimal_config(tmp_path: Path) -> None:
    spec = load_config(write_config(tmp_path))
    assert spec.run_dir == (tmp_path / "cfg" / "run").resolve()
    assert spec.limits == Limits(3, 3, 100, 0.0)
    assert spec.model_dir is None
    assert spec.model is None
    assert not spec.unsafe_kernel
    assert spec.run_json() == {
        "seed": 7,
        "editor": "scripted",
        "model": None,
        "limits": {"forward_passes": 3, "code_runs": 3, "tokens": 100, "dollars": 0.0},
        "kernel": {"timeout_s": 600.0, "mem_gib": 16},
        "forward": {"device": "auto", "dtype": "auto", "max_new_tokens": 4096},
        "export": {"rank": 8, "max_rel_residual": 0.05, "gguf_out_type": "q8_0"},
    }
    assert spec.editor_argv() == [str(spec.run_dir)]
    assert "text" not in repr(spec)


def test_relative_paths_resolve_against_the_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_config(tmp_path, run={"dir": "../runs/a", "model_dir": "m"})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    spec = load_config(Path("..") / "cfg" / "run.toml")
    assert spec.run_dir == (tmp_path / "runs" / "a").resolve()
    assert spec.model_dir == (tmp_path / "cfg" / "m").resolve()
    assert spec.source == path.resolve()
    absolute = write_config(tmp_path, run={"dir": str(tmp_path / "abs")})
    assert load_config(absolute).run_dir == (tmp_path / "abs").resolve()


def test_kernel_and_forward_options(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        kernel={"timeout_s": 30, "mem_gib": 2, "unsafe": True},
        forward={"device": "cpu", "dtype": "float32", "max_new_tokens": 8},
        limits={"dollars": 1},
    )
    spec = load_config(path)
    assert spec.unsafe_kernel
    assert spec.limits.dollars == 1.0
    assert spec.run_json()["kernel"] == {"timeout_s": 30.0, "mem_gib": 2}
    nested = load_config(write_config(tmp_path, kernel={"allow_nested_userns": True}))
    assert nested.run_json()["kernel"]["allow_nested_userns"] is True
    assert spec.run_json()["forward"] == {
        "device": "cpu",
        "dtype": "float32",
        "max_new_tokens": 8,
    }
    assert spec.editor_argv() == [str(spec.run_dir), "--unsafe-kernel"]


def test_export_options(tmp_path: Path) -> None:
    export = {
        "rank": 4,
        "max_rel_residual": 1,
        "base_gguf": "b.gguf",
        "gguf_out_type": "f16",
    }
    spec = load_config(write_config(tmp_path, export=export))
    assert spec.run_json()["export"] == {
        "rank": 4,
        "max_rel_residual": 1.0,
        "base_gguf": str((tmp_path / "cfg" / "b.gguf").resolve()),
        "gguf_out_type": "f16",
    }
    spec.run_dir.mkdir(parents=True)
    (spec.run_dir / "run.json").write_text(json.dumps(spec.run_json()))
    cfg = load_run_config(spec.run_dir)
    assert cfg.export_base_gguf == (tmp_path / "cfg" / "b.gguf").resolve()
    assert (cfg.export_rank, cfg.export_gguf_out_type) == (4, "f16")


def test_confirm_table_reaches_run_json(tmp_path: Path) -> None:
    confirm = {"items": "items.jsonl", "grader": "exact", "margin": 0, "max_queries": 3}
    spec = load_config(write_config(tmp_path, run={"model_dir": "m"}, confirm=confirm))
    items = str((tmp_path / "cfg" / "items.jsonl").resolve())
    assert spec.run_json()["confirm"] == {**confirm, "items": items}
    spec.run_dir.mkdir(parents=True)
    (spec.run_dir / "run.json").write_text(json.dumps(spec.run_json()))
    got = load_run_config(spec.run_dir).confirm
    assert got is not None
    assert (str(got.items), got.grader, got.margin, got.max_queries) == (
        items,
        "exact",
        0.0,
        3,
    )
    assert "confirm" not in load_config(write_config(tmp_path)).run_json()


def test_confirm_target_and_batch_size_reach_run_json(tmp_path: Path) -> None:
    confirm = {
        "items": "items.jsonl",
        "grader": "exact",
        "margin": 0.02,
        "max_queries": 5,
        "target": "tokens",
        "batch_size": 16,
    }
    spec = load_config(write_config(tmp_path, run={"model_dir": "m"}, confirm=confirm))
    assert spec.run_json()["confirm"]["target"] == "tokens"
    spec.run_dir.mkdir(parents=True)
    (spec.run_dir / "run.json").write_text(json.dumps(spec.run_json()))
    got = load_run_config(spec.run_dir).confirm
    assert got is not None
    assert (got.target, got.batch_size) == ("tokens", 16)


@pytest.mark.parametrize(
    ("tables", "message"),
    [
        ({"confirm": {"items": DEL}}, r"\[confirm\] items is missing"),
        ({"confirm": {"x": 1}}, r"unknown key \[confirm\] x"),
        ({"confirm": {"grader": "bleu"}}, r"confirm.grader must be one of"),
        ({"confirm": {"margin": 1}}, r"confirm.margin must be a number"),
        ({"confirm": {"max_queries": 0}}, r"\[confirm\] max_queries must be an int"),
        ({"run": {"model_dir": DEL}}, r"\[confirm\] needs \[run\] model_dir"),
        ({"confirm": {"items": "run/items.jsonl"}}, r"outside the run directory"),
        ({"confirm": {"target": "speed"}}, r"confirm.target must be one of"),
        ({"confirm": {"batch_size": 0}}, r"\[confirm\] batch_size must be an int"),
        ({"confirm": {"batch_size": -1}}, r"\[confirm\] batch_size must be an int"),
        ({"confirm": {"batch_size": True}}, r"\[confirm\] batch_size must be an int"),
    ],
)
def test_confirm_table_errors(
    tmp_path: Path, tables: dict[str, Any], message: str
) -> None:
    base = {"items": "items.jsonl", "grader": "exact", "margin": 0, "max_queries": 3}
    given: dict[str, dict[str, Any]] = {"run": {"model_dir": "m"}, "confirm": base}
    for name, table in tables.items():
        given[name] = {**given.get(name, {}), **table}
    with pytest.raises(RunError, match=message):
        load_config(write_config(tmp_path, **given))


def test_local_options_become_flags(tmp_path: Path) -> None:
    editor = {
        "kind": "local",
        "base_url": "http://127.0.0.1:9",
        "model": "m",
        "api_key_env": "MY_KEY",
        "max_turns": 3,
        "temperature": 0.5,
    }
    spec = load_config(write_config(tmp_path, editor=editor))
    assert spec.model == "m"
    assert spec.limits.dollars == 0.0
    assert spec.editor_argv()[1:] == [
        "--base-url",
        "http://127.0.0.1:9",
        "--model",
        "m",
        "--api-key-env",
        "MY_KEY",
        "--max-turns",
        "3",
        "--temperature",
        "0.5",
    ]
    bare = {"kind": "local", "base_url": "http://x", "model": "m"}
    spec = load_config(write_config(tmp_path, editor=bare))
    assert spec.editor_argv()[1:] == ["--base-url", "http://x", "--model", "m"]


def test_hosted_price_for_an_unknown_model_becomes_a_flag(tmp_path: Path) -> None:
    editor = {**HOSTED, "model": "claude-new", "price": "1,2,3,0.1,5"}
    spec = load_config(write_config(tmp_path, editor=editor))
    assert spec.editor_argv()[1:] == [
        "--model",
        "claude-new",
        "--price",
        "1,2,3,0.1,5",
        "--no-fallback",
        "--max-dollars",
        "20.0",
    ]


def test_hosted_dollars_default_to_20_and_cap_the_editor(tmp_path: Path) -> None:
    editor = {
        "kind": "hosted",
        "model": HOSTED_MODEL,
        "fallback_base_url": "http://127.0.0.1:9",
        "fallback_model": "local-m",
        "max_output_tokens": 1000,
    }
    spec = load_config(write_config(tmp_path, editor=editor))
    assert spec.limits.dollars == 20.0
    assert spec.editor_argv()[1:] == [
        "--model",
        HOSTED_MODEL,
        "--max-output-tokens",
        "1000",
        "--fallback-base-url",
        "http://127.0.0.1:9",
        "--fallback-model",
        "local-m",
        "--max-dollars",
        "20.0",
    ]
    alone = {"kind": "hosted", "model": HOSTED_MODEL, "no_fallback": True}
    spec = load_config(write_config(tmp_path, editor=alone, limits={"dollars": 2.5}))
    assert spec.editor_argv()[1:] == [
        "--model",
        HOSTED_MODEL,
        "--no-fallback",
        "--max-dollars",
        "2.5",
    ]


LOCAL = {"kind": "local", "base_url": "http://x", "model": "m"}
HOSTED = {"kind": "hosted", "model": HOSTED_MODEL, "no_fallback": True}

BAD: list[tuple[dict[str, Any], str]] = [
    ({"extra": {"a": 1}}, r"unknown table \[extra\]"),
    ({"run": DEL}, r"\[run\] table is missing"),
    ({"limits": DEL}, r"\[limits\] table is missing"),
    ({"editor": DEL}, r"\[editor\] table is missing"),
    ({"run": 3}, r"\[run\] must be a table"),
    ({"editor": 3}, r"\[editor\] must be a table"),
    ({"forward": 3}, r"\[forward\] must be a table"),
    ({"run": {"other": 1}}, r"unknown key \[run\] other"),
    ({"run": {"dir": DEL}}, r"\[run\] dir is missing"),
    ({"run": {"dir": 5}}, r"\[run\] dir must be a non-empty string"),
    ({"run": {"dir": " "}}, r"\[run\] dir must be a non-empty string"),
    ({"run": {"seed": DEL}}, r"\[run\] seed is missing"),
    ({"run": {"seed": "1"}}, r"\[run\] seed must be an int"),
    ({"run": {"seed": True}}, r"\[run\] seed must be an int"),
    ({"run": {"seed": 1.5}}, r"\[run\] seed must be an int"),
    ({"run": {"model_dir": 3}}, r"\[run\] model_dir must be a non-empty string"),
    ({"limits": {"tokens": DEL}}, r"\[limits\] tokens is missing"),
    ({"limits": {"forward_passes": -1}}, r"forward_passes must be an int >= 0"),
    ({"limits": {"code_runs": 1.5}}, r"code_runs must be an int >= 0"),
    ({"limits": {"tokens": True}}, r"tokens must be an int >= 0"),
    ({"limits": {"dollars": -1}}, r"dollars must be a finite number >= 0"),
    ({"limits": {"dollars": "1"}}, r"dollars must be a finite number >= 0"),
    ({"limits": {"dollars": math.inf}}, r"dollars must be a finite number >= 0"),
    ({"limits": {"dollars": math.nan}}, r"dollars must be a finite number >= 0"),
    ({"limits": {"dollars": 10**400}}, r"dollars must be a finite number >= 0"),
    ({"limits": {"turns": 1}}, r"unknown key \[limits\] turns"),
    ({"editor": {"kind": DEL}}, r"kind must be one of"),
    ({"editor": {"kind": "gpt"}}, r"kind must be one of"),
    ({"editor": {"kind": [1]}}, r"kind must be one of"),
    ({"editor": {"base_url": "http://x"}}, r"unknown key \[editor\] base_url"),
    ({"editor": {**LOCAL, "base_url": DEL}}, r"\[editor\] base_url is missing"),
    ({"editor": {**LOCAL, "model": DEL}}, r"\[editor\] model is missing"),
    ({"editor": {**LOCAL, "max_turns": 0}}, r"max_turns must be an int >= 1"),
    ({"editor": {**LOCAL, "temperature": -1}}, r"temperature must be a finite"),
    ({"editor": {**LOCAL, "api_key": "sk"}}, r"unknown key \[editor\] api_key"),
    ({"editor": {**HOSTED, "no_fallback": DEL}}, r"give fallback_base_url"),
    ({"editor": {**HOSTED, "fallback_model": "m"}}, r"no_fallback takes no"),
    ({"editor": {**HOSTED, "no_fallback": "yes"}}, r"no_fallback must be true or"),
    ({"editor": {**HOSTED, "model": "gpt-9"}}, r"no price for model 'gpt-9'"),
    ({"editor": {**HOSTED, "price": "1,2,3"}}, r"\[editor\] price must be"),
    ({"editor": {**HOSTED, "price": 4}}, r"\[editor\] price must be a non-empty"),
    ({"editor": HOSTED, "limits": {"dollars": 0}}, r"dollars must be > 0"),
    ({"editor": {**HOSTED, "max_output_tokens": 0}}, r"max_output_tokens must be"),
    ({"kernel": {"timeout_s": 0}}, r"kernel.timeout_s must be a positive number"),
    ({"kernel": {"timeout_s": math.inf}}, r"kernel.timeout_s must be a positive"),
    ({"kernel": {"timeout_s": "1"}}, r"\[kernel\] timeout_s must be a number"),
    ({"kernel": {"mem_gib": 0}}, r"kernel.mem_gib must be an int >= 1"),
    ({"kernel": {"unsafe": 1}}, r"\[kernel\] unsafe must be true or false"),
    ({"kernel": {"allow_nested_userns": 1}}, r"allow_nested_userns must be true or"),
    ({"kernel": {"cpus": 1}}, r"unknown key \[kernel\] cpus"),
    ({"forward": {"device": "tpu"}}, r"forward.device must be one of"),
    ({"forward": {"dtype": "int8"}}, r"forward.dtype must be one of"),
    ({"forward": {"max_new_tokens": 0}}, r"forward.max_new_tokens must be"),
    ({"forward": {"device": 1}}, r"\[forward\] device must be a non-empty string"),
    ({"export": {"rank": 0}}, r"\[export\] rank must be an int >= 1"),
    ({"export": {"max_rel_residual": -0.1}}, r"max_rel_residual must be a finite"),
    ({"export": {"base_gguf": ""}}, r"\[export\] base_gguf must be a non-empty"),
    ({"export": {"gguf_out_type": "q4"}}, r"export.gguf_out_type must be one of"),
    ({"export": {"path": "x"}}, r"unknown key \[export\] path"),
]


@pytest.mark.parametrize(("tables", "match"), BAD)
def test_bad_config_is_refused(
    tmp_path: Path, tables: dict[str, Any], match: str
) -> None:
    path = write_config(tmp_path, **tables)
    with pytest.raises(RunError, match=match) as err:
        load_config(path)
    assert str(path.resolve()) in str(err.value)


def test_unreadable_configs(tmp_path: Path) -> None:
    with pytest.raises(RunError, match="not found"):
        load_config(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[run\ndir = 1\n")
    with pytest.raises(RunError, match="not valid TOML"):
        load_config(bad)
    bad.write_bytes(b"\xff\xfe")
    with pytest.raises(RunError, match="not readable"):
        load_config(bad)
    with pytest.raises(RunError, match="not readable"):
        load_config(tmp_path)  # a directory


# --- start guard -------------------------------------------------------------------


def test_start_writes_run_json_and_config_copy(
    tmp_path: Path, editor: FakeEditor
) -> None:
    editor.code = 5
    path = write_config(tmp_path, kernel={"timeout_s": 30})
    spec = load_config(path)
    assert run.start(spec) == 5
    assert editor.calls == [[str(spec.run_dir)]]
    cfg = load_run_config(spec.run_dir)
    assert (cfg.seed, cfg.editor, cfg.model, cfg.limits) == (
        7,
        "scripted",
        None,
        spec.limits,
    )
    assert cfg.kernel_timeout_s == 30.0
    assert (spec.run_dir / "config.toml").read_bytes() == path.read_bytes()
    assert not list(spec.run_dir.glob(".*.tmp"))


def test_config_copy_keeps_crlf_bytes(tmp_path: Path, editor: FakeEditor) -> None:
    path = write_config(tmp_path)
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    spec = load_config(path)
    run.start(spec)
    assert (spec.run_dir / "config.toml").read_bytes() == path.read_bytes()


def test_start_in_an_empty_dir(tmp_path: Path, editor: FakeEditor) -> None:
    spec = load_config(write_config(tmp_path))
    spec.run_dir.mkdir(parents=True)
    assert run.start(spec) == 0
    assert load_run_config(spec.run_dir).seed == 7


@pytest.mark.parametrize("with_copy", [True, False])
def test_start_over_an_aborted_start_of_the_same_run(
    tmp_path: Path, editor: FakeEditor, with_copy: bool
) -> None:
    path = write_config(tmp_path)
    spec = load_config(path)
    spec.run_dir.mkdir(parents=True)
    # checks passed and the files were written, but the editor never ran
    (spec.run_dir / "run.json").write_text(json.dumps(spec.run_json()))
    if with_copy:
        (spec.run_dir / "config.toml").write_text("# an older copy\n")
    changed = load_config(write_config(tmp_path, kernel={"timeout_s": 5}))
    assert run.start(changed) == 0
    assert load_run_config(spec.run_dir).kernel_timeout_s == 5.0
    assert (spec.run_dir / "config.toml").read_bytes() == path.read_bytes()


@pytest.mark.parametrize(
    "files",
    [
        {"config.toml": "# my own project config\n", "run.json": "{}"},
        {"config.toml": "# my own project config\n"},
        {"run.json": "not json"},
        {"notes.txt": "mine"},
        {"run.json": "SAME", "notes.txt": "mine"},
        {"run.json": "OTHER"},
    ],
)
def test_fresh_start_never_overwrites_a_foreign_dir(
    tmp_path: Path, editor: FakeEditor, files: dict[str, str]
) -> None:
    spec = load_config(write_config(tmp_path))
    other = load_config(write_config(tmp_path / "o", run={"seed": 8}))
    saved = {"SAME": spec.run_json(), "OTHER": other.run_json()}
    spec.run_dir.mkdir(parents=True)
    for name, text in files.items():
        content = json.dumps(saved[text]) if text in saved else text
        (spec.run_dir / name).write_text(content)
    before = snapshot(spec.run_dir)
    with pytest.raises(RunError, match="is not empty and holds no run started"):
        run.start(spec)
    assert snapshot(spec.run_dir) == before
    assert editor.calls == []


def test_unlistable_run_dir_is_refused(
    tmp_path: Path, editor: FakeEditor, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_config(write_config(tmp_path))
    spec.run_dir.mkdir(parents=True)

    def denied(self: Path) -> Any:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "iterdir", denied)
    with pytest.raises(RunError, match="cannot read"):
        run.start(spec)


@pytest.mark.parametrize("marker", ["record.jsonl", "ledger.jsonl", "jobs", "edits"])
def test_existing_run_is_refused_without_resume(
    tmp_path: Path, editor: FakeEditor, marker: str
) -> None:
    spec = load_config(write_config(tmp_path))
    spec.run_dir.mkdir(parents=True)
    if marker.endswith(".jsonl"):
        (spec.run_dir / marker).write_text("")
    else:
        (spec.run_dir / marker).mkdir()
    (spec.run_dir / "run.json").write_text("{}")
    before = snapshot(spec.run_dir)
    with pytest.raises(RunError, match="already holds a run; pass --resume"):
        run.start(spec)
    assert snapshot(spec.run_dir) == before
    assert editor.calls == []


def test_paid_run_is_never_touched_by_a_refused_start_or_resume(
    tmp_path: Path, editor: FakeEditor
) -> None:
    spec = load_config(write_config(tmp_path))
    started_run(spec)
    (spec.run_dir / "config.toml").write_text("# the first config\n")
    before = snapshot(spec.run_dir)
    with pytest.raises(RunError, match="already holds a run"):
        run.start(spec)
    changes: list[dict[str, Any]] = [
        {"run": {"seed": 8}},
        {"limits": {"tokens": 1000}},
        {"limits": {"dollars": 1.0}},
        {"run": {"model_dir": "m"}},
        {"editor": LOCAL},
    ]
    for change in changes:
        other = load_config(write_config(tmp_path, **change))
        with pytest.raises(RunError, match="does not match") as err:
            run.start(other, resume=True)
        assert "nothing was changed" in str(err.value)
        assert snapshot(spec.run_dir) == before
    assert editor.calls == []


def test_resume_mismatch_lists_every_difference(
    tmp_path: Path, editor: FakeEditor
) -> None:
    spec = load_config(write_config(tmp_path))
    started_run(spec)
    other = load_config(
        write_config(tmp_path, run={"seed": 9}, limits={"code_runs": 1})
    )
    with pytest.raises(RunError) as err:
        run.start(other, resume=True)
    text = str(err.value)
    assert "seed: run has 7, config has 9" in text
    assert '"code_runs": 3' in text
    assert '"code_runs": 1' in text
    assert "editor" not in text.split("\n", 1)[1]


def test_resume_with_a_different_editor_model_is_refused(
    tmp_path: Path, editor: FakeEditor
) -> None:
    spec = load_config(write_config(tmp_path, editor=LOCAL))
    started_run(spec)
    other = load_config(write_config(tmp_path, editor={**LOCAL, "model": "m2"}))
    with pytest.raises(RunError, match='model: run has "m", config has "m2"'):
        run.start(other, resume=True)


def test_resume_goes_on_and_may_change_runtime_options(
    tmp_path: Path, editor: FakeEditor
) -> None:
    spec = load_config(write_config(tmp_path, editor=LOCAL))
    started_run(spec)
    ledger = (spec.run_dir / "ledger.jsonl").read_bytes()
    record = (spec.run_dir / "record.jsonl").read_bytes()
    changed = {**LOCAL, "base_url": "http://y", "max_turns": 2}
    path = write_config(tmp_path, editor=changed, kernel={"timeout_s": 5})
    assert run.start(load_config(path), resume=True) == 0
    assert "http://y" in editor.calls[0]
    assert load_run_config(spec.run_dir).kernel_timeout_s == 5.0
    assert (spec.run_dir / "ledger.jsonl").read_bytes() == ledger
    assert (spec.run_dir / "record.jsonl").read_bytes() == record
    assert b"http://y" in (spec.run_dir / "config.resume-1.toml").read_bytes()


def test_resume_keeps_every_earlier_config_copy(
    tmp_path: Path, editor: FakeEditor
) -> None:
    spec = load_config(write_config(tmp_path))
    started_run(spec)
    first = b"# the config the paid run started with\n"
    (spec.run_dir / "config.toml").write_bytes(first)
    one = write_config(tmp_path, kernel={"timeout_s": 5})
    assert run.start(load_config(one), resume=True) == 0
    second = (spec.run_dir / "config.resume-1.toml").read_bytes()
    assert second == one.read_bytes()
    two = write_config(tmp_path, kernel={"timeout_s": 9})
    assert run.start(load_config(two), resume=True) == 0
    assert (spec.run_dir / "config.toml").read_bytes() == first
    assert (spec.run_dir / "config.resume-1.toml").read_bytes() == second
    assert (spec.run_dir / "config.resume-2.toml").read_bytes() == two.read_bytes()
    assert not list(spec.run_dir.glob(".*.tmp"))


def test_resume_without_a_config_copy_writes_the_first(
    tmp_path: Path, editor: FakeEditor
) -> None:
    path = write_config(tmp_path)
    spec = load_config(path)
    started_run(spec)
    (spec.run_dir / "config.toml").unlink()
    assert run.start(spec, resume=True) == 0
    assert (spec.run_dir / "config.toml").read_bytes() == path.read_bytes()
    assert not (spec.run_dir / "config.resume-1.toml").exists()


def test_resume_needs_a_run(tmp_path: Path, editor: FakeEditor) -> None:
    spec = load_config(write_config(tmp_path))
    with pytest.raises(RunError, match="no run to resume"):
        run.start(spec, resume=True)
    assert not spec.run_dir.exists()


def test_resume_of_a_finished_run_is_refused(
    tmp_path: Path, editor: FakeEditor
) -> None:
    spec = load_config(write_config(tmp_path))
    started_run(spec, finish=True)
    before = snapshot(spec.run_dir)
    with pytest.raises(RunError, match="already finished"):
        run.start(spec, resume=True)
    assert snapshot(spec.run_dir) == before


@pytest.mark.parametrize("saved", [None, "not json", "[]"])
def test_resume_needs_the_saved_run_json(
    tmp_path: Path, editor: FakeEditor, saved: str | None
) -> None:
    spec = load_config(write_config(tmp_path))
    started_run(spec)
    if saved is None:
        (spec.run_dir / "run.json").unlink()
    else:
        (spec.run_dir / "run.json").write_text(saved)
    with pytest.raises(RunError, match="saved"):
        run.start(spec, resume=True)


def test_resume_with_an_unreadable_record_is_refused(
    tmp_path: Path, editor: FakeEditor
) -> None:
    spec = load_config(write_config(tmp_path))
    started_run(spec)
    (spec.run_dir / "record.jsonl").write_text("not json\n")
    with pytest.raises(RunError, match="cannot read"):
        run.start(spec, resume=True)


def test_run_dir_that_is_a_file(tmp_path: Path, editor: FakeEditor) -> None:
    spec = load_config(write_config(tmp_path))
    spec.run_dir.write_text("x")
    with pytest.raises(RunError, match="not a directory"):
        run.start(spec)


def test_unwritable_run_dir(tmp_path: Path, editor: FakeEditor) -> None:
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg" / "afile").write_text("x")
    spec = load_config(write_config(tmp_path, run={"dir": "afile/run"}))
    with pytest.raises(RunError, match="cannot write the run directory"):
        run.start(spec)
    assert editor.calls == []


def test_model_dir_must_exist(tmp_path: Path, editor: FakeEditor) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    spec = load_config(write_config(tmp_path, run={"model_dir": "nope"}))
    with pytest.raises(RunError, match="is not a directory"):
        run.start(spec)
    assert not spec.run_dir.exists()


def test_model_dir_needs_the_hf_extra(
    tmp_path: Path, editor: FakeEditor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forward, "hf_available", lambda: False)
    (tmp_path / "cfg" / "m").mkdir(parents=True)
    spec = load_config(write_config(tmp_path, run={"model_dir": "m"}))
    with pytest.raises(RunError, match=r"pip install 'astro-weights\[hf\]'"):
        run.start(spec)
    assert not spec.run_dir.exists()


def test_unavailable_forward_device_stops_the_run_before_the_editor(
    tmp_path: Path, editor: FakeEditor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forward, "hf_available", lambda: True)
    monkeypatch.setattr(
        forward, "device_problem", lambda d: f"{d}: this torch has no CUDA"
    )
    (tmp_path / "cfg" / "m").mkdir(parents=True)
    spec = load_config(
        write_config(tmp_path, run={"model_dir": "m"}, forward={"device": "cuda"})
    )
    with pytest.raises(RunError, match="cuda: this torch has no CUDA"):
        run.start(spec)
    assert not spec.run_dir.exists()
    assert editor.calls == []


def test_base_gguf_is_checked_before_the_run(
    tmp_path: Path, editor: FakeEditor, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_config(write_config(tmp_path, export={"base_gguf": "base.gguf"}))
    monkeypatch.setattr(run, "_gguf_installed", lambda: True)
    with pytest.raises(RunError, match=r"base\.gguf is not a file"):
        run.start(spec)
    monkeypatch.setattr(run, "_gguf_installed", lambda: False)
    with pytest.raises(RunError, match=r"pip install 'astro-weights\[gguf\]'"):
        run.start(spec)
    assert not spec.run_dir.exists()
    (tmp_path / "cfg" / "base.gguf").write_bytes(b"GGUF")
    monkeypatch.setattr(run, "_gguf_installed", lambda: True)
    assert run.start(spec) == 0


SECRET = "sk-ant-api03-SECRETSECRET"


@pytest.mark.parametrize("value", [SECRET, "1ABC", "A-B", "A B", "", "$KEY"])
def test_api_key_env_must_name_a_variable_and_is_never_echoed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    path = write_config(tmp_path, editor={**LOCAL, "api_key_env": value})
    with pytest.raises(RunError, match="api_key_env must be the name of an") as err:
        load_config(path)
    assert value not in str(err.value) or value == ""
    assert cli(["run", str(path)]) == 2
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err
    assert not (tmp_path / "cfg" / "run").exists()


def test_api_key_env_names_are_accepted(tmp_path: Path) -> None:
    for name in ("MY_KEY", "_k", "a1"):
        path = write_config(tmp_path, editor={**LOCAL, "api_key_env": name})
        assert load_config(path).options["api_key_env"] == name


def test_key_variables_are_checked_before_anything_is_written(
    tmp_path: Path, editor: FakeEditor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AW_RUN_TEST_KEY", raising=False)
    monkeypatch.delenv(run.HOSTED_KEY_ENV, raising=False)
    local = load_config(
        write_config(tmp_path, editor={**LOCAL, "api_key_env": "AW_RUN_TEST_KEY"})
    )
    with pytest.raises(RunError, match=r"\$AW_RUN_TEST_KEY is not set"):
        run.start(local)
    hosted = load_config(write_config(tmp_path / "h", editor=HOSTED))
    with pytest.raises(RunError, match=r"\$ANTHROPIC_API_KEY is not set"):
        run.start(hosted)
    monkeypatch.setenv("AW_RUN_TEST_KEY", "")
    with pytest.raises(RunError, match="is not set"):
        run.start(local)
    assert not local.run_dir.exists()
    assert not hosted.run_dir.exists()
    assert editor.calls == []
    monkeypatch.setenv("AW_RUN_TEST_KEY", "k")
    monkeypatch.setenv(run.HOSTED_KEY_ENV, "k")
    assert run.start(local) == 0
    assert run.start(hosted) == 0


def test_hosted_key_variable_is_the_editors() -> None:
    hosted = pytest.importorskip("astro_weights.mcp.editors.hosted")
    assert hosted.KEY_ENV == run.HOSTED_KEY_ENV


def test_editor_argparse_exit_is_its_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def exits(code: object) -> Callable[[list[str]], int]:
        def main(_argv: list[str]) -> int:
            raise SystemExit(code)

        return main

    for code, want in ((2, 2), (None, 2), ("message", 2), (0, 0)):
        monkeypatch.setattr(run, "_editor_main", lambda kind, c=code: exits(c))
        spec = load_config(write_config(tmp_path / str(code)))
        assert run.start(spec) == want


@pytest.mark.parametrize(
    ("kind", "blocked", "hint"),
    [
        ("scripted", ("mcp",), "astro-weights[mcp]"),
        ("local", ("mcp",), "astro-weights[mcp]"),
        ("hosted", ("anthropic",), "astro-weights[hosted]"),
    ],
)
def test_missing_extra_exits_2_with_the_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kind: str,
    blocked: tuple[str, ...],
    hint: str,
) -> None:
    for name in blocked:
        monkeypatch.setitem(sys.modules, name, None)
    for name in list(sys.modules):
        if name.startswith("astro_weights.mcp.editors."):
            monkeypatch.delitem(sys.modules, name)
    editors: dict[str, dict[str, Any]] = {
        "scripted": {},
        "local": LOCAL,
        "hosted": HOSTED,
    }
    path = write_config(tmp_path, editor={"kind": kind, **editors[kind]})
    assert cli(["run", str(path)]) == 2
    assert hint in capsys.readouterr().err
    assert not (tmp_path / "cfg" / "run").exists()


# --- status ------------------------------------------------------------------------


def test_status_of_a_run_not_started(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_config(tmp_path)
    spec = load_config(path)
    assert run.status(spec) == {"run_dir": str(spec.run_dir), "state": "not_started"}
    assert cli(["run", str(path), "--status"]) == 0
    assert "state: not_started" in capsys.readouterr().out
    assert not spec.run_dir.exists()


def test_status_reads_budget_events_and_finish_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_config(tmp_path)
    spec = load_config(path)
    started_run(spec, finish=True)
    edits = spec.run_dir / "edits"
    edits.mkdir()
    (edits / "manifest.json").write_text(json.dumps({"w.b": {}, "w.a": {}}))
    with (spec.run_dir / "ledger.jsonl").open("a") as f:
        f.write('{"op": "charge", "kind": "tok')  # a line a live server is writing
    before = snapshot(spec.run_dir)
    info = run.status(spec)
    assert info["state"] == "finished"
    assert info["finish"]["outcome"] == "finished"
    assert info["budget"]["used"]["tokens"] == 42
    assert info["budget"]["used"]["code_runs"] == 1
    assert info["edits"] == ["w.a", "w.b"]
    assert [e["event"] for e in info["events"]] == ["start", "usage", "finish"]
    assert run.status(spec, last=1)["events"][0]["event"] == "finish"
    assert run.status(spec, last=0)["events"] == []
    assert cli(["run", str(path), "--status", "--json"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["state"] == "finished"
    assert printed["budget"]["remaining"]["tokens"] == 58
    assert cli(["run", str(path), "--status"]) == 0
    text = capsys.readouterr().out
    assert "state: finished" in text
    assert "finish: finished at" in text
    assert "tokens: 42 / 100" in text
    assert "edits: w.a, w.b" in text
    assert "last 3 events:" in text
    assert snapshot(spec.run_dir) == before


def test_status_of_an_unfinished_run(tmp_path: Path) -> None:
    spec = load_config(write_config(tmp_path))
    started_run(spec)
    ledger = Ledger(spec.run_dir / "ledger.jsonl", spec.limits)
    for i in range(2):
        ledger.fail("forward_passes", "boom", ref=f"f{i}")
    record = RunRecord.open(spec.run_dir / "record.jsonl")
    record.event("summary", text="x" * 500)
    info = run.status(spec)
    assert info["state"] == "unfinished"
    assert info["finish"] is None
    assert info["edits"] == []
    text = run.format_status(info)
    assert "finish:" not in text
    assert "forward_passes: 0 / 3  failures in a row: 2" in text
    assert "edits: none" in text
    summary = next(line for line in text.splitlines() if "summary" in line)
    assert summary.endswith("...")


def test_status_with_run_json_only_is_not_started(tmp_path: Path) -> None:
    spec = load_config(write_config(tmp_path))
    spec.run_dir.mkdir(parents=True)
    (spec.run_dir / "run.json").write_text(json.dumps(spec.run_json()))
    info = run.status(spec)
    assert info["state"] == "not_started"
    assert info["budget"]["used"]["tokens"] == 0


@pytest.mark.parametrize(
    ("name", "content", "match"),
    [
        ("run.json", "{}", "seed must be an int"),
        ("record.jsonl", "[1]\n", "cannot read"),
        ("ledger.jsonl", '{"op": "steal", "kind": "tokens"}\n', "cannot read"),
        ("edits/manifest.json", "not json", None),
    ],
)
def test_status_of_a_damaged_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
    content: str,
    match: str | None,
) -> None:
    path = write_config(tmp_path)
    spec = load_config(path)
    started_run(spec)
    (spec.run_dir / name).parent.mkdir(exist_ok=True)
    (spec.run_dir / name).write_text(content)
    if match is None:
        assert run.status(spec)["edits"] == []
        return
    with pytest.raises(RunError, match=match):
        run.status(spec)
    assert cli(["run", str(path), "--status"]) == 2
    assert match in capsys.readouterr().err


def test_cli_bad_config_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_config(tmp_path, limits={"tokens": -1})
    assert cli(["run", str(path)]) == 2
    assert "tokens must be an int >= 0" in capsys.readouterr().err
    assert cli(["run", str(path), "--status"]) == 2


def test_resume_and_status_are_exclusive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as err:
        cli(["run", str(write_config(tmp_path)), "--resume", "--status"])
    assert err.value.code == 2


# --- end to end: real editors and server ----------------------------------------------


@pytest.fixture
def tiny_model(tmp_path: Path) -> Path:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("mcp")
    from mcp_tiny_hf import save_tiny_model

    return save_tiny_model(tmp_path / "model")


def _unsafe() -> bool:
    from astro_weights.mcp import sandbox

    return sandbox.probe() is None


def model_tables(model: Path) -> dict[str, Any]:
    return {
        "run": {"model_dir": str(model)},
        "kernel": {"timeout_s": 60, "unsafe": _unsafe()},
        "forward": {"device": "cpu", "max_new_tokens": 8},
    }


def events(run_dir: Path) -> list[str]:
    return [e["event"] for e in RunRecord(run_dir / "record.jsonl").read()]


def used(run_dir: Path) -> dict[str, float]:
    cfg = load_run_config(run_dir)
    return Ledger(run_dir / "ledger.jsonl", cfg.limits).used()


def test_scripted_run_end_to_end(
    tmp_path: Path, tiny_model: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tables = model_tables(tiny_model)
    path = write_config(
        tmp_path, limits={"forward_passes": 5, "code_runs": 5}, **tables
    )
    assert cli(["run", str(path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert len(printed["handles"]) == 4
    run_dir = tmp_path / "cfg" / "run"
    names = events(run_dir)
    assert names[0] == "start"
    assert names[-1] == "finish"
    assert "edit_saved" in names
    assert used(run_dir) == {
        "forward_passes": 2,
        "code_runs": 2,
        "tokens": 0,
        "dollars": 0,
    }
    assert names[-3:] == ["summary", "export", "finish"]
    from astro_weights.export import load_peft

    adapter = load_peft(run_dir / "export" / "adapter")
    assert len(adapter.deltas) == 1
    assert adapter.base_model == str(tiny_model.resolve())
    assert cli(["run", str(path), "--status", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["state"] == "finished"
    assert len(info["edits"]) == 1
    assert cli(["run", str(path)]) == 2  # never started over
    assert cli(["run", str(path), "--resume"]) == 2  # finished
    assert "already finished" in capsys.readouterr().err


@pytest.fixture
def endpoint() -> Iterator[tuple[str, list[Any]]]:
    """A fake OpenAI-compatible server; push ``(status, body)`` replies onto the list."""
    pytest.importorskip("mcp")
    from fake_endpoint import serve

    replies: list[Any] = []
    with serve(lambda _body: replies.pop(0)) as (url, _state):
        yield url, replies


def test_local_run_against_the_fake_endpoint(
    tmp_path: Path,
    endpoint: tuple[str, list[Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fake_endpoint import tool_reply

    url, replies = endpoint
    replies.append(tool_reply(("finish", {"summary": "nothing to do"}), tokens=(20, 5)))
    path = write_config(tmp_path, editor={**LOCAL, "base_url": url, "model": "tiny"})
    assert cli(["run", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["outcome"] == "finished"
    run_dir = tmp_path / "cfg" / "run"
    assert used(run_dir)["tokens"] == 25
    assert events(run_dir)[-1] == "finish"


def test_resume_keeps_budget_used(
    tmp_path: Path,
    endpoint: tuple[str, list[Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fake_endpoint import tool_reply

    url, replies = endpoint
    replies += [
        tool_reply(("status", {"handle": "nope"}), tokens=(10, 5)),
        (500, "server down"),
        (500, "server down"),
        (500, "server down"),
    ]
    path = write_config(tmp_path, editor={**LOCAL, "base_url": url, "model": "tiny"})
    assert cli(["run", str(path)]) == 1  # the endpoint died mid-run
    run_dir = tmp_path / "cfg" / "run"
    assert used(run_dir)["tokens"] == 15
    assert "finish" not in events(run_dir)
    assert cli(["run", str(path)]) == 2  # a paid run is never started over
    assert "pass --resume" in capsys.readouterr().err
    assert used(run_dir)["tokens"] == 15
    replies.append(tool_reply(("finish", {"summary": "done"}), tokens=(20, 5)))
    assert cli(["run", str(path), "--resume"]) == 0
    assert used(run_dir)["tokens"] == 40
    names = events(run_dir)
    assert names.count("start") == 1
    assert names[-1] == "finish"


def test_resume_keeps_edits(
    tmp_path: Path,
    tiny_model: Path,
    endpoint: tuple[str, list[Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fake_endpoint import tool_reply

    url, replies = endpoint
    edit = (
        "name = next(n for n in NAMES if 'mlp.down_proj' in n)\n"
        "W = load(name)\n"
        "rng = np.random.default_rng(0)\n"
        "d = np.outer(rng.standard_normal(W.shape[0]), rng.standard_normal(W.shape[1]))\n"
        "save_edit(name, (W + d * (0.3 * np.linalg.norm(W) / np.linalg.norm(d))).astype(np.float32))\n"
    )
    replies += [tool_reply(("exec_python", {"code": edit}))] + [(500, "down")] * 3
    tables = model_tables(tiny_model)
    editor = {**LOCAL, "base_url": url, "model": "tiny"}
    path = write_config(tmp_path, editor=editor, **tables)
    assert cli(["run", str(path)]) == 1
    run_dir = tmp_path / "cfg" / "run"
    manifest = (run_dir / "edits" / "manifest.json").read_bytes()
    assert len(json.loads(manifest)) == 1
    assert used(run_dir)["code_runs"] == 1
    replies += [
        tool_reply(("exec_python", {"code": "print(len(NAMES))"})),
        tool_reply(("finish", {"summary": "kept"})),
    ]
    assert cli(["run", str(path), "--resume"]) == 0
    capsys.readouterr()
    assert (run_dir / "edits" / "manifest.json").read_bytes() == manifest
    assert used(run_dir)["code_runs"] == 2
    assert events(run_dir)[-1] == "finish"


def _fake_anthropic(monkeypatch: pytest.MonkeyPatch, *contents: list[Any]) -> list[Any]:
    """Point the hosted editor at a fake client that sends ``contents`` in turn."""
    anthropic = pytest.importorskip("anthropic")
    pytest.importorskip("mcp")
    from anthropic.types import Message

    from astro_weights.mcp.editors import hosted

    def no_real_client(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("tests must not build a real Anthropic client")

    monkeypatch.setattr(anthropic, "Anthropic", no_real_client)
    monkeypatch.setenv(hosted.KEY_ENV, FAKE_KEY)
    queue = list(contents)
    sent: list[Any] = []

    class Stream:
        def __init__(self, message: Message) -> None:
            self.message = message

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def get_final_message(self) -> Message:
            return self.message

    def stream(**kw: Any) -> Stream:
        sent.append(kw)
        message = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": HOSTED_MODEL,
            "content": queue.pop(0),
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 10},
        }
        return Stream(Message.model_validate(message))

    def count_tokens(**_kw: Any) -> Any:
        return SimpleNamespace(input_tokens=1000)

    client = SimpleNamespace(
        messages=SimpleNamespace(stream=stream, count_tokens=count_tokens)
    )
    monkeypatch.setattr(hosted, "_client", lambda key: client)
    return sent


def test_hosted_run_with_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    finish = {
        "type": "tool_use",
        "id": "t1",
        "name": "finish",
        "input": {"summary": "ok"},
    }
    sent = _fake_anthropic(monkeypatch, [finish])
    editor = {**HOSTED, "max_output_tokens": 1000}
    path = write_config(
        tmp_path, editor=editor, limits={"dollars": 1.5, "tokens": 10_000}
    )
    assert cli(["run", str(path)]) == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["outcome"] == "finished"
    assert sent[0]["max_tokens"] == 1000
    run_dir = tmp_path / "cfg" / "run"
    spent = used(run_dir)
    assert spent["tokens"] == 110
    assert spent["dollars"] == pytest.approx(summary["dollars"])
    assert spent["dollars"] > 0
    assert load_run_config(run_dir).limits.dollars == 1.5
    files = "\n".join(p.read_text() for p in run_dir.rglob("*") if p.is_file())
    assert FAKE_KEY not in files + captured.out + captured.err


def test_hosted_run_without_a_key_exits_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _fake_anthropic(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    path = write_config(tmp_path, editor=HOSTED)
    assert cli(["run", str(path)]) == 2
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err
    assert not (tmp_path / "cfg" / "run").exists()  # checked before any write


def test_gguf_installed_matches_the_import_system() -> None:
    import importlib.util

    assert run._gguf_installed() == (importlib.util.find_spec("gguf") is not None)


def test_scripted_edit_paths_resolve_and_repeat_the_flag(tmp_path: Path) -> None:
    spec = load_config(write_config(tmp_path, editor={"edit": ["a", "/abs/b"]}))
    base = (tmp_path / "cfg").resolve()
    assert spec.editor_argv()[1:] == [
        "--edit",
        str(base / "a"),
        "--edit",
        str((base / "/abs/b").resolve()),  # Windows: joins base drive
    ]


@pytest.mark.parametrize("bad", [[], ["ok", ""], "a"])
def test_scripted_edit_must_be_a_list_of_paths(tmp_path: Path, bad: Any) -> None:
    with pytest.raises(RunError, match=r"\[editor\] edit must be"):
        load_config(write_config(tmp_path, editor={"edit": bad}))
