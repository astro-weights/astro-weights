"""0.3 exit test (PDR 0.3 section 7, criteria 2 to 8) on recorded files.

The `real` variant reads the directory named by $ASTRO_EXIT_DATA (layout and the
commands that produce it: tests/exit_0_3/README.md), carries the `realdata` marker and
skips without it. The `synthetic` variant runs the same tests on a small directory
generated here from tests/data/gate/make_fixture.py, so CI exercises the harness.

Every test copies the data to `tmp_path` first: a gate run writes a seal ledger and
`verdict.json`, and the real seal is never spent. Criterion 1 (planted overfit) is
covered by tests/test_eval_gate.py. Floors and the BFCL report are printed (`pytest -s`); a failed gate test shows its output.
"""

import contextlib
import dataclasses
import importlib.util
import io
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from astro_weights.cli import main
from astro_weights.eval import (
    FloorResult,
    Verdict,
    check_headers,
    config_hash,
    floor_from_records,
    format_tasks,
    header_path,
    load_floor,
    prompt_hash,
    read_header,
    read_jsonl,
    read_tasks,
    save_floor,
    write_header,
    write_jsonl,
)
from astro_weights.eval.gate_config import REPORT_ARMS, GateConfig, load_gate_config
from astro_weights.eval.seal import records_sha256
from astro_weights.eval.tasks import (
    TaskRecord,
    ToolCall,
    Turn,
    compare_tasks,
    write_tasks,
)
from astro_weights.mcp.record import RunRecord

N_CONTROLS = 21  # control-00 .. control-20, random_controls(edit, n=21)
REAL_CONTROLS = N_CONTROLS  # control gates in the real run
SYNTHETIC_CONTROLS = (
    3  # control gates in the synthetic run; the pools are the real ones
)
POOL = 20  # control files in each gate's `[confirm] controls`
CONTROL_NAMES = [f"control-{i:02d}" for i in range(N_CONTROLS)]
REAL_BFCL_TASKS = 200  # the full multi_turn_base run, not the 10-task pilot
SYNTHETIC_BFCL_TASKS = 12
NOT_CONFIRMED = ("not significant", "regressed")


def _load_fixture() -> ModuleType:
    path = Path(__file__).parent / "data" / "gate" / "make_fixture.py"
    spec = importlib.util.spec_from_file_location("exit_gate_make_fixture", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fixture = _load_fixture()


# ---------------------------------------------------------------- synthetic data


def _share_record(gate_dir: Path, shared: Path) -> None:
    """Point `gate_dir/GATE.toml` at the one agent run record, `run/record.jsonl`."""
    (gate_dir / "record.jsonl").unlink()
    toml = gate_dir / "GATE.toml"
    rel = os.path.relpath(shared, gate_dir).replace(os.sep, "/")
    toml.write_text(
        toml.read_text(encoding="utf-8").replace(
            'run_record = "record.jsonl"', f'run_record = "{rel}"'
        ),
        encoding="utf-8",
    )


def _as_control(gate_dir: Path, i: int) -> None:
    """Gate `gate_dir` as control `i`: its `[confirm] edit` is control `i`'s records."""
    toml = gate_dir / "GATE.toml"
    toml.write_text(
        toml.read_text(encoding="utf-8").replace(
            'edit = "confirm/edit.jsonl"',
            f'edit = "confirm/controls/control-{i:02d}.jsonl"',
        ),
        encoding="utf-8",
    )


def _pool(gate_dir: Path, exclude: int) -> None:
    """Point `gate_dir/GATE.toml` at a pool of the 20 controls other than `exclude`.

    `confirm/controls` holds all 21 control files; the pool is a copy of 20 of them
    (records and headers) in `confirm/pool`. The edit gate (and the planted bad edit's)
    excludes control-00, control `i` excludes itself.
    """
    pool = gate_dir / "confirm" / "pool"
    pool.mkdir()
    for f in sorted((gate_dir / "confirm" / "controls").glob("control-*")):
        if not f.name.startswith(f"control-{exclude:02d}."):
            shutil.copy(f, pool / f.name)
    toml = gate_dir / "GATE.toml"
    toml.write_text(
        toml.read_text(encoding="utf-8").replace(
            'controls = "confirm/controls"', 'controls = "confirm/pool"'
        ),
        encoding="utf-8",
    )


def _floor_runs(edit_dir: Path, floor: Path) -> None:
    """Two base runs under the edit's test config; the stored floor is the default one."""
    src = edit_dir / "test" / "base.jsonl"
    header = read_header(header_path(src))
    run1 = read_jsonl(src)
    deterministic = [dataclasses.replace(r, arm="base-b") for r in run1]
    default = [
        dataclasses.replace(
            r,
            arm="base-b",
            output=r.output + " (again)" if n % 10 == 0 else r.output,
            n_output_tokens=r.n_output_tokens + (2 if n % 20 == 0 else -2)
            if n % 10 == 0
            else r.n_output_tokens,
        )
        for n, r in enumerate(run1)
    ]
    flags = {"deterministic": {"serving_flags": {"VLLM_BATCH_INVARIANT": "1"}}}
    for name, second in (("deterministic", deterministic), ("default", default)):
        d = floor / name
        d.mkdir(parents=True)
        for stem, records in (("run1", run1), ("run2", second)):
            h = {**header, **flags.get(name, {}), "run_id": f"floor-{name}-{stem}"}
            write_jsonl(d / f"{stem}.jsonl", records)
            write_header(d / f"{stem}.header.json", h)
    save_floor(
        edit_dir / "floors",
        floor_from_records(
            read_jsonl(floor / "default" / "run1.jsonl"),
            read_jsonl(floor / "default" / "run2.jsonl"),
            header,
        ),
    )


def _bfcl(path: Path) -> None:
    """12 tasks x 2 trials, base and edit; the edit answers shorter."""
    path.mkdir(parents=True)
    for arm, cut in (("base", 0), ("edit", 25)):
        records = []
        for t in range(SYNTHETIC_BFCL_TASKS):
            for trial in range(2):
                calls = tuple(ToolCall("ls", "{}") for _ in range(1 + t % 3))
                turns = [
                    Turn("user", f"task {t}"),
                    Turn("assistant", "call", calls, 80 - cut + trial),
                    Turn("tool", "ok"),
                    Turn("assistant", "done", (), 20 - cut // 5),
                ]
                records.append(
                    TaskRecord.from_turns(
                        f"multi_turn_base_{t}",
                        trial,
                        arm,
                        prompt_hash(f"task {t}"),
                        turns,
                        passed=t % 4 != 0,
                    )
                )
        write_tasks(path / f"{arm}.jsonl", records)


def make_synthetic(root: Path) -> None:
    record = root / "run" / "record.jsonl"
    edit = fixture.build(root / "gate" / "edit", arms=True, n_controls=N_CONTROLS)
    _pool(edit.parent, 0)
    record.parent.mkdir(parents=True)
    shutil.move(edit.parent / "record.jsonl", record)
    edit.write_text(
        edit.read_text(encoding="utf-8").replace(
            'run_record = "record.jsonl"', 'run_record = "../../run/record.jsonl"'
        ),
        encoding="utf-8",
    )
    for i in range(SYNTHETIC_CONTROLS):  # every control fails the confirm stage
        c = fixture.build(
            root / "gate" / "controls" / f"{i:02d}",
            controls_beat=False,
            n_controls=N_CONTROLS,
        )
        _share_record(c.parent, record)
        _as_control(c.parent, i)
        _pool(c.parent, i)
    bad = fixture.build(
        root / "gate" / "bad", screens_fail="norm", n_controls=N_CONTROLS
    )
    _share_record(bad.parent, record)
    _pool(bad.parent, 0)
    _floor_runs(edit.parent, root / "floor")
    _bfcl(root / "bfcl")
    # the verdict a first run "recorded"; the seal it spent is not kept
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        assert main(["gate", str(edit)]) == 0
    shutil.rmtree(load_gate_config(edit).test.seal_dir)


# ------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Data:
    """A data directory copied to `tmp_path`, with the counts it must hold."""

    root: Path
    controls: int  # control gates
    bfcl_tasks: int  # tasks in the BFCL files

    @property
    def edit(self) -> Path:
        return self.root / "gate" / "edit" / "GATE.toml"

    @property
    def bad(self) -> Path:
        return self.root / "gate" / "bad" / "GATE.toml"

    def control_configs(self) -> list[Path]:
        return sorted((self.root / "gate" / "controls").glob("*/GATE.toml"))

    def copy(self, dst: Path) -> "Data":
        """Copy the tree to `dst`; refuse a gate whose seal would land outside it."""
        shutil.copytree(self.root, dst)
        data = dataclasses.replace(self, root=dst)
        configs = [data.edit, data.bad, *data.control_configs()]
        for cfg in configs:
            seal = load_gate_config(cfg).test.seal_dir.resolve()
            assert seal.is_relative_to(dst.resolve()), (
                f"{cfg}: seal_dir {seal} is outside the data dir"
            )
            shutil.rmtree(
                seal, ignore_errors=True
            )  # a seal left in the data is not spent here
            (cfg.parent / "verdict.json").unlink(missing_ok=True)
        return data


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> Data:
    root = tmp_path_factory.mktemp("exit_0_3")
    make_synthetic(root)
    return Data(root, SYNTHETIC_CONTROLS, SYNTHETIC_BFCL_TASKS)


@pytest.fixture(
    scope="module",
    params=["synthetic", pytest.param("real", marks=pytest.mark.realdata)],
)
def source(request: pytest.FixtureRequest, synthetic: Data) -> Data:
    if request.param == "synthetic":
        return synthetic
    env = os.environ.get("ASTRO_EXIT_DATA")
    if not env or not Path(env).is_dir():
        pytest.skip("ASTRO_EXIT_DATA not set to a directory (tests/exit_0_3/README.md)")
    return Data(Path(env), REAL_CONTROLS, REAL_BFCL_TASKS)


@pytest.fixture
def data(source: Data, tmp_path: Path) -> Data:
    return source.copy(tmp_path / "data")


def gate(
    cfg: Path, *flags: str, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str, str]:
    code = main(["gate", str(cfg), *flags])
    out, err = capsys.readouterr()
    return code, out, err


def outcome(out: str) -> str:
    m = re.search(r"^outcome: (.+)$", out, re.MULTILINE)
    assert m, out
    return m.group(1)


def ledger(cfg: Path) -> list[dict[str, object]]:
    seal = load_gate_config(cfg).test.seal_dir / "seal.jsonl"
    return [json.loads(line) for line in seal.read_text(encoding="utf-8").splitlines()]


def verdict_bytes(cfg: Path) -> bytes:
    return (cfg.parent / "verdict.json").read_bytes()


def _suites_present(cfg: GateConfig) -> None:
    assert cfg.suites, "no regression suites in GATE.toml"
    for name, s in cfg.suites.items():
        for arm in (s.base, s.edit):
            assert read_jsonl(arm), f"suite {name}: no records in {arm}"
            assert header_path(arm).is_file(), f"suite {name}: no header for {arm}"


def _pool_files(cfg: Path) -> dict[str, str]:
    """`{file stem: records sha256}` of a gate's `[confirm] controls` directory."""
    pool = load_gate_config(cfg).confirm.controls
    return {f.stem: records_sha256(read_jsonl(f)) for f in sorted(pool.glob("*.jsonl"))}


def check_pools(data: Data) -> None:
    """Every gate's pool is 20 distinct controls that do not contain the gated arm.

    The edit's pool is control-01..control-20 and the edit's records are in no pool.
    Control `i` is gated against the other 20 of control-00..control-20: a pool that
    holds the gated control lets the pool's best control beat the second best (itself
    excluded) by construction and always pass stage 3.
    """
    edit = load_gate_config(data.edit).confirm
    edit_sha = records_sha256(read_jsonl(edit.edit))
    pool = _pool_files(data.edit)
    assert sorted(pool) == CONTROL_NAMES[1:], f"edit pool: {sorted(pool)}"
    assert len(set(pool.values())) == POOL, "edit pool: two controls share records"
    assert edit_sha not in pool.values(), "the edit's records are in the edit pool"
    cfgs = data.control_configs()
    assert len(cfgs) == data.controls
    picked = []
    for cfg in [data.bad, *cfgs]:
        c = load_gate_config(cfg).confirm
        got = _pool_files(cfg)
        assert len(got) == POOL, f"{cfg}: pool has {len(got)} files, not {POOL}"
        assert len(set(got.values())) == POOL, f"{cfg}: pool files share records"
        assert edit_sha not in got.values(), (
            f"{cfg}: the edit's records are in the pool"
        )
        if cfg == data.bad:
            assert sorted(got) == CONTROL_NAMES[1:], f"{cfg}: pool {sorted(got)}"
            continue
        sha = records_sha256(read_jsonl(c.edit))
        assert c.edit.stem in CONTROL_NAMES, f"{cfg}: [confirm] edit is no control"
        assert c.edit.stem not in got, f"{cfg}: pool contains itself ({c.edit.stem})"
        assert sha not in got.values(), f"{cfg}: pool contains itself (same records)"
        assert sorted([*got, c.edit.stem]) == CONTROL_NAMES, (
            f"{cfg}: pool is not the other 20 controls: {sorted(got)}"
        )
        picked.append(sha)
    assert len(set(picked)) == len(picked), (
        "two control gates share a control's records"
    )


# ------------------------------------------------------------------- criterion 2


def test_2_edit_confirmed_every_control_not(
    data: Data, capsys: pytest.CaptureFixture[str]
) -> None:
    """Gate confirms the edit; the full gate confirms none of the controls.

    Each control is gated as if it were the edit, against the other 20 controls (never
    a pool it is part of). A control stopped at stage 1 would not prove the gate rejects
    it, so the outcome must be "not significant" or "regressed".
    """
    check_pools(data)
    code, out, _ = gate(data.edit, capsys=capsys)
    assert (code, outcome(out)) == (0, "confirmed")
    for cfg in data.control_configs():
        code, out, err = gate(cfg, capsys=capsys)
        assert code == 1, f"{cfg}: exit {code}\n{out}{err}"
        assert outcome(out) in NOT_CONFIRMED, cfg


# ------------------------------------------------------------------- criterion 3


def test_3_floor_zero_flips_deterministic_default_reported(
    data: Data, capsys: pytest.CaptureFixture[str]
) -> None:
    """Deterministic base-vs-base: flip_rate 0. Default serving: floor reported and used."""
    floor = data.root / "floor"
    floors: dict[str, FloorResult] = {}
    headers = {}
    for name in ("deterministic", "default"):
        run1, run2 = (read_jsonl(floor / name / f"{s}.jsonl") for s in ("run1", "run2"))
        header, other = (
            read_header(header_path(floor / name / f"{s}.jsonl"))
            for s in ("run1", "run2")
        )
        check_headers(header, other)  # one config
        assert header["run_id"] != other["run_id"], f"{name}: run2 is a copy of run1"
        floors[name], headers[name] = floor_from_records(run1, run2, header), header
        print(f"{name} floor: {floors[name].to_json()}")
    flags = headers["deterministic"].get("serving_flags")
    assert flags, "deterministic floor declares no serving_flags (batch-invariant mode)"
    assert flags != headers["default"].get("serving_flags")
    det, default = floors["deterministic"], floors["default"]
    assert det.flip_rate == 0.0
    assert (det.accuracy_delta, det.tokens_delta) == (0.0, 0.0)
    assert 0.0 <= default.flip_rate <= 1.0
    cfg = load_gate_config(data.edit)
    key = config_hash(read_header(header_path(cfg.test.base)))
    assert default.config_hash == key
    stored = load_floor(cfg.test.floor, key)
    assert stored is not None, "the edit's gate has no stored default-serving floor"
    keep = {f.name for f in dataclasses.fields(FloorResult)} - {
        "created",
        "library_version",
    }
    for name in sorted(keep):
        assert getattr(stored, name) == pytest.approx(getattr(default, name)), name
    code, out, _ = gate(
        data.edit, capsys=capsys
    )  # the floor is used by the tokens check
    assert code == 0
    assert "tokens lower beyond floor" in out


# ------------------------------------------------------------------- criterion 4


def test_4_second_scoring_refused_unseal_recorded(
    data: Data, capsys: pytest.CaptureFixture[str]
) -> None:
    assert gate(data.edit, capsys=capsys)[0] == 0
    first = ledger(data.edit)
    assert [x["unseal"] for x in first] == [False]
    code, out, err = gate(data.edit, capsys=capsys)
    assert (code, out) == (2, "")
    assert "already scored" in err
    assert ledger(data.edit) == first
    code, _, err = gate(data.edit, "--unseal", capsys=capsys)
    assert code == 0
    assert "--unseal" in err
    assert [x["unseal"] for x in ledger(data.edit)] == [False, True]


# ------------------------------------------------------------------- criterion 5


def test_5_report_has_mde_candidates_and_every_arm(
    data: Data, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = load_gate_config(data.edit)
    code, out, _ = gate(data.edit, capsys=capsys)
    assert code == 0
    v = Verdict.from_json(
        (data.edit.parent / "verdict.json").read_text(encoding="utf-8")
    )
    assert v.candidates_tried >= 1
    counts = RunRecord.open(cfg.run_record).confirm_counts()  # the verdict's counts
    assert (v.candidates_tried, v.confirm_queries) == (
        counts["candidates_tried"],
        counts["confirm_queries"],
    )
    assert f"candidates tried: {v.candidates_tried}" in out
    assert f"confirm queries: {v.confirm_queries}" in out
    assert "MDE (test)" in out
    for split in ("confirm", "test"):
        assert f"arms on the {split} split" in out
    assert re.search(r"^controls: n=\d+", out, re.MULTILINE)
    assert re.search(r"^edit +edit ", out, re.MULTILINE)
    assert set(cfg.arms) == set(REPORT_ARMS), "the edit gate needs every [arms] arm"
    for key in cfg.arms:
        name = re.escape(REPORT_ARMS[key][0])
        assert re.search(rf"^{name} +", out, re.MULTILINE), key
    assert out.count("MDE") >= 4  # one column per arm table, plus the gate line


# ------------------------------------------------------------------- criterion 6


def test_6_bfcl_paired_run_prints_tokens_and_tool_calls(data: Data) -> None:
    """Base vs edit on BFCL multi_turn_base. Published whatever it shows: no outcome assertion."""
    base = read_tasks(data.root / "bfcl" / "base.jsonl")
    edit = read_tasks(data.root / "bfcl" / "edit.jsonl")
    assert base
    assert {t.arm for t in base} != {t.arm for t in edit}
    cmp = compare_tasks(base, edit, n_boot=2000)
    text = format_tasks(cmp)
    print(text)
    assert cmp.n_tasks == len({t.task_id for t in base}) == data.bfcl_tasks
    assert all(t.task_id.startswith("multi_turn_base_") for t in base)
    for row in ("pass rate:", "tokens/task:", "tool calls/task:", "turns/task:"):
        assert row in text


# ------------------------------------------------------------------- criterion 7


def test_7_rerun_from_copied_files_gives_identical_verdict(
    source: Data, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recorded = (source.edit.parent / "verdict.json").read_bytes()  # as published
    a, b = (source.copy(tmp_path / n) for n in ("a", "b"))
    assert gate(a.edit, capsys=capsys)[0] == gate(b.edit, capsys=capsys)[0] == 0
    assert verdict_bytes(a.edit) == verdict_bytes(b.edit) == recorded
    assert (a.root / "run" / "record.jsonl").read_bytes() == (
        source.root / "run" / "record.jsonl"
    ).read_bytes()  # the run record was not touched by the gate
    assert (
        load_gate_config(a.edit).run_record.resolve()
        == (a.root / "run" / "record.jsonl").resolve()
    )


# ------------------------------------------------------------------- criterion 8


def test_8_planted_bad_edit_screened_out_or_regressed(
    data: Data, capsys: pytest.CaptureFixture[str]
) -> None:
    for cfg_path in (
        data.edit,
        data.bad,
    ):  # suites are present for the edit and the bad edit
        _suites_present(load_gate_config(cfg_path))
    code, out, _ = gate(data.bad, capsys=capsys)
    got = outcome(out)
    assert got in ("screened out", "regressed")
    assert code == (3 if got == "screened out" else 1)
    assert "keep the original weights" in out


# ------------------------------------------------------------- the harness itself


def test_a_seal_outside_the_copy_is_refused(synthetic: Data, tmp_path: Path) -> None:
    source = synthetic
    src = shutil.copytree(source.root, tmp_path / "src")
    cfg = src / "gate" / "edit" / "GATE.toml"
    cfg.write_text(
        cfg.read_text(encoding="utf-8").replace(
            'seal_dir = "seal"', 'seal_dir = "../../../elsewhere"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="outside the data dir"):
        dataclasses.replace(source, root=src).copy(tmp_path / "dst")
    assert not (tmp_path / "elsewhere").exists()


def test_a_leftover_seal_is_not_spent(
    synthetic: Data, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = synthetic
    src = shutil.copytree(source.root, tmp_path / "src")
    assert (
        gate(src / "gate" / "edit" / "GATE.toml", capsys=capsys)[0] == 0
    )  # spends src's seal
    copy = dataclasses.replace(source, root=src).copy(tmp_path / "dst")
    assert gate(copy.edit, capsys=capsys)[0] == 0


def test_a_pool_that_holds_the_gated_control_is_refused(
    synthetic: Data, tmp_path: Path
) -> None:
    data = synthetic.copy(tmp_path / "data")
    check_pools(data)
    cfg = data.control_configs()[1]  # control-01 against the edit's pool, which has it
    cfg.write_text(
        cfg.read_text(encoding="utf-8").replace(
            'controls = "confirm/pool"', 'controls = "../../edit/confirm/pool"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="pool contains itself"):
        check_pools(data)


def test_a_pool_that_holds_the_edit_or_a_copy_is_refused(
    synthetic: Data, tmp_path: Path
) -> None:
    data = synthetic.copy(tmp_path / "data")
    pool = load_gate_config(data.edit).confirm.controls
    victim = pool / "control-20.jsonl"
    edit = load_gate_config(data.edit).confirm.edit
    for src, dst in ((edit, victim), (header_path(edit), header_path(victim))):
        shutil.copy(src, dst)
    with pytest.raises(AssertionError, match="edit's records are in"):
        check_pools(data)


def test_a_short_or_duplicated_pool_is_refused(synthetic: Data, tmp_path: Path) -> None:
    data = synthetic.copy(tmp_path / "data")
    pool = load_gate_config(data.edit).confirm.controls
    (pool / "control-20.jsonl").unlink()
    with pytest.raises(AssertionError, match="edit pool"):
        check_pools(data)
