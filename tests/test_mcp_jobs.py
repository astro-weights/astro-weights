import json
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from astro_weights.mcp.budget import BudgetExceeded, Ledger, Limits, TooManyFailures
from astro_weights.mcp.handles import HandleStore, Job
from astro_weights.mcp.jobs import JobRunner, load_run_config

LIMITS = {"forward_passes": 3, "code_runs": 3, "tokens": 100, "dollars": 1.0}


def write_run(run_dir: Path, **over: Any) -> Path:
    cfg = {"seed": 7, "editor": "scripted", "model": None, "limits": LIMITS, **over}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(cfg))
    return run_dir


def wait_for(runner_store: HandleStore, handle: str, timeout: float = 10) -> Job:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        job = runner_store.get(handle)
        if job.state in {"done", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def make_runner(
    tmp_path: Path,
    fn: Callable[[dict[str, Any]], dict[str, Any]],
    code_runs: int = 3,
) -> tuple[JobRunner, HandleStore, Ledger]:
    store = HandleStore(tmp_path)
    ledger = Ledger(tmp_path / "ledger.jsonl", Limits(3, code_runs, 100, 1.0))
    return JobRunner(store, ledger, {"exec": fn}, {"exec": "code_runs"}), store, ledger


def test_job_runs_and_charges_once(tmp_path: Path) -> None:
    runner, store, ledger = make_runner(tmp_path, lambda p: {"echo": p["x"]})
    job = runner.submit("exec", {"x": 1})
    assert job.state == "queued"
    done = wait_for(store, job.handle)
    assert done.state == "done"
    assert done.result == {"echo": 1}
    assert ledger.used()["code_runs"] == 1


def test_failing_executor_is_free_and_counts_streak(tmp_path: Path) -> None:
    def boom(params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("nope")

    runner, store, ledger = make_runner(tmp_path, boom)
    job = wait_for(store, runner.submit("exec", {}).handle)
    assert job.state == "failed"
    assert job.error == "RuntimeError: nope"
    assert ledger.used()["code_runs"] == 0
    assert ledger.to_json()["streaks"]["code_runs"] == 1


def test_non_json_result_fails_the_job_free(tmp_path: Path) -> None:
    runner, store, ledger = make_runner(tmp_path, lambda p: {"x": object()})
    job = wait_for(store, runner.submit("exec", {}).handle)
    assert job.state == "failed"
    assert job.result is None
    assert ledger.used()["code_runs"] == 0


def test_exhausted_budget_refuses_free(tmp_path: Path) -> None:
    runner, store, ledger = make_runner(tmp_path, lambda p: {}, code_runs=0)
    with pytest.raises(BudgetExceeded):
        runner.submit("exec", {})
    assert store.list() == []
    lines = [
        json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text().split("\n") if x
    ]
    assert [x["op"] for x in lines] == ["refuse"]
    assert ledger.used()["code_runs"] == 0


def test_failure_streak_refuses(tmp_path: Path) -> None:
    def boom(params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("x")

    runner, store, _ = make_runner(tmp_path, boom)
    for _ in range(3):
        wait_for(store, runner.submit("exec", {}).handle)
    with pytest.raises(TooManyFailures):
        runner.submit("exec", {})


def test_unknown_kind_writes_nothing(tmp_path: Path) -> None:
    runner, store, _ = make_runner(tmp_path, lambda p: {})
    with pytest.raises(ValueError, match="no executor"):
        runner.submit("forward", {})
    assert store.list() == []
    assert not (tmp_path / "ledger.jsonl").exists()


def test_in_flight_jobs_past_limit_fail_budget_exhausted(tmp_path: Path) -> None:
    gate = threading.Event()

    def slow(params: dict[str, Any]) -> dict[str, Any]:
        gate.wait(10)
        return {"ok": True}

    runner, store, ledger = make_runner(tmp_path, slow, code_runs=1)
    jobs = [runner.submit("exec", {}) for _ in range(2)]  # both pass check()
    gate.set()
    done = [wait_for(store, j.handle) for j in jobs]
    assert sorted(j.state for j in done) == ["done", "failed"]
    failed = next(j for j in done if j.state == "failed")
    assert failed.error == "budget exhausted"
    assert failed.result is None
    assert ledger.used()["code_runs"] == 1


def test_disk_trouble_marks_job_failed(tmp_path: Path) -> None:
    runner, store, ledger = make_runner(tmp_path, lambda p: {})

    def bad_charge(*a: Any, **k: Any) -> None:
        raise OSError("disk full")

    ledger.charge = bad_charge  # type: ignore[method-assign]
    job = wait_for(store, runner.submit("exec", {}).handle)
    assert job.state == "failed"
    assert job.error == "OSError: disk full"


def test_disk_trouble_in_failing_too_is_swallowed(tmp_path: Path) -> None:
    runner, store, ledger = make_runner(tmp_path, lambda p: {})

    def bad(*a: Any, **k: Any) -> Any:
        raise OSError("gone")

    ledger.charge = bad  # type: ignore[method-assign]
    store.fail = bad  # type: ignore[method-assign]
    job = runner.submit("exec", {})
    time.sleep(0.2)  # the thread must end without raising; the job stays running
    assert store.get(job.handle).state == "running"


def test_exit_in_executor_fails_the_job(tmp_path: Path) -> None:
    def quit_(params: dict[str, Any]) -> dict[str, Any]:
        sys.exit(1)

    runner, store, ledger = make_runner(tmp_path, quit_)
    job = wait_for(store, runner.submit("exec", {}).handle)
    assert job.state == "failed"
    assert job.error == "SystemExit: 1"
    assert ledger.used()["code_runs"] == 0
    assert ledger.to_json()["streaks"]["code_runs"] == 1


def test_run_config_ok(tmp_path: Path) -> None:
    cfg = load_run_config(write_run(tmp_path, model="m"))
    assert (cfg.seed, cfg.editor, cfg.model) == (7, "scripted", "m")
    assert cfg.limits == Limits(3, 3, 100, 1.0)
    assert (cfg.model_dir, cfg.kernel_timeout_s, cfg.kernel_mem_gib) == (None, 600, 16)


def test_run_config_model_dir_and_kernel(tmp_path: Path) -> None:
    kernel = {"timeout_s": 30, "mem_gib": 4}
    cfg = load_run_config(write_run(tmp_path, model_dir="weights", kernel=kernel))
    assert cfg.model_dir == tmp_path / "weights"
    assert (cfg.kernel_timeout_s, cfg.kernel_mem_gib) == (30.0, 4)
    absolute = str(tmp_path.resolve() / "elsewhere")
    cfg = load_run_config(write_run(tmp_path, model_dir=absolute, kernel={}))
    assert cfg.model_dir == Path(absolute)
    assert (cfg.kernel_timeout_s, cfg.kernel_mem_gib) == (600, 16)


def test_run_config_forward(tmp_path: Path) -> None:
    cfg = load_run_config(write_run(tmp_path))
    assert (cfg.forward_device, cfg.forward_dtype, cfg.forward_max_new_tokens) == (
        "auto",
        "auto",
        4096,
    )
    forward = {"device": "cuda", "dtype": "bfloat16", "max_new_tokens": 32768}
    cfg = load_run_config(write_run(tmp_path, forward=forward))
    assert (cfg.forward_device, cfg.forward_dtype, cfg.forward_max_new_tokens) == (
        "cuda",
        "bfloat16",
        32768,
    )


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"seed": "7"}, "seed"),
        ({"seed": True}, "seed"),
        ({"editor": ""}, "editor"),
        ({"model": 3}, "model"),
        ({"limits": []}, "limits"),
        ({"limits": {"forward_passes": 1}}, "limits"),
        ({"limits": {**LIMITS, "code_runs": -1}}, "code_runs"),
        ({"limits": {**LIMITS, "tokens": 1.5}}, "tokens"),
        ({"model_dir": 3}, "model_dir"),
        ({"model_dir": ""}, "model_dir"),
        ({"kernel": []}, "kernel must be an object"),
        ({"kernel": {"timeout": 5}}, "kernel must be an object"),
        ({"kernel": {"timeout_s": 0}}, "timeout_s"),
        ({"kernel": {"timeout_s": True}}, "timeout_s"),
        ({"kernel": {"timeout_s": "5"}}, "timeout_s"),
        ({"kernel": {"timeout_s": float("inf")}}, "timeout_s"),
        ({"kernel": {"mem_gib": 0}}, "mem_gib"),
        ({"kernel": {"mem_gib": 2.5}}, "mem_gib"),
        ({"kernel": {"mem_gib": True}}, "mem_gib"),
        ({"forward": []}, "forward must be an object"),
        ({"forward": {"devices": "cpu"}}, "forward must be an object"),
        ({"forward": {"device": "tpu"}}, "forward.device"),
        ({"forward": {"dtype": "int8"}}, "forward.dtype"),
        ({"forward": {"max_new_tokens": 0}}, "forward.max_new_tokens"),
        ({"forward": {"max_new_tokens": 32769}}, "forward.max_new_tokens"),
        ({"forward": {"max_new_tokens": True}}, "forward.max_new_tokens"),
        ({"forward": {"max_new_tokens": 8.0}}, "forward.max_new_tokens"),
    ],
)
def test_run_config_bad(tmp_path: Path, over: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_run_config(write_run(tmp_path, **over))


def test_run_config_deeply_nested(tmp_path: Path) -> None:
    (tmp_path / "run.json").write_text("[" * 200_000)
    with pytest.raises(ValueError, match="not readable"):
        load_run_config(tmp_path)


def test_run_config_missing_garbled_and_not_object(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        load_run_config(tmp_path)
    (tmp_path / "run.json").write_text("{nope")
    with pytest.raises(ValueError, match="not readable"):
        load_run_config(tmp_path)
    (tmp_path / "run.json").write_text("[1]")
    with pytest.raises(ValueError, match="JSON object"):
        load_run_config(tmp_path)
    (tmp_path / "run.json").unlink()
    (tmp_path / "run.json").mkdir()
    with pytest.raises(ValueError, match="not readable"):
        load_run_config(tmp_path)
