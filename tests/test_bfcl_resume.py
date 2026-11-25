"""run_bfcl writes each record as it finishes, fsynced, and resumes after a kill."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from fake_endpoint import ok, serve

from astro_weights.eval import read_tasks
from astro_weights.eval.header import header_path, read_header
from astro_weights.runners import bfcl, endpoint
from astro_weights.runners.bfcl import BfclTask, load_bfcl, run_bfcl
from astro_weights.runners.endpoint import run_header_http

FIXTURE = Path(__file__).parent / "data" / "bfcl"
TRIALS = 2
SEED = 5


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(endpoint, "_RETRY_SLEEP", 0.0)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dest = tmp_path / "bfcl"
    shutil.copytree(FIXTURE, dest, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(bfcl, "PINS", _pins(dest))
    return dest


def _pins(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p.name != "TRIGGERED"
    }


@pytest.fixture
def tasks(root: Path) -> list[BfclTask]:
    return load_bfcl(root)


_CHILD = (
    textwrap.dedent(
        """
    import hashlib, sys
    from pathlib import Path
    from astro_weights.runners import bfcl
    from astro_weights.runners.endpoint import run_header_http

    root, url, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    bfcl.PINS = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p.name != "TRIGGERED"
    }
    tasks = bfcl.load_bfcl(root)
    bfcl.run_bfcl(
        tasks, url, "m", "base", trials=TRIALS, seed=SEED, max_new_tokens=16, out=out,
        header=run_header_http(url, "m", 16, SEED),
    )
    """
    )
    .replace("TRIALS", str(TRIALS))
    .replace("SEED", str(SEED))
)


def _run(tasks: list[BfclTask], url: str, out: Path, **kw: Any) -> list[Any]:
    args: dict[str, Any] = {
        "trials": TRIALS,
        "seed": SEED,
        "max_new_tokens": 16,
        "out": out,
        "header": run_header_http(url, "m", 16, SEED),
    }
    return run_bfcl(tasks, url, "m", kw.pop("arm", "base"), **{**args, **kw})


def _lines(path: Path) -> int:
    return path.read_bytes().count(b"\n") if path.exists() else 0


def _asked(requests: list[dict[str, Any]]) -> Counter[tuple[str, int]]:
    """(first user message, seed) of each chat request; the header's probe is left out."""
    return Counter(
        (r["body"]["messages"][0]["content"], r["body"]["seed"])
        for r in requests
        if "body" in r and r["body"]["messages"][0]["content"] != "Hi"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="needs SIGKILL")
def test_sigkill_then_resume(root: Path, tasks: list[BfclTask], tmp_path: Path) -> None:
    delay = [0.0]

    def respond(_b: dict[str, Any]) -> tuple[int, str]:
        time.sleep(delay[0])
        return ok("done")

    with serve(respond) as (url, state):
        whole = tmp_path / "whole.jsonl"
        expected = _run(tasks, url, whole)
        assert len(expected) == len(tasks) * TRIALS

        delay[0] = 0.15
        out = tmp_path / "r.jsonl"
        child = subprocess.Popen(
            [sys.executable, "-I", "-c", _CHILD, str(root), url, str(out)]
        )
        try:
            deadline = time.monotonic() + 300
            while _lines(out) < 2:
                assert child.poll() is None, "child ended before it could be killed"
                assert time.monotonic() < deadline, "child wrote no records in 300 s"
                time.sleep(0.02)
            os.kill(child.pid, signal.SIGKILL)
        finally:
            child.kill()
            child.wait()
        kept = out.read_bytes().split(b"\n")[
            :-1
        ]  # a kill can leave a partial last line
        assert 2 <= len(kept) < len(expected)
        assert header_path(out).exists()
        seen = len(state.requests)

        delay[0] = 0.0
        resumed = _run(tasks, url, out, resume=True)
        new = _asked(state.requests[seen:])

    assert resumed == expected
    assert [(r.task_id, r.trial) for r in read_tasks(out)] == [
        (t.task_id, k) for t in tasks for k in range(TRIALS)
    ]
    data = out.read_bytes()
    assert data.startswith(b"".join(line + b"\n" for line in kept))
    assert data == whole.read_bytes()
    first = {t.task_id: t.turns[0] for t in tasks}
    for line in kept:
        rec = json.loads(line)
        key = (first[rec["task_id"]], SEED + rec["trial"])
        assert new[key] == 0  # no request for a trial already written
    every = {(t.turns[0], SEED + k) for t in tasks for k in range(TRIALS)}
    written = {
        (first[json.loads(line)["task_id"]], SEED + json.loads(line)["trial"])
        for line in kept
    }
    assert set(new) == every - written  # every other trial was asked


def test_resume_equals_uninterrupted_and_skips_done(
    tasks: list[BfclTask], tmp_path: Path
) -> None:
    with serve(lambda _b: ok("done")) as (url, state):
        whole = _run(tasks, url, tmp_path / "w.jsonl")
        out = tmp_path / "r.jsonl"
        _run(tasks[:1], url, out)  # first task only, both trials
        before = out.read_bytes()
        n = len(state.requests)
        resumed = _run(tasks, url, out, resume=True)
        asked = _asked(state.requests[n:])
    assert resumed == whole
    assert out.read_bytes().startswith(before)
    assert tasks[0].turns[0] not in {k[0] for k in asked}
    assert set(asked) == {
        (t.turns[0], SEED + k) for t in tasks[1:] for k in range(TRIALS)
    }


def test_resume_a_finished_run_asks_nothing(
    tasks: list[BfclTask], tmp_path: Path
) -> None:
    out = tmp_path / "r.jsonl"
    with serve(lambda _b: ok("done")) as (url, state):
        first = _run(tasks, url, out)
        n = len(state.requests)
        again = _run(tasks, url, out, resume=True)
        assert _asked(state.requests[n:]) == Counter()
    assert again == first


def test_resume_more_trials_runs_only_the_new_ones(
    tasks: list[BfclTask], tmp_path: Path
) -> None:
    out = tmp_path / "r.jsonl"
    with serve(lambda _b: ok("done")) as (url, state):
        _run(tasks, url, out, trials=1)
        n = len(state.requests)
        recs = _run(tasks, url, out, resume=True)
        asked = _asked(state.requests[n:])
    assert len(recs) == len(tasks) * 2
    assert {k[1] for k in asked} == {SEED + 1}  # only trial 1 was asked


def test_torn_last_line_is_dropped_and_redone(
    tasks: list[BfclTask], tmp_path: Path
) -> None:
    out = tmp_path / "r.jsonl"
    with serve(lambda _b: ok("done")) as (url, _s):
        whole = _run(tasks, url, out)
        data = out.read_bytes()
        out.write_bytes(data[: len(data) - 20])  # cut inside the last record
        assert _run(tasks, url, out, resume=True) == whole
    assert out.read_bytes() == data


def test_whole_last_line_without_newline_is_kept(
    tasks: list[BfclTask], tmp_path: Path
) -> None:
    out = tmp_path / "r.jsonl"
    with serve(lambda _b: ok("done")) as (url, state):
        whole = _run(tasks, url, out)
        data = out.read_bytes()
        out.write_bytes(data[:-1])
        n = len(state.requests)
        assert _run(tasks, url, out, resume=True) == whole
        assert _asked(state.requests[n:]) == Counter()
    assert out.read_bytes() == data


def test_existing_records_need_resume(tasks: list[BfclTask], tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    with serve(lambda _b: ok("done")) as (url, _s):
        _run(tasks, url, out)
        before = out.read_bytes()
        with pytest.raises(FileExistsError, match="resume=True"):
            _run(tasks, url, out)
    assert out.read_bytes() == before


def test_resume_needs_out(tasks: list[BfclTask]) -> None:
    with pytest.raises(ValueError, match="resume=True needs out"):
        run_bfcl(tasks, "http://x", "m", "a", resume=True)


def test_resume_of_a_missing_file_runs_everything(
    tasks: list[BfclTask], tmp_path: Path
) -> None:
    out = tmp_path / "new.jsonl"
    with serve(lambda _b: ok("done")) as (url, _s):
        recs = _run(tasks, url, out, resume=True)
    assert read_tasks(out) == recs


def test_no_out_is_as_before(tasks: list[BfclTask], tmp_path: Path) -> None:
    with serve(lambda _b: ok("done")) as (url, _s):
        recs = run_bfcl(tasks, url, "m", "a", trials=2, seed=SEED, max_new_tokens=16)
    assert [(r.task_id, r.trial) for r in recs] == [
        (t.task_id, k) for t in tasks for k in range(2)
    ]
    assert list(tmp_path.glob("*.jsonl")) == []


def test_header_is_written_beside_out(tasks: list[BfclTask], tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    with serve(lambda _b: ok("done")) as (url, _s):
        head = run_header_http(url, "m", 16, SEED)
        run_bfcl(
            tasks[:1], url, "m", "a", seed=SEED, max_new_tokens=16, out=out, header=head
        )
    assert read_header(header_path(out)) == head


def test_resume_refuses_another_config_hash(
    tasks: list[BfclTask], tmp_path: Path
) -> None:
    out = tmp_path / "r.jsonl"
    with serve(lambda _b: ok("done")) as (url, state):
        _run(tasks, url, out)
        before, old = out.read_bytes(), header_path(out).read_bytes()
        other = run_header_http(url, "m", 16, SEED, serving_flags={"--parallel": "4"})
        n = len(state.requests)
        with pytest.raises(ValueError, match="config_hash"):
            _run(tasks, url, out, resume=True, header=other)
        assert _asked(state.requests[n:]) == Counter()
    assert out.read_bytes() == before
    assert header_path(out).read_bytes() == old


def test_resume_refuses_a_call_the_header_file_disagrees_with(
    tasks: list[BfclTask], tmp_path: Path
) -> None:
    out = tmp_path / "r.jsonl"
    with serve(lambda _b: ok("done")) as (url, _s):
        _run(tasks, url, out)
        with pytest.raises(ValueError, match="seed"):
            _run(tasks, url, out, resume=True, seed=SEED + 1, header=None)
        with pytest.raises(ValueError, match="max_new_tokens"):
            _run(tasks, url, out, resume=True, max_new_tokens=17, header=None)


def test_header_must_agree_with_the_call_only_on_resume_too(
    tasks: list[BfclTask], tmp_path: Path
) -> None:
    out = tmp_path / "r.jsonl"
    with serve(lambda _b: ok("done")) as (url, _s):
        _run(tasks, url, out)
        n = out.stat().st_size
        _run(tasks, url, out, resume=True, header=None)  # the file's own header is kept
    assert out.stat().st_size == n


def test_resume_refuses_another_arm(tasks: list[BfclTask], tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    with serve(lambda _b: ok("done")) as (url, _s):
        _run(tasks, url, out, arm="base")
        before = out.read_bytes()
        with pytest.raises(ValueError, match="arm 'base'"):
            _run(tasks, url, out, resume=True, arm="edit")
    assert out.read_bytes() == before


def test_resume_refuses_an_unknown_task(tasks: list[BfclTask], tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    with serve(lambda _b: ok("done")) as (url, _s):
        _run(tasks, url, out)
        with pytest.raises(ValueError, match="not in tasks"):
            _run(tasks[1:], url, out, resume=True)


def test_resume_refuses_a_trial_beyond_trials(
    tasks: list[BfclTask], tmp_path: Path
) -> None:
    out = tmp_path / "r.jsonl"
    with serve(lambda _b: ok("done")) as (url, _s):
        _run(tasks, url, out)
        with pytest.raises(ValueError, match=r"outside trials 0\.\.0"):
            _run(tasks, url, out, resume=True, trials=1)


def test_resume_refuses_a_duplicate_record(
    tasks: list[BfclTask], tmp_path: Path
) -> None:
    out = tmp_path / "r.jsonl"
    with serve(lambda _b: ok("done")) as (url, _s):
        _run(tasks, url, out)
        first = out.read_bytes().split(b"\n")[0] + b"\n"
        with out.open("ab") as f:
            f.write(first)
        with pytest.raises(ValueError, match="appears twice"):
            _run(tasks, url, out, resume=True)


def test_resume_refuses_a_changed_task(tasks: list[BfclTask], tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    with serve(lambda _b: ok("done")) as (url, _s):
        _run(tasks, url, out)
        lines = out.read_bytes().split(b"\n")
        rec = json.loads(lines[0])
        rec["prompt_hash"] = "0" * 64
        lines[0] = json.dumps(rec).encode()
        out.write_bytes(b"\n".join(lines))
        with pytest.raises(ValueError, match="differs"):
            _run(tasks, url, out, resume=True)


def test_a_refused_resume_leaves_a_torn_file_alone(
    tasks: list[BfclTask], tmp_path: Path
) -> None:
    out = tmp_path / "r.jsonl"
    with serve(lambda _b: ok("done")) as (url, _s):
        _run(tasks, url, out, arm="base")
        torn = out.read_bytes()[:-20]
        out.write_bytes(torn)
        with pytest.raises(ValueError, match="arm"):
            _run(tasks, url, out, resume=True, arm="edit")
    assert out.read_bytes() == torn


def test_server_error_keeps_the_records_already_written(
    tasks: list[BfclTask], tmp_path: Path
) -> None:
    out = tmp_path / "r.jsonl"
    calls = [0]

    def respond(_b: dict[str, Any]) -> tuple[int, str]:
        calls[0] += 1
        return ok("done") if calls[0] <= 4 else (500, "no")

    with (
        serve(respond) as (url, _s),
        pytest.raises(RuntimeError, match="server error"),
    ):
        run_bfcl(tasks, url, "m", "a", trials=2, max_new_tokens=16, out=out)
    assert len(read_tasks(out)) == 2


def test_run_fsyncs_each_record(
    tasks: list[BfclTask], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = os.fsync

    def counting(fd: int) -> None:
        calls.append(fd)
        real(fd)

    monkeypatch.setattr(os, "fsync", counting)
    with serve(lambda _b: ok("done")) as (url, _s):
        recs = run_bfcl(
            tasks, url, "m", "a", trials=2, max_new_tokens=16, out=tmp_path / "r.jsonl"
        )
    assert len(calls) >= len(recs) == len(tasks) * 2


def test_header_must_agree_with_this_call(
    tasks: list[BfclTask], tmp_path: Path
) -> None:
    out = tmp_path / "r.jsonl"
    with serve(lambda _b: ok("done")) as (url, state):
        head = run_header_http(url, "m", 32, SEED)
        n = len(state.requests)
        with pytest.raises(ValueError, match="max_new_tokens"):
            run_bfcl(
                tasks, url, "m", "a", seed=SEED, max_new_tokens=16, out=out, header=head
            )
        with pytest.raises(ValueError, match="seed"):
            run_bfcl(
                tasks, url, "m", "a", seed=SEED + 1, max_new_tokens=32, header=head
            )
        assert _asked(state.requests[n:]) == Counter()
    assert not out.exists()
    assert not header_path(out).exists()
