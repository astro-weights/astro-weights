"""confirm survives a kill: per-item checkpoints, an interrupted query is not counted.

The kill test runs a Confirmer query on a tiny saved model in a child process,
SIGKILLs it mid-way and resumes in this process.
"""

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from astro_weights.mcp import forward
from astro_weights.mcp.confirm import Confirmer, _load_rows, _restore, generator
from astro_weights.mcp.handles import HandleStore
from astro_weights.mcp.jobs import ConfirmConfig
from astro_weights.mcp.record import RunRecord

HERE = Path(__file__).parent
HF = forward.hf_available()
PROMPTS = ["w1 w2", "w3", "w4 w5 w6", "w7", "w8 w9"]
GOLDS = ["a", "b", "c", "d", "e"]
CAP = 4
CFG = ConfirmConfig(items=Path("unused"), grader="exact", margin=0.0, max_queries=3)


def new_run(run: Path) -> Path:
    run.mkdir(parents=True)
    RunRecord.start(run / "record.jsonl", seed=0, config={}, editor="e", model=None)
    return run


def make_confirmer(run: Path, tiny: Path) -> Confirmer:
    """Confirmer on the tiny model, checkpointing under ``run`` (also used by the child)."""
    from astro_weights.mcp.edit_store import EditStore

    backend = forward.ForwardBackend(tiny, EditStore(run, tiny), device="cpu")
    items = (PROMPTS, GOLDS, "sha")
    gen = generator(backend, CAP, run)
    return Confirmer(
        CFG, items, gen, HandleStore(run), RunRecord.open(run / "record.jsonl")
    )


def save_edit(run: Path, tiny: Path) -> None:
    from mcp_tiny_hf import weight_key

    from astro_weights.mcp.edit_store import EditStore

    store = EditStore(run, tiny)
    name = weight_key(tiny, "layers.0.mlp.down_proj.weight")
    w = store.original(name)
    store.save(name, w + 0.05 * np.ones_like(w) * float(np.abs(w).mean()))


def wait_done(run: Path, handle: str) -> str:
    store = HandleStore(run)
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        state = store.get(handle).state
        if state in {"done", "failed"}:
            return state
        time.sleep(0.02)
    raise AssertionError("confirm job did not finish")


def events(run: Path, name: str) -> list[dict[str, Any]]:
    return [
        e for e in RunRecord.open(run / "record.jsonl").read() if e["event"] == name
    ]


def strip(e: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in e.items() if k not in {"time", "query"}}


CHILD = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    sys.path.insert(0, sys.argv[1])
    import astro_weights.runners.generate as g
    real, calls = g._run_one, [0]
    def slow(*a, **k):
        calls[0] += 1
        if calls[0] > 2:
            time.sleep(600)  # killed here, with two rows checkpointed
        return real(*a, **k)
    g._run_one = slow
    from test_mcp_confirm_resume import make_confirmer
    make_confirmer(Path(sys.argv[2]), Path(sys.argv[3])).submit()
    time.sleep(600)
    """
)


def lines(path: Path) -> int:
    try:
        return path.read_bytes().count(b"\n")
    except FileNotFoundError:
        return 0


@pytest.mark.skipif(not HF, reason="needs the hf extra")
@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="needs SIGKILL")
def test_kill_mid_query_then_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp_tiny_hf import save_tiny_model

    import astro_weights.runners.generate as g

    tiny = save_tiny_model(tmp_path / "tiny")
    run = new_run(tmp_path / "run")
    save_edit(run, tiny)
    ckpt = run / "confirm" / f"base-1-{CAP}.jsonl"
    proc = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(HERE), str(run), str(tiny)]
    )
    try:
        deadline = time.monotonic() + 300
        while lines(ckpt) < 2:
            assert proc.poll() is None, "child exited early"
            assert time.monotonic() < deadline, "no checkpoint rows"
            time.sleep(0.05)
        os.kill(proc.pid, signal.SIGKILL)
    finally:
        proc.kill()
        proc.wait()
    kept = ckpt.read_bytes()
    assert kept.count(b"\n") == 2
    assert [e["query"] for e in events(run, "confirm_query")] == [1]
    assert not events(run, "confirm_result")

    calls: list[str] = []
    real = g._run_one

    def counted(*a: Any, **k: Any) -> Any:
        calls.append(a[2])
        return real(*a, **k)

    monkeypatch.setattr(g, "_run_one", counted)
    confirmer = make_confirmer(run, tiny)
    assert confirmer._queries == 0  # the killed query is not counted
    make_confirmer(run, tiny)  # a second start marks it no second time
    assert [e["query"] for e in events(run, "confirm_interrupted")] == [1]
    state = wait_done(run, confirmer.submit()["handle"])
    assert state == "done"
    assert [e["query"] for e in events(run, "confirm_query")] == [1, 2]
    assert len(calls) == 2 * len(PROMPTS) - 2  # base: only the 3 not checkpointed
    assert ckpt.read_bytes().startswith(kept)
    counts = RunRecord.open(run / "record.jsonl").confirm_counts()
    assert (counts["confirm_queries"], counts["confirm_interrupted"]) == (2, 1)
    assert make_confirmer(run, tiny)._queries == 1

    clean = new_run(tmp_path / "clean")
    save_edit(clean, tiny)
    calls.clear()
    state = wait_done(clean, make_confirmer(clean, tiny).submit()["handle"])
    assert state == "done"
    assert len(calls) == 2 * len(PROMPTS)
    (got,) = events(run, "confirm_result")
    (want,) = events(clean, "confirm_result")
    assert got["query"] == 2
    assert strip(got) == strip(want)
    names = sorted(p.name for p in (clean / "confirm").iterdir())
    assert names == sorted(p.name for p in (run / "confirm").iterdir())
    for name in names:
        assert (run / "confirm" / name).read_bytes() == (
            clean / "confirm" / name
        ).read_bytes()


def fake(params: dict[str, Any]) -> dict[str, Any]:
    raise AssertionError("no generation expected")


def test_old_record_restores_as_before(tmp_path: Path) -> None:
    run = new_run(tmp_path / "run")
    rec = RunRecord.open(run / "record.jsonl")
    rec.event("confirm_query", query=1)
    rec.event("confirm_base", n=5, base_correct=2, base_tokens=40, base_hit_cap=0)
    rec.event(
        "confirm_result",
        query=1,
        passed=True,
        best=3,
        best_candidate="c1",
        candidate="c1",
        tokens=30,
    )
    rec.event("confirm_query", query=2)
    rec.event("confirm_result", query=2, passed=False, best=3, candidate="c2")
    before = (run / "record.jsonl").read_bytes()
    counts = rec.confirm_counts()
    c = Confirmer(CFG, (PROMPTS, GOLDS, "s"), fake, HandleStore(run), rec)
    assert (run / "record.jsonl").read_bytes() == before  # nothing appended
    assert not (run / "confirm").exists()
    assert (c._queries, c._last, c._bar, c.best_candidate) == (2, 2, 3, "c1")
    assert (c._st.base, c._st.bar_tokens) == ((2, 40), 30)
    assert (
        counts
        == rec.confirm_counts()
        == {
            "candidates_tried": 2,
            "candidates_saved": 0,
            "confirm_queries": 2,
            "confirm_refused": 0,
        }
    )


def test_interrupted_query_frees_the_limit_numbers_go_on(tmp_path: Path) -> None:
    run = new_run(tmp_path / "run")
    rec = RunRecord.open(run / "record.jsonl")
    rec.event("confirm_query", query=1)
    rec.event("confirm_result", query=1, passed=False, best=1, candidate="a")
    rec.event("confirm_query", query=2)  # killed
    rec.event("confirm_query", query=3)  # killed again after a resume
    st = _restore(rec.read())
    assert (st.queries, st.last) == (3, 3)  # old code would count 3
    c = Confirmer(CFG, (PROMPTS, GOLDS, "s"), fake, HandleStore(run), rec)
    assert [e["query"] for e in events(run, "confirm_interrupted")] == [2, 3]
    assert (c._queries, c._last, c._bar) == (1, 3, 1)
    assert rec.confirm_counts()["confirm_interrupted"] == 2


def test_load_rows_cuts_a_torn_line_and_skips_stale_rows(tmp_path: Path) -> None:
    path = tmp_path / "ck.jsonl"
    assert _load_rows(path, ["s0"]) == {}
    good = {"index": 0, "sha256": "s0", "text": "x", "tokens": 1, "hit_cap": False}
    rows = [
        good,
        {**good, "index": 1, "sha256": "old"},  # the prompt changed
        {**good, "index": 2, "tokens": "1"},  # bad type
        {**good, "index": 9},  # out of range
        {"index": 1},  # missing fields
    ]
    body = "".join(json.dumps(r) + "\n" for r in rows) + "not json\n"
    path.write_text(body + '{"index": 1, "sha2')
    assert _load_rows(path, ["s0", "s1", "s2"]) == {0: ("x", 1, False)}
    assert path.read_text() == body


@pytest.mark.skipif(not HF, reason="needs the hf extra")
def test_generator_batches_resume_from_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp_tiny_hf import save_tiny_model

    import astro_weights.runners.generate as g
    from astro_weights.mcp.edit_store import EditStore

    tiny = save_tiny_model(tmp_path / "tiny")
    run = tmp_path / "run"
    backend = forward.ForwardBackend(tiny, EditStore(run, tiny), device="cpu")
    backend._ensure()
    if backend._tok.pad_token_id is None:
        backend._tok.pad_token = backend._tok.eos_token
    synced: list[int] = []
    real_fsync = os.fsync

    def fsync(fd: int) -> None:
        synced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    gen = generator(backend, CAP, run)
    params = {"arm": "base", "prompts": PROMPTS, "batch_size": 2}
    first = gen(params)
    assert len(synced) == 3  # one fsync per chunk written: [0, 1], [2, 3], [4]
    ckpt = run / "confirm" / f"base-2-{CAP}.jsonl"
    full = ckpt.read_bytes()
    ckpt.write_bytes(b"".join(full.splitlines(keepends=True)[:3]))  # rows 0-2
    calls: list[int] = []
    real = g._run_batch

    def counted(*a: Any, **k: Any) -> Any:
        calls.append(len(a[2]))
        return real(*a, **k)

    monkeypatch.setattr(g, "_run_batch", counted)
    assert gen(params) == first
    assert calls == [2, 1]  # chunk [2, 3] whole again, then [4]
    assert gen(params) == first
    assert calls == [2, 1]
