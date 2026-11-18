"""Sandboxed numpy kernel: cells, edit requests, faults, restarts.

Behaviour tests run twice: in the OS sandbox (skipped where none works) and with
``allow_unsafe`` and no sandbox, which runs everywhere, Windows included. Hostile
requests are sent the way a cell can send them: through the child's own server
object, reached from a closure in the cell namespace.
"""

import ast
import contextlib
import importlib.util
import io
import itertools
import json
import os
import queue
import secrets
import socket
import sys
import threading
import time
import tracemalloc
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mcp_fakes import fake_model_dir

from astro_weights.mcp import forward, kernel_child, sandbox
from astro_weights.mcp import kernel as kmod
from astro_weights.mcp.edit_store import EditStore
from astro_weights.mcp.handles import HandleStore
from astro_weights.mcp.kernel import (
    Kernel,
    KernelError,
    child_env,
    popen_options,
    read_capped,
    read_npy,
    write_npy,
)

MODE = sandbox.probe()
needs_sandbox = pytest.mark.skipif(
    MODE is None, reason=f"no working OS sandbox: {sandbox._detect()[1]}"
)
posix = pytest.mark.skipif(sys.platform == "win32", reason="POSIX only")
N0 = "model.layers.0.mlp.down_proj.weight"  # (32, 64)
N1 = "model.layers.0.self_attn.q_proj.weight"  # (32, 32)
# the cell's handle on the child's server object, as hostile code would find it
SRV = "srv = edited.__closure__[0].cell_contents\n"

Events = list[tuple[str, dict[str, Any]]]
Make = Callable[..., tuple[Kernel, EditStore, Events]]


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return fake_model_dir(tmp_path_factory.mktemp("model"))


def _no_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_detect", lambda: (None, "disabled for this test"))


@pytest.fixture(params=["sandbox", "unsafe"])
def unsafe(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> bool:
    if request.param == "sandbox":
        if MODE is None:
            pytest.skip(f"no working OS sandbox: {sandbox._detect()[1]}")
        return False
    _no_sandbox(monkeypatch)
    return True


@pytest.fixture
def make(tmp_path: Path, model: Path) -> Iterator[Make]:
    made: list[Kernel] = []

    def build(
        unsafe: bool, run: Path | None = None, **kw: Any
    ) -> tuple[Kernel, EditStore, Events]:
        run = run or tmp_path / "run"
        store = EditStore(run, model)
        events: Events = []
        k = Kernel(
            run,
            store,
            allow_unsafe=unsafe,
            on_event=lambda n, f: events.append((n, f)),
            **kw,
        )
        made.append(k)
        return k, store, events

    yield build
    for k in made:
        k.close()


def ok(k: Kernel, code: str) -> str:
    r = k.exec(code)
    assert r["ok"], r["output"]
    return str(r["output"])


def fail(k: Kernel, code: str) -> str:
    r = k.exec(code)
    assert not r["ok"], r["output"]
    return str(r["output"])


# ---------------------------------------------------------------- cells


def test_vars_persist_and_last_expression_echo(make: Make, unsafe: bool) -> None:
    k, _, _ = make(unsafe)
    r = k.exec("x = 41")
    assert r == {"ok": True, "output": "", "restarted": False}
    assert ok(k, "x + 1") == "42\n"
    assert ok(k, "print('a')\nNone") == "a\n"
    assert ok(k, "NAMES[:1], type(np.zeros(1)).__name__") == (
        "(('model.layers.0.mlp.down_proj.weight',), 'ndarray')\n"
    )


def test_exception_and_exit_keep_kernel(make: Make, unsafe: bool) -> None:
    k, _, _ = make(unsafe)
    ok(k, "y = 3")
    out = fail(k, "raise ValueError('boom')")
    assert out.startswith("Traceback")
    assert "ValueError: boom" in out
    assert "kernel_child" not in out  # only the cell's frames
    assert "SystemExit: 3" in fail(k, "import sys; sys.exit(3)")
    assert "KeyboardInterrupt" in fail(k, "raise KeyboardInterrupt")
    assert "SyntaxError" in fail(k, "def (")
    assert ok(k, "y") == "3\n"


def test_stdout_cap_marker(make: Make, unsafe: bool) -> None:
    k, _, _ = make(unsafe, stdout_cap=100)
    out = ok(k, "print('a' * 5000)")
    assert out == "a" * 100 + "\n[output truncated at 100 characters]"
    assert ok(k, "print('é' * 10)") == "é" * 10 + "\n"


def test_fake_protocol_lines_do_not_corrupt(make: Make, unsafe: bool) -> None:
    k, _, _ = make(unsafe)
    fake = '{"done": true, "ok": true}'
    code = (
        f"import os, sys\nprint({fake!r})\nos.write(1, b'{fake}\\n')\n"
        f"os.write(2, b'{fake}\\n')\nsys.__stdout__.write({fake!r} + '\\n')\n"
        "sys.stderr.write('err\\n')\n7"
    )
    out = ok(k, code)
    assert out.count(fake) == 4
    assert out.rstrip().endswith("7")
    assert "err" in out
    assert ok(k, "8") == "8\n"


def test_timeout_kills_and_loses_vars(make: Make, unsafe: bool) -> None:
    k, _, events = make(unsafe, timeout_s=1.5)
    ok(k, "v = 1")
    assert k._proc is not None
    old = k._proc
    out = fail(k, "import time\nprint('partial', flush=True)\ntime.sleep(60)")
    assert out == "partial\n[kernel killed after 1.5 s; its variables are lost]"
    assert old.poll() is not None
    r = k.exec("'v' in globals()")
    assert r == {"ok": True, "output": "False\n", "restarted": True}
    assert ok(k, "1") == "1\n"  # restarted only once
    assert ("kernel_restarted", {}) in events


def test_child_death_reports_died(make: Make, unsafe: bool) -> None:
    k, _, _ = make(unsafe)
    ok(k, "v = 1")
    out = fail(k, "import os\nprint('bye', flush=True)\nos._exit(4)")
    assert out == "bye\n[kernel died (exit code 4); its variables are lost]"
    r = k.exec("'v' in globals()")
    assert r["restarted"]
    assert r["output"] == "False\n"


def test_death_between_cells_detected(make: Make, unsafe: bool) -> None:
    k, _, _ = make(unsafe)
    ok(
        k,
        "import threading, os, time\nv = 1\n"
        "threading.Thread(target=lambda: (time.sleep(0.3), os._exit(9))).start()",
    )
    time.sleep(1.5)
    r = k.exec("'v' in globals()")
    assert r == {"ok": True, "output": "False\n", "restarted": True}


def test_restart_marker_survives_server_restart(
    make: Make, unsafe: bool, tmp_path: Path
) -> None:
    k, _store, _ = make(unsafe)
    assert not k.exec("1")["restarted"]
    ok(k, f"W = load('{N0}'); W[0] += 0.3; save_edit('{N0}', W)")
    k.close()
    k2, _store2, events = make(unsafe)
    r = k2.exec("edited(), 'W' in globals()")
    assert r["restarted"]
    assert r["output"] == f"(['{N0}'], False)\n"
    assert not k2.exec("1")["restarted"]
    assert events == [("kernel_restarted", {})]
    k3, _, _ = make(unsafe, run=tmp_path / "other")
    assert not k3.exec("1")["restarted"]


def test_cell_cannot_hide_or_fake_a_restart(
    make: Make, unsafe: bool, tmp_path: Path
) -> None:
    """The marker lives outside the work dir; work-dir files named like it do nothing."""
    k, _, _ = make(unsafe)
    ok(k, "v = 1")
    marker = tmp_path / "run" / kmod.MARKER
    assert marker.exists()
    for name in ("kernel.pid", kmod.MARKER):
        ok(k, f"open({name!r}, 'w').close()\nimport os; os.remove({name!r})")
    if not unsafe:  # in the sandbox the real marker cannot be touched
        assert "Error" in fail(k, f"import os; os.remove({str(marker.resolve())!r})")
    k.close()
    k2, _, _ = make(unsafe)
    assert k2.exec("1")["restarted"]
    other = tmp_path / "other"
    k3, _, _ = make(unsafe, run=other)
    ok(k3, f"open({kmod.MARKER!r}, 'w').close()")  # planted in work
    k3.close()
    (other / kmod.MARKER).unlink()
    k4, _, _ = make(unsafe, run=other)
    assert not k4.exec("1")["restarted"]


def test_survives_the_thread_that_started_it(make: Make, unsafe: bool) -> None:
    """bwrap --die-with-parent fires when the starting thread ends."""
    k, _, _ = make(unsafe)
    t = threading.Thread(target=lambda: ok(k, "kept = 5"))
    t.start()
    t.join()
    time.sleep(0.5)
    r = k.exec("kept")
    assert r == {"ok": True, "output": "5\n", "restarted": False}


def test_protocol_flood_keeps_parent_memory_bounded(make: Make, unsafe: bool) -> None:
    """A cell writing requests it never reads replies to: back-pressure, then kill."""
    k, _, _ = make(unsafe, timeout_s=5)
    ok(k, "1")
    flood = (
        f"{SRV}import json\n"
        "line = json.dumps({'rpc': 'names', 'pad': 'x' * ((1 << 20) - 100)}).encode()\n"
        "for _ in range(300):\n    srv._out.write(line + b'\\n')\n"
        "srv._out.flush()\nimport time; time.sleep(60)"
    )
    tracemalloc.start()
    try:
        out = fail(k, flood)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert out.endswith("[kernel killed after 5 s; its variables are lost]")
    assert peak < 64 << 20  # 300 MiB were sent
    assert not k._reader.is_alive()
    assert ok(k, "2") == "2\n"


def test_concurrent_cells_run_one_at_a_time(make: Make, unsafe: bool) -> None:
    k, _, _ = make(unsafe)
    ok(k, "import time\nspans = []")
    cell = (
        "t0 = time.monotonic(); time.sleep(0.2); spans.append((t0, time.monotonic()))"
    )
    results: list[dict[str, Any]] = []
    threads = [
        threading.Thread(target=lambda: results.append(k.exec(cell))) for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [r["ok"] for r in results] == [True] * 4
    spans = json.loads(ok(k, "import json; print(json.dumps(sorted(spans)))"))
    assert len(spans) == 4
    assert all(a[1] <= b[0] for a, b in itertools.pairwise(spans))


def test_close(make: Make, unsafe: bool) -> None:
    k, _, _ = make(unsafe)
    ok(k, "1")
    running: list[dict[str, Any]] = []
    queued: list[BaseException] = []

    def wait_queued() -> None:
        try:
            k.exec("2")
        except RuntimeError as e:
            queued.append(e)

    t1 = threading.Thread(
        target=lambda: running.append(k.exec("import time; time.sleep(60)"))
    )
    t1.start()
    time.sleep(0.5)
    t2 = threading.Thread(target=wait_queued)
    t2.start()
    time.sleep(0.2)
    t0 = time.monotonic()
    k.close()
    t1.join(10)
    t2.join(10)
    assert time.monotonic() - t0 < 10
    assert running
    assert not running[0]["ok"]
    assert "variables are lost" in running[0]["output"]
    assert queued
    assert "closed" in str(queued[0])
    k.close()  # twice is fine
    with pytest.raises(RuntimeError, match="closed"):
        k.exec("1")


def test_exec_needs_str(make: Make) -> None:
    k, _, _ = make(True)
    with pytest.raises(TypeError):
        k.exec(b"1")  # type: ignore[arg-type]


# ---------------------------------------------------------------- edits


def test_save_edit_lands_in_store(make: Make, unsafe: bool) -> None:
    k, store, events = make(unsafe)
    out = ok(k, f"W = load('{N0}'); W[0] += 0.3; r = save_edit('{N0}', W); r")
    assert N0 in out
    assert "rel_frobenius" in out
    assert store.edited() == [N0]
    assert N0 in store.validate()
    name, fields = events[-1]
    assert name == "edit_saved"
    assert fields["tensor"] == N0
    assert 0 < fields["rel_frobenius"] < 0.5
    assert [p.name for p in k.work.glob("*_*.npy")] == []  # no files left behind


def test_rejected_edit_errors_in_cell(make: Make, unsafe: bool) -> None:
    k, store, events = make(unsafe)
    out = fail(k, f"save_edit('{N1}', load_original('{N1}') * 1.1)")
    assert "ValueError" in out
    assert "scalar multiple" in out
    assert store.edited() == []
    assert events[-1][0] == "edit_rejected"
    assert events[-1][1]["tensor"] == N1
    for code, msg in [
        ("save_edit('model.norm.weight', np.zeros(32))", "not an editable"),
        (f"save_edit('{N0}', np.zeros((3, 3)))", "does not match"),
        (f"save_edit('{N0}', np.zeros((32, 64), dtype=np.int32))", "float"),
        (f"save_edit('{N0}', np.array([[None]]))", "pickle"),
        ("load('../../etc/passwd')", "not an editable"),
    ]:
        assert msg in fail(k, code), code
    assert store.edited() == []


def test_drop_edited_load_round_trip(make: Make, unsafe: bool) -> None:
    k, store, events = make(unsafe)
    code = (
        f"W0 = load_original('{N0}')\nW = W0.copy(); W[1] += 0.25\n"
        f"save_edit('{N0}', W)\n"
        f"print(edited(), load('{N0}').dtype, np.abs(load('{N0}') - W).max() < 0.05,"
        f" bool((load('{N0}') == W0).all()), names() == list(NAMES))\n"
        f"drop_edit('{N0}')\nprint(edited(), bool((load('{N0}') == W0).all()))"
    )
    assert ok(k, code) == f"['{N0}'] float32 True False True\n[] True\n"
    assert store.edited() == []
    assert ("edit_dropped", {"tensor": N0}) in events
    assert "has no edit" in fail(k, f"drop_edit('{N0}')")


@needs_sandbox
def test_kernel_cannot_write_edits(make: Make) -> None:
    k, store, _ = make(False)
    ok(k, f"W = load('{N0}'); W[0] += 0.3; save_edit('{N0}', W)")
    before = store.validate()
    edits = store.dir
    for target in (edits / "manifest.json", edits / "x.safetensors"):
        out = fail(k, f"open({str(target)!r}, 'w').write('{{}}')")
        assert "Error" in out
    assert store.validate() == before
    assert not (edits / "x.safetensors").exists()


# ---------------------------------------------------------------- forward results


def forward_job(run: Path, keys: list[str], state: str = "done", **params: Any) -> str:
    """A forward job in ``run`` with result files for ``keys``; its handle."""
    jobs = HandleStore(run)
    out = secrets.token_hex(16)
    job = jobs.submit("forward", {"request": {}, "out": out, **params})
    d = run / forward.RESULTS / out
    d.mkdir(parents=True)
    for key in keys:
        if key.startswith("texts"):
            forward._save_texts(d / f"{key}.json", ["one", "two"])
        else:
            forward._save_array(d / f"{key}.npy", np.arange(6, dtype=np.int64) << 40)
    if state != "queued":
        jobs.start(job.handle)
    if state == "done":
        jobs.finish(job.handle, {"keys": keys})
    elif state == "failed":
        jobs.fail(job.handle, "boom")
    return job.handle


def test_forward_result_reads_arrays_and_texts(
    make: Make, unsafe: bool, tmp_path: Path
) -> None:
    run = tmp_path / "run"
    k, _, _ = make(unsafe, jobs=HandleStore(run))
    h = forward_job(run, ["logprobs_top_ids", "texts"])
    out = ok(
        k,
        f"a = forward_result('{h}', 'logprobs_top_ids')\n"
        f"print(a.dtype, a.tolist() == [i << 40 for i in range(6)])\n"
        f"forward_result('{h}', 'texts')",
    )
    assert out == "int64 True\n['one', 'two']\n"
    assert list(k.work.glob("in_*")) == []


def test_forward_result_refusals(make: Make, unsafe: bool, tmp_path: Path) -> None:
    run = tmp_path / "run"
    k, _, _ = make(unsafe, jobs=HandleStore(run))
    good = forward_job(run, ["hidden_l0"])
    queued = forward_job(run, ["hidden_l0"], state="queued")
    failed = forward_job(run, ["hidden_l0"], state="failed")
    exec_job = HandleStore(run).submit("exec", {"out": "0" * 32}).handle
    other = forward_job(tmp_path / "other", ["hidden_l0"])  # another run's job
    cases = [
        ("../../x", "hidden_l0", "bad handle"),
        (good, "../../../etc/passwd", "bad key"),
        (good, "HIDDEN", "bad key"),
        (good, None, "bad key"),
        (other, "hidden_l0", "unknown handle"),
        (exec_job, "hidden_l0", "not a forward_pass job"),
        (queued, "hidden_l0", "queued, not done"),
        (failed, "hidden_l0", "failed, not done"),
        (good, "mlp_l0", "no result"),
    ]
    for handle, key, msg in cases:
        out = fail(k, f"forward_result({handle!r}, {key!r})")
        assert msg in out, (handle, key, out)
    out = fail(k, f"{SRV}srv.call('forward_result', handle=[1], key={{}})")
    assert "bad handle" in out
    assert ok(k, f"forward_result('{good}', 'hidden_l0').shape") == "(6,)\n"


def test_forward_result_needs_jobs(make: Make) -> None:
    k, _, _ = make(True)
    assert "not available in this kernel" in fail(
        k, "forward_result('forward-0123456789abcdef', 'texts')"
    )


@needs_sandbox
def test_kernel_cannot_write_forward_results(make: Make, tmp_path: Path) -> None:
    run = tmp_path / "run"
    k, _, _ = make(False, jobs=HandleStore(run))
    h = forward_job(run, ["hidden_l0"])
    target = next((run / forward.RESULTS).iterdir()) / "hidden_l0.npy"
    before = target.read_bytes()
    assert "Error" in fail(k, f"open({str(target)!r}, 'wb').write(b'x')")
    assert target.read_bytes() == before
    assert ok(k, f"forward_result('{h}', 'hidden_l0').shape") == "(6,)\n"


# ---------------------------------------------------------------- hostile requests


def test_bad_tokens_rejected(make: Make, unsafe: bool, tmp_path: Path) -> None:
    k, store, _ = make(unsafe)
    secret = tmp_path / "secret.npy"
    np.save(secret, np.ones((32, 64), dtype=np.float32))
    for token in ("../x", "../secret", str(secret), "A" * 32, "0" * 31, 5, None):
        out = fail(k, f"{SRV}srv.call('save_edit', name='{N0}', token={token!r})")
        assert "bad array token" in out, token
    out = fail(k, f"{SRV}srv.call('save_edit', name='{N0}', token='{'0' * 32}')")
    assert "cannot open" in out
    assert store.edited() == []
    assert secret.exists()


def _planted(header_shape: str, dtype: str, extra: str = "") -> str:
    """Cell that writes a hand-made .npy as the outbox file and asks to save it."""
    return (
        f"{SRV}import secrets\nt = secrets.token_hex(16)\n"
        "f = open(f'out_{t}.npy', 'wb')\n"
        f"np.lib.format.write_array_header_1_0(f, {{'descr': {dtype!r}, "
        f"'fortran_order': False, 'shape': {header_shape}}})\n{extra}f.close()\n"
        f"srv.call('save_edit', name='{N0}', token=t)"
    )


@pytest.mark.parametrize(
    ("shape", "dtype", "msg"),
    [
        ("(10**6, 10**6)", "<f4", "shape"),
        ("(64, 32)", "<f4", "shape"),
        ("(32, 64)", "|O", "dtype"),
        ("(32, 64)", "<i4", "dtype"),
        ("(32, 64)", ">f4", "dtype"),
        ("(32, 64)", "<f4", "size does not match"),
    ],
)
def test_bad_header_rejected_before_data(
    make: Make, unsafe: bool, shape: str, dtype: str, msg: str
) -> None:
    k, store, events = make(unsafe)
    out = fail(k, _planted(shape, dtype))
    assert msg in out
    assert store.edited() == []
    assert events[-1][0] == "edit_rejected"
    assert list(k.work.glob("out_*")) == []  # the parent removed the file


def test_unknown_request_answered(make: Make, unsafe: bool) -> None:
    k, _, _ = make(unsafe)
    out = fail(k, f"{SRV}srv.call('rm_rf', name='/')")
    assert "unknown request 'rm_rf'" in out
    assert ok(k, "1") == "1\n"


@pytest.mark.parametrize(
    ("raw", "why"),
    [
        ("b'not json\\n'", "line is not JSON"),
        ("b'[1]\\n'", "line is not a JSON object"),
        ("b'{\"hello\": 1}\\n'", "unexpected message"),
        ("b'[' * 100000 + b'\\n'", "line is not JSON"),
        ("b'x' * (2 << 20)", "line longer than 1048576 bytes or cut off"),
    ],
)
def test_protocol_faults_kill_kernel(
    make: Make, unsafe: bool, raw: str, why: str
) -> None:
    k, _, _ = make(unsafe)
    ok(k, "v = 1")
    out = fail(
        k, f"{SRV}srv._out.write({raw}); srv._out.flush()\nimport time; time.sleep(30)"
    )
    assert out.endswith(
        f"protocol error: {why}; kernel killed; its variables are lost]"
    )
    assert k.exec("'v' in globals()")["restarted"]


# ---------------------------------------------------------------- sandbox


@needs_sandbox
def test_torch_and_site_packages_not_importable(make: Make) -> None:
    k, _, _ = make(False)
    assert "ModuleNotFoundError" in fail(k, "import torch")
    assert "ModuleNotFoundError" in fail(k, "import safetensors")
    assert "ModuleNotFoundError" in fail(k, "import astro_weights")


def test_parent_env_secret_absent(
    make: Make, unsafe: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-sentinel-123")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid")
    k, _, _ = make(unsafe)
    out = ok(k, "import os; print(dict(os.environ))")
    assert "sk-sentinel-123" not in out
    assert "proxy.invalid" not in out
    assert ok(k, "import os; os.environ['HOME']") == f"{str(k.work)!r}\n"


@needs_sandbox
def test_network_blocked(make: Make) -> None:
    k, _, _ = make(False)
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(8)
        port = srv.getsockname()[1]
        out = fail(
            k, f"import socket\nsocket.create_connection(('127.0.0.1', {port}), 3)"
        )
    assert "Error" in out


@needs_sandbox
def test_outside_files_not_readable(make: Make, tmp_path: Path) -> None:
    k, store, _ = make(False)
    secret = tmp_path / "secret.txt"
    secret.write_text("sk-sentinel-123")
    assert "Error" in fail(k, f"open({str(secret.resolve())!r}).read()")
    model_file = store.model_dir.resolve() / "model.safetensors"
    assert "Error" in fail(k, f"open({str(model_file)!r}, 'rb').read(8)")


@pytest.mark.skipif(MODE != "bwrap", reason="RLIMIT_AS is set on Linux only")
def test_linux_memory_limit(make: Make) -> None:
    k, _, _ = make(False, mem_gib=4)
    assert "MemoryError" in fail(k, "a = np.ones(6 << 30, dtype=np.uint8)")
    assert ok(k, "float(np.ones(1 << 20).sum())") == "1048576.0\n"


# ---------------------------------------------------------------- start-up


def test_sandbox_unavailable_refused(
    tmp_path: Path, model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_sandbox(monkeypatch)
    with pytest.raises(sandbox.SandboxUnavailable, match="disabled for this test"):
        Kernel(tmp_path / "run", EditStore(tmp_path / "run", model))


@pytest.mark.parametrize(
    ("code", "msg"),
    [
        ("import sys; sys.stderr.write('boom'); sys.exit(5)", "exit code 5): boom"),
        ("print('garbage')", "protocol error: line is not JSON"),
        ("import time; time.sleep(30)", "did not start within 0.5 s"),
    ],
)
def test_start_failures(
    make: Make, monkeypatch: pytest.MonkeyPatch, code: str, msg: str
) -> None:
    k, _, _ = make(True)
    monkeypatch.setattr(kmod, "START_TIMEOUT_S", 0.5)
    k._argv = [sys.executable, "-c", code]
    with pytest.raises(KernelError, match=msg.replace("(", r"\(").replace(")", r"\)")):
        k.exec("1")


def test_start_popen_error_then_recovers(make: Make) -> None:
    k, _, _ = make(True)
    argv = k._argv
    k._argv = [str(k.work / "no-such-program")]
    with pytest.raises(KernelError, match="failed to start"):
        k.exec("1")
    k._argv = argv
    assert ok(k, "2") == "2\n"


def test_start_cleans_planted_files(make: Make, tmp_path: Path) -> None:
    k, _, _ = make(True)
    secret = tmp_path / "secret.txt"
    secret.write_text("keep")
    k.work.mkdir(parents=True, exist_ok=True)
    stale = k.work / f"in_{'a' * 32}.npy"
    stale.write_bytes(b"x")
    with contextlib.suppress(OSError, NotImplementedError):
        (k.work / kmod.LOG).symlink_to(secret)
    ok(k, "1")
    assert secret.read_text() == "keep"
    assert not stale.exists()


def test_restart_when_old_files_cannot_be_deleted(
    make: Make, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows can hold a killed child's files locked; restart must not need unlink."""
    k, _, _ = make(True)
    ok(k, "v = 1")

    def locked(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError(13, "file in use", str(self))

    monkeypatch.setattr(Path, "unlink", locked)
    out = fail(k, f"{SRV}srv._out.write(b'not json\\n'); srv._out.flush()")
    assert out.endswith("kernel killed; its variables are lost]")
    res = k.exec("'v' in globals()")
    assert res["restarted"]
    assert res["output"] == "False\n"
    assert not (k.work / kmod.LOG).exists()


# ---------------------------------------------------------------- file helpers


def test_read_npy_accepts_good_files(tmp_path: Path) -> None:
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    for dt in ("<f2", "<f4", "<f8"):
        for version in ((1, 0), (2, 0)):
            p = tmp_path / "a.npy"
            with p.open("wb") as f:
                np.lib.format.write_array(f, a.astype(dt), version=version)
            got = read_npy(p, (2, 3))
            assert got.dtype == np.dtype(dt)
            assert (got == a).all()


def _npy_bytes() -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.zeros((2, 3), np.float32))
    return buf.getvalue()


@pytest.mark.parametrize(
    ("make_file", "msg"),
    [
        (lambda f: f.write(b"not an npy file"), "magic"),
        (
            lambda f: np.lib.format.write_array(
                f, np.zeros((2, 3), np.float32), version=(3, 0)
            ),
            "version",
        ),
        (
            lambda f: np.lib.format.write_array(
                f, np.zeros((2, 3), np.float32).T.copy().T
            ),
            "C-ordered",
        ),
        (
            lambda f: f.write(_npy_bytes() + b"extra"),
            "size does not match",
        ),
        (lambda f: f.write(b"\x93NUMPY\x01\x00\x10"), "cut short"),
        (lambda f: f.write(b"\x93NUMPY\x01\x00\x40\x00" + b" " * 60), "cut short"),
        (lambda f: f.write(b"\x93NUMPY\x01\x00\xff\xff" + b" " * 60), "over the"),
        (lambda f: f.write(b"\x93NUMPY\x01\x00\x04\x00{}  "), "correct keys"),
    ],
)
def test_read_npy_rejects(tmp_path: Path, make_file: Any, msg: str) -> None:
    p = tmp_path / "a.npy"
    with p.open("wb") as f:
        make_file(f)
    with pytest.raises(ValueError, match=msg):
        read_npy(p, (2, 3))


@posix
def test_huge_declared_header_read_bounded(tmp_path: Path) -> None:
    """numpy would read the whole declared header (here ~4 GiB) before checking it."""
    p = tmp_path / "a.npy"
    with p.open("wb") as f:
        f.write(b"\x93NUMPY\x02\x00" + (0xFFFFFFF0).to_bytes(4, "little"))
        f.truncate(3 << 30)  # sparse: no disk used
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="over the 65536 byte cap"):
            read_npy(p, (2, 3))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1 << 20


def test_read_npy_cut_short(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "a.npy"
    np.save(p, np.zeros((2, 3), np.float32))
    real = os.fstat
    seen: list[os.stat_result] = []

    def fstat(
        fd: int,
    ) -> os.stat_result:  # stale size: the file shrinks after the check
        if not seen:
            seen.append(real(fd))
            os.truncate(p, seen[0].st_size - 4)
        return seen[0]

    monkeypatch.setattr("astro_weights.mcp.kernel.os.fstat", fstat)
    with pytest.raises(ValueError, match="cut short"):
        read_npy(p, (2, 3))


@posix
def test_planted_links_and_fifos_not_followed(tmp_path: Path) -> None:
    secret = tmp_path / "secret.npy"
    np.save(secret, np.ones((2, 3), np.float32))
    (tmp_path / "secret.txt").write_text("sk-sentinel-123")
    link = tmp_path / "link.npy"
    link.symlink_to(secret)
    hard = tmp_path / "hard.npy"
    os.link(secret, hard)
    fifo = tmp_path / "fifo.npy"
    os.mkfifo(fifo)
    out_link = tmp_path / "stdout.txt"
    out_link.symlink_to(tmp_path / "secret.txt")
    with pytest.raises(ValueError, match="cannot open"):
        read_npy(link, (2, 3))
    for p in (hard, fifo, tmp_path):
        with pytest.raises(ValueError, match="not a plain file"):
            read_npy(p, (2, 3))
    assert read_capped(out_link, 100) == ""
    assert read_capped(tmp_path / "missing.txt", 100) == ""


def test_write_npy_atomic(tmp_path: Path) -> None:
    p = tmp_path / "in_x.npy"
    write_npy(p, np.ones((2, 3), np.float64))
    assert np.load(p).dtype == np.float32
    q = tmp_path / "in_y.npy"
    write_npy(q, np.ones((2, 3), np.int64), None)
    assert np.load(q).dtype == np.int64
    q.unlink()
    assert list(tmp_path.iterdir()) == [p]
    p.with_suffix(".tmp").write_bytes(b"planted")
    with pytest.raises(FileExistsError):
        write_npy(tmp_path / "in_x.npy", np.ones((2, 3)))


def test_put_failure_answers_error(make: Make, monkeypatch: pytest.MonkeyPatch) -> None:
    k, _, _ = make(True)
    monkeypatch.setattr(
        "astro_weights.mcp.kernel.secrets.token_hex", lambda n: "b" * 32
    )
    (k.work / f"in_{'b' * 32}.tmp").write_bytes(b"planted")
    reply = k._answer({"rpc": "load", "name": N0})
    assert reply["ok"] is False
    assert reply["error"].startswith("FileExistsError")


def test_event_sink_fault_ignored(make: Make) -> None:
    k, _, _ = make(True)

    def boom(name: str, fields: dict[str, Any]) -> None:
        raise OSError("disk full")

    k._on_event = boom
    k._emit("edit_dropped", {"tensor": N0})
    k._on_event = None
    k._emit("edit_dropped", {"tensor": N0})


def test_popen_options_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert popen_options("win32", 4) == {}
    assert popen_options("darwin", 4) == {"start_new_session": True}
    linux = popen_options("linux", 4)
    assert linux["start_new_session"] is True
    assert callable(linux["preexec_fn"])
    monkeypatch.setenv("SYSTEMROOT", "C:\\Windows")
    work = tmp_path.resolve()
    assert child_env(work, "win32")["SYSTEMROOT"] == "C:\\Windows"
    assert "SYSTEMROOT" not in child_env(work, "linux")


@posix
def test_kill_tolerates_gone_process() -> None:
    import subprocess

    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    proc.wait()
    kmod._kill(proc)  # group and process gone: no error


# ---------------------------------------------------------------- child, in process


def test_child_imports_only_stdlib_and_numpy() -> None:
    spec = importlib.util.find_spec("astro_weights.mcp.kernel_child")
    assert spec is not None
    assert spec.origin is not None
    tree = ast.parse(Path(spec.origin).read_text())
    mods: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            assert n.module is not None
            mods.add(n.module.split(".")[0])
    assert mods - set(sys.stdlib_module_names) == {"numpy"}


def test_run_cell_and_print_error(capsys: pytest.CaptureFixture[str]) -> None:
    ns: dict[str, Any] = {}
    kernel_child.run_cell("a = 2\na * 3", ns)
    kernel_child.run_cell("", ns)
    kernel_child.run_cell("None", ns)
    assert capsys.readouterr().out == "6\n"
    try:
        kernel_child.run_cell("1 / 0", ns)
    except ZeroDivisionError as e:
        kernel_child.print_error(e)
    out = capsys.readouterr().out
    assert "ZeroDivisionError" in out
    assert "in run_cell" not in out


def test_exit_with_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    ppids = iter([10, 10, 1])
    monkeypatch.setattr(
        "astro_weights.mcp.kernel_child.os.getppid", lambda: next(ppids)
    )
    monkeypatch.setattr("astro_weights.mcp.kernel_child.time.sleep", lambda s: None)

    class Exited(Exception):
        pass

    def fake_exit(code: int) -> None:
        raise Exited(code)

    monkeypatch.setattr("astro_weights.mcp.kernel_child.os._exit", fake_exit)
    with pytest.raises(Exited):
        kernel_child.exit_with_parent(10)


def test_child_serve_in_process(
    make: Make, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """serve() and the cell functions, against the parent's request handler."""
    run = tmp_path / "run"
    k, store, _ = make(True, jobs=HandleStore(run))
    h = forward_job(run, ["hidden_l0", "texts"])
    k.work.mkdir(parents=True, exist_ok=True)
    c2p_r, c2p_w = os.pipe()
    p2c_r, p2c_w = os.pipe()
    server = kernel_child.Server(os.fdopen(p2c_r, "rb"), os.fdopen(c2p_w, "wb"), k.work)
    parent_in, parent_out = os.fdopen(c2p_r, "rb"), os.fdopen(p2c_w, "wb")
    cells = [
        f"W = load('{N0}'); W[0] += 0.3; save_edit('{N0}', W)",
        "edited()",
        "raise SystemExit(2)",
        f"drop_edit('{N0}'); len(load_original('{N0}'))",
        f"forward_result('{h}', 'hidden_l0').shape",
        f"forward_result('{h}', 'texts')",
    ]
    done: queue.Queue[dict[str, Any]] = queue.Queue()

    def parent() -> None:
        todo = list(cells)
        for line in iter(parent_in.readline, b""):
            msg = json.loads(line)
            if "rpc" in msg:
                reply = k._answer(msg)
            elif todo:
                if "done" in msg:
                    done.put(msg)
                reply = {"code": todo.pop(0)}
            else:
                done.put(msg)
                parent_out.close()
                return
            parent_out.write(json.dumps(reply).encode() + b"\n")
            parent_out.flush()

    t = threading.Thread(target=parent)
    t.start()
    ns = kernel_child.namespace(server)
    assert ns["NAMES"] == store.names
    server.send({"ready": True})
    out = tmp_path / "out.txt"
    with out.open("wb") as f:
        kernel_child.serve(server, f.fileno(), ns)
    t.join(10)
    oks = [done.get_nowait()["ok"] for _ in range(6)]
    assert oks == [True, True, False, True, True, True]
    printed = capsys.readouterr().out
    assert f"['{N0}']" in printed
    assert "SystemExit: 2" in printed
    assert "32" in printed
    assert "(6,)" in printed
    assert "['one', 'two']" in printed
    assert store.edited() == []
    with pytest.raises(ConnectionError):
        server.call("names")


@pytest.mark.parametrize(
    ("argv", "compat"),
    [
        (["/usr/bin/bwrap", "--unshare-all", "--", "py"], True),
        (["/usr/bin/bwrap", "--unshare-all", "--disable-userns", "--", "py"], False),
        (["/usr/bin/sandbox-exec", "-p", "x", "--", "py"], False),
    ],
)
def test_sandbox_compat_event_only_for_bwrap_without_disable_userns(
    monkeypatch: pytest.MonkeyPatch, make: Make, argv: list[str], compat: bool
) -> None:
    seen: dict[str, Any] = {}

    def fake_wrap(_argv: list[str], **kw: Any) -> list[str]:
        seen.update(kw)
        return argv

    monkeypatch.setattr(sandbox, "wrap", fake_wrap)
    _, _, events = make(False, allow_nested_userns=True)
    assert seen["allow_nested_userns"] is True
    names = [n for n, _ in events]
    assert names == (["sandbox_compat"] if compat else [])
