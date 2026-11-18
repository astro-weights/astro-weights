"""OS sandbox: command builders on any OS, escape attempts where a sandbox works.

Each escape test runs the attack inside ``wrap`` and expects it to fail, and runs the
same attack unsandboxed as a control, so a test that cannot see a leak fails too.
"""

import ctypes
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from astro_weights.mcp import sandbox
from astro_weights.mcp.sandbox import (
    SandboxUnavailable,
    bwrap_command,
    child_env,
    harden_parent,
    hidden_site_dirs,
    probe,
    python_paths,
    seatbelt_command,
    wrap,
)

NAMESPACES = [
    "--unshare-all",
    "--unshare-user",
    "--disable-userns",
    "--die-with-parent",
    "--new-session",
]
MODE = probe()
posix_paths = pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX paths")
needs_sandbox = pytest.mark.skipif(
    MODE is None, reason=f"no working OS sandbox: {sandbox._detect()[1]}"
)
linux_sandbox = pytest.mark.skipif(MODE != "bwrap", reason="needs a working bwrap")
macos_sandbox = pytest.mark.skipif(MODE != "seatbelt", reason="needs sandbox-exec")
_numpy = importlib.util.find_spec("numpy")
assert _numpy is not None
assert _numpy.origin is not None
SITE = Path(_numpy.origin).resolve().parent.parent
# numpy must import in every run, so a blocked attack never means a broken child
PRELUDE = f"import sys; sys.path.insert(0, {str(SITE)!r}); import numpy\n"
HOSTILE = 'we"ird)\n(allow default) =x'


# ---------------------------------------------------------------- helpers


@pytest.fixture
def fresh_probe() -> Iterator[None]:
    sandbox._detect.cache_clear()
    sandbox._detect_nested_userns.cache_clear()
    yield
    sandbox._detect.cache_clear()
    sandbox._detect_nested_userns.cache_clear()


def _attack(code: str) -> str:
    """Python source printing ESCAPED if ``code`` runs, else BLOCKED <error>."""
    body = "\n".join("    " + line for line in code.strip().splitlines())
    return (
        f"{PRELUDE}try:\n{body}\n    print('ESCAPED')\n"
        "except Exception as e:\n    print('BLOCKED', type(e).__name__, e)\n"
    )


def _run(cmd: list[str], work: Path) -> subprocess.CompletedProcess[str]:
    (work / "tmp").mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        cmd,
        env=child_env(work),
        cwd=work,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _inside(
    code: str, work: Path, ro: tuple[Path, ...] = ()
) -> subprocess.CompletedProcess[str]:
    argv = [sys.executable, "-I", "-S", "-c", code]
    return _run(wrap(argv, ro=[*python_paths(), *ro], rw=work), work)


def _outside(code: str, work: Path) -> subprocess.CompletedProcess[str]:
    return _run([sys.executable, "-I", "-S", "-c", code], work)


def _blocked(code: str, work: Path, ro: tuple[Path, ...] = ()) -> None:
    """``code`` escapes unsandboxed (control) and is blocked in the sandbox."""
    control = _outside(_attack(code), work)
    assert control.stdout.startswith("ESCAPED"), control.stdout + control.stderr
    r = _inside(_attack(code), work, ro)
    assert r.stdout.startswith("BLOCKED"), r.stdout + r.stderr


@pytest.fixture
def work(tmp_path: Path) -> Path:
    w = tmp_path / "work"
    w.mkdir()
    return w.resolve()


# ---------------------------------------------------------------- builders


@posix_paths
def test_bwrap_command_flags_and_order() -> None:
    ro = [Path("/opt/py"), Path("/sp/x/numpy")]
    hide = [Path("/sp/x")]
    cmd = bwrap_command(["/opt/py/bin/python", "-c", "1"], ro, Path("/w"), hide)
    assert cmd[:6] == ["bwrap", *NAMESPACES]
    assert cmd[-4:] == ["--", "/opt/py/bin/python", "-c", "1"]
    s = " ".join(cmd)
    assert "--proc /proc" in s
    assert "--dev /dev" in s
    assert "--ro-bind-try /usr /usr" in s
    assert "--ro-bind-try /etc/ld.so.cache /etc/ld.so.cache" in s
    for absent in (
        "/sys",
        "/run",
        "/var",
        "/home",
        "/tmp",
        "--share-net",
        "--dev-bind",
    ):
        assert absent not in cmd
    # rw bound last, then everything but rw read-only
    tail = "--bind /w /w --remount-ro /dev --remount-ro / --chdir /w --"
    assert tail in s
    assert cmd.index("/opt/py") < cmd.index("--bind")
    i_hide = cmd.index("/sp/x")
    assert cmd[i_hide - 1] == "--tmpfs"
    rebinds = [i for i, a in enumerate(cmd) if a == "/sp/x/numpy"]
    assert len(rebinds) == 4  # bound, then bound again after the tmpfs hides it
    assert rebinds[0] < i_hide < rebinds[2]
    i_ro = max(i for i, a in enumerate(cmd) if a == "/sp/x")
    assert cmd[i_ro - 1] == "--remount-ro"
    assert rebinds[2] < i_ro


@posix_paths
def test_seatbelt_profile_rules_and_params() -> None:
    ro = [Path("/opt/py"), Path("/sp/x/numpy")]
    cmd = seatbelt_command(
        ["/opt/py/bin/python", "-c", "1"], ro, Path("/w"), [Path("/sp/x")]
    )
    assert cmd[:2] == ["/usr/bin/sandbox-exec", "-p"]
    profile = cmd[2]
    assert profile.startswith("(version 1)\n(deny default)\n")
    lines = [ln for ln in profile.splitlines() if not ln.startswith(";")]
    rules = "\n".join(lines)
    # deny default covers these; no rule may re-allow them
    for op in ("network", "iokit", "mach-lookup", "process-fork", "signal", "ipc"):
        assert op not in rules
    assert "/opt" not in profile
    assert "/w" not in profile
    deny = next(i for i, ln in enumerate(lines) if ln.startswith("(deny file-read*"))
    allow_r = next(i for i, ln in enumerate(lines) if '"R0"' in ln)
    allow_i = next(i for i, ln in enumerate(lines) if '"I0"' in ln)
    assert allow_r < deny < allow_i  # later Seatbelt rules win
    assert lines[-1] == '(allow file-read* file-write* (subpath (param "RW")))'
    defs = dict(cmd[i + 1].split("=", 1) for i, a in enumerate(cmd) if a == "-D")
    assert defs["EXE"] == "/opt/py/bin/python"
    assert defs["RW"] == "/w"
    assert defs["R0"] == "/opt/py"
    assert defs["R1"] == "/sp/x/numpy"
    assert defs["H0"] == "/sp/x"
    assert defs["I0"] == "/sp/x/numpy"
    assert {defs["P0"], defs["P1"]} == {"/opt", "/sp/x"}
    assert "/" in defs.values()  # ancestors for metadata include the root
    assert cmd[-4:] == ["--", "/opt/py/bin/python", "-c", "1"]


@posix_paths
def test_seatbelt_allows_framework_app_exec(tmp_path: Path) -> None:
    """python.org framework builds re-exec Resources/Python.app at start."""
    exe = tmp_path / "bin" / "python3.12"
    app = tmp_path / "Resources/Python.app/Contents/MacOS/Python"
    for f in (exe, app):
        f.parent.mkdir(parents=True)
        f.write_text("")
    cmd = seatbelt_command([str(exe)], [tmp_path], tmp_path / "w")
    defs = dict(cmd[i + 1].split("=", 1) for i, a in enumerate(cmd) if a == "-D")
    assert defs["EXE2"] == str(app)
    assert '(allow process-exec (literal (param "EXE2")))' in cmd[2]
    app.unlink()
    cmd = seatbelt_command([str(exe)], [tmp_path], tmp_path / "w")
    assert "EXE2" not in cmd[2]


@posix_paths
def test_seatbelt_hostile_path_stays_a_parameter() -> None:
    rw = Path("/tmp") / HOSTILE
    cmd = seatbelt_command(["/bin/x"], [], rw)
    assert HOSTILE not in cmd[2]
    assert f"RW={rw}" in cmd


@pytest.mark.parametrize(
    "call",
    [
        lambda: bwrap_command(["/x"], [Path("rel")], Path("/w")),
        lambda: bwrap_command(["/x"], [], Path("w")),
        lambda: seatbelt_command(["/x"], [], Path("w")),
        lambda: seatbelt_command(["x"], [], Path("/w")),
        lambda: seatbelt_command(["/x"], [], Path("/w\0")),
        lambda: seatbelt_command([], [], Path("/w")),
        lambda: wrap([sys.executable], ro=[Path("rel")], rw=Path("/")),
        lambda: wrap([sys.executable], ro=[], rw=Path("rel")),
        lambda: wrap(["python"], ro=[], rw=Path("/")),
        lambda: wrap([], ro=[], rw=Path("/")),
        lambda: child_env(Path("rel")),
    ],
)
def test_relative_or_bad_path_rejected(call: Any) -> None:
    with pytest.raises(ValueError, match=r"absolute|NUL|empty"):
        call()


def test_wrap_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        wrap([sys.executable], ro=[tmp_path / "nope"], rw=tmp_path)


# ---------------------------------------------------------------- probe, wrap


def _fake_run(rc: int, calls: list[list[str]]) -> Any:
    def run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, rc, "", "denied by policy")

    return run


@pytest.mark.usefixtures("fresh_probe")
def test_probe_linux_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(sandbox, "_platform", lambda: "linux")
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/bwrap")
    monkeypatch.setattr(subprocess, "run", _fake_run(0, calls))
    assert probe() == "bwrap"
    assert probe() == "bwrap"
    assert len(calls) == 1
    # the probe runs the same namespace flags as wrap, --disable-userns included
    assert calls[0][:6] == ["bwrap", *NAMESPACES]


@pytest.mark.usefixtures("fresh_probe")
def test_probe_linux_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_platform", lambda: "linux")
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/bwrap")
    monkeypatch.setattr(subprocess, "run", _fake_run(1, []))
    assert probe() is None
    assert "denied by policy" in sandbox._detect()[1]


@pytest.mark.usefixtures("fresh_probe")
def test_probe_linux_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_platform", lambda: "linux")
    monkeypatch.setattr(shutil, "which", lambda _: None)
    assert probe() is None
    assert "not installed" in sandbox._detect()[1]


@pytest.mark.usefixtures("fresh_probe")
@pytest.mark.parametrize("err", [OSError("boom"), subprocess.TimeoutExpired("x", 30)])
def test_probe_run_error(monkeypatch: pytest.MonkeyPatch, err: Exception) -> None:
    def run(*_: Any, **__: Any) -> None:
        raise err

    monkeypatch.setattr(sandbox, "_platform", lambda: "linux")
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/bwrap")
    monkeypatch.setattr(subprocess, "run", run)
    assert probe() is None
    assert "failed to run" in sandbox._detect()[1]


@pytest.mark.usefixtures("fresh_probe")
def test_probe_macos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    exe = tmp_path / "sandbox-exec"
    monkeypatch.setattr(sandbox, "_platform", lambda: "darwin")
    monkeypatch.setattr(sandbox, "SANDBOX_EXEC", str(exe))
    assert probe() is None
    assert "missing" in sandbox._detect()[1]
    sandbox._detect.cache_clear()
    exe.touch()
    monkeypatch.setattr(subprocess, "run", _fake_run(0, calls))
    assert probe() == "seatbelt"
    assert calls[0][0] == str(exe)


@pytest.mark.usefixtures("fresh_probe")
def test_probe_other_os(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_platform", lambda: "win32")
    assert probe() is None
    assert "win32" in sandbox._detect()[1]


def test_wrap_refuses_without_sandbox(
    monkeypatch: pytest.MonkeyPatch, work: Path
) -> None:
    monkeypatch.setattr(sandbox, "_detect", lambda: (None, "bwrap is not installed"))
    argv = [sys.executable, "-c", "1"]
    with pytest.raises(SandboxUnavailable, match=r"not installed.*allow_unsafe=True"):
        wrap(argv, ro=[], rw=work)
    assert wrap(argv, ro=[], rw=work, allow_unsafe=True) == argv


@pytest.mark.parametrize("mode", ["bwrap", "seatbelt"])
def test_wrap_builds_for_mode(
    monkeypatch: pytest.MonkeyPatch, work: Path, mode: str
) -> None:
    monkeypatch.setattr(sandbox, "_detect", lambda: (mode, ""))
    cmd = wrap(
        [sys.executable, "-c", "1"], ro=python_paths(), rw=work, allow_unsafe=True
    )
    real = str(Path(sys.executable).resolve())
    assert cmd[-3:] == [real, "-c", "1"]
    assert str(work) in cmd if mode == "bwrap" else f"RW={work}" in cmd


# ---------------------------------------------------------------- paths, env


def test_python_paths() -> None:
    paths = python_paths()
    assert Path(sys.executable).resolve() in paths
    assert SITE / "numpy" in paths
    assert SITE not in paths
    assert all(p.is_absolute() and p.exists() for p in paths)
    with pytest.raises(ModuleNotFoundError, match="not installed"):
        python_paths(("no_such_package_xyz",))


def test_hidden_site_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    std = tmp_path / "lib"
    (std / "site-packages").mkdir(parents=True)
    (std / "dist-packages").mkdir()
    explicit = tmp_path / "other" / "site-packages"
    explicit.mkdir(parents=True)
    monkeypatch.setattr(sandbox, "_platform", lambda: "darwin")
    found = hidden_site_dirs([std.resolve(), explicit.resolve(), tmp_path / "missing"])
    assert found == [
        (std / "site-packages").resolve(),
        (std / "dist-packages").resolve(),
    ]
    monkeypatch.setattr(sandbox, "_platform", lambda: "linux")
    assert set(found) <= set(hidden_site_dirs([std.resolve()]))


def test_child_env_allowlist(monkeypatch: pytest.MonkeyPatch, work: Path) -> None:
    for k in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "HTTPS_PROXY",
        "PYTHONPATH",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        monkeypatch.setenv(k, "sk-sentinel-123")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.delenv("LC_ALL", raising=False)
    env = child_env(work)
    assert env == {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "HOME": str(work),
        "TMPDIR": str(work / "tmp"),
        "CUDA_VISIBLE_DEVICES": "",
    }


class _Libc:
    def __init__(self, rc: int) -> None:
        self.rc = rc
        self.args: tuple[int, ...] = ()

    def prctl(self, *args: int) -> int:
        self.args = args
        return self.rc


def test_harden_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_platform", lambda: "darwin")
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: pytest.fail("called"))
    harden_parent()  # no-op off Linux
    monkeypatch.setattr(sandbox, "_platform", lambda: "linux")
    ok = _Libc(0)
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: ok)
    harden_parent()
    assert ok.args == (4, 0, 0, 0, 0)
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: _Libc(-1))
    with pytest.raises(OSError, match="PR_SET_DUMPABLE"):
        harden_parent()


@pytest.mark.skipif(sys.platform != "linux", reason="prctl and /proc are Linux only")
def test_harden_parent_hides_environ_from_child() -> None:
    child = (
        "import os\ntry:\n    open(f'/proc/{os.getppid()}/environ','rb').read()\n"
        "    print('READ')\nexcept PermissionError:\n    print('DENIED')"
    )
    parent = (
        "import subprocess, sys\n"
        "from astro_weights.mcp.sandbox import harden_parent\n"
        "if sys.argv[1] == '1': harden_parent()\n"
        f"subprocess.run([sys.executable, '-c', {child!r}])"
    )
    out = [
        subprocess.run(
            [sys.executable, "-c", parent, flag],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for flag in ("0", "1")
    ]
    assert out == ["READ", "DENIED"]


# ---------------------------------------------------------------- escapes


@needs_sandbox
def test_numpy_runs_and_rw_is_writable(work: Path) -> None:
    code = (
        f"{PRELUDE}import numpy as np, os\n"
        "a = np.random.default_rng(0).standard_normal((64, 64))\n"
        "print(round(float(np.trace(np.linalg.inv(a) @ a)), 6))\n"
        "open('out.txt', 'w').write('ok')\n"
        "import tempfile; tempfile.NamedTemporaryFile().close()\n"
        "print(os.getcwd())"
    )
    r = _inside(code, work)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["64.0", str(work)]
    assert (work / "out.txt").read_text() == "ok"


@needs_sandbox
def test_hostile_rw_name_still_sandboxed(tmp_path: Path) -> None:
    w = tmp_path / HOSTILE
    w.mkdir()
    w = w.resolve()
    secret = tmp_path / "secret.txt"
    secret.write_text("sk-sentinel-123")
    r = _inside(f"{PRELUDE}open('ok.txt', 'w').write('ok')", w)
    assert r.returncode == 0, r.stderr
    assert (w / "ok.txt").read_text() == "ok"
    _blocked(f"open({str(secret)!r}).read()", w)


@needs_sandbox
def test_tcp_to_parent_listener_blocked(work: Path) -> None:
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(8)
        port = srv.getsockname()[1]
        _blocked(
            f"import socket\ns = socket.create_connection(('127.0.0.1', {port}), 3)",
            work,
        )


@needs_sandbox
def test_unix_socket_outside_rw_blocked(work: Path) -> None:
    d = Path(tempfile.mkdtemp(dir="/tmp")).resolve()  # short: sun_path is ~104 bytes
    path = d / "s.sock"
    try:
        with socket.socket(socket.AF_UNIX) as srv:
            srv.bind(str(path))
            srv.listen(8)
            _blocked(
                "import socket\ns = socket.socket(socket.AF_UNIX)\n"
                f"s.connect({str(path)!r})",
                work,
            )
    finally:
        path.unlink(missing_ok=True)
        d.rmdir()


@linux_sandbox
def test_abstract_unix_socket_blocked(work: Path) -> None:
    name = f"\0aw-{uuid.uuid4().hex}"
    with socket.socket(socket.AF_UNIX) as srv:
        srv.bind(name)
        srv.listen(8)
        _blocked(
            f"import socket\ns = socket.socket(socket.AF_UNIX)\ns.connect({name!r})",
            work,
        )


@needs_sandbox
def test_read_outside_ro_rw_blocked(tmp_path: Path, work: Path) -> None:
    secret = tmp_path / "secret" / "key.txt"
    secret.parent.mkdir()
    secret.write_text("sk-sentinel-123")
    _blocked(f"open({str(secret.resolve())!r}).read()", work)
    _blocked(f"import os\nos.listdir({str(secret.parent.resolve())!r})", work)


@needs_sandbox
def test_write_outside_rw_blocked(tmp_path: Path, work: Path) -> None:
    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    ro_dir = ro_dir.resolve()
    other = tmp_path.resolve()
    _blocked(f"open({str(other / 'x.txt')!r}, 'w').write('x')", work)
    (other / "x.txt").unlink()  # the control wrote it
    _blocked(f"open({str(ro_dir / 'x.txt')!r}, 'w').write('x')", work, (ro_dir,))
    (ro_dir / "x.txt").unlink()
    numpy_dir = SITE / "numpy"
    r = _inside(_attack(f"open({str(numpy_dir / 'aw_x.txt')!r}, 'w').write('x')"), work)
    assert r.stdout.startswith("BLOCKED"), r.stdout + r.stderr
    assert not (numpy_dir / "aw_x.txt").exists()
    assert not (other / "x.txt").exists()
    assert not (ro_dir / "x.txt").exists()


@linux_sandbox
@pytest.mark.parametrize("target", ["/aw_x", "/dev/aw_x", "/dev/shm/aw_x", "/tmp/aw_x"])
def test_linux_no_writable_memory_fs(work: Path, target: str) -> None:
    """bwrap's own root and /dev are tmpfs: writes would use RAM no rlimit counts."""
    r = _inside(_attack(f"open({target!r}, 'w').write('x' * 1024)"), work)
    assert r.stdout.startswith("BLOCKED"), r.stdout + r.stderr


@needs_sandbox
def test_site_dir_under_ro_is_hidden(tmp_path: Path, work: Path) -> None:
    """A stdlib-like ro dir hides its site-packages, except packages listed in ro."""
    std = tmp_path / "std"
    for pkg in ("aw_good", "aw_evil"):
        (std / "site-packages" / pkg).mkdir(parents=True)
        (std / "site-packages" / pkg / "__init__.py").write_text("X = 1\n")
    std = std.resolve()
    good = std / "site-packages" / "aw_good"
    sp = str(std / "site-packages")
    ro = (std, good)
    r = _inside(f"{PRELUDE}sys.path.insert(0, {sp!r})\nimport aw_good", work, ro)
    assert r.returncode == 0, r.stderr
    control = _outside(_attack(f"sys.path.insert(0, {sp!r})\nimport aw_evil"), work)
    assert control.stdout.startswith("ESCAPED")
    r = _inside(_attack(f"sys.path.insert(0, {sp!r})\nimport aw_evil"), work, ro)
    assert r.stdout.startswith("BLOCKED"), r.stdout + r.stderr


@needs_sandbox
def test_unlisted_package_not_importable(work: Path) -> None:
    name = f"aw_fake_torch_{uuid.uuid4().hex[:8]}"
    pkg = SITE / name
    try:
        pkg.mkdir()
        (pkg / "__init__.py").write_text("X = 1\n")
    except OSError:
        pytest.skip("site-packages is not writable")
    try:
        _blocked(f"import {name}", work)
    finally:
        shutil.rmtree(pkg)
    if importlib.util.find_spec("torch") is not None:
        _blocked("import torch", work)


@needs_sandbox
def test_env_has_no_secret(monkeypatch: pytest.MonkeyPatch, work: Path) -> None:
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HTTPS_PROXY", "PYTHONPATH"):
        monkeypatch.setenv(k, "sk-sentinel-123")
    r = _inside("import json, os; print(json.dumps(dict(os.environ)))", work)
    env = json.loads(r.stdout)
    assert "sk-sentinel-123" not in r.stdout
    assert env["HOME"] == str(work)
    assert env["CUDA_VISIBLE_DEVICES"] == ""


@linux_sandbox
def test_linux_hides_devices_procs_and_host_dirs(work: Path) -> None:
    token = f"aw-sentinel-{uuid.uuid4().hex}"
    host = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)", token]
    )
    code = (
        "import glob, json, os\n"
        "cmds = []\n"
        "for p in glob.glob('/proc/[0-9]*/cmdline'):\n"
        "    try: cmds.append(open(p, 'rb').read().decode(errors='replace'))\n"
        "    except OSError: pass\n"
        f"token = {token[::-1]!r}[::-1]  # reversed: not in this process's own argv\n"
        "print(json.dumps(dict(gpu=glob.glob('/dev/nvidia*') + glob.glob('/dev/dri'),\n"
        "    host=any(token in c for c in cmds), n=len(cmds), ppid=os.getppid(),\n"
        "    seen=[p for p in ('/sys', '/run', '/var', '/home', '/root',\n"
        "        '/var/run/docker.sock', '/run/docker.sock', os.path.expanduser('~'))\n"
        "        if os.path.exists(p) and p != os.environ['HOME']])))"
    )
    try:
        control = json.loads(_outside(code, work).stdout)
        out = json.loads(_inside(code, work).stdout)
    finally:
        host.kill()
        host.wait()
    assert control["host"]
    assert out["gpu"] == []
    assert not out["host"]
    assert out["n"] <= 3
    assert out["ppid"] <= 1
    # /home may exist only as the bare path to a bound file (venv under home)
    assert [p for p in out["seen"] if p not in ("/home", "/root")] == []


@linux_sandbox
def test_child_dies_with_parent(work: Path) -> None:
    beat = work / "beat"
    child = wrap(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            (
                "import time\nwhile True:\n    open('beat', 'w').write(str(time.time()))\n"
                "    time.sleep(0.05)"
            ),
        ],
        ro=python_paths(),
        rw=work,
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", f"import subprocess; subprocess.run({child!r})"],
        env=child_env(work),
        cwd=work,
    )
    deadline = time.monotonic() + 30
    while not beat.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert beat.exists()
    os.kill(parent.pid, signal.SIGKILL)
    parent.wait()
    time.sleep(1)
    last = beat.read_text()
    time.sleep(1)
    assert beat.read_text() == last


_METAL = (
    "import ctypes\n"
    "m = ctypes.CDLL('/System/Library/Frameworks/Metal.framework/Metal')\n"
    "m.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p\n"
    "assert m.MTLCreateSystemDefaultDevice(), 'no Metal device'"
)
_IOSURFACE = (
    "import ctypes\n"
    "k = ctypes.CDLL('/System/Library/Frameworks/IOKit.framework/IOKit')\n"
    "k.IOServiceMatching.restype = ctypes.c_void_p\n"
    "k.IOServiceMatching.argtypes = [ctypes.c_char_p]\n"
    "k.IOServiceGetMatchingService.restype = ctypes.c_uint\n"
    "k.IOServiceGetMatchingService.argtypes = [ctypes.c_uint, ctypes.c_void_p]\n"
    "k.IOServiceOpen.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,\n"
    "    ctypes.POINTER(ctypes.c_uint)]\n"
    "task = ctypes.c_uint.in_dll(ctypes.CDLL(None), 'mach_task_self_').value\n"
    "svc = k.IOServiceGetMatchingService(0, k.IOServiceMatching(b'IOSurfaceRoot'))\n"
    "conn = ctypes.c_uint(0)\n"
    "kr = k.IOServiceOpen(svc, task, 0, ctypes.byref(conn))\n"
    "assert kr == 0, f'IOServiceOpen returned {kr:#x}'"
)


@macos_sandbox
@pytest.mark.parametrize("code", [_METAL, _IOSURFACE], ids=["metal", "iosurface"])
def test_macos_gpu_not_openable(work: Path, code: str) -> None:
    control = _outside(_attack(code), work)
    if not control.stdout.startswith("ESCAPED"):
        pytest.skip(f"no GPU service on this Mac: {control.stdout.strip()}")
    r = _inside(_attack(code), work)
    # Metal's init may trap (SIGTRAP) when denied, instead of returning NULL
    assert "ESCAPED" not in r.stdout
    assert r.stdout.startswith("BLOCKED") or r.returncode < 0, r.stdout + r.stderr


_SYSCTL = """
import ctypes, os
libc = ctypes.CDLL(None, use_errno=True)
def sysctl(mib):
    arr = (ctypes.c_int * len(mib))(*mib)
    size = ctypes.c_size_t(0)
    if libc.sysctl(arr, len(mib), None, ctypes.byref(size), None, 0) != 0:
        return None
    buf = ctypes.create_string_buffer(size.value)
    if libc.sysctl(arr, len(mib), buf, ctypes.byref(size), None, 0) != 0:
        return None
    return buf.raw[: size.value]
"""


@macos_sandbox
def test_macos_parent_argv_env_and_process_list_hidden(work: Path) -> None:
    """KERN_PROCARGS2 shows another process's exec-time argv and env, KERN_PROC_ALL
    the process list. The secret is in the intermediate parent's exec-time env."""
    secret = f"sk-parent-{uuid.uuid4().hex}"
    code = (
        PRELUDE
        + _SYSCTL
        + (
            f"secret = {secret[::-1]!r}[::-1].encode()  # reversed: not in any argv\n"
            "args = sysctl([1, 49, os.getppid()])  # CTL_KERN, KERN_PROCARGS2\n"
            "procs = sysctl([1, 14, 0])  # CTL_KERN, KERN_PROC, KERN_PROC_ALL\n"
            "print('args', 'none' if args is None else secret in args,\n"
            "      'procs', 'none' if procs is None else len(procs))\n"
        )
    )
    child = [sys.executable, "-I", "-S", "-c", code]
    parent = (
        "import json, subprocess, sys\n"
        "cmd, env = json.loads(sys.argv[1]), json.loads(sys.argv[2])\n"
        "r = subprocess.run(cmd, env=env, capture_output=True, text=True)\n"
        "print(r.stdout, r.stderr)"
    )
    (work / "tmp").mkdir(exist_ok=True)

    def via_parent(cmd: list[str]) -> list[str]:
        env = {**child_env(work), "AW_PARENT_SECRET": secret}
        r = subprocess.run(
            [
                sys.executable,
                "-c",
                parent,
                json.dumps(cmd),
                json.dumps(child_env(work)),
            ],
            env=env,
            cwd=work,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return r.stdout.split()

    control = via_parent(child)
    assert control[:2] == ["args", "True"], control
    assert int(control[3]) > 0
    out = via_parent(wrap(child, ro=python_paths(), rw=work))
    assert out == ["args", "none", "procs", "none"], out


_MACH = (
    "import ctypes\n"
    "libc = ctypes.CDLL(None)\n"
    "bp = ctypes.c_uint.in_dll(libc, 'bootstrap_port').value\n"
    "port = ctypes.c_uint(0)\n"
    "kr = libc.bootstrap_look_up(bp, b'com.apple.pasteboard.1', ctypes.byref(port))\n"
    "assert kr == 0, f'bootstrap_look_up returned {kr}'"
)


@macos_sandbox
@pytest.mark.parametrize(
    "code", ["import os\nos.kill(os.getppid(), 0)", _MACH], ids=["signal", "mach"]
)
def test_macos_ipc_denied(work: Path, code: str) -> None:
    _blocked(code, work)


@linux_sandbox
def test_linux_no_nested_user_namespace(work: Path) -> None:
    # no numpy import here: its BLAS threads make unshare(CLONE_NEWUSER) fail anyway
    code = (
        "import ctypes, errno\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "rc = libc.unshare(0x10000000)  # CLONE_NEWUSER\n"
        "print('ESCAPED' if rc == 0 else errno.errorcode[ctypes.get_errno()])"
    )
    cmd = wrap([sys.executable, "-I", "-S", "-c", code], ro=python_paths(), rw=work)
    assert "--disable-userns" in cmd
    # control: the same sandbox without --disable-userns lets the child unshare
    control = _run([a for a in cmd if a != "--disable-userns"], work)
    assert control.stdout.strip() == "ESCAPED", control.stdout + control.stderr
    r = _run(cmd, work)
    # --disable-userns sets user.max_user_namespaces to 0 in the sandbox
    assert r.stdout.strip() == "ENOSPC", r.stdout + r.stderr


# ------------------------------------------------- bwrap < 0.8 (no --disable-userns)

OLD_BWRAP = (
    "bwrap is installed but cannot sandbox here (exit 1): "
    "bwrap: Unknown option --disable-userns"
)


def _old_bwrap(monkeypatch: pytest.MonkeyPatch, nested_ok: bool = True) -> None:
    monkeypatch.setattr(sandbox, "_detect", lambda: (None, OLD_BWRAP))
    monkeypatch.setattr(
        sandbox,
        "_detect_nested_userns",
        lambda: ("bwrap", "") if nested_ok else (None, "bwrap still fails"),
    )
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/bwrap")


@posix_paths
def test_old_bwrap_refused_by_default_with_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _old_bwrap(monkeypatch)
    with pytest.raises(SandboxUnavailable) as e:
        wrap([sys.executable], ro=[], rw=tmp_path)
    assert "allow_nested_userns" in str(e.value)
    assert "nested user namespaces" in str(e.value)


@posix_paths
def test_old_bwrap_opt_in_drops_only_disable_userns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _old_bwrap(monkeypatch)
    cmd = wrap([sys.executable], ro=[], rw=tmp_path, allow_nested_userns=True)
    assert cmd[0] == "/usr/bin/bwrap"
    assert cmd[1:5] == [f for f in NAMESPACES if f != "--disable-userns"]
    assert "--disable-userns" not in cmd
    nested = bwrap_command(cmd[-1:], [], tmp_path, exe="b", nested_userns=True)
    assert "--disable-userns" not in nested
    assert "--disable-userns" in bwrap_command(cmd[-1:], [], tmp_path, exe="b")


@posix_paths
def test_opt_in_does_not_cover_other_bwrap_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _old_bwrap(monkeypatch)
    monkeypatch.setattr(sandbox, "_detect", lambda: (None, "denied by policy"))
    with pytest.raises(SandboxUnavailable, match="denied by policy"):
        wrap([sys.executable], ro=[], rw=tmp_path, allow_nested_userns=True)


@posix_paths
def test_opt_in_when_old_bwrap_still_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _old_bwrap(monkeypatch, nested_ok=False)
    with pytest.raises(SandboxUnavailable, match="bwrap still fails"):
        wrap([sys.executable], ro=[], rw=tmp_path, allow_nested_userns=True)
    # unsafe still wins over a failed compat probe, as before
    argv = [sys.executable, "-c", "1"]
    assert (
        wrap(argv, ro=[], rw=tmp_path, allow_nested_userns=True, allow_unsafe=True)
        == argv
    )


@pytest.mark.usefixtures("fresh_probe")
def test_nested_probe_runs_without_disable_userns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_run(0, calls))
    assert sandbox._detect_nested_userns() == ("bwrap", "")
    assert calls[0][:5] == [
        "bwrap",
        *(f for f in NAMESPACES if f != "--disable-userns"),
    ]


@linux_sandbox
def test_linux_old_bwrap_compat_still_sandboxes(
    monkeypatch: pytest.MonkeyPatch, work: Path
) -> None:
    # real bwrap, the strict probe forced to fail as on bwrap < 0.8
    monkeypatch.setattr(sandbox, "_detect", lambda: (None, OLD_BWRAP))
    sandbox._detect_nested_userns.cache_clear()
    code = (
        "import ctypes, os\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "print('home', os.path.exists('/home'))\n"
        "print('nested', libc.unshare(0x10000000) == 0)"
    )
    argv = [sys.executable, "-I", "-S", "-c", code]
    cmd = wrap(argv, ro=python_paths(), rw=work, allow_nested_userns=True)
    assert "--disable-userns" not in cmd
    assert "--unshare-all" in cmd
    r = _run(cmd, work)
    # the protection given up: the child can create a user namespace
    assert r.stdout.split() == ["home", "False", "nested", "True"], r.stdout + r.stderr
