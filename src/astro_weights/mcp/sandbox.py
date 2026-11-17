"""OS sandbox for agent-written code: bubblewrap on Linux, Seatbelt on macOS.

``wrap`` turns a command into one that runs with no network, no GPU or accelerator
devices, no view of other processes, read access to the given paths plus the system
libraries the interpreter needs, and write access to one directory. With no working
sandbox it raises ``SandboxUnavailable`` unless the caller passes ``allow_unsafe``.

Paths are passed to the sandbox as separate arguments (bwrap options, Seatbelt
``-D`` parameters), never spliced into a profile, so no path can change the rules.

Each module-level import works on Linux, macOS and Windows. Standard library only.
"""

import ctypes
import functools
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

__all__ = [
    "Mode",
    "SandboxUnavailable",
    "bwrap_command",
    "child_env",
    "harden_parent",
    "hidden_site_dirs",
    "probe",
    "python_paths",
    "seatbelt_command",
    "wrap",
]

Mode = Literal["bwrap", "seatbelt"]

UNSAFE_HINT = "pass allow_unsafe=True to run agent code without a sandbox"
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
# Every namespace (user, ipc, pid, net, uts, cgroup); no nested user namespaces (the
# kernel surface behind most unprivileged exploits; needs bwrap >= 0.8, older bwrap
# fails the probe); die with the parent; own session, so the child cannot push
# keystrokes into the parent's terminal.
BWRAP_NAMESPACES = (
    "--unshare-all",
    "--unshare-user",
    "--disable-userns",
    "--die-with-parent",
    "--new-session",
)
# System files the interpreter and numpy load from. /sys, /run, /var, /home and the
# rest of /etc stay out of view.
BWRAP_SYSTEM = ("/usr", "/bin", "/lib", "/lib64", "/etc/ld.so.cache")
# System site-packages that would expose packages such as torch; hidden by wrap.
SYSTEM_SITE_GLOBS = (
    "/usr/lib*/python3*/*-packages",
    "/usr/local/lib*/python3*/*-packages",
)
PR_SET_DUMPABLE = 4

# Seatbelt rules. Later rules override earlier ones. Paths arrive as -D parameters.
SEATBELT_HEAD = """\
(version 1)
(deny default)
; run the wrapped program only; process-fork stays denied, so nothing else runs
(allow process-exec (literal (param "EXE")))
; sysctls by name only. hw.*: CPU count and features (libSystem at start, BLAS
; threads). kern.ostype, osrelease, version, hostname: os.uname(), which numpy's
; import calls. Not kern.procargs2 or kern.proc: other processes' argv, env, list.
(allow sysctl-read (sysctl-name-prefix "hw.") (sysctl-name "kern.ostype")
  (sysctl-name "kern.osrelease") (sysctl-name "kern.version")
  (sysctl-name "kern.hostname"))
; KERN_PROCARGS2 / KERN_PROC_ALL leak unless both the allowlist and this deny apply
(deny process-info*)
; dyld shared cache, system frameworks and libSystem
(allow file-read* (subpath "/System/Library") (subpath "/usr/lib"))
; Python lists the root directory while it computes its paths
(allow file-read* (literal "/"))
(allow file-read* file-write-data (literal "/dev/null"))
"""


class SandboxUnavailable(RuntimeError):
    """No working OS sandbox, and the caller did not allow running without one."""


def _platform() -> str:
    return sys.platform


@functools.cache
def _detect() -> tuple[Mode | None, str]:
    """(mode, reason it is None). Runs a trivial command in the sandbox."""
    system = _platform()
    if system == "linux":
        if shutil.which("bwrap") is None:
            return None, "bwrap is not installed"
        cmd = ["bwrap", *BWRAP_NAMESPACES, "--ro-bind", "/", "/", "--proc", "/proc"]
        return _runs([*cmd, "--dev", "/dev", "--", "true"], "bwrap")
    if system == "darwin":
        if not Path(SANDBOX_EXEC).exists():
            return None, f"{SANDBOX_EXEC} is missing"
        profile = "(version 1)(allow default)(deny network*)"
        return _runs([SANDBOX_EXEC, "-p", profile, "/usr/bin/true"], "seatbelt")
    return None, f"no sandbox is supported on {system}"


def _runs(cmd: list[str], mode: Mode) -> tuple[Mode | None, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"{cmd[0]} failed to run: {e}"
    if r.returncode != 0:
        err = r.stderr.strip()[-300:]
        rc = r.returncode
        return None, f"{cmd[0]} is installed but cannot sandbox here (exit {rc}): {err}"
    return mode, ""


def probe() -> Mode | None:
    """The sandbox that works on this machine, or None. Cached per process.

    bwrap can be installed yet blocked (AppArmor user namespace rules), and
    sandbox-exec fails inside another sandbox, so both are run once for real.
    """
    return _detect()[0]


def _check_abs(p: Path) -> Path:
    if not p.is_absolute():
        raise ValueError(f"sandbox path must be absolute: {str(p)!r}")
    if "\0" in str(p):
        raise ValueError(f"sandbox path contains a NUL byte: {str(p)!r}")
    return p


def _resolve(p: Path) -> Path:
    return _check_abs(p).resolve(strict=True)


def _unique(paths: Sequence[Path]) -> list[Path]:
    return list(dict.fromkeys(paths))


def _under(p: Path, dirs: Sequence[Path]) -> bool:
    return any(p != d and p.is_relative_to(d) for d in dirs)


def python_paths(packages: Sequence[str] = ("numpy",)) -> list[Path]:
    """Directories and files ``python -I -S`` plus ``packages`` must read.

    The real interpreter (venv symlinks resolved), its standard library,
    lib-dynload and shared libpython, each package directory and the
    ``<package>.libs`` directory that Linux wheels bundle native libraries in.
    Not site-packages as a whole. The child adds each package's parent directory
    to ``sys.path`` itself.
    """
    base = Path(sys.base_prefix)
    out = [Path(sys.executable).resolve(), Path(sysconfig.get_paths()["stdlib"])]
    out += [Path(p) for p in sys.path if Path(p).name == "lib-dynload"]
    out += sorted((base / "lib").glob("libpython3*"))
    out += [f for f in (base / "Python",) if f.is_file()]  # macOS framework build
    for name in packages:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            raise ModuleNotFoundError(f"package {name!r} is not installed")
        pkg = Path(next(iter(spec.submodule_search_locations)))
        libs = pkg.parent / f"{name}.libs"
        out += [pkg, *([libs] if libs.is_dir() else [])]
    return _unique([p.resolve() for p in out if p.exists()])


def hidden_site_dirs(ro: Sequence[Path]) -> list[Path]:
    """site-packages / dist-packages that would be readable through ``ro`` or /usr.

    A stdlib directory often holds the base interpreter's site-packages, and
    /usr holds the system ones; either may contain torch.
    """
    found = [
        d for r in ro for d in (*r.glob("site-packages"), *r.glob("dist-packages"))
    ]
    if _platform() == "linux":
        found += [p for g in SYSTEM_SITE_GLOBS for p in Path("/").glob(g.lstrip("/"))]
    keep = set(ro)
    return _unique(
        [d.resolve() for d in found if d.is_dir() and d.resolve() not in keep]
    )


def bwrap_command(
    argv: Sequence[str],
    ro: Sequence[Path],
    rw: Path,
    hide: Sequence[Path] = (),
    exe: str = "bwrap",
) -> list[str]:
    """bwrap command line; ``hide`` dirs become empty, ``ro`` inside them stay.

    bwrap builds the root and /dev as in-memory tmpfs. Both are remounted
    read-only at the end, as are hidden dirs: writes there would use RAM that
    no rlimit counts. ``rw`` is the one writable place.
    """
    for p in (*ro, rw, *hide):
        _check_abs(p)
    cmd = [exe, *BWRAP_NAMESPACES, "--proc", "/proc", "--dev", "/dev"]
    for d in BWRAP_SYSTEM:
        cmd += ["--ro-bind-try", d, d]
    for p in ro:
        cmd += ["--ro-bind", str(p), str(p)]
    for h in hide:
        cmd += ["--tmpfs", str(h)]
    for p in ro:
        if _under(p, hide):
            cmd += ["--ro-bind", str(p), str(p)]
    for h in hide:
        cmd += ["--remount-ro", str(h)]
    cmd += ["--bind", str(rw), str(rw), "--remount-ro", "/dev", "--remount-ro", "/"]
    return [*cmd, "--chdir", str(rw), "--", *argv]


def _ancestors(paths: Sequence[Path]) -> list[Path]:
    return _unique([a for p in paths for a in p.parents])


def seatbelt_command(
    argv: Sequence[str], ro: Sequence[Path], rw: Path, hide: Sequence[Path] = ()
) -> list[str]:
    """sandbox-exec command line. Profile text holds no path; ``-D`` passes them."""
    if not argv:
        raise ValueError("argv is empty")
    for p in (*ro, rw, *hide):
        _check_abs(p)
    exe = _check_abs(Path(argv[0]))
    params: list[tuple[str, Path]] = [("EXE", exe), ("RW", rw)]
    lines = [SEATBELT_HEAD]

    def rule(allow: str, kind: str, key: str, paths: Sequence[Path]) -> None:
        if paths:
            names = [f"{key}{i}" for i in range(len(paths))]
            params.extend(zip(names, paths, strict=True))
            refs = " ".join(f'({kind} (param "{n}"))' for n in names)
            lines.append(f"({allow} {refs})\n")

    lines.append("; stat of every directory above a granted path (path resolution)\n")
    rule("allow file-read-metadata", "literal", "A", _ancestors([*ro, rw]))
    lines.append(
        "; read-only paths, minus hidden site-packages, re-allowing ro under them\n"
    )
    rule("allow file-read*", "subpath", "R", ro)
    rule("deny file-read*", "subpath", "H", hide)
    rule("allow file-read*", "subpath", "I", [p for p in ro if _under(p, hide)])
    lines.append("; listing of each read-only path's parent, so imports can find it\n")
    rule("allow file-read*", "literal", "P", _unique([p.parent for p in ro]))
    lines.append("; the one writable directory\n")
    lines.append('(allow file-read* file-write* (subpath (param "RW")))\n')
    defs = [a for k, p in params for a in ("-D", f"{k}={p}")]
    return [SANDBOX_EXEC, "-p", "".join(lines), *defs, "--", *argv]


def wrap(
    argv: Sequence[str],
    *,
    ro: Sequence[Path],
    rw: Path,
    allow_unsafe: bool = False,
) -> list[str]:
    """``argv`` as a sandboxed command line: read ``ro``, write ``rw``, no network.

    ``argv[0]`` must be an absolute path; in the sandbox it is resolved, so a venv
    python runs as its base interpreter. Paths must be absolute and exist. Start
    the command with ``cwd=rw``: sandbox-exec cannot change directory itself.
    Without a working sandbox: ``SandboxUnavailable``, or ``list(argv)`` unchanged
    when ``allow_unsafe`` is True.
    """
    if not argv:
        raise ValueError("argv is empty")
    exe = _resolve(Path(argv[0]))
    ro_r = _unique([_resolve(p) for p in ro])
    rw_r = _resolve(rw)
    mode, why = _detect()
    if mode is None:
        if allow_unsafe:
            return list(argv)
        raise SandboxUnavailable(f"no OS sandbox: {why}; {UNSAFE_HINT}")
    run = [str(exe), *argv[1:]]
    hide = hidden_site_dirs(ro_r)
    if mode == "bwrap":
        return bwrap_command(
            run, ro_r, rw_r, hide, exe=shutil.which("bwrap") or "bwrap"
        )
    return seatbelt_command(run, ro_r, rw_r, hide)


def child_env(work: Path) -> dict[str, str]:
    """Allowlisted environment for the sandboxed child. Keys and proxies never pass.

    The caller creates ``work / "tmp"``.
    """
    _check_abs(work)
    env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL") if k in os.environ}
    env.update(HOME=str(work), TMPDIR=str(work / "tmp"), CUDA_VISIBLE_DEVICES="")
    return env


def harden_parent() -> None:
    """Linux: make this process undumpable, so a same-user child cannot read its
    ``/proc/<pid>/environ`` or memory. No-op elsewhere."""
    if _platform() != "linux":
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_DUMPABLE, 0) failed")
