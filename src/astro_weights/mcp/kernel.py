"""Persistent numpy kernel behind ``exec_python``, run in the OS sandbox.

``Kernel.exec(code)`` runs one cell in a child process (``kernel_child.py``) and
returns ``{"ok", "output", "restarted"}``. Cells share variables until the child is
killed (wall clock limit, protocol error) or dies; the next cell then starts a fresh
child and says ``restarted: true``. The first cell after a server restart says so too
(``run_dir/kernel.started`` exists).

The child can read the interpreter and numpy, and write only ``run_dir/kernel``.
It never sees the edits or the model: ``save_edit``, ``drop_edit``, ``edited``,
``load``, ``load_original`` and ``names`` are requests the parent checks and
answers through the ``EditStore``; ``forward_result`` reads a finished
``forward_pass`` job's result, found through the ``HandleStore``. Arrays cross as ``.npy`` files in the work dir;
the parent builds every path itself from a validated token, and opens files there
without following links, since the child may have planted one.

One long-lived thread per Kernel starts the child and talks to it, so cells run
one at a time and the child outlives the job thread that asked for a cell (bwrap's
``--die-with-parent`` fires when the starting thread exits).

Every module-level import works on Linux, macOS and Windows. numpy and the
standard library only.
"""

import atexit
import contextlib
import io
import json
import os
import queue
import re
import secrets
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import IO, Any

import numpy as np

from astro_weights.mcp import forward, sandbox
from astro_weights.mcp.edit_store import EditStore
from astro_weights.mcp.handles import HandleStore

__all__ = ["EventSink", "Kernel", "KernelError"]

EventSink = Callable[[str, dict[str, Any]], None]

CHILD = Path(__file__).resolve().with_name("kernel_child.py")
MAX_LINE = 1 << 20
START_TIMEOUT_S = 60.0
ERROR_MAX = 2000
OUTPUT = "stdout.txt"
LOG = "kernel.log"  # in run_dir: the child never needs it, and cannot plant it
MARKER = "kernel.started"  # in run_dir, where the child cannot write
MAX_HEADER = 1 << 16
QUEUE_LINES = 4  # lines read ahead of the kernel thread; then the child's write blocks
DRAIN_S = 5.0
_TOKEN = re.compile(r"[0-9a-f]{32}")
_SCRATCH = re.compile(r"(?:in|out)_[0-9a-f]{32}\.(?:npy|tmp)")
_FLOATS = tuple(np.dtype(t) for t in ("<f2", "<f4", "<f8"))
_READ = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)  # a planted FIFO must not block the open
    | getattr(os, "O_BINARY", 0)
)
_CREATE = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_BINARY", 0)
)
_TRUNCATE = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)

# reader-thread queue items: a line, a protocol fault (str), or end of stream (None)
_Item = bytes | str | None


class KernelError(RuntimeError):
    """The kernel process could not be started."""


def _open_plain(path: Path) -> IO[bytes]:
    """Open a file the child could have replaced: no links, no FIFOs, no hard links."""
    try:
        fd = os.open(path, _READ)
    except OSError as e:
        raise ValueError(f"{path.name}: cannot open: {e.strerror}") from None
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        os.close(fd)
        raise ValueError(f"{path.name}: not a plain file")
    return os.fdopen(fd, "rb")


def read_npy(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """A float array of ``shape`` from an untrusted ``.npy`` file.

    dtype, order, shape and file size are checked from the header before any data
    is read, so a bad file costs no allocation; then exactly the data is read.
    """
    with _open_plain(path) as f:
        got, fortran, dtype = _read_header(f)
        if dtype not in _FLOATS:
            raise ValueError(f"array dtype {dtype} is not little-endian float16/32/64")
        if tuple(got) != shape:
            raise ValueError(f"array shape {tuple(got)} does not match {shape}")
        if fortran:
            raise ValueError("array must be C-ordered")
        nbytes = shape[0] * shape[1] * dtype.itemsize
        if os.fstat(f.fileno()).st_size != f.tell() + nbytes:
            raise ValueError("array file size does not match its header")
        data = f.read(nbytes)
    if len(data) != nbytes:
        raise ValueError("array file was cut short")
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def _read_header(f: IO[bytes]) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    """(shape, fortran_order, dtype) of a .npy file; at most MAX_HEADER bytes read.

    numpy's own readers read the whole declared header length (up to 4 GiB)
    before checking it, so the length field is checked here first.
    """
    magic = f.read(8)
    if len(magic) != 8 or magic[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy file (bad magic string)")
    version = (magic[6], magic[7])
    if version not in ((1, 0), (2, 0)):
        raise ValueError(f"unsupported .npy version {version}")
    size_field = f.read(2 if version == (1, 0) else 4)
    if len(size_field) not in (2, 4):
        raise ValueError(".npy header is cut short")
    n = int.from_bytes(size_field, "little")
    if n > MAX_HEADER or n > os.fstat(f.fileno()).st_size:
        raise ValueError(f".npy header length {n} is over the {MAX_HEADER} byte cap")
    raw = f.read(n)
    if len(raw) != n:
        raise ValueError(".npy header is cut short")
    buf = io.BytesIO(size_field + raw)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(buf, max_header_size=MAX_HEADER)
    return np.lib.format.read_array_header_2_0(buf, max_header_size=MAX_HEADER)


def write_npy(path: Path, arr: np.ndarray, dtype: Any = np.float32) -> None:
    """Write ``arr`` as ``.npy`` (float32; None keeps its dtype): temp file, then rename."""
    tmp = path.with_suffix(".tmp")
    try:
        with os.fdopen(os.open(tmp, _CREATE, 0o600), "wb") as f:
            a = np.ascontiguousarray(arr, dtype=dtype)
            np.lib.format.write_array(f, a, allow_pickle=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def read_capped(path: Path, cap: int) -> str:
    """The first ``cap`` characters of a cell's output, with a marker when cut."""
    try:
        with _open_plain(path) as f:
            raw = f.read(cap * 4 + 4)
            more = f.read(1) != b""
    except ValueError:
        return ""
    s = raw.decode("utf-8", "replace")
    if len(s) > cap or more:
        s = s[:cap] + f"\n[output truncated at {cap} characters]"
    return s


def popen_options(platform: str, mem_gib: int) -> dict[str, Any]:
    """POSIX: own session, so the whole group can be killed; Linux: RLIMIT_AS."""
    if platform == "win32":
        return {}
    opts: dict[str, Any] = {"start_new_session": True}
    if platform == "linux":
        opts["preexec_fn"] = _limit_memory(mem_gib << 30)
    return opts


def _limit_memory(nbytes: int) -> Callable[[], None]:
    def apply() -> None:  # runs in the forked child, before exec
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))

    return apply


def child_env(work: Path, platform: str) -> dict[str, str]:
    """``sandbox.child_env``; on Windows plus SYSTEMROOT, without which Python cannot start."""
    env = sandbox.child_env(work)
    if platform == "win32" and "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return env


def _kill(proc: "subprocess.Popen[bytes]") -> None:
    """SIGKILL the child's process group (POSIX) or the child (Windows)."""
    if sys.platform != "win32":
        with contextlib.suppress(OSError):  # gone, or a zombie group (macOS EPERM)
            os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        proc.kill()


def _read_lines(stream: IO[bytes], lines: "queue.Queue[_Item]") -> None:
    """Reader thread: queue each protocol line, a fault, then None at the end."""
    with stream:
        while True:
            line = stream.readline(MAX_LINE + 1)
            if not line.endswith(b"\n"):
                if line:
                    lines.put(f"line longer than {MAX_LINE} bytes or cut off")
                break
            lines.put(line)
    lines.put(None)


def _send(proc: "subprocess.Popen[bytes]", msg: dict[str, Any]) -> None:
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(msg).encode() + b"\n")
    proc.stdin.flush()


class Kernel:
    """A sandboxed numpy kernel for one run; variables persist between cells."""

    def __init__(
        self,
        run_dir: Path,
        store: EditStore,
        *,
        allow_unsafe: bool = False,
        timeout_s: float = 600,
        mem_gib: int = 16,
        stdout_cap: int = 10_000,
        on_event: EventSink | None = None,
        jobs: HandleStore | None = None,
    ) -> None:
        """Check the sandbox and start the kernel thread; the child starts on first use.

        ``SandboxUnavailable`` when there is no OS sandbox and ``allow_unsafe`` is
        False. ``jobs`` answers ``forward_result``; without it that request fails. ``on_event(event, fields)`` hears ``edit_saved`` (tensor,
        rel_frobenius), ``edit_rejected`` (tensor, error), ``edit_dropped``
        (tensor) and ``kernel_restarted``.
        """
        work = Path(run_dir) / "kernel"
        (work / "tmp").mkdir(parents=True, exist_ok=True)
        self.work = work.resolve()
        self._store = store
        self._timeout_s = float(timeout_s)
        self._mem_gib = mem_gib
        self._stdout_cap = stdout_cap
        self._on_event = on_event
        self._jobs_store = jobs
        self._run_dir = Path(run_dir).resolve()
        site = Path(np.__file__).resolve().parent.parent
        argv = [sys.executable, "-I", "-S", str(CHILD), str(self.work), str(site)]
        sandbox.harden_parent()
        self._argv = sandbox.wrap(
            argv,
            ro=[*sandbox.python_paths(), CHILD],
            rw=self.work,
            allow_unsafe=allow_unsafe,
        )
        # a kernel ran here before this process: its variables are gone
        self._marker = Path(run_dir).resolve() / MARKER
        self._log = Path(run_dir).resolve() / LOG
        self._lost = self._marker.exists()
        self._proc: subprocess.Popen[bytes] | None = None
        self._lines: queue.Queue[_Item] = queue.Queue(QUEUE_LINES)
        self._reader = threading.Thread()  # replaced by each start
        self._jobs: queue.Queue[tuple[str, Future[dict[str, Any]]] | None]
        self._jobs = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(target=self._loop, name="kernel", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    # caller side

    def exec(self, code: str) -> dict[str, Any]:
        """Run one cell and wait for it; cells from many threads run one at a time.

        ``KernelError`` if the child cannot start, ``RuntimeError`` after ``close``.
        """
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        fut: Future[dict[str, Any]] = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("kernel is closed")
            self._jobs.put((code, fut))
        return fut.result()

    def close(self) -> None:
        """Kill the child, stop the kernel thread. Queued cells raise RuntimeError."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._jobs.put(None)
        atexit.unregister(self.close)
        proc = self._proc
        if proc is not None:
            _kill(proc)  # ends a running cell now
        self._thread.join(timeout=30)

    # kernel thread

    def _loop(self) -> None:
        while (item := self._jobs.get()) is not None:
            code, fut = item
            if self._closed:
                fut.set_exception(RuntimeError("kernel is closed"))
                continue
            try:
                fut.set_result(self._exec(code))
            except BaseException as e:  # noqa: BLE001 - the thread must outlive any fault
                fut.set_exception(e)
        self._stop()

    def _exec(self, code: str) -> dict[str, Any]:
        proc = self._proc
        restarted = False
        # anything queued between cells means the child ended or misbehaved
        if proc is None or proc.poll() is not None or not self._lines.empty():
            self._lost = self._lost or proc is not None
            self._stop()
            proc = self._start()
            restarted, self._lost = self._lost, False
            if restarted:
                self._emit("kernel_restarted", {})
        fired = threading.Event()
        end = self._converse(proc, {"code": code}, "done", self._timeout_s, fired)
        output = read_capped(self.work / OUTPUT, self._stdout_cap)
        if isinstance(end, dict):
            if fired.is_set():  # the deadline hit just as the cell ended
                self._stop()
                self._lost = True
            return {
                "ok": end.get("ok") is True,
                "output": output,
                "restarted": restarted,
            }
        rc = self._stop()
        self._lost = True
        if fired.is_set():
            why = f"killed after {self._timeout_s:g} s"
        elif end is None:
            why = f"died (exit code {rc})"
        else:
            why = f"protocol error: {end}; kernel killed"
        note = f"[kernel {why}; its variables are lost]"
        out = f"{output.rstrip(chr(10))}\n{note}" if output else note
        return {"ok": False, "output": out, "restarted": restarted}

    def _start(self) -> "subprocess.Popen[bytes]":
        for p in self.work.iterdir():
            if p.name == OUTPUT or _SCRATCH.fullmatch(p.name):
                with contextlib.suppress(OSError):
                    p.unlink()
        # truncate, not unlink + create: Windows can keep a just-killed
        # child's file locked for a moment, and unlink then fails
        log_fd = os.open(self._log, _TRUNCATE, 0o600)
        try:
            proc = subprocess.Popen(
                self._argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log_fd,
                env=child_env(self.work, sys.platform),
                cwd=self.work,
                **popen_options(sys.platform, self._mem_gib),
            )
        except OSError as e:
            raise KernelError(f"kernel failed to start: {e}") from e
        finally:
            os.close(log_fd)
        self._proc = proc
        self._lines = queue.Queue(QUEUE_LINES)
        assert proc.stdout is not None
        self._reader = threading.Thread(
            target=_read_lines, args=(proc.stdout, self._lines), daemon=True
        )
        self._reader.start()
        fired = threading.Event()
        end = self._converse(proc, None, "ready", START_TIMEOUT_S, fired)
        if not isinstance(end, dict):
            rc = self._stop()
            if fired.is_set():
                what = f"did not start within {START_TIMEOUT_S:g} s"
            elif end is None:
                what = f"failed to start (exit code {rc})"
            else:
                what = f"failed to start: protocol error: {end}"
            tail = read_capped(self._log, 100_000)[-1500:]
            raise KernelError(f"kernel {what}: {tail}".strip())
        self._marker.write_text(f"{proc.pid}\n", encoding="utf-8")
        return proc

    def _stop(self) -> int | None:
        """Kill and reap the child; its exit code, or None if there was none."""
        proc, self._proc = self._proc, None
        if proc is None:
            return None
        _kill(proc)
        rc = proc.wait()
        if proc.stdin is not None:
            with contextlib.suppress(OSError):
                proc.stdin.close()
        # empty the queue until the reader thread reaches the end of the stream
        deadline = time.monotonic() + DRAIN_S
        while self._reader.is_alive() and time.monotonic() < deadline:
            with contextlib.suppress(queue.Empty):
                self._lines.get(timeout=0.05)
        return rc

    def _converse(
        self,
        proc: "subprocess.Popen[bytes]",
        msg: dict[str, Any] | None,
        until: str,
        timeout_s: float,
        fired: threading.Event,
    ) -> dict[str, Any] | str | None:
        """Send ``msg``, answer requests until a line with ``until``: true.

        Returns that line, None if the child ended (or was killed at the deadline,
        then ``fired`` is set), or a protocol fault as a string.
        """
        lines = self._lines

        def expire() -> None:
            fired.set()
            with contextlib.suppress(queue.Full):  # full: the next get returns anyway
                lines.put_nowait(None)
            _kill(proc)  # also unblocks a write to a child that stopped reading

        timer = threading.Timer(timeout_s, expire)
        timer.daemon = True
        timer.start()
        try:
            if msg is not None:
                with contextlib.suppress(OSError):  # child gone: end of stream follows
                    _send(proc, msg)
            while True:
                item = lines.get()
                if fired.is_set():
                    return None
                if not isinstance(item, bytes):
                    return item
                try:
                    reply = json.loads(item)
                except (ValueError, RecursionError):
                    return "line is not JSON"
                if not isinstance(reply, dict):
                    return "line is not a JSON object"
                if reply.get(until) is True:
                    return reply
                if "rpc" not in reply:
                    return "unexpected message"
                with contextlib.suppress(OSError):
                    _send(proc, self._answer(reply))
        finally:
            timer.cancel()

    # requests from the child

    def _answer(self, req: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"ok": True, "result": self._dispatch(req)}
        except Exception as e:  # noqa: BLE001 - a bad request must not stop the kernel
            err = str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"
            return {"ok": False, "error": err[:ERROR_MAX]}

    def _dispatch(self, req: dict[str, Any]) -> Any:
        op = req.get("rpc")
        name: Any = req.get("name")
        store = self._store
        if op == "names":
            return list(store.names)
        if op == "edited":
            return store.edited()
        if op == "save_edit":
            return self._save(name, req.get("token"))
        if op == "drop_edit":
            store.drop(name)
            self._emit("edit_dropped", {"tensor": name})
            return None
        if op == "load":
            return self._put(store.load(name))
        if op == "load_original":
            return self._put(store.original(name))
        if op == "forward_result":
            return self._forward_result(req.get("handle"), req.get("key"))
        raise ValueError(f"unknown request {str(op)[:80]!r}")

    def _save(self, name: Any, token: Any) -> dict[str, Any]:
        try:
            shape = self._store.shape(name)
            result = self._store.save(name, self._take(token, shape))
        except ValueError as e:
            fields = {"tensor": str(name)[:200], "error": str(e)[:ERROR_MAX]}
            self._emit("edit_rejected", fields)
            raise
        self._emit(
            "edit_saved", {"tensor": name, "rel_frobenius": result["rel_frobenius"]}
        )
        return result

    def _take(self, token: Any, shape: tuple[int, int]) -> np.ndarray:
        if not isinstance(token, str) or _TOKEN.fullmatch(token) is None:
            raise ValueError("bad array token")
        path = self.work / f"out_{token}.npy"
        try:
            return read_npy(path, shape)
        finally:
            with contextlib.suppress(OSError):
                path.unlink()

    def _put(self, arr: np.ndarray, dtype: Any = np.float32) -> str:
        token = secrets.token_hex(16)
        write_npy(self.work / f"in_{token}.npy", arr, dtype)
        return token

    def _forward_result(self, handle: Any, key: Any) -> dict[str, Any]:
        if self._jobs_store is None:
            raise ValueError("forward results are not available in this kernel")
        got = forward.read_result(self._run_dir, self._jobs_store, handle, key)
        if isinstance(got, list):
            return {"texts": got}
        return {"array": self._put(got, None)}

    def _emit(self, name: str, fields: dict[str, Any]) -> None:
        if self._on_event is None:
            return
        with contextlib.suppress(
            OSError, ValueError
        ):  # a full disk must not stop cells
            self._on_event(name, fields)
