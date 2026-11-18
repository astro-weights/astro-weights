"""Kernel process behind ``exec_python``: runs agent cells, asks the server for edits.

A standalone script: standard library and numpy only, never ``astro_weights``.
``kernel.py`` starts it as ``python -I -S kernel_child.py WORK SITE_DIR...`` inside
the OS sandbox. Each ``SITE_DIR`` goes on ``sys.path``, so numpy imports.

Protocol: one JSON line each way, on private copies of the original stdin and
stdout. File descriptors 0, 1 and 2 are pointed away before any cell runs, so
nothing a cell prints can look like a protocol line. Cell output goes to
``WORK/stdout.txt``. Edits are requests to the server (``save_edit``,
``drop_edit``, ``edited``, ``load``, ``load_original``, ``names``,
``forward_result``); arrays cross as
``.npy`` files in ``WORK`` named by a random token.
"""

import ast
import json
import os
import secrets
import sys
import threading
import time
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

__all__: list[str] = []

OUTPUT = "stdout.txt"
PARENT_POLL_S = 1.0


class Server:
    """Requests from cell code to the server process, one at a time."""

    def __init__(self, proto_in: BinaryIO, proto_out: BinaryIO, work: Path) -> None:
        self._in = proto_in
        self._out = proto_out
        self.work = work
        self._lock = threading.Lock()

    def send(self, msg: dict[str, Any]) -> None:
        """Write one protocol line."""
        self._out.write(json.dumps(msg).encode() + b"\n")
        self._out.flush()

    def lines(self) -> Iterator[bytes]:
        """The server's lines (cells to run), until it closes the pipe."""
        return iter(self._in.readline, b"")

    def call(self, op: str, **args: Any) -> Any:
        """Send a request and return its result; ValueError with the server's error."""
        with self._lock:
            self.send({"rpc": op, **args})
            line = self._in.readline()
        if not line:
            raise ConnectionError("the server closed the kernel's pipe")
        reply = json.loads(line)
        if reply.get("ok") is not True:
            raise ValueError(reply.get("error"))
        return reply.get("result")


def namespace(server: Server) -> dict[str, Any]:
    """The globals cells run in: numpy as ``np``, ``NAMES`` and the edit functions."""
    import numpy as np

    def save_edit(name: str, w: Any) -> Any:
        """Ask the server to check and store ``w`` as the full edited weight ``name``."""
        token = secrets.token_hex(16)
        path = server.work / f"out_{token}.npy"
        try:
            with path.open("xb") as f:
                np.save(f, np.ascontiguousarray(w), allow_pickle=False)
            return server.call("save_edit", name=name, token=token)
        finally:
            path.unlink(missing_ok=True)

    def fetch(op: str, name: str) -> Any:
        token = server.call(op, name=name)
        path = server.work / f"in_{token}.npy"
        try:
            return np.load(path, allow_pickle=False)
        finally:
            path.unlink(missing_ok=True)

    def load(name: str) -> Any:
        """The stored edit of ``name``, else its base weight; float32."""
        return fetch("load", name)

    def load_original(name: str) -> Any:
        """The base weight of ``name``; float32."""
        return fetch("load_original", name)

    def drop_edit(name: str) -> None:
        """Remove the stored edit of ``name``."""
        server.call("drop_edit", name=name)

    def edited() -> Any:
        """Names with a stored edit."""
        return server.call("edited")

    def names() -> Any:
        """Names of the editable weights."""
        return server.call("names")

    def forward_result(handle: str, key: str) -> Any:
        """Result ``key`` of a done forward_pass job: an array, or a list of texts."""
        got = server.call("forward_result", handle=handle, key=key)
        if "texts" in got:
            return got["texts"]
        path = server.work / f"in_{got['array']}.npy"
        try:
            return np.load(path, allow_pickle=False)
        finally:
            path.unlink(missing_ok=True)

    return {
        "__name__": "__main__",
        "np": np,
        "NAMES": tuple(names()),
        "save_edit": save_edit,
        "drop_edit": drop_edit,
        "edited": edited,
        "load": load,
        "load_original": load_original,
        "names": names,
        "forward_result": forward_result,
    }


def run_cell(code: str, ns: dict[str, Any]) -> None:
    """Run ``code`` in ``ns``; print the repr of a last bare expression, if not None."""
    tree = ast.parse(code)
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body[-1].value)
        tree.body.pop()
    exec(compile(tree, "<cell>", "exec"), ns)  # noqa: S102 - running cells is the job
    if last is not None:
        value = eval(compile(last, "<cell>", "eval"), ns)
        if value is not None:
            print(repr(value))


def print_error(e: BaseException) -> None:
    """Print the traceback of a cell's exception, without this file's frames."""
    exc = traceback.TracebackException.from_exception(e)
    exc.stack = traceback.StackSummary.from_list(
        [f for f in exc.stack if f.filename != __file__]
    )
    print("".join(exc.format()), end="")


def serve(server: Server, out_fd: int, ns: dict[str, Any]) -> None:
    """Run cells until the server closes the pipe. No cell can end the loop."""
    for line in server.lines():
        os.ftruncate(out_fd, 0)
        ok = True
        try:
            run_cell(json.loads(line)["code"], ns)
        except BaseException as e:  # noqa: BLE001 - SystemExit too: a cell must not end the kernel
            ok = False
            print_error(e)
        sys.stdout.flush()
        sys.stderr.flush()
        server.send({"done": True, "ok": ok})


def exit_with_parent(ppid: int, poll_s: float = PARENT_POLL_S) -> None:
    """Exit once the parent is gone, even in the middle of a cell."""
    while os.getppid() == ppid:
        time.sleep(poll_s)
    os._exit(1)


def main(argv: list[str]) -> None:
    """Point fds 0-2 away from the protocol pipes, then serve."""
    work = Path(argv[1])
    sys.path[:0] = argv[2:]
    proto_in = os.fdopen(os.dup(0), "rb")  # dup: not inherited by subprocesses
    proto_out = os.fdopen(os.dup(1), "wb")
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    out = os.open(work / OUTPUT, flags, 0o600)
    os.dup2(out, 1)
    # utf-8 and \n on every OS (Windows defaults: locale code page, \r\n)
    sys.stdout.reconfigure(  # type: ignore[union-attr]
        encoding="utf-8", newline="\n", line_buffering=True
    )
    sys.stderr.reconfigure(encoding="utf-8", newline="\n")  # type: ignore[union-attr]
    ppid = os.getppid()
    threading.Thread(target=exit_with_parent, args=(ppid,), daemon=True).start()
    server = Server(proto_in, proto_out, work)
    ns = namespace(server)
    server.send({"ready": True})
    os.dup2(out, 2)  # until now, start-up errors went to the parent's log file
    serve(server, out, ns)


if __name__ == "__main__":
    main(sys.argv)
