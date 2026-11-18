"""Scripted editor: a fixed list of tool calls sent to the MCP server, no model.

An MCP client. It starts the server over stdio, runs the script, polls each job to
its end and prints a JSON summary. It uses no tokens and no dollars, so it reports
no usage. It needs the ``mcp`` extra and never imports the server's internals.

    python -m astro_weights.mcp.editors.scripted RUN_DIR [--unsafe-kernel]
"""

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import anyio
    from mcp import Client, ClientSession, StdioServerParameters
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"{__name__} needs the official MCP SDK: pip install 'astro-weights[mcp]'",
        name=e.name,
    ) from e

__all__ = ["DEFAULT_SCRIPT", "ScriptFailed", "Step", "main", "run_scripted"]

Args = Mapping[str, Any] | Callable[[Sequence[str]], Mapping[str, Any]]


@dataclass(frozen=True)
class Step:
    """One tool call, or (``tool`` None) a poll of the last handle until it ends.

    ``args`` may be a function of the handles returned so far, oldest first.
    """

    tool: str | None = None
    args: Args = field(default_factory=dict)


class ScriptFailed(RuntimeError):
    """A step returned an error, a failed job, or did not end in time."""

    def __init__(self, index: int, result: Mapping[str, Any]) -> None:
        super().__init__(
            f"step {index} failed: {json.dumps(result, default=str)[:500]}"
        )
        self.index = index
        self.result = dict(result)


_EDIT = """\
name = next((n for n in NAMES if "mlp.down_proj" in n), NAMES[0])
W = load(name)
rng = np.random.default_rng(0)
d = np.outer(rng.standard_normal(W.shape[0]), rng.standard_normal(W.shape[1]))
W2 = W + d * (0.45 * np.linalg.norm(W) / np.linalg.norm(d))
save_edit(name, W2.astype(np.float32))
print(name, float(np.linalg.norm(W2 - W) / np.linalg.norm(W)))
"""

_TEXT = "The quick brown fox jumps over the lazy dog."


def _forward(arm: str) -> dict[str, Any]:
    request = {
        "kind": "teacher_forced",
        "texts": [_TEXT],
        "return": ["hidden"],
        "layers": [0],
        "last_tokens": 4,
        "arm": arm,
    }
    return {"request": request}


def _compare(handles: Sequence[str]) -> dict[str, Any]:
    h_base, h_edit = handles[1], handles[2]
    code = (
        f'a = forward_result("{h_base}", "hidden_l0")\n'
        f'b = forward_result("{h_edit}", "hidden_l0")\n'
        "print(float(abs(a - b).max()))\n"
    )
    return {"code": code}


DEFAULT_SCRIPT: tuple[Step, ...] = (
    Step("exec_python", {"code": _EDIT}),
    Step(),
    Step("forward_pass", _forward("base")),
    Step(),
    Step("forward_pass", _forward("edited")),
    Step(),
    Step("exec_python", _compare),
    Step(),
    Step(
        "finish", {"summary": "scripted run: edited one weight, compared hidden states"}
    ),
)


async def _call(
    session: Client | ClientSession, tool: str, args: Mapping[str, Any]
) -> dict[str, Any]:
    result = await session.call_tool(tool, dict(args))
    out = result.structured_content
    if result.is_error or out is None:
        return {"error": f"{tool} returned an error", "content": str(result.content)}
    return dict(out)


async def _poll(
    session: Client | ClientSession, handle: str, poll_s: float, deadline: float
) -> dict[str, Any] | None:
    """The job's status once it is done or failed; None if the deadline passes."""
    while True:
        out = await _call(session, "status", {"handle": handle})
        if out.get("state") in ("done", "failed") or out.get("error"):
            return out
        if time.monotonic() >= deadline:
            return None
        await anyio.sleep(poll_s)


def _failure(result: Mapping[str, Any]) -> bool:
    inner = result.get("result")
    return (
        bool(result.get("error"))
        or result.get("state") == "failed"
        or (isinstance(inner, dict) and inner.get("ok") is False)
    )


async def run_scripted(
    session: Client | ClientSession,
    script: Sequence[Step] = DEFAULT_SCRIPT,
    *,
    poll_s: float = 0.2,
    timeout_s: float = 600,
) -> list[dict[str, Any]]:
    """Run ``script`` on ``session``; return every tool result in order.

    A poll step adds the job's final status. ``ScriptFailed`` names the step index
    and holds its result, on an error, a failed job, a cell that raised, or a poll
    that outlasts ``timeout_s``.
    """
    results: list[dict[str, Any]] = []
    handles: list[str] = []
    for index, step in enumerate(script):
        if step.tool is None:
            if not handles:
                raise ScriptFailed(index, {"error": "poll with no handle"})
            deadline = time.monotonic() + timeout_s
            out = await _poll(session, handles[-1], poll_s, deadline)
            if out is None:
                raise ScriptFailed(
                    index, {"error": "poll timed out", "handle": handles[-1]}
                )
        else:
            args = step.args(handles) if callable(step.args) else step.args
            out = await _call(session, step.tool, args)
            if isinstance(out.get("handle"), str):
                handles.append(out["handle"])
        results.append(out)
        if _failure(out):
            raise ScriptFailed(index, out)
    return results


async def _run(run_dir: Path, unsafe: bool) -> list[dict[str, Any]]:
    args = ["-m", "astro_weights.mcp", str(run_dir)] + (
        ["--unsafe-kernel"] if unsafe else []
    )
    params = StdioServerParameters(command=sys.executable, args=args)
    failure: ScriptFailed | None = None
    async with Client(params) as client:
        try:
            return await run_scripted(client)
        except ScriptFailed as exc:  # keep it out of the transport's exception group
            failure = exc
    raise failure


def main(argv: list[str] | None = None) -> int:
    """Run the default script against a stdio server for RUN_DIR; print a JSON summary."""
    parser = argparse.ArgumentParser(
        prog="python -m astro_weights.mcp.editors.scripted"
    )
    parser.add_argument("run_dir", metavar="RUN_DIR", type=Path, help="run directory")
    parser.add_argument(
        "--unsafe-kernel", action="store_true", help="pass through to the server"
    )
    args = parser.parse_args(argv)
    if not (args.run_dir / "run.json").is_file():
        print(f"error: {args.run_dir / 'run.json'} not found", file=sys.stderr)
        return 2
    try:
        results = anyio.run(_run, args.run_dir, args.unsafe_kernel)
    except ScriptFailed as exc:
        print(json.dumps({"error": str(exc), "step": exc.index, "result": exc.result}))
        return 1
    except Exception as exc:  # noqa: BLE001 - the server did not start or died
        while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
            exc = exc.exceptions[0]
        print(f"error: could not run the server: {exc!r}", file=sys.stderr)
        return 2
    handles = list(dict.fromkeys(r["handle"] for r in results if r.get("handle")))
    outputs = [
        r["result"]["output"]
        for r in results
        if isinstance(r.get("result"), dict) and "output" in r["result"]
    ]
    summary = {
        "steps": len(results),
        "handles": handles,
        "outputs": outputs,
        "budget": results[-1].get("budget"),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
