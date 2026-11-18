"""Local editor: any OpenAI-compatible server with tool calls drives the MCP server.

An MCP client. It starts the server over stdio and lets a model behind
``{base_url}/v1/chat/completions`` (vLLM, llama-server) call its tools until the model
calls ``finish`` or a stop rule of ``run_loop`` fires. Tokens go to the ledger,
dollars are 0. The API key is read from the environment variable named by
``--api-key-env``, never from the command line. It needs the ``mcp`` extra.

    python -m astro_weights.mcp.editors.local RUN_DIR --base-url URL --model NAME
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    import anyio
    from mcp import Client, StdioServerParameters
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"{__name__} needs the official MCP SDK: pip install 'astro-weights[mcp]'",
        name=e.name,
    ) from e

from astro_weights.mcp.editors.loop import (
    ChatFn,
    LoopError,
    Message,
    Refused,
    Reply,
    run_loop,
    system_prompt,
)
from astro_weights.runners import endpoint

__all__ = ["main", "make_chat"]

_TIMEOUT_S = 300.0
_DETAIL_CHARS = 500


def _retryable(status: int) -> bool:
    return status == 429 or status >= 500


def _post(url: str, body: dict[str, Any], key: str | None) -> tuple[int, str]:
    """POST ``body``; retry connection errors and 429/5xx, then return the last answer."""
    for attempt in range(endpoint._RETRIES + 1):
        if attempt:
            time.sleep(endpoint._RETRY_SLEEP)
        try:
            status, text = endpoint._request(url, body, key, _TIMEOUT_S)
        except RuntimeError as exc:
            raise LoopError(str(exc)) from None
        if not _retryable(status):
            break
    return status, text


def _tokens(usage: object, seen: str) -> int:
    if isinstance(usage, dict):
        total = usage.get("total_tokens")
        if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
            return total
        parts = [usage.get("prompt_tokens"), usage.get("completion_tokens")]
        if all(
            isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in parts
        ):
            return int(parts[0] or 0) + int(parts[1] or 0)
    raise LoopError(f"response lacks usage token counts: {seen}")


def _tokens_or_zero(usage: object) -> int:
    try:
        return _tokens(usage, "")
    except LoopError:
        return 0


def make_chat(
    base_url: str, model: str, api_key: str | None, temperature: float
) -> ChatFn:
    """A chat function for ``run_loop`` over one OpenAI-compatible server.

    Raises ``Refused("content_filter")`` when the server stops a reply with
    ``finish_reason`` ``content_filter`` and ``LoopError`` on any other failure;
    the key is scrubbed from every message.
    """
    url = f"{endpoint._root(base_url)}/v1/chat/completions"
    endpoint._check_key(api_key)

    def chat(messages: list[Message], tools: list[dict[str, Any]]) -> Reply:
        body = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
        }
        status, text = _post(url, body, api_key)
        if status >= 400:
            raise LoopError(str(endpoint._fail("server error", status, text, api_key)))
        seen = endpoint._scrub(text, api_key)[:_DETAIL_CHARS]
        try:
            obj = json.loads(text)
            choice = obj["choices"][0]
            message = choice["message"]
        except (ValueError, RecursionError, KeyError, IndexError, TypeError):
            raise LoopError(f"response is not a chat completion: {seen}") from None
        if not isinstance(message, dict):
            raise LoopError(f"response has no message: {seen}")
        if choice.get("finish_reason") == "content_filter":
            raise Refused("content_filter", seen, _tokens_or_zero(obj.get("usage")))
        return Reply(message, _tokens(obj.get("usage"), seen))

    return chat


async def _run(
    run_dir: Path, unsafe: bool, chat: ChatFn, model: str, max_turns: int
) -> dict[str, Any]:
    args = ["-m", "astro_weights.mcp", str(run_dir)] + (
        ["--unsafe-kernel"] if unsafe else []
    )
    params = StdioServerParameters(command=sys.executable, args=args)
    failure: Refused | LoopError | None = None
    async with Client(params) as client:
        try:
            return await run_loop(
                client,
                chat,
                max_turns=max_turns,
                system=system_prompt(),
                model=model,
                on_turn=lambda n, r: print(
                    f"turn {n}: {r.tokens} tokens", file=sys.stderr
                ),
            )
        except (Refused, LoopError) as exc:  # keep it out of the transport's group
            failure = exc
    raise failure


def _positive_int(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m astro_weights.mcp.editors.local")
    p.add_argument("run_dir", metavar="RUN_DIR", type=Path, help="run directory")
    p.add_argument("--base-url", required=True, help="OpenAI-compatible server URL")
    p.add_argument("--model", required=True, help="model name to send")
    p.add_argument(
        "--api-key-env", metavar="VAR", help="environment variable holding the API key"
    )
    p.add_argument("--max-turns", type=_positive_int, default=40)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--unsafe-kernel", action="store_true", help="pass through to the server"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the loop against a stdio server for RUN_DIR and print a JSON summary.

    Exit 0 when the run finished, 1 when the model refused or the loop failed (the
    refusal or error is printed), 2 for bad arguments, a bad run directory or a server
    that did not start.
    """
    args = _parser().parse_args(argv)
    if not (args.run_dir / "run.json").is_file():
        print(f"error: {args.run_dir / 'run.json'} not found", file=sys.stderr)
        return 2
    key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not key:
        print(f"error: ${args.api_key_env} is not set", file=sys.stderr)
        return 2
    try:
        chat = make_chat(args.base_url, args.model, key, args.temperature)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        summary = anyio.run(
            _run, args.run_dir, args.unsafe_kernel, chat, args.model, args.max_turns
        )
    except Refused as exc:
        print(json.dumps({"refused": {"category": exc.category, "detail": exc.detail}}))
        return 1
    except LoopError as exc:
        print(json.dumps({"error": endpoint._scrub(str(exc), key)}))
        return 1
    except Exception as exc:  # noqa: BLE001 - the server did not start or died
        err: BaseException = exc
        while isinstance(err, BaseExceptionGroup) and err.exceptions:
            err = err.exceptions[0]
        print(
            f"error: could not run the server: {endpoint._scrub(repr(err), key)}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(summary, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
