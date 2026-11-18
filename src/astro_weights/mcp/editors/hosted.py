"""Hosted editor: Claude, on the user's own Anthropic API key, drives the MCP server.

An MCP client like the local editor: it starts the server over stdio and runs the
shared ``run_loop``; each turn is one Messages API call with the server's tools.
Before every call it counts the prompt's tokens, bounds the call's cost
(``prices.max_cost``) and asks the server's ``check_spend``; a call that could take
the run past its dollar limit is not made, and the run finishes with "dollar cap".
After every call its tokens and dollars go to the ledger. When Claude refuses (stop
reason ``refusal``) the refused turn is dropped, the refusal is recorded and the
same run goes on with a local OpenAI-compatible model, unless ``--no-fallback``.

The key is read from ``ANTHROPIC_API_KEY`` only. It is never printed or recorded,
and the server process does not get it. Needs the ``hosted`` and ``mcp`` extras.

    python -m astro_weights.mcp.editors.hosted RUN_DIR --model ID \\
        --fallback-base-url URL --fallback-model NAME
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import anthropic
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"{__name__} needs the Anthropic SDK: pip install 'astro-weights[hosted]'",
        name=e.name,
    ) from e

try:
    import anyio
    import anyio.from_thread
    from mcp import Client, StdioServerParameters
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"{__name__} needs the official MCP SDK: pip install 'astro-weights[mcp]'",
        name=e.name,
    ) from e

from astro_weights.mcp.editors import prices
from astro_weights.mcp.editors.local import make_chat
from astro_weights.mcp.editors.loop import (
    ChatFn,
    LoopError,
    Message,
    Refused,
    Reply,
    Session,
    run_loop,
    system_prompt,
)
from astro_weights.mcp.editors.scripted import _call
from astro_weights.mcp.jobs import load_run_config
from astro_weights.runners.endpoint import _scrub

__all__ = ["HANDOFF", "KEY_ENV", "main"]

KEY_ENV = "ANTHROPIC_API_KEY"
HANDOFF = (
    "The previous model stopped. You now drive the same run: the tool results above "
    "are real. Go on from there and call finish when you are done."
)
_COUNT_MARGIN = 1.05  # token counting is an estimate; bound the prompt 5% higher
_DETAIL_CHARS = 500

Raw = dict[int, tuple[tuple[str, ...], list[dict[str, Any]]]]


class _Sent(LoopError):
    """The call failed after the request went out: it may have been billed."""


class _Stop(Exception):
    """End the run now; the editor calls ``finish`` with ``summary``."""

    def __init__(self, outcome: str, summary: str) -> None:
        super().__init__(summary)
        self.outcome, self.summary = outcome, summary


def _client(key: str) -> Any:
    """The Anthropic client; its own function so tests can put a fake in its place."""
    return anthropic.Anthropic(api_key=key)


def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI-style function tools (from ``ToolBox``) as Messages API tools."""
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"],
        }
        for t in tools
    ]


def _convert(messages: list[Message], raw: Raw) -> tuple[str, list[dict[str, Any]]]:
    """The loop's OpenAI-style messages as a Messages API system prompt and turns.

    Assistant turns are sent back exactly as the API returned them (``raw``, keyed by
    their place in ``messages``), thinking blocks included; tool results and user
    text after them are joined into one user turn.
    """
    system: list[str] = []
    out: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        role, content = m.get("role"), m.get("content")
        if role == "system":
            system.append(str(content or ""))
            continue
        if role == "assistant":
            ids = tuple(c.get("id") for c in m.get("tool_calls") or [])
            stored = raw.get(i)
            if stored is None or stored[0] != ids:
                raise LoopError(f"assistant turn {i} was not made by this editor")
            side, blocks = "assistant", stored[1]
        elif role == "tool":
            result = {
                "type": "tool_result",
                "tool_use_id": str(m.get("tool_call_id")),
                "content": str(content or ""),
            }
            side, blocks = "user", [result]
        elif role == "user":
            text = content if isinstance(content, str) else ""
            side, blocks = "user", [{"type": "text", "text": text}] if text else []
        else:
            raise LoopError(f"unknown message role {role!r}")
        if not blocks:
            continue
        if out and out[-1]["role"] == side:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": side, "content": list(blocks)})
    return "\n\n".join(system), out


def _as_openai(content: list[dict[str, Any]]) -> Message:
    """One API reply as the loop's assistant message: its text and its tool calls."""
    text = "\n".join(b["text"] for b in content if b.get("type") == "text")
    calls = [
        {
            "id": b["id"],
            "type": "function",
            "function": {"name": b["name"], "arguments": json.dumps(b.get("input"))},
        }
        for b in content
        if b.get("type") == "tool_use"
    ]
    message: Message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = calls
    return message


def _tokens(usage: dict[str, Any]) -> int:
    keys = (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    )
    return sum(int(usage.get(k) or 0) for k in keys)


@dataclass
class _Hosted:
    """One Claude model behind the Messages API."""

    client: Any = field(repr=False)
    model: str
    max_output: int
    key: str = field(repr=False)
    raw: Raw = field(default_factory=dict, repr=False)

    def _text(self, what: str, exc: BaseException) -> str:
        text = _scrub(f"{what}: {type(exc).__name__}: {exc}", self.key)
        return text[:_DETAIL_CHARS]

    def _fail(self, what: str, exc: BaseException) -> LoopError:
        return LoopError(self._text(what, exc))

    def request(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        system, turns = _convert(messages, self.raw)
        return {
            "model": self.model,
            "system": system,
            "messages": turns,
            "tools": _tools(tools),
        }

    def bound(self, req: dict[str, Any]) -> tuple[int, float]:
        """Upper bounds on the tokens and dollars of sending ``req``, from a token count."""
        try:
            counted = int(self.client.messages.count_tokens(**req).input_tokens)
        except anthropic.AnthropicError as exc:
            raise self._fail("token count failed", exc) from None
        prompt = math.ceil(counted * _COUNT_MARGIN)
        dollars = prices.max_cost(self.model, prompt, self.max_output)
        return prompt + self.max_output, dollars

    def call(self, req: dict[str, Any]) -> Any:
        """Send ``req`` (streamed, so a large ``max_tokens`` cannot time out)."""
        try:
            with self.client.messages.stream(
                **req,
                max_tokens=self.max_output,
                cache_control={"type": "ephemeral"},
            ) as stream:
                return stream.get_final_message()
        except anthropic.AnthropicError as exc:
            raise self._fail("Messages API call failed", exc) from None
        except Exception as exc:  # noqa: BLE001 - transport error mid-stream
            raise _Sent(self._text("Messages API call broke off", exc)) from None


@dataclass
class _Editor:
    """The loop's chat function: Claude first, the local model after a refusal.

    It reports every turn's usage to the ledger itself, with the model that made
    it and its dollars, and hands ``run_loop`` that report in ``Reply.usage``.
    """

    hosted: _Hosted
    session: Session
    fallback: ChatFn | None = None
    fallback_model: str = ""
    turns: int = 0
    tokens: int = 0
    dollars: float = 0.0
    handoff: dict[str, Any] | None = None

    def _tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return anyio.from_thread.run(_call, self.session, name, args)

    def _report(self, tokens: int, dollars: float, model: str) -> dict[str, Any]:
        self.tokens += tokens
        self.dollars += dollars
        out = self._tool(
            "report_usage",
            {"tokens": tokens, "dollars": dollars, "turn": self.turns, "model": model},
        )
        if out.get("error"):
            raise LoopError(f"report_usage failed: {json.dumps(out, default=str)}")
        print(
            f"turn {self.turns}: {model} {tokens} tokens ${dollars:.4f}",
            file=sys.stderr,
        )
        return out

    def _refusal(self, refusal: Refused, model: str) -> None:
        args = {"category": refusal.category, "model": model, "detail": refusal.detail}
        out = self._tool("report_refusal", args)
        if out.get("error"):
            raise LoopError(f"report_refusal failed: {json.dumps(out, default=str)}")

    def __call__(self, messages: list[Message], tools: list[dict[str, Any]]) -> Reply:
        self.turns += 1
        if self.handoff is None:
            try:
                return self._hosted_turn(messages, tools)
            except Refused as refusal:
                self._refusal(refusal, self.hosted.model)
                if self.fallback is None:
                    raise _Stop("refused", "refused") from None
                self.handoff = {
                    "turn": self.turns,
                    "category": refusal.category,
                    "model": self.fallback_model,
                }
                messages.append({"role": "user", "content": HANDOFF})
        return self._local_turn(messages, tools)

    def _hosted_turn(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> Reply:
        hosted = self.hosted
        req = hosted.request(messages, tools)
        most_tokens, estimate = hosted.bound(req)
        spend = self._tool("check_spend", {"estimate": estimate})
        if spend.get("ok") is False:
            raise _Stop("dollar_cap", "dollar cap")
        if spend.get("ok") is not True:
            raise LoopError(f"check_spend failed: {json.dumps(spend, default=str)}")
        try:
            reply = hosted.call(req)
        except _Sent:  # usage unknown: charge the bound, never less than was spent
            self._report(most_tokens, estimate, hosted.model)
            raise
        usage = reply.usage.to_dict(mode="json")
        try:
            cost = prices.turn_cost(hosted.model, usage)
        except ValueError as exc:
            raise LoopError(f"cannot price the reply: {exc}") from None
        tokens = _tokens(usage)
        report = self._report(tokens, cost, hosted.model)
        if reply.stop_reason == "refusal":  # dropped: nothing goes to the messages
            details = reply.stop_details
            category = getattr(details, "category", None) or "refusal"
            explanation = getattr(details, "explanation", None) or ""
            raise Refused(category, explanation)
        content = [
            b.to_dict(mode="json", exclude_none=True) for b in reply.content or []
        ]
        message = _as_openai(content)
        ids = tuple(c["id"] for c in message.get("tool_calls", []))
        hosted.raw[len(messages)] = (ids, content)  # where run_loop appends it
        return Reply(message, tokens, report)

    def _local_turn(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> Reply:
        assert self.fallback is not None
        try:
            reply = self.fallback(messages, tools)
        except Refused as refusal:
            if refusal.tokens > 0:
                self._report(refusal.tokens, 0.0, self.fallback_model)
            self._refusal(refusal, self.fallback_model)
            raise _Stop("refused", "refused") from None
        report = self._report(reply.tokens, 0.0, self.fallback_model)
        return Reply(reply.message, reply.tokens, report)


@dataclass(frozen=True)
class _EditorSpec:
    """What ``main`` checked; ``build`` makes the chat function for one session."""

    hosted: _Hosted
    fallback: ChatFn | None
    fallback_model: str

    def build(self, session: Session) -> _Editor:
        return _Editor(self.hosted, session, self.fallback, self.fallback_model)


def _server_params(run_dir: Path, unsafe: bool) -> StdioServerParameters:
    """The server command. ``env`` stays unset: the MCP SDK then passes only a short
    list of safe variables (HOME, PATH, ...), never ``ANTHROPIC_API_KEY``."""
    args = ["-m", "astro_weights.mcp", str(run_dir)]
    return StdioServerParameters(
        command=sys.executable, args=args + (["--unsafe-kernel"] if unsafe else [])
    )


async def _drive(
    session: Session, editor: _EditorSpec, max_turns: int, *, poll_s: float = 0.2
) -> dict[str, Any]:
    """Run the loop on an open session; the loop's summary plus turns, tokens,
    dollars and the fallback handoff (None when Claude did not refuse)."""
    chat = editor.build(session)
    try:
        state = await run_loop(
            session,
            chat,
            max_turns=max_turns,
            system=system_prompt(),
            model=chat.hosted.model,
            poll_s=poll_s,
        )
    except _Stop as stop:
        final = await _call(session, "finish", {"summary": stop.summary})
        state = {
            "outcome": stop.outcome,
            "summary": stop.summary,
            "budget": final.get("budget"),
        }
    return {
        **state,
        "turns": chat.turns,
        "tokens": chat.tokens,
        "dollars": chat.dollars,
        "fallback": chat.handoff,
    }


async def _run(
    run_dir: Path, unsafe: bool, editor: _EditorSpec, max_turns: int
) -> dict[str, Any]:
    failure: LoopError | None = None
    async with Client(_server_params(run_dir, unsafe)) as client:
        try:
            return await _drive(client, editor, max_turns)
        except LoopError as exc:  # keep it out of the transport's group
            failure = exc
    raise failure


def _positive_int(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def _dollars(text: str) -> float:
    x = float(text)
    if not math.isfinite(x) or x < 0:
        raise argparse.ArgumentTypeError("must be a finite number >= 0")
    return x


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m astro_weights.mcp.editors.hosted")
    p.add_argument("run_dir", metavar="RUN_DIR", type=Path, help="run directory")
    p.add_argument(
        "--model", required=True, help="Claude model id, e.g. claude-opus-5-5"
    )
    p.add_argument(
        "--max-dollars",
        type=_dollars,
        default=20.0,
        help="refuse a run.json whose dollar limit is above this (default 20)",
    )
    p.add_argument("--max-turns", type=_positive_int, default=40)
    p.add_argument("--max-output-tokens", type=_positive_int, default=16_000)
    p.add_argument("--fallback-base-url", help="OpenAI-compatible server for refusals")
    p.add_argument("--fallback-model", help="model name on the fallback server")
    p.add_argument(
        "--no-fallback", action="store_true", help="end the run when Claude refuses"
    )
    p.add_argument(
        "--unsafe-kernel", action="store_true", help="pass through to the server"
    )
    return p


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    """Run Claude against a stdio server for RUN_DIR and print a JSON summary.

    Exit 0 when the run finished (by the model, a turn cap, a spent budget or the
    dollar cap), 1 when the model refused with no fallback left or the loop failed,
    2 for bad arguments, a bad run directory, an unknown model, a dollar limit above
    ``--max-dollars``, a missing key or a server that did not start.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    given = [args.fallback_base_url, args.fallback_model]
    if args.no_fallback and any(given):
        parser.error("--no-fallback takes no --fallback-base-url or --fallback-model")
    if not args.no_fallback and not all(given):
        parser.error("give --fallback-base-url and --fallback-model, or --no-fallback")
    try:
        cfg = load_run_config(args.run_dir)
        prices.price(args.model)
    except ValueError as exc:
        return _fail(str(exc))
    limit = cfg.limits.dollars
    if limit > args.max_dollars:
        return _fail(
            f"run.json limits.dollars is {limit}, above --max-dollars "
            f"{args.max_dollars}; raise --max-dollars to allow it"
        )
    if limit <= 0:
        return _fail("run.json limits.dollars is 0; a hosted run needs a dollar budget")
    key = os.environ.get(KEY_ENV)
    if not key:
        return _fail(f"${KEY_ENV} is not set")
    fallback: ChatFn | None = None
    if not args.no_fallback:
        try:
            fallback = make_chat(args.fallback_base_url, args.fallback_model, None, 0.0)
        except ValueError as exc:
            return _fail(str(exc))
    hosted = _Hosted(_client(key), args.model, args.max_output_tokens, key)
    spec = _EditorSpec(hosted, fallback, args.fallback_model or "")
    try:
        summary = anyio.run(
            _run, args.run_dir, args.unsafe_kernel, spec, args.max_turns
        )
    except LoopError as exc:
        print(json.dumps({"error": _scrub(str(exc), key)}))
        return 1
    except Exception as exc:  # noqa: BLE001 - the server did not start or died
        err: BaseException = exc
        while isinstance(err, BaseExceptionGroup) and err.exceptions:
            err = err.exceptions[0]
        print(
            f"error: could not run the server: {_scrub(repr(err), key)}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(summary, default=str))
    return 1 if summary["outcome"] == "refused" else 0


if __name__ == "__main__":
    sys.exit(main())
