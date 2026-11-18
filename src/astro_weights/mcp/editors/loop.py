"""Shared pieces of a model-driven editor: the tool list, a result cap and the turn loop.

An MCP client side. ``ToolBox`` turns the server's tools into OpenAI chat-completions
``tools`` (the three editor-only tools stay hidden from the model), runs one call and
waits for a job to end, so the model sees one result per call. ``run_loop`` asks a chat
function for a reply each turn, reports the turn's usage to the ledger and runs the
reply's tool calls. It needs the ``mcp`` extra and never imports the server's internals.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

try:
    import anyio
    import anyio.to_thread
    from mcp import Client, ClientSession
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"{__name__} needs the official MCP SDK: pip install 'astro-weights[mcp]'",
        name=e.name,
    ) from e

from astro_weights.mcp.editors.scripted import _call, _poll

__all__ = [
    "EDITOR_ONLY",
    "RESULT_CAP",
    "ChatFn",
    "LoopError",
    "Message",
    "OnTurn",
    "Refused",
    "Reply",
    "ToolBox",
    "run_loop",
    "system_prompt",
]

Session = Client | ClientSession
Message = dict[str, Any]
EDITOR_ONLY = frozenset({"report_usage", "check_spend", "report_refusal"})
_JOB_TOOLS = frozenset({"exec_python", "forward_pass"})
RESULT_CAP = 16 * 1024
_NUDGE = "Call one of the tools. Call finish when you are done."


class LoopError(RuntimeError):
    """The model endpoint or the server failed in a way the loop cannot go on from."""


class Refused(Exception):
    """The model refused the request; ``category`` and ``detail`` go to the record."""

    def __init__(self, category: str, detail: str = "") -> None:
        super().__init__(f"{category}: {detail}" if detail else category)
        self.category = category
        self.detail = detail


@dataclass(frozen=True)
class Reply:
    """One assistant turn: its message and the tokens it used (prompt plus output)."""

    message: Message
    tokens: int


ChatFn = Callable[[list[Message], list[dict[str, Any]]], Reply]
OnTurn = Callable[[int, Reply], None]


def system_prompt() -> str:
    """The packaged agent system prompt."""
    return (
        resources.files("astro_weights.mcp.editors")
        .joinpath("prompts/agent.md")
        .read_text("utf-8")
    )


def _strip_titles(schema: Any) -> Any:
    """Drop the generated ``title`` strings from a JSON schema; they only cost tokens."""
    if isinstance(schema, dict):
        return {
            k: _strip_titles(v)
            for k, v in schema.items()
            if not (k == "title" and isinstance(v, str))
        }
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


def _cap(text: str) -> str:
    if len(text) <= RESULT_CAP:
        return text
    marker = f" ...[cut at {RESULT_CAP} bytes, {len(text)} in all]"
    return text[: RESULT_CAP - len(marker)] + marker


@dataclass
class ToolBox:
    """The server's tools as the model sees them, and a way to call them."""

    session: Session
    tools: list[dict[str, Any]]
    summaries: list[str] = field(default_factory=list)
    poll_s: float = 0.2
    timeout_s: float = 600

    @classmethod
    async def open(
        cls, session: Session, *, poll_s: float = 0.2, timeout_s: float = 600
    ) -> "ToolBox":
        """List the server's tools, drop the editor-only ones, convert the rest."""
        listed = await session.list_tools()
        tools: list[dict[str, Any]] = []
        summaries: list[str] = []
        for tool in listed.tools:
            if tool.name in EDITOR_ONLY:
                continue
            description = tool.description or ""
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": description,
                        "parameters": _strip_titles(tool.input_schema),
                    },
                }
            )
            first = description.strip().split("\n", 1)[0]
            summaries.append(f"- {tool.name}: {first}")
        return cls(session, tools, summaries, poll_s, timeout_s)

    @property
    def names(self) -> list[str]:
        return [t["function"]["name"] for t in self.tools]

    async def run(self, name: str, args: object) -> dict[str, Any]:
        """Call one tool; for a job tool, wait until the job is done or failed."""
        if name not in self.names:
            return {"error": f"unknown tool {name!r}; tools: {', '.join(self.names)}"}
        if not isinstance(args, dict):
            return {"error": "arguments must be a JSON object"}
        out = await _call(self.session, name, args)
        handle = out.get("handle")
        if name not in _JOB_TOOLS or not isinstance(handle, str):
            return out
        deadline = anyio.current_time() + self.timeout_s
        done = await _poll(self.session, handle, self.poll_s, deadline)
        if done is None:
            return {"error": "job did not end in time", "handle": handle}
        return {"handle": handle, **done}

    @staticmethod
    def render(out: Mapping[str, Any]) -> str:
        """The result as JSON text, cut at 16 KiB with a marker."""
        return _cap(json.dumps(out, default=str))

    async def call(self, name: str, args: object) -> str:
        """``run`` and render the result for the model."""
        return self.render(await self.run(name, args))


def _first_message(box: ToolBox, budget: object) -> str:
    return (
        "Find a weight edit that cuts output tokens without losing accuracy.\n"
        f"Budget: {json.dumps(budget, default=str)}\n"
        "Tools:\n" + "\n".join(box.summaries)
    )


@dataclass(frozen=True)
class _Call:
    id: str
    name: str
    raw: str  # the arguments as the model sent them
    args: object  # parsed; None when not valid JSON


def _parse_calls(message: Message, turn: int) -> list[_Call]:
    calls: list[_Call] = []
    raw = message.get("tool_calls")
    for i, tc in enumerate(raw if isinstance(raw, list) else []):
        fn = tc.get("function") if isinstance(tc, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            continue
        given = tc.get("id")
        call_id = given if isinstance(given, str) and given else f"call_{turn}_{i}"
        args = fn.get("arguments")
        text = args if isinstance(args, str) else json.dumps(args or {})
        try:
            parsed: object = json.loads(text or "{}")
        except ValueError:
            parsed = None
        calls.append(_Call(call_id, fn["name"], text, parsed))
    return calls


def _assistant(message: Message, calls: list[_Call]) -> Message:
    content = message.get("content")
    out: Message = {
        "role": "assistant",
        "content": content if isinstance(content, str) else "",
    }
    if calls:
        out["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.raw},
            }
            for c in calls
        ]
    return out


async def _report(box: ToolBox, tokens: int, turn: int, model: str) -> dict[str, Any]:
    out = await _call(
        box.session,
        "report_usage",
        {"tokens": tokens, "dollars": 0.0, "turn": turn, "model": model},
    )
    if out.get("error"):
        raise LoopError(f"report_usage failed: {json.dumps(out, default=str)[:500]}")
    return out


async def _execute(
    box: ToolBox, calls: list[_Call], messages: list[Message]
) -> tuple[str, Any] | None:
    """Run the calls in order; (summary, budget) once the model's ``finish`` worked."""
    for call in calls:
        out = await box.run(call.name, call.args)
        text = box.render(out)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
        if call.name == "finish" and "error" not in out:
            given = call.args.get("summary") if isinstance(call.args, dict) else None
            return str(given), out.get("budget")
    return None


async def run_loop(
    session: Session,
    chat_fn: ChatFn,
    *,
    max_turns: int,
    system: str,
    model: str,
    on_turn: OnTurn | None = None,
    poll_s: float = 0.2,
    timeout_s: float = 600,
) -> dict[str, Any]:
    """Drive the server with a model until it calls ``finish`` or a stop rule fires.

    Each turn ``chat_fn(messages, tools)`` makes one model call; its tokens go to the
    ledger (dollars 0) before its tool calls run, in order. Stops when the model calls
    ``finish``, when ``max_turns`` is reached or the budget is spent (the loop calls
    ``finish`` itself), or when the model twice in a row replies without a tool call
    (one nudge first). A ``Refused`` from ``chat_fn`` is sent to the record, then
    raised again. Returns ``{outcome, turns, tokens, summary, budget}``.
    """
    box = await ToolBox.open(session, poll_s=poll_s, timeout_s=timeout_s)
    start = await _call(session, "check_spend", {"estimate": 0.0})
    messages: list[Message] = [
        {"role": "system", "content": system},
        {"role": "user", "content": _first_message(box, start.get("budget"))},
    ]
    state: dict[str, Any] = {
        "outcome": "turn_cap",
        "turns": 0,
        "tokens": 0,
        "summary": "turn cap",
        "budget": start.get("budget"),
    }
    quiet = 0
    for turn in range(1, max_turns + 1):
        try:
            reply = await anyio.to_thread.run_sync(chat_fn, messages, box.tools)
        except Refused as refusal:
            await _call(
                session,
                "report_refusal",
                {
                    "category": refusal.category,
                    "model": model,
                    "detail": refusal.detail,
                },
            )
            raise
        state["turns"] = turn
        state["tokens"] += reply.tokens
        usage = await _report(box, reply.tokens, turn, model)
        state["budget"] = usage.get("budget")
        if on_turn is not None:
            on_turn(turn, reply)
        calls = _parse_calls(reply.message, turn)
        messages.append(_assistant(reply.message, calls))
        if usage.get("over"):
            state["outcome"], state["summary"] = "budget", "budget spent"
            break
        if not calls:
            quiet += 1
            if quiet == 2:
                state["outcome"], state["summary"] = "no_tool_call", "no tool call"
                break
            messages.append({"role": "user", "content": _NUDGE})
            continue
        quiet = 0
        ended = await _execute(box, calls, messages)
        if ended is not None:
            state["outcome"], state["summary"] = "finished", ended[0]
            state["budget"] = ended[1] or state["budget"]
            return state
    final = await box.run("finish", {"summary": state["summary"]})
    state["budget"] = final.get("budget", state["budget"])
    return state
