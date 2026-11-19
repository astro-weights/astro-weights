"""Hosted editor tests: a fake Anthropic client, no network, no key."""

import json
import math
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self, cast

import pytest

from astro_weights.mcp.editors import prices
from astro_weights.mcp.editors.prices import Price, max_cost, price, turn_cost

MODEL = "claude-opus-5-5"
KEY = "sk-ant-test-key-0123456789"
LIMITS: dict[str, Any] = {
    "forward_passes": 5,
    "code_runs": 5,
    "tokens": 100_000,
    "dollars": 1.0,
}


# --- prices (standard library only) ----------------------------------------------


def test_turn_cost_is_exact_from_the_table() -> None:
    usage = {
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_read_input_tokens": 5000,
        "cache_creation_input_tokens": 3000,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 2000,
            "ephemeral_1h_input_tokens": 1000,
        },
    }
    want = (1000 * 4 + 2000 * 5 + 1000 * 8 + 5000 * 0.20 + 200 * 20) / 1e6
    assert turn_cost(MODEL, usage) == pytest.approx(want, rel=1e-12)


def test_cache_writes_without_a_split_are_five_minute_writes() -> None:
    usage: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 10,
    }
    assert turn_cost(MODEL, usage) == pytest.approx(10 * 5 / 1e6)
    usage["cache_creation"] = None
    assert turn_cost(MODEL, usage) == pytest.approx(10 * 5 / 1e6)


def test_us_inference_costs_more() -> None:
    usage: dict[str, Any] = {"input_tokens": 1_000_000, "output_tokens": 0}
    assert turn_cost(MODEL, usage) == pytest.approx(4.0)
    usage["inference_geo"] = "global"
    assert turn_cost(MODEL, usage) == pytest.approx(4.0)
    usage["inference_geo"] = "us"
    assert turn_cost(MODEL, usage) == pytest.approx(4.4)


def test_long_prompts_switch_price_tier() -> None:
    model = "claude-haiku-5-5"
    assert price(model, 100_000).input == 0.10
    assert price(model, 100_001).input == 0.50
    usage = {"input_tokens": 60_000, "cache_read_input_tokens": 50_000}
    assert turn_cost(model, {**usage, "output_tokens": 0}) == pytest.approx(
        (60_000 * 0.50 + 50_000 * 0.05) / 1e6
    )


def test_unknown_model_has_no_price() -> None:
    with pytest.raises(ValueError, match="no price for model 'claude-x'"):
        price("claude-x")
    with pytest.raises(ValueError, match="no price"):
        turn_cost("gpt-4", {"input_tokens": 1, "output_tokens": 1})
    with pytest.raises(ValueError, match="no price"):
        max_cost("claude-x", 1, 1)


@pytest.mark.parametrize(
    "usage",
    [
        {"output_tokens": 1},
        {"input_tokens": 1},
        {"input_tokens": -1, "output_tokens": 1},
        {"input_tokens": True, "output_tokens": 1},
        {"input_tokens": 1.5, "output_tokens": 1},
        {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": "3"},
        {"input_tokens": 1, "output_tokens": 1, "cache_creation": [1]},
    ],
)
def test_bad_usage_is_refused(usage: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="usage"):
        turn_cost(MODEL, usage)


def test_max_cost_bounds_any_turn_with_that_prompt() -> None:
    p = price(MODEL)
    assert max_cost(MODEL, 1000, 100) == pytest.approx(
        (1000 * p.cache_write_5m + 100 * p.output) / 1e6 * prices.US_GEO
    )
    worst = {
        "input_tokens": 0,
        "cache_creation_input_tokens": 1000,
        "output_tokens": 100,
        "inference_geo": "us",
    }
    assert turn_cost(MODEL, worst) <= max_cost(MODEL, 1000, 100)
    with pytest.raises(ValueError, match=">= 0"):
        max_cost(MODEL, -1, 1)


@pytest.mark.parametrize(
    "text",
    [
        "1,2,3,0.1",
        "1,2,3,0.1,5,6",
        "1,2,x,0.1,5",
        "1,2,3,-0.1,5",
        "1,2,inf,0,5",
        "nan,1,2,0,5",
        "",
    ],
)
def test_bad_price_text_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match="price"):
        prices.parse_price(text)


def test_given_price_overrides_the_table() -> None:
    given = prices.parse_price("1, 2,3,0.5,10")
    assert given == Price(1, 2, 3, 0.5, 10)
    assert price("claude-x", given=given) is given
    assert price("claude-haiku-5-5", 200_000, given) is given  # no long tier
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    assert turn_cost("claude-x", usage, given) == pytest.approx(11)
    assert max_cost("claude-x", 1_000_000, 0, given) == pytest.approx(2 * prices.US_GEO)
    with pytest.raises(ValueError, match=r"\[editor\] price; hosted editor: --price"):
        price("claude-x")


def test_every_price_is_sane() -> None:
    for model, p in prices.PRICES.items():
        assert model.startswith("claude-")
        assert isinstance(p, Price)
        assert 0 < p.cache_read < p.input < p.cache_write_5m < p.cache_write_1h
        assert p.input < p.output


# --- the hosted editor -----------------------------------------------------------

anthropic = pytest.importorskip("anthropic")
pytest.importorskip("mcp")
httpx2 = pytest.importorskip("httpx2")

from anthropic.types import Message
from fake_endpoint import serve, tool_reply
from mcp import Client
from mcp.client.stdio import get_default_environment

from astro_weights.mcp.budget import Ledger, Limits
from astro_weights.mcp.editors import hosted
from astro_weights.mcp.editors.hosted import (
    HANDOFF,
    KEY_ENV,
    _convert,
    _drive,
    _EditorSpec,
    _Hosted,
    _server_params,
    main,
)
from astro_weights.mcp.editors.loop import LoopError, Refused, Reply, system_prompt
from astro_weights.mcp.record import RunRecord
from astro_weights.mcp.server import build_server
from astro_weights.runners import endpoint


@pytest.fixture(autouse=True)
def _no_key_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("tests must not build a real Anthropic client")

    monkeypatch.setattr(anthropic, "Anthropic", refuse)
    monkeypatch.setattr(endpoint, "_RETRY_SLEEP", 0.0)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def write_run(run_dir: Path, **over: Any) -> Path:
    cfg = {"seed": 7, "editor": "hosted", "model": None, "limits": LIMITS, **over}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(cfg))
    return run_dir


def use(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"type": "tool_use", "id": call_id, "name": name, "input": args}


THINK = {"type": "thinking", "thinking": "", "signature": "sig-1"}


def reply(
    *blocks: dict[str, Any],
    stop: str = "tool_use",
    usage: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> Message:
    return Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": list(blocks),
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": usage or {"input_tokens": 100, "output_tokens": 10},
            "stop_details": details,
        }
    )


def refusal(category: str | None = "cyber") -> Message:
    details = {"type": "refusal", "category": category, "explanation": "declined"}
    return reply(
        {"type": "text", "text": "partial answer"},
        stop="refusal",
        usage={"input_tokens": 300, "output_tokens": 5},
        details=details,
    )


class FakeStream:
    def __init__(self, message: "Message | MidStream") -> None:
        self.message = message

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def get_final_message(self) -> Message:
        if isinstance(self.message, MidStream):
            raise self.message.error
        return self.message


class MidStream:
    """A stream that breaks off after the request went out."""

    def __init__(self, error: Exception) -> None:
        self.error = error


class FakeMessages:
    def __init__(
        self, replies: list[Message | MidStream | Exception], count: int
    ) -> None:
        self.replies = replies
        self.count = count
        self.counted: list[dict[str, Any]] = []
        self.sent: list[dict[str, Any]] = []
        self.count_error: Exception | None = None

    def count_tokens(self, **kw: Any) -> Any:
        self.counted.append(kw)
        if self.count_error is not None:
            raise self.count_error
        return SimpleNamespace(input_tokens=self.count)

    def stream(self, **kw: Any) -> FakeStream:
        self.sent.append(json.loads(json.dumps(kw)))  # a copy, as sent
        nxt = self.replies.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return FakeStream(nxt)


class FakeClient:
    def __init__(
        self, *replies: Message | MidStream | Exception, count: int = 1000
    ) -> None:
        self.messages = FakeMessages(list(replies), count)


def api_error(text: str) -> Exception:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    err: Exception = anthropic.APIConnectionError(message=text, request=request)
    return err


def scripted(*replies: Reply | Exception) -> Any:
    queue = list(replies)
    seen: list[list[dict[str, Any]]] = []

    def chat(messages: list[dict[str, Any]], _tools: list[dict[str, Any]]) -> Reply:
        seen.append([dict(m) for m in messages])
        nxt = queue.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    chat.seen = seen  # type: ignore[attr-defined]
    return chat


def finish_call(summary: str = "done") -> Reply:
    call = {
        "id": "local_1",
        "type": "function",
        "function": {"name": "finish", "arguments": json.dumps({"summary": summary})},
    }
    return Reply({"role": "assistant", "content": "", "tool_calls": [call]}, 30)


EXECUTORS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "exec": lambda p: {"ok": True, "output": "hi"},
    "forward": lambda p: {"ok": True},
}


async def drive(
    run_dir: Path,
    client: FakeClient,
    fallback: Any = None,
    *,
    max_output: int = 1000,
    max_turns: int = 6,
    wrap: Any = None,
) -> dict[str, Any]:
    spec = _EditorSpec(
        _Hosted(client, MODEL, max_output, KEY), fallback, "local-m" if fallback else ""
    )
    failure: LoopError | None = None
    async with Client(build_server(run_dir, EXECUTORS)) as session:
        s = session if wrap is None else wrap(session)
        try:
            return await _drive(s, spec, max_turns, poll_s=0.01)
        except LoopError as exc:  # keep it out of the transport's group
            failure = exc
    raise failure


def events(run_dir: Path) -> list[dict[str, Any]]:
    return RunRecord.open(run_dir / "record.jsonl").read()


def used(run_dir: Path) -> dict[str, float]:
    return Ledger(run_dir / "ledger.jsonl", Limits(**LIMITS)).used()


# --- message conversion ----------------------------------------------------------


def test_convert_joins_tool_results_and_keeps_raw_assistant_turns() -> None:
    raw_blocks = [THINK, use("status", {"handle": "h"}, "toolu_1")]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "toolu_1", "function": {}}],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": '{"x": 1}'},
        {"role": "user", "content": "more"},
        {"role": "user", "content": ""},
    ]
    system, turns = _convert(messages, {2: (("toolu_1",), raw_blocks)})
    assert system == "sys"
    assert turns == [
        {"role": "user", "content": [{"type": "text", "text": "go"}]},
        {"role": "assistant", "content": raw_blocks},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": '{"x": 1}',
                },
                {"type": "text", "text": "more"},
            ],
        },
    ]


def test_convert_refuses_foreign_turns_and_roles() -> None:
    turn = {"role": "assistant", "content": "x"}
    with pytest.raises(LoopError, match="not made by this editor"):
        _convert([turn], {})
    with pytest.raises(LoopError, match="not made by this editor"):
        _convert([turn], {0: (("other",), [])})
    with pytest.raises(LoopError, match="unknown message role"):
        _convert([{"role": "developer", "content": "x"}], {})


# --- runs against the real server (in process) -----------------------------------


@pytest.mark.anyio
async def test_each_turn_is_capped_counted_and_charged(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)
    usage1 = {
        "input_tokens": 900,
        "output_tokens": 40,
        "cache_creation_input_tokens": 100,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 100,
            "ephemeral_1h_input_tokens": 0,
        },
    }
    usage2 = {"input_tokens": 50, "output_tokens": 20, "cache_read_input_tokens": 1000}
    client = FakeClient(
        reply(
            THINK,
            {"type": "text", "text": "trying"},
            use("exec_python", {"code": "1"}, "toolu_1"),
            usage=usage1,
        ),
        reply(use("finish", {"summary": "all done"}, "toolu_2"), usage=usage2),
    )
    out = await drive(run_dir, client)
    assert out["outcome"] == "finished"
    assert out["summary"] == "all done"
    assert out["fallback"] is None
    cost = turn_cost(MODEL, usage1) + turn_cost(MODEL, usage2)
    assert out["dollars"] == pytest.approx(cost)
    assert out["tokens"] == 1040 + 1070
    ledger = used(run_dir)
    assert ledger["dollars"] == pytest.approx(cost)
    assert ledger["tokens"] == 2110
    usage = [e for e in events(run_dir) if e["event"] == "usage"]
    charged = [(e["turn"], e["model"], e["tokens"]) for e in usage]
    assert charged == [(1, MODEL, 1040), (2, MODEL, 1070)]
    assert [e["dollars"] for e in usage] == pytest.approx(
        [turn_cost(MODEL, usage1), turn_cost(MODEL, usage2)]
    )
    # each request: the four model tools, streamed, auto-cached, capped output
    first, second = client.messages.sent
    assert [t["name"] for t in first["tools"]] == [
        "exec_python",
        "forward_pass",
        "status",
        "finish",
    ]
    assert first["max_tokens"] == 1000
    assert first["cache_control"] == {"type": "ephemeral"}
    assert first["model"] == MODEL
    assert first["system"] == system_prompt()
    # the assistant turn goes back exactly as returned, thinking block included
    assert second["messages"][1] == {
        "role": "assistant",
        "content": [
            THINK,
            {"type": "text", "text": "trying"},
            use("exec_python", {"code": "1"}, "toolu_1"),
        ],
    }
    result = second["messages"][2]["content"][0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "toolu_1"
    assert len(client.messages.counted) == 2  # one count before every call


@pytest.mark.anyio
async def test_dollar_cap_stops_before_the_call(tmp_path: Path) -> None:
    # estimate per call: (4000 * 1.05 * $5 + 1000 * $20) / 1M * 1.1 = $0.0451
    run_dir = write_run(tmp_path, limits={**LIMITS, "dollars": 0.06})
    spend = {"input_tokens": 4000, "output_tokens": 1000}  # $0.036
    client = FakeClient(
        reply(use("status", {"handle": "h"}, "toolu_1"), usage=spend),
        reply(use("finish", {"summary": "never"}, "toolu_2")),
        count=4000,
    )
    out = await drive(run_dir, client)
    assert out["outcome"] == "dollar_cap"
    assert len(client.messages.sent) == 1  # no call past the cap
    assert len(client.messages.counted) == 2
    assert used(run_dir)["dollars"] == pytest.approx(0.036)
    names = [e["event"] for e in events(run_dir)]
    assert names[-1] == "finish"
    summary = next(e for e in events(run_dir) if e["event"] == "summary")
    assert summary["text"] == "dollar cap"


@pytest.mark.anyio
async def test_refusal_hands_the_run_to_the_local_model(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)
    client = FakeClient(
        reply(use("exec_python", {"code": "1"}, "toolu_1")),
        refusal("cyber"),
    )
    local = scripted(finish_call("local finished"))
    out = await drive(run_dir, client, local)
    assert out["outcome"] == "finished"
    assert out["summary"] == "local finished"
    assert out["fallback"] == {"turn": 2, "category": "cyber", "model": "local-m"}
    record = events(run_dir)
    refused = [e for e in record if e["event"] == "refusal"]
    assert [(e["category"], e["model"], e["detail"]) for e in refused] == [
        ("cyber", MODEL, "declined")
    ]
    # one usage event per model call, each with the model that made it
    charged = [
        (e["turn"], e["model"], e["tokens"]) for e in record if e["event"] == "usage"
    ]
    assert charged == [(1, MODEL, 110), (2, MODEL, 305), (2, "local-m", 30)]
    names = [e["event"] for e in record]
    assert names.index("refusal") > max(
        i for i, e in enumerate(record) if e.get("tokens") == 305
    )
    ledger = used(run_dir)
    assert ledger["tokens"] == 110 + 305 + 30
    refused_cost = turn_cost(MODEL, {"input_tokens": 300, "output_tokens": 5})
    assert ledger["dollars"] == pytest.approx(
        turn_cost(MODEL, {"input_tokens": 100, "output_tokens": 10}) + refused_cost
    )
    # the refused partial never reaches the local model; the handoff is last
    (seen,) = local.seen
    assert seen[-1] == {"role": "user", "content": HANDOFF}
    assert "partial answer" not in json.dumps(seen)
    assert any(m["role"] == "tool" for m in seen)


@pytest.mark.anyio
async def test_refusal_without_a_category(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)
    client = FakeClient(refusal(None))
    out = await drive(run_dir, client, scripted(finish_call()))
    assert out["fallback"]["category"] == "refusal"
    bare = reply(stop="refusal")  # no stop_details at all
    run2 = write_run(tmp_path / "two")
    out = await drive(run2, FakeClient(bare), scripted(finish_call()))
    assert out["fallback"]["category"] == "refusal"
    refused = next(e for e in events(run2) if e["event"] == "refusal")
    assert refused["detail"] == ""


@pytest.mark.anyio
async def test_refusal_with_no_fallback_finishes_refused(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)
    out = await drive(run_dir, FakeClient(refusal()))
    assert out["outcome"] == "refused"
    assert out["fallback"] is None
    record = events(run_dir)
    assert next(e for e in record if e["event"] == "summary")["text"] == "refused"
    assert record[-1]["event"] == "finish"


@pytest.mark.anyio
async def test_local_refusal_also_finishes_refused(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)
    local = scripted(Refused("content_filter", "no", tokens=12))
    out = await drive(run_dir, FakeClient(refusal()), local)
    assert out["outcome"] == "refused"
    refused = [
        (e["category"], e["model"]) for e in events(run_dir) if e["event"] == "refusal"
    ]
    assert refused == [("cyber", MODEL), ("content_filter", "local-m")]
    assert used(run_dir)["tokens"] == 305 + 12
    no_tokens = write_run(tmp_path / "two")
    out = await drive(no_tokens, FakeClient(refusal()), scripted(Refused("x")))
    assert out["outcome"] == "refused"
    assert used(no_tokens)["tokens"] == 305


@pytest.mark.anyio
async def test_text_only_turn_and_local_turns_after_handoff(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)
    client = FakeClient(
        reply({"type": "text", "text": "thinking aloud"}, stop="end_turn"), refusal()
    )
    status = Reply(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "l1",
                    "type": "function",
                    "function": {"name": "status", "arguments": '{"handle": "h"}'},
                }
            ],
        },
        8,
    )
    local = scripted(status, finish_call())
    out = await drive(run_dir, client, local)
    assert out["outcome"] == "finished"
    assert out["turns"] == 3
    assert out["tokens"] == 110 + 305 + 8 + 30
    usage = [(e["turn"], e["model"]) for e in events(run_dir) if e["event"] == "usage"]
    assert usage == [(1, MODEL), (2, MODEL), (2, "local-m"), (3, "local-m")]
    # the nudge after the text-only turn reached Claude in the same user turn
    assert client.messages.sent[1]["messages"][-1]["role"] == "user"
    assert len(local.seen) == 2


@pytest.mark.anyio
async def test_api_errors_fail_without_the_key(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)
    with pytest.raises(LoopError) as info:
        await drive(run_dir, FakeClient(api_error(f"bad key {KEY}")))
    assert "Messages API call failed" in str(info.value)
    assert KEY not in str(info.value)
    client = FakeClient()
    client.messages.count_error = api_error(f"count {KEY}")
    with pytest.raises(LoopError) as info:
        await drive(write_run(tmp_path / "two"), client)
    assert "token count failed" in str(info.value)
    assert KEY not in str(info.value)


@pytest.mark.anyio
async def test_stream_that_breaks_off_is_charged_the_bound(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)
    broken = MidStream(httpx2.ReadError(f"reset while reading {KEY}"))
    with pytest.raises(LoopError) as info:
        await drive(run_dir, FakeClient(broken, count=2000), max_output=500)
    assert "broke off: ReadError" in str(info.value)
    assert KEY not in str(info.value)
    prompt = math.ceil(2000 * 1.05)
    ledger = used(run_dir)
    assert ledger["dollars"] == pytest.approx(max_cost(MODEL, prompt, 500))
    assert ledger["tokens"] == prompt + 500
    (usage,) = [e for e in events(run_dir) if e["event"] == "usage"]
    assert usage["model"] == MODEL


@pytest.mark.anyio
async def test_unpriceable_usage_is_a_loop_error(tmp_path: Path) -> None:
    bad = reply(usage={"input_tokens": -1, "output_tokens": 1})
    with pytest.raises(LoopError, match="cannot price"):
        await drive(write_run(tmp_path), FakeClient(bad))


class Tap:
    """A session that answers one editor tool with an error."""

    def __init__(self, inner: Any, tool: str, after: int = 0) -> None:
        self.inner, self.tool, self.after = inner, tool, after

    async def list_tools(self) -> Any:
        return await self.inner.list_tools()

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        if name == self.tool:
            if self.after == 0:
                return SimpleNamespace(
                    structured_content={"error": "nope"}, is_error=False, content=[]
                )
            self.after -= 1
        return await self.inner.call_tool(name, args)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool", "after", "match"),
    [
        ("check_spend", 1, "check_spend failed"),  # the first is run_loop's own
        ("report_usage", 0, "report_usage failed"),
        ("report_refusal", 0, "report_refusal failed"),
    ],
)
async def test_editor_tool_errors_are_loop_errors(
    tmp_path: Path, tool: str, after: int, match: str
) -> None:
    run_dir = write_run(tmp_path)
    with pytest.raises(LoopError, match=match):
        await drive(
            run_dir,
            FakeClient(refusal()),
            scripted(finish_call()),
            wrap=lambda s: Tap(s, tool, after),
        )


# --- the key and the server's environment ----------------------------------------


def test_server_child_gets_no_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(KEY_ENV, KEY)
    params = _server_params(tmp_path, unsafe=False)
    env = get_default_environment() | (params.env or {})
    assert KEY_ENV not in env
    assert KEY not in json.dumps(env)
    assert params.args[-1] == str(tmp_path)
    assert _server_params(tmp_path, unsafe=True).args[-1] == "--unsafe-kernel"


def test_key_and_client_stay_out_of_repr() -> None:
    client = FakeClient()
    spec = _EditorSpec(_Hosted(client, MODEL, 10, KEY), None, "")
    for obj in (spec, spec.hosted, spec.build(cast("Any", SimpleNamespace()))):
        assert KEY not in repr(obj)
        assert "FakeClient" not in repr(obj)
    assert MODEL in repr(spec)


def test_client_gets_the_key_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[dict[str, Any]] = []

    def make(**kw: Any) -> str:
        made.append(kw)
        return "c"

    monkeypatch.setattr(anthropic, "Anthropic", make)
    assert hosted._client(KEY) == "c"
    assert made == [{"api_key": KEY}]


# --- main ------------------------------------------------------------------------


def argv(run: Path, *extra: str) -> list[str]:
    return [str(run), "--model", MODEL, *extra]


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--fallback-model", "m"],
        ["--no-fallback", "--fallback-model", "m"],
        ["--no-fallback", "--max-dollars", "nan"],
        ["--no-fallback", "--max-turns", "0"],
    ],
)
def test_main_bad_args_exit_2(tmp_path: Path, extra: list[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(argv(write_run(tmp_path), *extra))
    assert info.value.code == 2


def test_main_refuses_to_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = write_run(tmp_path)

    def err(*args: str) -> str:
        assert main(list(args)) == 2
        return capsys.readouterr().err

    assert "not found" in err(str(tmp_path / "none"), "--model", MODEL, "--no-fallback")
    assert "no price for model" in err(str(run), "--model", "claude-x", "--no-fallback")
    # a given price passes that check; then the missing key stops it
    priced = ["--model", "claude-x", "--price", "1,2,3,0.1,5", "--no-fallback"]
    assert KEY_ENV in err(str(run), *priced)
    high = write_run(tmp_path / "high", limits={**LIMITS, "dollars": 25})
    assert "above --max-dollars 20.0" in err(*argv(high, "--no-fallback"))
    # raising the flag passes that check; then the missing key stops it
    assert KEY_ENV in err(*argv(high, "--no-fallback", "--max-dollars", "30"))
    zero = write_run(tmp_path / "zero", limits={**LIMITS, "dollars": 0})
    assert "needs a dollar budget" in err(*argv(zero, "--no-fallback"))
    monkeypatch.setenv(KEY_ENV, "")
    assert KEY_ENV in err(*argv(run, "--no-fallback"))
    monkeypatch.setenv(KEY_ENV, KEY)
    bad_url = ["--fallback-base-url", "ftp://x", "--fallback-model", "m"]
    assert "http" in err(*argv(run, *bad_url))


def fake_client(monkeypatch: pytest.MonkeyPatch, client: FakeClient) -> None:
    monkeypatch.setenv(KEY_ENV, KEY)
    seen: list[str] = []

    def build(key: str) -> FakeClient:
        seen.append(key)
        return client

    monkeypatch.setattr(hosted, "_client", build)


def all_text(run: Path, out: str, err: str) -> str:
    files = [p.read_text(errors="replace") for p in run.rglob("*") if p.is_file()]
    return "\n".join([out, err, *files])


def test_main_refusal_falls_back_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = write_run(tmp_path / "run")
    client = FakeClient(
        reply(use("status", {"handle": "nope"}, "toolu_1")), refusal("bio")
    )
    fake_client(monkeypatch, client)
    replies = [tool_reply(("finish", {"summary": "local done"}), tokens=(200, 10))]
    with serve(lambda _b: replies.pop(0)) as (url, state):
        extra = ["--fallback-base-url", url, "--fallback-model", "local-m"]
        assert main(argv(run, *extra)) == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["outcome"] == "finished"
    assert summary["fallback"]["category"] == "bio"
    assert summary["dollars"] > 0
    ledger = used(run)
    assert ledger["tokens"] == 110 + 305 + 210
    assert ledger["dollars"] == pytest.approx(summary["dollars"])
    handoff = state.requests[0]["body"]["messages"][-1]
    assert handoff == {"role": "user", "content": HANDOFF}
    assert state.requests[0]["auth"] is None  # the local server never sees the key
    assert KEY not in all_text(run, captured.out, captured.err)


def test_main_no_fallback_refusal_exits_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = write_run(tmp_path)
    fake_client(monkeypatch, FakeClient(refusal()))
    assert main(argv(run, "--no-fallback", "--max-turns", "3")) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["outcome"] == "refused"
    assert KEY not in all_text(run, captured.out, captured.err)


def test_main_dollar_cap_exits_0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = write_run(tmp_path, limits={**LIMITS, "dollars": 0.01})
    client = FakeClient(reply(use("finish", {"summary": "x"}, "t")), count=10_000)
    fake_client(monkeypatch, client)
    assert main(argv(run, "--no-fallback")) == 0
    assert json.loads(capsys.readouterr().out)["outcome"] == "dollar_cap"
    assert client.messages.sent == []


def test_main_api_failure_exits_1_without_the_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = write_run(tmp_path)
    fake_client(monkeypatch, FakeClient(api_error(f"denied for {KEY}")))
    assert main(argv(run, "--no-fallback")) == 1
    captured = capsys.readouterr()
    assert "Messages API call failed" in json.loads(captured.out)["error"]
    assert KEY not in all_text(run, captured.out, captured.err)


def test_main_server_that_does_not_start_exits_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mcp import StdioServerParameters

    run = write_run(tmp_path)
    fake_client(monkeypatch, FakeClient())
    missing = str(tmp_path / f"no-such-{KEY}")
    monkeypatch.setattr(
        hosted,
        "_server_params",
        lambda *_a: StdioServerParameters(command=missing, args=[]),
    )
    assert main(argv(run, "--no-fallback")) == 2
    captured = capsys.readouterr()
    assert "could not run the server" in captured.err
    assert KEY not in captured.out + captured.err


def test_hosted_editor_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from mcp_tiny_hf import save_tiny_model

    from astro_weights.mcp import sandbox

    model = save_tiny_model(tmp_path / "model")
    limits = {"forward_passes": 5, "code_runs": 5, "tokens": 100_000, "dollars": 2}
    run = write_run(
        tmp_path / "run",
        limits=limits,
        model_dir=str(model),
        kernel={"timeout_s": 60},
        forward={"device": "cpu", "max_new_tokens": 8},
    )
    edit = (
        "name = next(n for n in NAMES if 'mlp.down_proj' in n)\n"
        "W = load(name)\n"
        "rng = np.random.default_rng(0)\n"
        "d = np.outer(rng.standard_normal(W.shape[0]), rng.standard_normal(W.shape[1]))\n"
        "W2 = W + d * (0.3 * np.linalg.norm(W) / np.linalg.norm(d))\n"
        "save_edit(name, W2.astype(np.float32))\n"
        "print(name)\n"
    )
    forward = {
        "kind": "teacher_forced",
        "texts": ["The quick brown fox."],
        "return": ["hidden"],
        "layers": [0],
        "last_tokens": 4,
        "arm": "edited",
    }
    client = FakeClient(
        reply(THINK, use("exec_python", {"code": edit}, "toolu_1")),
        reply(use("forward_pass", {"request": forward}, "toolu_2")),
        refusal("frontier_llm"),
    )
    fake_client(monkeypatch, client)
    replies = [tool_reply(("finish", {"summary": "scaled one weight"}), tokens=(9, 1))]
    with serve(lambda _b: replies.pop(0)) as (url, _state):
        flags = ["--unsafe-kernel"] if sandbox.probe() is None else []
        extra = ["--fallback-base-url", url, "--fallback-model", "tiny", *flags]
        assert main(argv(run, *extra)) == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["outcome"] == "finished"
    ledger = Ledger(run / "ledger.jsonl", Limits(**limits)).used()
    assert ledger["forward_passes"] == 1
    assert ledger["code_runs"] == 1
    assert ledger["tokens"] == 110 + 110 + 305 + 10
    assert math.isclose(ledger["dollars"], summary["dollars"])
    record = RunRecord.open(run / "record.jsonl").read()
    names = [e["event"] for e in record]
    assert {"edit_saved", "refusal", "summary", "finish"} <= set(names)
    models = {e["model"] for e in record if e["event"] == "usage" and e["tokens"]}
    assert models == {MODEL, "tiny"}
    tool_result = client.messages.sent[1]["messages"][2]["content"][0]
    assert json.loads(tool_result["content"])["state"] == "done"
    assert KEY not in all_text(run, captured.out, captured.err)
