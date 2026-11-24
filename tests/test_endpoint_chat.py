import json
from typing import Any

import pytest
from fake_endpoint import ok as _ok
from fake_endpoint import serve, tool_reply

from astro_weights.eval import ToolCall
from astro_weights.runners import endpoint
from astro_weights.runners.endpoint import ChatReply, chat_http

KEY = "sk-secret-key-123"
MSGS = [{"role": "user", "content": "hi"}]
TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {}}}]


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(endpoint, "_RETRY_SLEEP", 0.0)


def _chat(url: str, **kw: Any) -> ChatReply:
    return chat_http(url, "m", MSGS, max_new_tokens=16, **kw)


def _raw(message: dict[str, Any]) -> tuple[int, str]:
    obj = {"choices": [{"message": message}], "usage": {"completion_tokens": 1}}
    return 200, json.dumps(obj)


def test_text_reply() -> None:
    usage = {"completion_tokens": 7, "x": 1}
    with serve(lambda b: _ok("hello", usage=usage)) as (u, s):
        r = _chat(u, api_key=KEY)
    msg = {"role": "assistant", "content": "hello"}
    assert r == ChatReply("hello", (), 7, "stop", usage, msg)
    body = s.requests[0]["body"]
    assert (body["temperature"], body["max_tokens"], body["seed"]) == (0, 16, 0)
    assert "tools" not in body
    assert s.requests[0]["auth"] == f"Bearer {KEY}"


def test_tool_call_reply_and_request_body() -> None:
    with serve(lambda b: tool_reply(("f", {"x": 1}), ("g", "{}"), tokens=(3, 9))) as (
        u,
        s,
    ):
        r = _chat(u, tools=TOOLS, extra_body={"seed": 5, "top_k": 1}, seed=2)
    assert r.tool_calls == (ToolCall("f", '{"x": 1}'), ToolCall("g", "{}"))
    assert (r.output_tokens, r.finish_reason, r.text) == (9, "tool_calls", "")
    assert r.message["content"] is None
    assert [c["id"] for c in r.message["tool_calls"]] == ["call_0", "call_1"]
    fn = r.message["tool_calls"][0]["function"]
    assert fn == {"name": "f", "arguments": '{"x": 1}'}
    body = s.requests[0]["body"]
    assert (body["tools"], body["seed"], body["top_k"]) == (TOOLS, 5, 1)


def test_arguments_object_missing_arguments_and_missing_id() -> None:
    calls = [
        {"function": {"name": "a", "arguments": {"k": [1]}}},
        {"id": "", "function": {"name": "b"}},
        {"id": "x9", "function": {"name": "c", "arguments": "{}"}},
    ]
    with serve(lambda b: _raw({"content": None, "tool_calls": calls})) as (u, _):
        r = _chat(u)
    assert [c.arguments for c in r.tool_calls] == ['{"k": [1]}', "{}", "{}"]
    assert [c["id"] for c in r.message["tool_calls"]] == ["call_0", "call_1", "x9"]
    assert r.finish_reason is None


def test_retries_503_then_ok() -> None:
    n = {"n": 0}

    def respond(b: dict[str, Any]) -> tuple[int, str]:
        n["n"] += 1
        return (503, "busy") if n["n"] < 3 else _ok("fine")

    with serve(respond) as (u, s):
        assert _chat(u).text == "fine"
    assert len(s.requests) == 3


def test_server_error_is_scrubbed() -> None:
    with (
        serve(lambda b: (500, f"boom {KEY} boom")) as (u, _),
        pytest.raises(RuntimeError, match=r"server error after 3 attempts") as e,
    ):
        _chat(u, api_key=KEY)
    assert KEY not in str(e.value)
    assert "***" in str(e.value)


@pytest.mark.parametrize(
    ("reply", "match"),
    [
        (_ok("x", usage={}), "usage.completion_tokens"),
        (_raw({"content": 5}), "message content"),
        (_raw({"tool_calls": "no"}), "not a list"),
        (_raw({"tool_calls": [1]}), "malformed tool call 0"),
        (
            _raw({"tool_calls": [{"function": {"name": "f", "arguments": 3}}]}),
            "malformed tool call 0",
        ),
    ],
)
def test_bad_reply_is_value_error_without_key(
    reply: tuple[int, str], match: str
) -> None:
    with (
        serve(lambda b: reply) as (u, _),
        pytest.raises(ValueError, match=match) as e,
    ):
        _chat(u, api_key=KEY)
    assert KEY not in str(e.value)


def test_bad_url_and_key() -> None:
    with pytest.raises(ValueError, match="http"):
        chat_http("ftp://x", "m", MSGS, max_new_tokens=1)
    with pytest.raises(ValueError, match="api_key"):
        chat_http("http://127.0.0.1:1", "m", MSGS, max_new_tokens=1, api_key="a b")
