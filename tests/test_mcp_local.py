import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fake_endpoint import Reply as Wire
from fake_endpoint import ok, serve, tool_reply

from astro_weights.mcp.budget import Ledger, Limits
from astro_weights.mcp.record import RunRecord

LIMITS = {"forward_passes": 3, "code_runs": 3, "tokens": 1000, "dollars": 0}
KEY = "sk-secret-key-123"


def write_run(run_dir: Path, **over: Any) -> Path:
    cfg = {"seed": 7, "editor": "local", "model": None, "limits": LIMITS, **over}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(cfg))
    return run_dir


pytest.importorskip("mcp")

from mcp import Client

from astro_weights.mcp.editors import loop
from astro_weights.mcp.editors.local import main, make_chat
from astro_weights.mcp.editors.loop import (
    RESULT_CAP,
    LoopError,
    Refused,
    Reply,
    ToolBox,
    run_loop,
    system_prompt,
)
from astro_weights.mcp.server import build_server
from astro_weights.runners import endpoint

BUDGET = {"limits": LIMITS, "used": {"tokens": 0}}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(endpoint, "_RETRY_SLEEP", 0.0)


SERVER_TOOLS = {
    "exec_python": "Start a Python code run.\n\nMore.",
    "forward_pass": "Start a model forward pass.",
    "status": "Return a job's state.",
    "finish": "End the run.",
    "report_usage": "For editors; not meant for the model.",
    "check_spend": "For editors; not meant for the model.",
    "report_refusal": "For editors; not meant for the model.",
}


class FakeSession:
    """A duck-typed MCP session: canned tool results, every call recorded."""

    def __init__(self, **results: Any) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.results = results  # tool -> dict, or function of the arguments
        self.jobs: dict[str, list[dict[str, Any]]] = {}

    async def list_tools(self) -> Any:
        tools = [
            SimpleNamespace(
                name=name,
                description=text,
                input_schema={
                    "type": "object",
                    "title": f"{name}Arguments",
                    "properties": {"code": {"title": "Code", "type": "string"}},
                },
            )
            for name, text in SERVER_TOOLS.items()
        ]
        return SimpleNamespace(tools=tools)

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        self.calls.append((name, args))
        out = self._answer(name, args)
        return SimpleNamespace(structured_content=out, is_error=False, content=[])

    def _answer(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        if name in self.results:
            given = self.results[name]
            if callable(given):
                return cast("dict[str, Any] | None", given(args))
            return cast("dict[str, Any] | None", given)
        if name == "status":
            queue = self.jobs[args["handle"]]
            return queue.pop(0) if len(queue) > 1 else queue[0]
        if name == "report_usage":
            return {"budget": BUDGET, "over": []}
        if name == "check_spend":
            return {"ok": True, "budget": BUDGET}
        if name == "finish":
            return {"state": "finished", "budget": BUDGET}
        if name == "report_refusal":
            return {"recorded": True}
        return {"handle": "h1", "state": "queued"}

    def names(self) -> list[str]:
        return [n for n, _ in self.calls]


def as_session(fake: FakeSession) -> Any:
    return cast("Any", fake)


def scripted_chat(*replies: Reply | Exception) -> loop.ChatFn:
    queue = list(replies)

    def chat(messages: list[loop.Message], tools: list[dict[str, Any]]) -> Reply:
        scripted_chat.seen.append((list(messages), tools))  # type: ignore[attr-defined]
        nxt = queue.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    scripted_chat.seen = []  # type: ignore[attr-defined]
    return chat


def call(name: str, args: object = None, call_id: str = "c1") -> dict[str, Any]:
    raw = args if isinstance(args, str) else json.dumps(args or {})
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": raw},
    }


def turn(*calls: dict[str, Any], tokens: int = 7, content: Any = None) -> Reply:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = list(calls)
    return Reply(message, tokens)


async def run(fake: FakeSession, chat: loop.ChatFn, **kw: Any) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "max_turns": 5,
        "system": "sys",
        "model": "m",
        "poll_s": 0.001,
        "timeout_s": 2,
    }
    return await run_loop(as_session(fake), chat, **{**opts, **kw})


# --- ToolBox ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_toolbox_hides_editor_tools_and_converts_schemas() -> None:
    box = await ToolBox.open(as_session(FakeSession()))
    assert box.names == ["exec_python", "forward_pass", "status", "finish"]
    first = box.tools[0]
    assert first["type"] == "function"
    fn = first["function"]
    assert fn["name"] == "exec_python"
    assert fn["description"].startswith("Start a Python code run.")
    assert fn["parameters"] == {
        "type": "object",
        "properties": {"code": {"type": "string"}},
    }
    assert box.summaries[0] == "- exec_python: Start a Python code run."


@pytest.mark.anyio
async def test_toolbox_will_not_call_hidden_or_unknown_tools() -> None:
    fake = FakeSession()
    box = await ToolBox.open(as_session(fake))
    for name in ("report_usage", "nope"):
        out = json.loads(await box.call(name, {}))
        assert "unknown tool" in out["error"]
    assert "arguments" in json.loads(await box.call("status", "x"))["error"]
    assert fake.calls == []


@pytest.mark.anyio
async def test_result_is_capped_with_a_marker() -> None:
    big = {"state": "done", "result": {"output": "x" * 50_000}}
    fake = FakeSession(status=big)
    box = await ToolBox.open(as_session(fake))
    text = await box.call("status", {"handle": "h"})
    assert len(text) == RESULT_CAP
    assert text.endswith(f"{len(json.dumps(big))} in all]")
    assert "cut at" in text
    small = await box.call("finish", {"summary": "s"})
    assert json.loads(small)["state"] == "finished"


@pytest.mark.anyio
async def test_job_tools_are_polled_to_done() -> None:
    fake = FakeSession()
    fake.jobs["h1"] = [
        {"state": "running"},
        {"state": "running"},
        {"state": "done", "result": {"ok": True, "output": "hi"}},
    ]
    box = await ToolBox.open(as_session(fake), poll_s=0.001)
    out = json.loads(await box.call("exec_python", {"code": "1"}))
    assert out["handle"] == "h1"
    assert out["state"] == "done"
    assert out["result"]["output"] == "hi"
    assert fake.names() == ["exec_python", "status", "status", "status"]


@pytest.mark.anyio
async def test_failed_job_and_refused_submit_are_returned() -> None:
    fake = FakeSession()
    fake.jobs["h1"] = [{"state": "failed", "error": "boom"}]
    box = await ToolBox.open(as_session(fake), poll_s=0.001)
    assert (
        json.loads(await box.call("forward_pass", {"request": {}}))["error"] == "boom"
    )
    refused = FakeSession(exec_python={"error": "budget", "budget": BUDGET})
    box = await ToolBox.open(as_session(refused))
    assert json.loads(await box.call("exec_python", {"code": ""}))["error"] == "budget"
    assert refused.names() == ["exec_python"]  # no handle, nothing to poll


@pytest.mark.anyio
async def test_poll_timeout_names_the_handle() -> None:
    fake = FakeSession()
    fake.jobs["h1"] = [{"state": "running"}]
    box = await ToolBox.open(as_session(fake), poll_s=0.001, timeout_s=0.02)
    out = json.loads(await box.call("exec_python", {"code": "1"}))
    assert out == {"error": "job did not end in time", "handle": "h1"}


@pytest.mark.anyio
async def test_tool_error_result_has_an_error_key() -> None:
    fake = FakeSession(status=None)
    box = await ToolBox.open(as_session(fake))
    out = json.loads(await box.call("status", {"handle": "h"}))
    assert "returned an error" in out["error"]


# --- run_loop --------------------------------------------------------------------


@pytest.mark.anyio
async def test_finish_ends_the_loop_and_usage_is_reported_each_turn() -> None:
    fake = FakeSession()
    fake.jobs["h1"] = [{"state": "done", "result": {"ok": True}}]
    chat = scripted_chat(
        turn(call("exec_python", {"code": "1"}), tokens=11),
        turn(call("status", {"handle": "h1"}, "c2"), content="thinking", tokens=13),
        turn(call("finish", {"summary": "all done"}, "c3"), tokens=17),
    )
    seen: list[tuple[int, int]] = []
    out = await run(fake, chat, on_turn=lambda n, r: seen.append((n, r.tokens)))
    assert out["outcome"] == "finished"
    assert out["summary"] == "all done"
    assert (out["turns"], out["tokens"]) == (3, 41)
    assert seen == [(1, 11), (2, 13), (3, 17)]
    usage = [a for n, a in fake.calls if n == "report_usage"]
    assert usage == [
        {"tokens": t, "dollars": 0.0, "turn": i, "model": "m"}
        for i, t in enumerate((11, 13, 17), 1)
    ]
    first_seen, tools = scripted_chat.seen[0]  # type: ignore[attr-defined]
    assert [m["role"] for m in first_seen] == ["system", "user"]
    assert "exec_python" in first_seen[1]["content"]
    assert "report_usage" not in first_seen[1]["content"]
    assert [t["function"]["name"] for t in tools] == [
        "exec_python",
        "forward_pass",
        "status",
        "finish",
    ]
    last_messages = scripted_chat.seen[-1][0]  # type: ignore[attr-defined]
    tool_msgs = [m for m in last_messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]
    assert fake.names().count("finish") == 1


@pytest.mark.anyio
async def test_no_tool_call_gets_one_nudge_then_stops() -> None:
    fake = FakeSession()
    chat = scripted_chat(turn(content="hm"), turn(content="still hm"), turn())
    out = await run(fake, chat)
    assert out["outcome"] == "no_tool_call"
    assert out["turns"] == 2
    second = scripted_chat.seen[1][0]  # type: ignore[attr-defined]
    assert second[-1]["role"] == "user"
    assert "Call one of the tools" in second[-1]["content"]
    assert ("finish", {"summary": "no tool call"}) in fake.calls


@pytest.mark.anyio
async def test_a_tool_call_resets_the_nudge() -> None:
    fake = FakeSession()
    chat = scripted_chat(
        turn(),
        turn(call("finish", {"summary": "s"})),
    )
    out = await run(fake, chat)
    assert out["outcome"] == "finished"
    chat = scripted_chat(
        turn(),
        turn(call("exec_python", {"code": "1"})),
        turn(),
        turn(call("finish", {"summary": "s"})),
    )
    fake = FakeSession()
    fake.jobs["h1"] = [{"state": "done"}]
    assert (await run(fake, chat))["outcome"] == "finished"


@pytest.mark.anyio
async def test_turn_cap_calls_finish() -> None:
    fake = FakeSession()
    fake.jobs["h1"] = [{"state": "done"}]
    chat = scripted_chat(*[turn(call("exec_python", {"code": "1"}))] * 3)
    out = await run(fake, chat, max_turns=3)
    assert out["outcome"] == "turn_cap"
    assert out["turns"] == 3
    assert fake.calls[-1] == ("finish", {"summary": "turn cap"})
    assert out["budget"] == BUDGET


@pytest.mark.anyio
async def test_spent_budget_stops_the_loop() -> None:
    fake = FakeSession(report_usage={"budget": BUDGET, "over": ["tokens"]})
    chat = scripted_chat(turn(call("exec_python", {"code": "1"})))
    out = await run(fake, chat)
    assert out["outcome"] == "budget"
    assert fake.names().count("exec_python") == 0  # the turn's calls never ran
    assert fake.calls[-1] == ("finish", {"summary": "budget spent"})


@pytest.mark.anyio
async def test_bad_tool_calls_become_tool_results() -> None:
    fake = FakeSession()
    odd: dict[str, Any] = {
        "function": {"name": "exec_python", "arguments": {"code": "1"}}
    }
    chat = scripted_chat(
        Reply(
            {
                "content": "x",
                "tool_calls": [
                    call("exec_python", "{not json", "a"),
                    {"id": 5, "function": "nope"},
                    {"function": {"name": 5}},
                    odd,
                    call("exec_python", "", "b"),
                ],
            },
            3,
        ),
        turn(call("finish", {"summary": "s"})),
    )
    fake.jobs["h1"] = [{"state": "done"}]
    await run(fake, chat)
    msgs = scripted_chat.seen[1][0]  # type: ignore[attr-defined]
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["a", "call_1_3", "b"]
    assert "JSON object" in tool_msgs[0]["content"]
    assert "JSON object" not in tool_msgs[1]["content"]  # dict arguments are accepted
    assistant = msgs[2]
    assert assistant["tool_calls"][0]["function"]["arguments"] == "{not json"
    assert assistant["tool_calls"][1]["function"]["arguments"] == '{"code": "1"}'


@pytest.mark.anyio
async def test_finish_with_bad_arguments_does_not_end_the_run() -> None:
    fake = FakeSession(
        finish=lambda a: {"error": "bad"} if not a else {"state": "finished"}
    )
    chat = scripted_chat(
        turn(call("finish", {})), turn(call("finish", {"summary": "ok"}))
    )
    out = await run(fake, chat)
    assert out["outcome"] == "finished"
    assert out["turns"] == 2
    assert out["budget"] == BUDGET  # last finish carried none; the last report stays


@pytest.mark.anyio
async def test_refusal_goes_to_the_record_then_raises() -> None:
    fake = FakeSession()
    chat = scripted_chat(Refused("content_filter", "stopped"))
    with pytest.raises(Refused) as err:
        await run(fake, chat)
    assert err.value.category == "content_filter"
    assert str(err.value) == "content_filter: stopped"
    assert fake.calls[-1] == (
        "report_refusal",
        {"category": "content_filter", "model": "m", "detail": "stopped"},
    )
    assert str(Refused("x")) == "x"


@pytest.mark.anyio
async def test_refused_turn_reports_its_usage_first() -> None:
    fake = FakeSession()
    with pytest.raises(Refused):
        await run(fake, scripted_chat(Refused("content_filter", "x", tokens=9)))
    tail = fake.calls[-2:]
    assert tail[0] == (
        "report_usage",
        {"tokens": 9, "dollars": 0.0, "turn": 1, "model": "m"},
    )
    assert tail[1][0] == "report_refusal"


@pytest.mark.anyio
async def test_deeply_nested_arguments_are_a_tool_error() -> None:
    fake = FakeSession()
    chat = scripted_chat(
        turn(call("exec_python", "[" * 200_000)),
        turn(call("finish", {"summary": "s"})),
    )
    out = await run(fake, chat)
    assert out["outcome"] == "finished"
    tool = next(m for m in scripted_chat.seen[1][0] if m["role"] == "tool")  # type: ignore[attr-defined]
    assert "JSON object" in tool["content"]
    assert "exec_python" not in fake.names()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "start",
    [
        {"error": "run already finished"},
        {"ok": False, "error": "3 failures in a row", "budget": BUDGET},
    ],
)
async def test_run_that_cannot_start_makes_no_model_call(
    start: dict[str, Any],
) -> None:
    def chat(_m: list[loop.Message], _t: list[dict[str, Any]]) -> Reply:
        raise AssertionError("the model must not be called")

    with pytest.raises(LoopError, match="cannot start"):
        await run(FakeSession(check_spend=start), chat)


@pytest.mark.anyio
async def test_usage_failure_is_a_loop_error() -> None:
    fake = FakeSession(report_usage={"error": "run already finished"})
    with pytest.raises(LoopError, match="report_usage failed"):
        await run(fake, scripted_chat(turn()))


@pytest.mark.anyio
async def test_chat_errors_pass_through() -> None:
    with pytest.raises(LoopError, match="down"):
        await run(FakeSession(), scripted_chat(LoopError("down")))


# --- chat function over HTTP -----------------------------------------------------


def chat_once(url: str, key: str | None = None, **kw: Any) -> Reply:
    chat = make_chat(url, "m", key, kw.pop("temperature", 0.0))
    return chat([{"role": "user", "content": "hi"}], [{"type": "function"}])


def test_chat_posts_tools_and_reads_tokens() -> None:
    with serve(lambda _b: tool_reply(("status", {"handle": "h"}), tokens=(4, 6))) as (
        url,
        state,
    ):
        reply = chat_once(url + "/v1/", KEY, temperature=0.5)
    assert reply.tokens == 10
    assert reply.message["tool_calls"][0]["function"]["name"] == "status"
    sent = state.requests[0]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["auth"] == f"Bearer {KEY}"
    assert sent["body"] == {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function"}],
        "temperature": 0.5,
    }


def test_tokens_fall_back_to_prompt_plus_completion() -> None:
    usage = {"prompt_tokens": 3, "completion_tokens": 4}
    with serve(lambda _b: ok("x", usage=usage)) as (url, _):
        assert chat_once(url).tokens == 7


@pytest.mark.parametrize(
    "usage",
    [{}, {"total_tokens": True}, {"total_tokens": -1}, {"prompt_tokens": 1}, "x"],
)
def test_missing_usage_fails_clean(usage: Any) -> None:
    with (
        serve(lambda _b: ok("x", usage=usage if usage != {} else {"x": 1})) as (
            url,
            _,
        ),
        pytest.raises(LoopError, match="usage"),
    ):
        chat_once(url)


@pytest.mark.parametrize(
    ("usage", "tokens"),
    [({"prompt_tokens": 8, "completion_tokens": 7}, 15), ({"x": 1}, 0)],
)
def test_refusal_carries_the_turn_s_tokens(usage: dict[str, int], tokens: int) -> None:
    with (
        serve(lambda _b: ok("", finish="content_filter", usage=usage)) as (url, _),
        pytest.raises(Refused) as err,
    ):
        chat_once(url)
    assert err.value.tokens == tokens


def test_deeply_nested_response_fails_clean() -> None:
    with (
        serve(lambda _b: (200, "[" * 200_000)) as (url, _),
        pytest.raises(LoopError, match="chat completion"),
    ):
        chat_once(url)


def test_content_filter_is_a_refusal() -> None:
    with (
        serve(lambda _b: ok("", finish="content_filter")) as (url, _),
        pytest.raises(Refused) as err,
    ):
        chat_once(url, KEY)
    assert err.value.category == "content_filter"


@pytest.mark.parametrize("status", [500, 503, 429])
def test_server_errors_are_retried_then_fail_clean(status: int) -> None:
    body = f"oops {KEY}"
    with (
        serve(lambda _b: (status, body)) as (url, state),
        pytest.raises(LoopError) as err,
    ):
        chat_once(url, KEY)
    assert len(state.requests) == 3
    assert f"HTTP {status}" in str(err.value)
    assert KEY not in str(err.value)
    assert "***" in str(err.value)


def test_a_retry_can_succeed() -> None:
    replies: list[Wire] = [(503, "busy"), ok("fine", usage={"total_tokens": 4})]
    with serve(lambda _b: replies.pop(0)) as (url, state):
        assert chat_once(url).message["content"] == "fine"
    assert len(state.requests) == 2


def test_client_errors_are_not_retried() -> None:
    with (
        serve(lambda _b: (401, "no")) as (url, state),
        pytest.raises(LoopError, match="HTTP 401"),
    ):
        chat_once(url)
    assert len(state.requests) == 1


@pytest.mark.parametrize(
    "reply",
    [
        (200, "not json"),
        (200, "[]"),
        (200, json.dumps({"choices": []})),
        (200, json.dumps({"choices": [{"message": "x"}], "usage": {}})),
    ],
)
def test_odd_responses_fail_clean(reply: Wire) -> None:
    with (
        serve(lambda _b: reply) as (url, _),
        pytest.raises(LoopError, match=r"chat completion|message"),
    ):
        chat_once(url)


def test_unreachable_server_fails_clean() -> None:
    with serve(lambda _b: ok()) as (url, _):
        pass
    with pytest.raises(LoopError, match="cannot reach"):
        chat_once(url, KEY)


def test_bad_url_and_key_are_refused_before_any_request() -> None:
    with pytest.raises(ValueError, match="http"):
        make_chat("ftp://x", "m", None, 0.0)
    with pytest.raises(ValueError, match="api_key"):
        make_chat("http://x", "m", "bad key", 0.0)


# --- prompt ---------------------------------------------------------------------


def test_system_prompt_is_short_and_names_the_tools() -> None:
    text = system_prompt()
    assert len(text.splitlines()) < 60
    for name in ("exec_python", "forward_pass", "status", "finish", "save_edit"):
        assert name in text
    for hidden in ("report_usage", "check_spend", "report_refusal"):
        assert hidden not in text


# --- main -----------------------------------------------------------------------

ARGS = ["--base-url", "http://127.0.0.1:1", "--model", "m"]


def test_main_without_run_json_exits_2(tmp_path: Path) -> None:
    assert main([str(tmp_path), *ARGS]) == 2


def test_main_bad_args_exit_2(tmp_path: Path) -> None:
    for argv in ([], [str(tmp_path)], [str(tmp_path), *ARGS, "--max-turns", "0"]):
        with pytest.raises(SystemExit) as err:
            main(argv)
        assert err.value.code == 2


def test_main_missing_key_variable_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("AW_TEST_KEY", raising=False)
    run_dir = write_run(tmp_path)
    assert main([str(run_dir), *ARGS, "--api-key-env", "AW_TEST_KEY"]) == 2
    assert "AW_TEST_KEY is not set" in capsys.readouterr().err


def test_main_bad_url_or_key_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = write_run(tmp_path)
    assert main([str(run_dir), "--base-url", "ftp://x", "--model", "m"]) == 2
    monkeypatch.setenv("AW_TEST_KEY", "bad key!")
    assert main([str(run_dir), *ARGS, "--api-key-env", "AW_TEST_KEY"]) == 2
    err = capsys.readouterr().err
    assert "bad key" not in err


def test_main_bad_run_json_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "run.json").write_text("{}")
    assert main([str(tmp_path), *ARGS]) == 2
    assert "could not run the server" in capsys.readouterr().err


def test_main_refusal_exits_1_and_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AW_TEST_KEY", KEY)
    run_dir = write_run(tmp_path)
    with serve(lambda _b: ok("", finish="content_filter")) as (url, state):
        argv = [str(run_dir), "--base-url", url, "--model", "m"]
        assert main([*argv, "--api-key-env", "AW_TEST_KEY"]) == 1
    out = capsys.readouterr()
    assert json.loads(out.out)["refused"]["category"] == "content_filter"
    assert KEY not in out.out + out.err
    assert state.requests[0]["auth"] == f"Bearer {KEY}"
    events = RunRecord.open(run_dir / "record.jsonl").read()
    assert KEY not in json.dumps(events)


def test_main_endpoint_failure_exits_1_without_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AW_TEST_KEY", KEY)
    run_dir = write_run(tmp_path)
    with serve(lambda _b: (500, f"bad {KEY}")) as (url, _):
        argv = [str(run_dir), "--base-url", url, "--model", "m"]
        assert main([*argv, "--api-key-env", "AW_TEST_KEY"]) == 1
    out = capsys.readouterr()
    assert "HTTP 500" in json.loads(out.out)["error"]
    assert KEY not in out.out + out.err


# --- against the real server ------------------------------


@pytest.mark.anyio
async def test_loop_against_the_server_charges_tokens(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)
    server = build_server(
        run_dir,
        {
            "exec": lambda p: {"ok": True, "output": "hi"},
            "forward": lambda p: {"ok": True},
        },
    )
    chat = scripted_chat(
        turn(call("exec_python", {"code": "1"}), tokens=100),
        turn(call("forward_pass", {"request": {}}), tokens=50),
        turn(call("finish", {"summary": "done"}), tokens=25),
    )
    async with Client(server) as client:
        out = await run_loop(
            client, chat, max_turns=5, system="s", model="m", poll_s=0.01
        )
    assert out["outcome"] == "finished"
    assert out["budget"]["used"]["tokens"] == 175
    assert out["budget"]["used"]["dollars"] == 0
    events = RunRecord.open(run_dir / "record.jsonl").read()
    usage = [e for e in events if e["event"] == "usage"]
    assert [(e["turn"], e["tokens"], e["model"]) for e in usage] == [
        (1, 100, "m"),
        (2, 50, "m"),
        (3, 25, "m"),
    ]
    assert events[-1]["event"] == "finish"


@pytest.mark.anyio
async def test_refusal_is_recorded_by_the_server(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)
    async with Client(build_server(run_dir)) as client:
        with pytest.raises(Refused):
            await run_loop(
                client,
                scripted_chat(Refused("content_filter", "stopped", tokens=40)),
                max_turns=2,
                system="s",
                model="m",
            )
    events = RunRecord.open(run_dir / "record.jsonl").read()
    refusal = next(e for e in events if e["event"] == "refusal")
    assert refusal["category"] == "content_filter"
    assert refusal["model"] == "m"
    names = [e["event"] for e in events]
    assert names.index("usage") < names.index("refusal")
    assert Ledger(run_dir / "ledger.jsonl", Limits(**LIMITS)).used()["tokens"] == 40


def never_chat(_m: list[loop.Message], _t: list[dict[str, Any]]) -> Reply:
    raise AssertionError("the model must not be called")


@pytest.mark.anyio
async def test_finished_run_makes_no_model_call(tmp_path: Path) -> None:
    async with Client(build_server(write_run(tmp_path))) as client:
        await client.call_tool("finish", {"summary": "done"})
        with pytest.raises(LoopError, match="already finished"):
            await run_loop(client, never_chat, max_turns=3, system="s", model="m")


@pytest.mark.anyio
async def test_failure_streak_makes_no_model_call(tmp_path: Path) -> None:
    run_dir = write_run(tmp_path)
    ledger = Ledger(run_dir / "ledger.jsonl", Limits(**LIMITS))
    for i in range(3):
        ledger.fail("dollars", "x", ref=f"r{i}")
    async with Client(build_server(run_dir)) as client:
        with pytest.raises(LoopError, match="cannot start"):
            await run_loop(client, never_chat, max_turns=3, system="s", model="m")


def test_local_editor_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from mcp_tiny_hf import save_tiny_model

    from astro_weights.mcp import sandbox

    model = save_tiny_model(tmp_path / "model")
    limits = {"forward_passes": 5, "code_runs": 5, "tokens": 1000, "dollars": 0}
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
    replies = [
        tool_reply(("exec_python", {"code": edit}), tokens=(100, 20)),
        tool_reply(("forward_pass", {"request": forward}), tokens=(150, 30)),
        tool_reply(("finish", {"summary": "scaled one weight"}), tokens=(200, 10)),
    ]
    with serve(lambda _b: replies.pop(0)) as (url, state):
        flags = ["--unsafe-kernel"] if sandbox.probe() is None else []
        argv = [str(run), "--base-url", url, "--model", "tiny", *flags]
        assert main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["outcome"] == "finished"
    assert summary["tokens"] == 510
    used = Ledger(run / "ledger.jsonl", Limits(**limits)).used()
    assert used == {"forward_passes": 1, "code_runs": 1, "tokens": 510, "dollars": 0}
    events = RunRecord.open(run / "record.jsonl").read()
    names = [e["event"] for e in events]
    assert names.count("usage") == 3
    assert {"edit_saved", "summary", "finish"} <= set(names)
    assert names[-1] == "finish"
    assert sum(e["tokens"] for e in events if e["event"] == "usage") == 510
    assert all(e["dollars"] == 0 for e in events if e["event"] == "usage")
    sent = [r["body"] for r in state.requests]
    tool_msg = next(m for m in sent[1]["messages"] if m["role"] == "tool")
    assert json.loads(tool_msg["content"])["state"] == "done"
