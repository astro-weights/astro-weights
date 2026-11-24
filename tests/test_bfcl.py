import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from fake_endpoint import ok as _ok
from fake_endpoint import serve, tool_reply

from astro_weights.eval import TaskRecord, ToolCall, read_tasks
from astro_weights.runners import bfcl, endpoint
from astro_weights.runners.bfcl import BfclTask, load_bfcl, run_bfcl, run_task
from astro_weights.runners.endpoint import ChatReply

FIXTURE = Path(__file__).parent / "data" / "bfcl"
SRC = "bfcl_eval/eval_checker/multi_turn_eval/func_source_code"
DATA = "bfcl_eval/data"


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(endpoint, "_RETRY_SLEEP", 0.0)


def _pin(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pins = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p.name != "TRIGGERED"
    }
    monkeypatch.setattr(bfcl, "PINS", pins)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A copy of the fixture checkout, pinned as it is."""
    dest = tmp_path / "bfcl"
    shutil.copytree(FIXTURE, dest, ignore=shutil.ignore_patterns("__pycache__"))
    _pin(dest, monkeypatch)
    return dest


@pytest.fixture
def tasks(root: Path) -> dict[str, BfclTask]:
    return {t.task_id: t for t in load_bfcl(root)}


def reply(
    *calls: tuple[str, dict[str, Any] | str], text: str = "", tokens: int = 5
) -> ChatReply:
    made = [
        (f"c{i}", ToolCall(n, a if isinstance(a, str) else json.dumps(a)))
        for i, (n, a) in enumerate(calls)
    ]
    msg: dict[str, Any] = {"role": "assistant", "content": text or None}
    if made:
        msg["tool_calls"] = [
            {
                "id": i,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments},
            }
            for i, c in made
        ]
    usage = {"completion_tokens": tokens}
    finish = "tool_calls" if made else "stop"
    return ChatReply(text, tuple(c for _, c in made), tokens, finish, usage, msg)


class Script:
    """A fake model: the next canned reply per call; remembers what it was shown."""

    def __init__(self, *replies: ChatReply) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict[str, Any]]] = []

    def __call__(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ChatReply:
        self.seen.append(copy.deepcopy(messages))
        return self.replies.pop(0)


def done(text: str = "done") -> ChatReply:
    return reply(text=text)


def test_load(tasks: dict[str, BfclTask], root: Path) -> None:
    assert list(tasks) == ["toy_0", "toy_1", "toy_2"]
    t = tasks["toy_1"]
    assert t.turns == ("Add 2 and 3, then post the sum.", "Thanks.")
    assert t.involved_classes == ("TwitterAPI", "MathAPI")
    assert t.ground_truth == (
        (("add", (), {"a": 2, "b": 3}), ("post_tweet", ("5",), {})),
        (),
    )
    names = [x["function"]["name"] for x in t.tools]
    assert names == ["post_tweet", "list_tweets", "delete_tweet", "add"]
    add = t.tools[-1]["function"]
    assert add["parameters"]["type"] == "object"
    assert add["parameters"]["properties"]["a"]["type"] == "number"
    assert add["parameters"]["properties"]["a"]["description"].endswith(
        "float type value."
    )
    assert "response field has the following schema" in add["description"]
    assert "response" not in add
    assert t.root == root.resolve()
    again = {x.task_id: x for x in load_bfcl(root)}
    assert [x.prompt_hash for x in again.values()] == [
        x.prompt_hash for x in tasks.values()
    ]
    assert len({x.prompt_hash for x in tasks.values()}) == 3


def test_perfect_run_totals_and_history(tasks: dict[str, BfclTask]) -> None:
    model = Script(
        reply(("post_tweet", {"content": "hello"}), tokens=4),
        done(),
        reply(("list_tweets", {}), tokens=6),
        done("these"),
    )
    rec = run_task(tasks["toy_0"], model, arm="base", trial=3)
    assert isinstance(rec, TaskRecord)
    assert rec.passed
    assert (rec.arm, rec.trial, rec.task_id) == ("base", 3, "toy_0")
    assert (rec.output_tokens, rec.tool_calls, rec.n_turns) == (4 + 5 + 6 + 5, 2, 8)
    assert [t.role for t in rec.turns] == [
        *["user", "assistant", "tool", "assistant"] * 2
    ]
    assert rec.turns[2].text == '{"posted": 2}'
    assert rec.turns[6].text == '{"tweets": ["first", "hello"]}'
    # the tool result goes back under the call's id; the second turn sees the first
    last = model.seen[1][-1]
    assert last == {"role": "tool", "tool_call_id": "c0", "content": '{"posted": 2}'}
    assert [m["role"] for m in model.seen[3]] == [
        *["user", "assistant", "tool", "assistant"] * 2,
    ][:7]
    assert rec.prompt_hash == tasks["toy_0"].prompt_hash


def test_wrong_call_fails(tasks: dict[str, BfclTask]) -> None:
    model = Script(
        reply(("post_tweet", {"content": "bye"})),
        done(),
        reply(("list_tweets", {})),
        done(),
    )
    rec = run_task(tasks["toy_0"], model)
    assert not rec.passed
    assert rec.n_turns == 8  # the conversation still ran to the end


def test_no_call_in_a_turn_with_ground_truth_fails(tasks: dict[str, BfclTask]) -> None:
    rec = run_task(tasks["toy_2"], Script(done("it is 2")))
    assert not rec.passed
    assert (rec.tool_calls, rec.n_turns) == (0, 2)


def test_parallel_calls_and_empty_ground_truth_turn(tasks: dict[str, BfclTask]) -> None:
    model = Script(
        reply(("add", {"a": 2, "b": 3}), ("post_tweet", {"content": "5"})),
        done(),
        reply(("list_tweets", {})),  # turn 1 has no ground truth: runs, not checked
        done(),
    )
    rec = run_task(tasks["toy_1"], model)
    assert rec.passed
    assert rec.tool_calls == 3
    assert model.seen[1][-2]["tool_call_id"] == "c0"
    assert model.seen[1][-1]["tool_call_id"] == "c1"


def test_state_right_but_result_missing_fails(tasks: dict[str, BfclTask]) -> None:
    # posts '5' without the add call: same state as the ground truth, but the add result
    # `{"result": 5}` of the ground truth is nowhere in the model's results
    model = Script(reply(("post_tweet", {"content": "5"})), done(), done())
    assert not run_task(tasks["toy_1"], model).passed


def test_result_mismatch_fails(tasks: dict[str, BfclTask]) -> None:
    rec = run_task(tasks["toy_2"], Script(reply(("add", {"a": 2, "b": 2})), done()))
    assert not rec.passed


def test_unknown_private_and_dunder_names_are_refused(
    tasks: dict[str, BfclTask], root: Path
) -> None:
    names = ["__class__", "_secret", "__import__", "__init__", "nope", "_load_scenario"]
    model = Script(
        reply(*[(n, {}) for n in names]),
        done(),
        done(),
    )
    rec = run_task(tasks["toy_0"], model)
    tool_turns = [t.text for t in rec.turns if t.role == "tool"]
    assert len(tool_turns) == len(names)
    assert all(t.startswith("Error during execution:") for t in tool_turns)
    assert rec.tool_calls == len(names)  # refused calls still count
    assert not rec.passed
    assert not (root / SRC / "TRIGGERED").exists()


def test_model_text_is_never_run(
    tasks: dict[str, BfclTask], root: Path, tmp_path: Path
) -> None:
    pwned = tmp_path / "PWNED"
    code = f"__import__('pathlib').Path({str(pwned)!r}).write_text('x')"
    model = Script(
        reply(("post_tweet", {"content": code}), text=f"```python\n{code}\n```"),
        done(code),
        reply(("list_tweets", {})),
        done(code),
    )
    rec = run_task(tasks["toy_0"], model)
    assert not pwned.exists()
    assert not rec.passed  # the code only ended up as (truncated) tweet text
    assert rec.turns[2].text == '{"posted": 2}'


def test_bad_arguments_become_tool_results(tasks: dict[str, BfclTask]) -> None:
    model = Script(
        reply(
            ("post_tweet", "{not json"),
            ("post_tweet", "[1]"),
            ("post_tweet", {"wrong": 1}),
            ("delete_tweet", {"index": 99}),
            ("post_tweet", "[]"),
        ),
        done(),
        done(),
    )
    rec = run_task(tasks["toy_0"], model)
    got = [t.text for t in rec.turns if t.role == "tool"]
    assert got[0].startswith("Error during execution: arguments are not valid JSON")
    assert got[1] == "Error during execution: arguments must be a JSON object"
    assert "unexpected keyword argument 'wrong'" in got[2]
    assert got[3] == "Error during execution: list assignment index out of range"
    assert not rec.passed


def test_step_cap_stops_the_task(tasks: dict[str, BfclTask]) -> None:
    model = Script(*[reply(("list_tweets", {})) for _ in range(3)])
    rec = run_task(tasks["toy_0"], model, max_steps_per_turn=2)
    assert not rec.passed
    assert rec.tool_calls == 3  # BFCL: `count > MAXIMUM_STEP_LIMIT` runs one step more
    assert rec.turns[-1].role == "tool"
    assert model.replies == []


def test_step_cap_on_the_last_turn_is_scored_normally(
    tasks: dict[str, BfclTask],
) -> None:
    # toy_2 has one turn: the cap is hit there, the right call was made, BFCL passes it
    model = Script(*[reply(("add", {"a": 1, "b": 1})) for _ in range(2)])
    rec = run_task(tasks["toy_2"], model, max_steps_per_turn=1)
    assert rec.passed
    assert rec.tool_calls == 2
    assert rec.turns[-1].role == "tool"
    # and a wrong state on that turn still fails
    model = Script(*[reply(("add", {"a": 2, "b": 2})) for _ in range(2)])
    assert not run_task(tasks["toy_2"], model, max_steps_per_turn=1).passed


def test_deeply_nested_arguments_are_a_tool_error(tasks: dict[str, BfclTask]) -> None:
    deep = "[" * 400_000
    model = Script(reply(("post_tweet", deep)), done(), done(), done())
    rec = run_task(tasks["toy_0"], model)
    assert rec.turns[2].text.startswith(
        "Error during execution: arguments are not valid"
    )
    assert not rec.passed


def test_long_context_restored_when_exec_fails_after_the_import(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = root / SRC / "posting_api.py"
    _edit(src, src.read_text() + "\nraise RuntimeError('boom')\n")
    ts = load_bfcl(root, verify=False)
    name = bfcl._LONG_CONTEXT_NAME
    before = set(sys.modules)
    sys.modules.pop(name, None)
    with pytest.raises(RuntimeError, match="boom"):
        bfcl._Env(ts[0])
    assert name not in sys.modules
    sentinel = type(sys)("sentinel")
    monkeypatch.setitem(sys.modules, name, sentinel)
    with pytest.raises(RuntimeError, match="boom"):
        bfcl._Env(ts[0])
    assert sys.modules[name] is sentinel
    left = {
        m for m in set(sys.modules) - before if m.startswith("astro_weights_bfcl_env")
    }
    assert all("posting_api" not in m for m in left)


def test_long_context_import_is_ours_and_restored(
    tasks: dict[str, BfclTask], monkeypatch: pytest.MonkeyPatch
) -> None:
    name = bfcl._LONG_CONTEXT_NAME
    sentinel = type(sys)("sentinel")
    monkeypatch.setitem(sys.modules, name, sentinel)
    long = "x" * 30
    model = Script(reply(("post_tweet", {"content": long})), done(), done(), done())
    rec = run_task(tasks["toy_0"], model)
    assert rec.turns[2].text == '{"posted": 2}'
    assert sys.modules[name] is sentinel
    env = bfcl._Env(tasks["toy_0"])
    env.call("post_tweet", (), {"content": long})
    assert env.instances["TwitterAPI"].tweets[-1] == "x" * 20  # MAX_LEN of the fixture


def test_long_context_name_is_removed_when_it_was_absent(
    tasks: dict[str, BfclTask], root: Path
) -> None:
    sys.modules.pop(bfcl._LONG_CONTEXT_NAME, None)
    bfcl._Env(tasks["toy_0"])
    assert bfcl._LONG_CONTEXT_NAME not in sys.modules


def _edit(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_hash_mismatch_is_refused(root: Path) -> None:
    data = root / DATA / "BFCL_v4_multi_turn_base.json"
    _edit(data, data.read_text() + "\n")
    with pytest.raises(
        ValueError, match=r"BFCL_v4_multi_turn_base\.json: sha256 .* pin"
    ):
        load_bfcl(root)
    assert len(load_bfcl(root, verify=False)) == 3


def test_tampered_env_source_is_refused_at_load_time(root: Path) -> None:
    ts = load_bfcl(root)
    src = root / SRC / "posting_api.py"
    _edit(src, src.read_text() + "\nimport os\n")
    with pytest.raises(ValueError, match=r"posting_api\.py: sha256"):
        run_task(ts[0], Script())
    with pytest.raises(ValueError, match=r"posting_api\.py: sha256"):
        run_bfcl(ts, "http://127.0.0.1:1", "m", "a")


def test_file_not_in_pins_is_refused(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bfcl, "PINS", {})
    with pytest.raises(ValueError, match="not in the pinned file table"):
        load_bfcl(root)


def test_pins_cover_the_checkout_layout() -> None:
    assert len(bfcl.PINS) == 2 + 8 + 8 + 1
    assert all(len(v) == 64 for v in bfcl.PINS.values())
    assert len(bfcl.BFCL_COMMIT) == 40


def test_class_name_cannot_pick_a_path(root: Path) -> None:
    f = root / DATA / "BFCL_v4_multi_turn_base.json"
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    rows[0]["involved_classes"] = ["../../../evil"]
    _edit(f, "".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match=r"unknown involved class '[.][.]"):
        load_bfcl(root, verify=False)


def test_files_outside_the_root_are_never_read(root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    _edit(outside, "raise SystemExit('imported')\n")
    src = root / SRC / "posting_api.py"
    src.unlink()
    src.symlink_to(outside)
    ts = load_bfcl(root, verify=False)
    with pytest.raises(ValueError, match="resolves outside bfcl_root"):
        run_task(ts[0], Script())
    data = root / DATA / "possible_answer" / "BFCL_v4_multi_turn_base.json"
    data.unlink()
    data.symlink_to(outside)
    with pytest.raises(ValueError, match="resolves outside bfcl_root"):
        load_bfcl(root, verify=False)


def test_missing_module_stops_before_any_request(root: Path) -> None:
    _edit(root / SRC / "math_api.py", "import not_a_real_module_xyz\n")
    ts = load_bfcl(root, verify=False)
    before = set(sys.modules)
    with pytest.raises(
        ModuleNotFoundError, match="MathAPI needs the module 'not_a_real_module_xyz'"
    ):
        run_bfcl(ts, "http://127.0.0.1:1", "m", "a")
    left = {
        m for m in set(sys.modules) - before if m.startswith("astro_weights_bfcl_env")
    }
    assert not left  # the failed import left nothing behind


def test_missing_long_context_is_a_module_error(root: Path) -> None:
    (root / SRC / "long_context.py").unlink()
    ts = load_bfcl(root, verify=False)
    with pytest.raises(ModuleNotFoundError, match="TwitterAPI needs the module"):
        run_bfcl(ts, "http://127.0.0.1:1", "m", "a")


def test_not_a_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        load_bfcl(tmp_path / "nope")


def _rewrite(root: Path, rel: str, edit: Any) -> None:
    f = root / rel
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    edit(rows)
    _edit(f, "".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.parametrize(
    ("rel", "edit", "match"),
    [
        (
            f"{DATA}/BFCL_v4_multi_turn_base.json",
            lambda r: r[0]["question"][0].append({}),
            "not one user message",
        ),
        (
            f"{DATA}/BFCL_v4_multi_turn_base.json",
            lambda r: r[0]["question"][0][0].update(role="system"),
            "not one user message",
        ),
        (
            f"{DATA}/BFCL_v4_multi_turn_base.json",
            lambda r: r[0]["question"].pop(),
            "turns but",
        ),
        (
            f"{DATA}/possible_answer/BFCL_v4_multi_turn_base.json",
            lambda r: r.pop(),
            "no possible_answer",
        ),
        (
            f"{DATA}/possible_answer/BFCL_v4_multi_turn_base.json",
            lambda r: r[0].update(ground_truth=[["a.b()"], []]),
            "not a plain function call",
        ),
        (
            f"{DATA}/possible_answer/BFCL_v4_multi_turn_base.json",
            lambda r: r[0].update(ground_truth=[["f(**x)"], []]),
            r"\*\* arguments",
        ),
        (
            f"{DATA}/possible_answer/BFCL_v4_multi_turn_base.json",
            lambda r: r[0].update(ground_truth=[["f(x=g())"], []]),
            "ground truth call",
        ),
        (
            f"{DATA}/possible_answer/BFCL_v4_multi_turn_base.json",
            lambda r: r[0].update(ground_truth=[["f("], []]),
            "ground truth call",
        ),
    ],
)
def test_malformed_data_is_refused(root: Path, rel: str, edit: Any, match: str) -> None:
    _rewrite(root, rel, edit)
    with pytest.raises(ValueError, match=match):
        load_bfcl(root, verify=False)


def test_jsonl_row_must_be_an_object(root: Path) -> None:
    _edit(root / DATA / "BFCL_v4_multi_turn_base.json", "[1]\n")
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_bfcl(root, verify=False)


def test_helpers() -> None:
    assert bfcl._text("s") == "s"
    assert bfcl._text({"a": 1}) == '{"a": 1}'
    assert bfcl._text({"a": {1, 2}}).startswith("{'a'")
    assert bfcl._text(None) == "None"
    assert bfcl._has_all(["a", "b", "a"], ["a", "a"])
    assert not bfcl._has_all(["a", "b"], ["a", "a"])
    assert bfcl._has_all([], [])
    s = bfcl._schema(
        {
            "type": "dict",
            "properties": {"x": {"type": "float"}, "y": {}},
            "items": {"type": "weird"},
        }
    )
    assert s["type"] == "object"
    assert s["properties"]["x"]["format"] == "float"
    assert s["properties"]["y"]["type"] == "string"
    assert s["items"]["type"] == "string"


def test_run_bfcl_over_http_with_trials_and_out(
    tasks: dict[str, BfclTask], tmp_path: Path
) -> None:
    def respond(body: dict[str, Any]) -> tuple[int, str]:
        last = body["messages"][-1]
        if last["role"] == "user":
            return tool_reply(("add", {"a": 1, "b": 1}), tokens=(10, 4))
        return tool_reply(content="2", tokens=(10, 3))

    out = tmp_path / "tasks.jsonl"
    with serve(respond) as (url, state):
        recs = run_bfcl(
            [tasks["toy_2"]],
            url,
            "m",
            "base",
            trials=2,
            seed=7,
            out=out,
            max_new_tokens=64,
        )
    assert [(r.task_id, r.trial, r.arm, r.passed) for r in recs] == [
        ("toy_2", 0, "base", True),
        ("toy_2", 1, "base", True),
    ]
    assert all((r.output_tokens, r.tool_calls, r.n_turns) == (7, 1, 4) for r in recs)
    assert recs[0].prompt_hash == recs[1].prompt_hash == tasks["toy_2"].prompt_hash
    assert read_tasks(out) == recs
    bodies = [r["body"] for r in state.requests]
    assert [b["seed"] for b in bodies] == [7, 7, 8, 8]
    assert all(b["max_tokens"] == 64 and b["tools"] for b in bodies)


def test_run_bfcl_server_error_writes_nothing(
    tasks: dict[str, BfclTask], tmp_path: Path
) -> None:
    out = tmp_path / "tasks.jsonl"
    with (
        serve(lambda b: (500, "no")) as (url, _),
        pytest.raises(RuntimeError, match="server error"),
    ):
        run_bfcl([tasks["toy_2"]], url, "m", "base", out=out)
    assert not out.exists()


def test_run_bfcl_checks_its_arguments(tasks: dict[str, BfclTask]) -> None:
    ts = list(tasks.values())
    with pytest.raises(ValueError, match="trials"):
        run_bfcl(ts, "http://x", "m", "a", trials=0)
    with pytest.raises(ValueError, match="trials"):
        run_bfcl(ts, "http://x", "m", "a", trials=True)
    with pytest.raises(ValueError, match="max_steps_per_turn"):
        run_bfcl(ts, "http://x", "m", "a", max_steps_per_turn=0)
    with pytest.raises(ValueError, match="duplicate task_id"):
        run_bfcl([ts[0], ts[0]], "http://x", "m", "a")


def test_default_chat_uses_ok_reply_shape() -> None:
    # a server that answers without tool calls ends every turn at once
    with serve(lambda b: _ok("hi")) as (url, _):
        r = bfcl._chat_once(
            url, "m", [{"role": "user", "content": "x"}], [], max_new_tokens=4
        )
    assert r.text == "hi"


def test_unparsable_tool_call_is_a_turn_without_a_call(
    tasks: dict[str, BfclTask],
) -> None:
    body = json.dumps({"error": {"code": 500, "message": "Failed to parse tool call"}})
    with serve(lambda b: (500, body)) as (url, _):
        records = run_bfcl(list(tasks.values())[:1], url, "m", "a", max_new_tokens=7)
    (r,) = records
    assert not r.passed
    replies = sum(t.role == "assistant" for t in r.turns)
    assert replies >= 1
    assert (r.tool_calls, r.output_tokens) == (0, 7 * replies)


def test_context_overflow_ends_the_task_as_failed_and_the_run_goes_on(
    tasks: dict[str, BfclTask],
) -> None:
    body = json.dumps({"error": {"type": "exceed_context_size_error", "message": "x"}})
    ts = list(tasks.values())[:2]
    with serve(lambda b: (400, body)) as (url, s):
        records = run_bfcl(ts, url, "m", "a", max_new_tokens=7)
    assert [r.task_id for r in records] == [t.task_id for t in ts]
    assert not any(r.passed for r in records)
    assert len(s.requests) == 2  # one request per task, no retry, no later turn
