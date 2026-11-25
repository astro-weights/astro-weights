"""BFCL `multi_turn_base` as multi-turn tasks run against an OpenAI-compatible server.

Berkeley Function-Calling Leaderboard v4, from a local checkout of
https://github.com/ShishirPatil/gorilla (`berkeley-function-call-leaderboard/`, the
directory that holds `bfcl_eval/`) at commit `BFCL_COMMIT`. There is no pip
dependency: `bfcl-eval` on PyPI pins numpy 1.26 and pulls in about 30 packages. The
data files and environment classes are read from the checkout and checked against
`PINS` (sha256) before use; a file outside the table, or one that resolves outside
the checkout, is refused. Environment source is `exec`'d only after its hash
matched, and only from files in `PINS`; model output is never evaluated. Standard
library only, except that `MathAPI` imports `mpmath` (25 of the 200 tasks).

Scoring follows `bfcl_eval/eval_checker/multi_turn_eval/multi_turn_checker.py`
(`multi_turn_checker`, `state_checker`, `response_checker`) and the inference loop in
`bfcl_eval/model_handler/base_handler.py` (`inference_multi_turn_FC`); see `run_task`.
"""

import ast
import contextlib
import copy
import hashlib
import inspect
import json
import os
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

from astro_weights.eval import TaskRecord, ToolCall, Turn, prompt_hash, write_header
from astro_weights.eval.header import header_path
from astro_weights.eval.tasks import read_tasks_tail
from astro_weights.runners.endpoint import (
    ChatReply,
    ContextOverflowError,
    ToolCallParseError,
    _header_to_keep,
    chat_http,
)

BFCL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"

_DATA = "bfcl_eval/data"
_SRC = "bfcl_eval/eval_checker/multi_turn_eval/func_source_code"
_TASKS = f"{_DATA}/BFCL_v4_multi_turn_base.json"
_ANSWERS = f"{_DATA}/possible_answer/BFCL_v4_multi_turn_base.json"
# class name -> (source file, function doc file); BFCL `CLASS_FILE_PATH_MAPPING` and
# `MULTI_TURN_FUNC_DOC_FILE_MAPPING` in `bfcl_eval/constants/executable_backend_config.py`.
_CLASSES = {
    "GorillaFileSystem": ("gorilla_file_system", "gorilla_file_system"),
    "MathAPI": ("math_api", "math_api"),
    "MessageAPI": ("message_api", "message_api"),
    "TwitterAPI": ("posting_api", "posting_api"),
    "TicketAPI": ("ticket_api", "ticket_api"),
    "TradingBot": ("trading_bot", "trading_bot"),
    "TravelAPI": ("travel_booking", "travel_booking"),
    "VehicleControlAPI": ("vehicle_control", "vehicle_control"),
}
_STATELESS = frozenset({"MathAPI"})  # BFCL `STATELESS_CLASSES`: no `_load_scenario`
_LONG_CONTEXT = "long_context"  # imported by four env files by its package path
_LONG_CONTEXT_NAME = (
    f"bfcl_eval.eval_checker.multi_turn_eval.func_source_code.{_LONG_CONTEXT}"
)

PINS: dict[str, str] = {
    f"{_DATA}/BFCL_v4_multi_turn_base.json": "1a21a995d06fd6f20ba55de7bced30ef953ec35e998f502ec2ecf4d66ef1c43a",
    f"{_DATA}/possible_answer/BFCL_v4_multi_turn_base.json": "1fee67823b317571649177dd89d63969feaae4e810cc7448ee55ba797fb7c8fc",
    f"{_DATA}/multi_turn_func_doc/gorilla_file_system.json": "c4c1b741c71e2a17c97a5dc9c4a91d89978c4eaece56494d14272d5df6c650e9",
    f"{_DATA}/multi_turn_func_doc/math_api.json": "83fa31708c89442bdcf12ac4dfbe3be8663ec9183a1b1524a9fda98164bc7e0b",
    f"{_DATA}/multi_turn_func_doc/message_api.json": "4d58ea933a5d2b280d7a52366617fb47fecd333d7b0e08e724db6fa12fb5f847",
    f"{_DATA}/multi_turn_func_doc/posting_api.json": "87f9fc404a06e4107d7c366f1399c596440e84c16cb119faf374b2f532d86e8b",
    f"{_DATA}/multi_turn_func_doc/ticket_api.json": "31324e0380782664fe82ba05bd23517808ad55692b45a5fe7c3d4d395fe4f0f8",
    f"{_DATA}/multi_turn_func_doc/trading_bot.json": "1a7933fd8f0cb8fbec38aeae05cb8c24132747cad4904fe1c13ac3d01fc22c2d",
    f"{_DATA}/multi_turn_func_doc/travel_booking.json": "f17b950c13adddf41d0848077df58788252e4c2e7cad5cfa71c8c4bf04f57b26",
    f"{_DATA}/multi_turn_func_doc/vehicle_control.json": "0c8a66292844874ef7b168f343bc394d8615d2d9e1f4387999a9ee23011eac78",
    f"{_SRC}/gorilla_file_system.py": "84173750930b09aaa31b90547c9bd79ce75714682608b35e78a9c1ea259faf2b",
    f"{_SRC}/math_api.py": "abde32cc25e71b2ff3acf4d028a3c005b98409c174aba4abddbfe1c018afb9a8",
    f"{_SRC}/message_api.py": "cf72af1635654147af17df1891a7fb354549e78a61930139f741a31f4be21121",
    f"{_SRC}/posting_api.py": "32b887aaa37c7167ad9c84ddfa7bbb855c75d3c1edab19ba868aba29acc4cfa1",
    f"{_SRC}/ticket_api.py": "6a9375aa6425410c07a151aef27f3e44e9ed1023bb5cc5c25f14f4a9b3c99116",
    f"{_SRC}/trading_bot.py": "cc2e3949035b561e9d07418b081561aa07b211672200e2e511aec074bf79fcb2",
    f"{_SRC}/travel_booking.py": "e698e22adccf0d6d41b297f2d5e44faebd828cef4752df6a4ffb1570c0c35296",
    f"{_SRC}/vehicle_control.py": "f5f7a2f28968728927a98ae12ec68a32d70781cfb74c2d7b14801772362b12d1",
    f"{_SRC}/long_context.py": "5c9a5710115609c779e5b2ccb43f6ad58271ed96f2ea7a797edb0d65c385aa38",
}

# BFCL `GORILLA_TO_OPENAPI` (bfcl_eval/constants/type_mappings.py), the types the docs use.
_TYPES = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
}

_Call = tuple[
    str, tuple[Any, ...], dict[str, Any]
]  # name, positional args, keyword args
Chat = Callable[[list[dict[str, Any]], list[dict[str, Any]]], ChatReply]


@dataclass(frozen=True)
class BfclTask:
    """One `multi_turn_base` entry.

    `turns` are the user questions, one per turn. `ground_truth` holds, per turn, the
    calls (name, args, kwargs) that BFCL's `possible_answer` makes; a turn with none
    is one where the model should not have to act. `tools` are the OpenAI function
    tools of the involved classes. `prompt_hash` hashes the questions, the tools and
    the initial config. `root` and `verify` say where the environment classes come
    from and whether their hash is checked again when they are loaded.
    """

    task_id: str
    root: Path
    verify: bool
    initial_config: dict[str, Any]
    involved_classes: tuple[str, ...]
    turns: tuple[str, ...]
    ground_truth: tuple[tuple[_Call, ...], ...]
    tools: tuple[dict[str, Any], ...]
    prompt_hash: str


def _read(root: Path, rel: str, verify: bool) -> bytes:
    """Bytes of `root/rel`; ValueError if it resolves outside `root` or fails its pin."""
    path = (root / rel).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{rel} resolves outside bfcl_root ({path})")
    data = path.read_bytes()
    if verify:
        want = PINS.get(rel)
        if want is None:
            raise ValueError(f"{rel} is not in the pinned file table")
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            raise ValueError(
                f"{rel}: sha256 {got} does not match the pin {want} "
                f"(BFCL commit {BFCL_COMMIT}); use that checkout or verify=False"
            )
    return data


def _jsonl(data: bytes, rel: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for num, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{rel} line {num}: expected a JSON object")
            rows.append(row)
    return rows


def _schema(prop: dict[str, Any]) -> dict[str, Any]:
    """A BFCL parameter schema as OpenAI JSON schema (`_cast_to_openai_type` in
    `bfcl_eval/model_handler/utils.py`; unknown types become string)."""
    out = dict(prop)
    kind = out.get("type")
    out["type"] = _TYPES.get(kind, "string") if isinstance(kind, str) else "string"
    if kind == "float":
        out["format"] = "float"
        out["description"] = out.get("description", "") + " This is a float type value."
    if isinstance(out.get("properties"), dict):
        out["properties"] = {k: _schema(v) for k, v in out["properties"].items()}
    if isinstance(out.get("items"), dict):
        out["items"] = _schema(out["items"])
    return out


def _tool(doc: dict[str, Any]) -> dict[str, Any]:
    """One function doc as an OpenAI tool; the `response` schema moves into the description."""
    fn = copy.deepcopy(doc)
    params = fn.setdefault("parameters", {})
    params["type"] = "object"
    params["properties"] = {
        k: _schema(v) for k, v in params.get("properties", {}).items()
    }
    if "response" in fn:
        fn["description"] += (
            f" The response field has the following schema: {json.dumps(fn.pop('response'))}"
        )
    return {"type": "function", "function": fn}


def _parse_call(text: str, task_id: str) -> _Call:
    """`name(a, k=v)` with literal arguments, by `ast`; nothing is evaluated."""
    try:
        node = ast.parse(text, mode="eval").body
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            raise ValueError("not a plain function call")  # noqa: TRY004
        args = tuple(ast.literal_eval(a) for a in node.args)
        kwargs = {}
        for k in node.keywords:
            if k.arg is None:
                raise ValueError("** arguments")
            kwargs[k.arg] = ast.literal_eval(k.value)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"{task_id}: ground truth call {text!r}: {e}") from e
    return node.func.id, args, kwargs


def _check_turns(entry: dict[str, Any]) -> tuple[str, ...]:
    questions = entry["question"]
    out: list[str] = []
    for turn in questions:
        if (
            not isinstance(turn, list)
            or len(turn) != 1
            or turn[0].get("role") != "user"
            or not isinstance(turn[0].get("content"), str)
        ):
            raise ValueError(f"{entry['id']}: a turn is not one user message")
        out.append(turn[0]["content"])
    return tuple(out)


def load_bfcl(bfcl_root: str | Path, *, verify: bool = True) -> list[BfclTask]:
    """The 200 `multi_turn_base` tasks of the BFCL checkout at `bfcl_root`.

    `bfcl_root` is `berkeley-function-call-leaderboard/` of a gorilla checkout at
    `BFCL_COMMIT`. With `verify` (default) every file read, and every environment
    source file later loaded for these tasks, must match `PINS`, else `ValueError`.
    A task whose involved class is not one of the eight multi-turn classes, whose
    turns are not one user message each, whose ground truth does not parse as flat
    calls with literal arguments, or whose turn counts differ from its ground
    truth's, raises `ValueError`.
    """
    root = Path(bfcl_root).resolve()
    if not root.is_dir():
        raise ValueError(f"bfcl_root {str(bfcl_root)!r} is not a directory")
    entries = _jsonl(_read(root, _TASKS, verify), _TASKS)
    answers = {
        a["id"]: a["ground_truth"]
        for a in _jsonl(_read(root, _ANSWERS, verify), _ANSWERS)
    }
    docs: dict[str, list[dict[str, Any]]] = {}
    tasks: list[BfclTask] = []
    for e in entries:
        task_id = str(e["id"])
        classes = tuple(e["involved_classes"])
        unknown = [c for c in classes if c not in _CLASSES]
        if unknown:
            raise ValueError(f"{task_id}: unknown involved class {unknown[0]!r}")
        if task_id not in answers:
            raise ValueError(f"{task_id}: no possible_answer entry")
        turns = _check_turns(e)
        truth = tuple(
            tuple(_parse_call(c, task_id) for c in t) for t in answers[task_id]
        )
        if len(truth) != len(turns):
            raise ValueError(
                f"{task_id}: {len(turns)} turns but {len(truth)} ground truth turns"
            )
        for c in classes:
            if c not in docs:
                rel = f"{_DATA}/multi_turn_func_doc/{_CLASSES[c][1]}.json"
                docs[c] = _jsonl(_read(root, rel, verify), rel)
        tools = tuple(_tool(d) for c in classes for d in docs[c])
        config = e.get("initial_config", {})
        digest = prompt_hash(
            json.dumps(
                {"turns": turns, "tools": tools, "config": config}, sort_keys=True
            )
        )
        tasks.append(
            BfclTask(
                task_id, root, verify, config, classes, turns, truth, tools, digest
            )
        )
    return tasks


# --- environment classes --------------------------------------------------------------

_LOCK = threading.RLock()
_MODULES: dict[tuple[str, str, str], ModuleType] = {}


def _module(root: Path, stem: str, verify: bool) -> ModuleType:
    """The env source file `stem` of `root`, executed once per content.

    The source is `exec`'d from the bytes that were hashed, so the file cannot change
    between check and use. A file that needs `long_context` finds it under its BFCL
    package name for the duration of the import.
    """
    rel = f"{_SRC}/{stem}.py"
    data = _read(root, rel, verify)
    digest = hashlib.sha256(data).hexdigest()
    key = (str(root), rel, digest)
    with _LOCK:
        mod = _MODULES.get(key)
        if mod is None:
            mod = ModuleType(f"astro_weights_bfcl_env.{stem}_{digest[:12]}")
            mod.__file__ = str(root / rel)
            sys.modules[mod.__name__] = mod
            try:
                with _long_context(root, stem, verify):
                    exec(compile(data, mod.__file__, "exec"), mod.__dict__)  # noqa: S102
            except BaseException:
                del sys.modules[mod.__name__]
                raise
            _MODULES[key] = mod
        return mod


@contextlib.contextmanager
def _long_context(root: Path, stem: str, verify: bool) -> Iterator[None]:
    if stem == _LONG_CONTEXT or not (root / _SRC / f"{_LONG_CONTEXT}.py").is_file():
        yield
        return
    mod = _module(root, _LONG_CONTEXT, verify)
    before = sys.modules.get(_LONG_CONTEXT_NAME)
    sys.modules[_LONG_CONTEXT_NAME] = mod
    try:
        yield
    finally:
        if before is None:
            del sys.modules[_LONG_CONTEXT_NAME]
        else:
            sys.modules[_LONG_CONTEXT_NAME] = before


def _text(value: Any) -> str:
    """A call result as BFCL's `execute_multi_turn_func_call` makes it: str as is, dict as JSON."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        with contextlib.suppress(TypeError, ValueError):
            return json.dumps(value)
    return str(value)


class _Env:
    """Fresh instances of a task's classes and the only methods a call may reach.

    A call is looked up by name in a table built from the public bound methods of the
    instances that the task's function docs name (later classes win a shared name, as
    in BFCL). Nothing is resolved with `getattr` from a name the model chose.
    """

    def __init__(self, task: BfclTask) -> None:
        names = {t["function"]["name"] for t in task.tools}
        self.instances: dict[str, Any] = {}
        self.table: dict[str, Callable[..., Any]] = {}
        for cname in task.involved_classes:
            module = _module(task.root, _CLASSES[cname][0], task.verify)
            inst = getattr(module, cname)()
            if cname not in _STATELESS:
                inst._load_scenario(
                    copy.deepcopy(task.initial_config.get(cname, {})),
                    long_context=False,
                )
            self.instances[cname] = inst
            for name, method in inspect.getmembers(inst, predicate=inspect.ismethod):
                if not name.startswith("_") and name in names:
                    self.table[name] = method

    def call(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        """Run one call; the result text, or `Error during execution: ...` as BFCL words it."""
        method = self.table.get(name)
        if method is None:
            return f"Error during execution: {name!r} is not a function of this task"
        try:
            return _text(method(*args, **kwargs))
        except Exception as e:  # noqa: BLE001  (BFCL turns any error into the result text)
            return f"Error during execution: {e}"


def _model_call(env: _Env, call: ToolCall) -> str:
    try:
        kwargs = json.loads(call.arguments)
    except (ValueError, RecursionError) as e:  # RecursionError: absurdly deep nesting
        return f"Error during execution: arguments are not valid JSON: {e}"
    if not isinstance(kwargs, dict):
        return "Error during execution: arguments must be a JSON object"
    return env.call(call.name, (), kwargs)


# --- scoring --------------------------------------------------------------------------

_MISSING = object()


def _same_state(model: _Env, truth: _Env) -> bool:
    """BFCL `state_checker`/`_compare_instances`: every public attribute of the
    ground-truth instance equals the model's, for every class."""
    return all(
        vars(model.instances[c]).get(a, _MISSING) == v
        for c, inst in truth.instances.items()
        for a, v in vars(inst).items()
        if not a.startswith("_")
    )


def _has_all(results: list[str], wanted: list[str]) -> bool:
    """BFCL `response_checker`: each wanted result is among `results`, any order,
    duplicates used up one by one."""
    left = list(results)
    for w in wanted:
        if w not in left:
            return False
        left.remove(w)
    return True


def run_task(
    task: BfclTask,
    chat: Chat,
    *,
    arm: str = "",
    trial: int = 0,
    max_steps_per_turn: int = 20,
) -> TaskRecord:
    """Run one task through `chat(messages, tools) -> ChatReply` and score it.

    Per user turn the model calls tools, each step's results going back to it, until
    it answers without a call. More than `max_steps_per_turn` steps with calls in one
    turn force-quits (BFCL's `MAXIMUM_STEP_LIMIT` is 20, `count > limit` in
    `base_handler.py`): before the last turn the task fails and the rest is not run; on
    the last turn the responses so far are scored as usual, as BFCL does. A reply the
    server refuses because the conversation no longer fits its context
    (`ContextOverflowError`) ends the task as failed. Calls are dispatched only to the public methods the task's
    function docs name; an unknown name, bad arguments or an exception becomes the
    tool result `Error during execution: ...` and is still a counted call. The model's
    text is never evaluated.

    Passed means every turn with ground truth passes BFCL's `multi_turn_checker`: the
    model made a call that turn; every public attribute of each ground-truth
    instance, after the ground-truth calls of turns 0..i ran on fresh instances,
    equals the model's (`state_checker`); and the ground-truth results of the turn are
    among all the model's results so far, in any order (`response_checker`). A turn
    with no ground truth is not checked; its calls still run. The conversation runs to
    the end even after a check failed, as in BFCL, so the token counts are comparable.
    """
    model, truth = _Env(task), _Env(task)
    messages: list[dict[str, Any]] = []
    tools = list(task.tools)
    turns: list[Turn] = []
    results: list[str] = []
    passed = True
    for turn, (question, wanted) in enumerate(
        zip(task.turns, task.ground_truth, strict=True)
    ):
        messages.append({"role": "user", "content": question})
        turns.append(Turn("user", question))
        steps, acted, forced, overflow = 0, False, False, False
        while True:
            try:
                reply = chat(messages, tools)
            except ContextOverflowError:
                overflow = True  # the conversation no longer fits: it ends here, failed
                break
            messages.append(reply.message)
            turns.append(
                Turn("assistant", reply.text, reply.tool_calls, reply.output_tokens)
            )
            if not reply.tool_calls:
                break
            acted = True
            ids = [c["id"] for c in reply.message.get("tool_calls", [])]
            for i, call in enumerate(reply.tool_calls):
                out = _model_call(model, call)
                results.append(out)
                cid = ids[i] if i < len(ids) else f"call_{i}"
                messages.append({"role": "tool", "tool_call_id": cid, "content": out})
                turns.append(Turn("tool", out))
            steps += 1
            if steps > max_steps_per_turn:
                forced = True
                break
        if overflow:
            passed = False
            break
        if forced and turn < len(task.turns) - 1:
            passed = False  # BFCL force-quit: the rest of the task is not run
            break
        expected = [truth.call(n, a, k) for n, a, k in wanted]
        if wanted and not (
            acted and _same_state(model, truth) and _has_all(results, expected)
        ):
            passed = False
    return TaskRecord.from_turns(
        task.task_id, trial, arm, task.prompt_hash, turns, passed
    )


def _check_envs(tasks: Sequence[BfclTask]) -> None:
    """Load every environment class the tasks need, so a missing file, a bad hash or
    a missing `mpmath` stops the run before the first request."""
    seen: set[tuple[Path, str, bool]] = set()
    for t in tasks:
        for c in t.involved_classes:
            if (t.root, c, t.verify) not in seen:
                seen.add((t.root, c, t.verify))
                try:
                    _module(t.root, _CLASSES[c][0], t.verify)
                except ModuleNotFoundError as e:
                    raise ModuleNotFoundError(
                        f"{c} needs the module {e.name!r}; install it, or drop the tasks "
                        f"that involve {c} (task.involved_classes)"
                    ) from e


def _load_done(
    path: Path, tasks: Sequence[BfclTask], arm: str, trials: int
) -> dict[tuple[str, int], TaskRecord]:
    """Records of an earlier run in `path`, checked against this run.

    The file is only repaired (a crash's partial last line) after it checked out.
    """
    records, cut, whole = read_tasks_tail(path)
    hashes = {t.task_id: t.prompt_hash for t in tasks}
    done: dict[tuple[str, int], TaskRecord] = {}
    for r in records:
        where = f"{path}: task {r.task_id!r} trial {r.trial}"
        if r.arm != arm:
            raise ValueError(
                f"{where} is arm {r.arm!r}, this run is arm {arm!r}; "
                "the file is another run's"
            )
        if r.task_id not in hashes:
            raise ValueError(f"{path}: task_id {r.task_id!r} is not in tasks")
        if not 0 <= r.trial < trials:
            raise ValueError(f"{where} is outside trials=1..{trials}")
        if r.prompt_hash != hashes[r.task_id]:
            raise ValueError(
                f"{where}: the task differs from the one that would be run now; "
                "the file is another run's"
            )
        if (r.task_id, r.trial) in done:
            raise ValueError(f"{where} appears twice")
        done[r.task_id, r.trial] = r
    if cut is not None:
        with path.open("r+b") as f:
            f.truncate(cut)
    elif whole:
        with path.open("ab") as f:
            f.write(b"\n")
    return done


def run_bfcl(
    tasks: Sequence[BfclTask],
    base_url: str,
    model: str,
    arm: str,
    *,
    trials: int = 1,
    max_new_tokens: int = 1024,
    max_steps_per_turn: int = 20,
    seed: int = 0,
    api_key: str | None = None,
    extra_body: Mapping[str, Any] | None = None,
    timeout: float = 120.0,
    out: str | Path | None = None,
    resume: bool = False,
    header: Mapping[str, Any] | None = None,
) -> list[TaskRecord]:
    """Run each task `trials` times against `{base_url}/v1/chat/completions`.

    Returns the records task by task, trial by trial. With `out`, each record is
    appended to the JSONL file (flushed and fsynced) as soon as its trial finishes,
    so a killed run keeps what it finished. Trial `t` uses request seed `seed + t`.
    `max_new_tokens` caps each reply. A server error raises as in `chat_http`; the
    records already written stay. Serve the base and the edit under two names and call
    this once per name, with the same `tasks`.

    A non-empty `out` raises `FileExistsError` unless `resume=True`. Resuming keeps
    the records in the file and runs only the (task, trial) pairs not in it, asking
    each the same as an uninterrupted run would (same seed); a last line cut short by
    a crash is dropped and redone. A record of another arm, of a task not in `tasks`,
    with `trial >= trials`, whose prompt hash differs, or a second record of one
    (task, trial) raises `ValueError`. The returned list holds the kept and the new
    records together.

    `header` is this run's `run_header_http` result. It must agree with this call
    (`max_new_tokens`, `seed`, `extra_body`), else `ValueError`. With `out`, it is
    written to `header_path(out)`. Resuming, an existing header file there must
    agree with this call too and, when `header` is given, have the same
    `config_hash`; it is kept as it is.
    """
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError(f"trials must be an int >= 1, got {trials!r}")
    if max_steps_per_turn < 1:
        raise ValueError(f"max_steps_per_turn must be >= 1, got {max_steps_per_turn!r}")
    ids = [t.task_id for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task_id in tasks")
    if resume and out is None:
        raise ValueError("resume=True needs out")
    _check_envs(tasks)
    path = Path(out) if out is not None else None
    done: dict[tuple[str, int], TaskRecord] = {}
    kept = None if header is None else dict(header)
    if path is not None and path.exists() and path.stat().st_size > 0:
        if not resume:
            raise FileExistsError(
                f"{path} already has records; pass resume=True to continue it "
                "or choose another path"
            )
        call = {
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "extra_body": json.loads(json.dumps(dict(extra_body or {}))),
        }
        kept = _header_to_keep(path, header, call)  # before _load_done repairs the file
        done = _load_done(path, tasks, arm, trials)
    if path is not None and kept is not None:
        write_header(header_path(path), kept)
    sink = None
    try:
        for task in tasks:
            for trial in range(trials):
                if (task.task_id, trial) in done:
                    continue
                chat = partial(
                    _chat_once,
                    base_url,
                    model,
                    max_new_tokens=max_new_tokens,
                    seed=seed + trial,
                    api_key=api_key,
                    extra_body=extra_body,
                    timeout=timeout,
                )
                record = run_task(
                    task,
                    chat,
                    arm=arm,
                    trial=trial,
                    max_steps_per_turn=max_steps_per_turn,
                )
                done[task.task_id, trial] = record
                if path is not None:
                    if sink is None:
                        sink = path.open("a", encoding="utf-8")
                    sink.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                    sink.flush()
                    os.fsync(sink.fileno())
    finally:
        if sink is not None:
            sink.close()
    return [done[t.task_id, k] for t in tasks for k in range(trials)]


def _chat_once(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    **kwargs: Any,
) -> ChatReply:
    try:
        return chat_http(base_url, model, messages, tools=tools, **kwargs)
    except ToolCallParseError:
        # the model's call was cut off or not JSON: a turn with no call (BFCL scores it
        # so), charged the whole budget; raising would lose every record of the run
        n = int(kwargs["max_new_tokens"])
        message = {"role": "assistant", "content": None}
        return ChatReply("", (), n, "error", {"completion_tokens": n}, message)
