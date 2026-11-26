"""confirm tool: pass/fail vs the current best, coarse feedback, limits and counters.

A fake generator stands in for the model, except in the one tiny-model test.
"""

import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from astro_weights.mcp.confirm import candidate_id, passes, read_items
from astro_weights.mcp.jobs import load_run_config
from astro_weights.mcp.record import RunRecord

pytest.importorskip("mcp")

import anyio  # type: ignore[import-not-found, unused-ignore]
from mcp import Client
from mcp_fakes import fake_model_dir

from astro_weights.mcp import forward
from astro_weights.mcp.edit_store import EditStore
from astro_weights.mcp.server import build_server

LIMITS = {"forward_passes": 3, "code_runs": 3, "tokens": 100, "dollars": 1.0}
N0 = "model.layers.0.mlp.down_proj.weight"
GOLDS = ["leakgold-0", "leakgold-1", "leakgold-2", "leakgold-3"]
PROMPTS = ["leakprompt-0", "leakprompt-1", "leakprompt-2", "leakprompt-3"]
PROBES = [*GOLDS, *PROMPTS, "leakid-0", "items.jsonl"]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def write_items(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"item_id": f"leakid-{i}", "prompt": p, "gold": g}
        for i, (p, g) in enumerate(zip(PROMPTS, GOLDS, strict=True))
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def confirm_run(tmp_path: Path, **confirm: Any) -> Path:
    """Run dir with a fake model; the items live outside it, in tmp_path/secret."""
    run = tmp_path / "run"
    fake_model_dir(run / "model")
    items = write_items(tmp_path / "secret" / "items.jsonl")
    cfg = {
        "seed": 7,
        "editor": "scripted",
        "model": None,
        "limits": LIMITS,
        "model_dir": "model",
        "confirm": {
            "items": str(items),
            "grader": "exact",
            "margin": 0.25,
            "max_queries": 4,
            **confirm,
        },
    }
    (run / "run.json").write_text(json.dumps(cfg))
    return run


def save_edit(run: Path, scale: float) -> None:
    store = EditStore(run, run / "model")
    w = store.original(N0)
    rng = np.random.default_rng(0)
    d = np.outer(rng.standard_normal(w.shape[0]), rng.standard_normal(w.shape[1]))
    store.save(N0, w + scale * d / np.linalg.norm(d) * np.linalg.norm(w))


class FakeModel:
    """Base gets ``base`` items right; edited gets the next count in ``script``."""

    def __init__(self, run: Path, base: int, script: list[int]) -> None:
        self.run = run
        self.base = base
        self.script = list(script)
        self.calls: list[str] = []
        self.gate: threading.Event | None = None

    def __call__(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.gate is not None:
            self.gate.wait(10)
        self.calls.append(params["arm"])
        assert params["prompts"] == PROMPTS
        if params["arm"] == "base":
            right, cand = self.base, "base"
        else:
            right = self.script.pop(0)
            manifest = json.loads((self.run / "edits" / "manifest.json").read_text())
            cand = candidate_id({k: v["sha256"] for k, v in manifest.items()})
        texts = [g if i < right else "wrong" for i, g in enumerate(GOLDS)]
        return {"texts": texts, "candidate": cand}


async def call(client: Client, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    result = await client.call_tool(tool, args)
    assert not result.is_error
    assert result.structured_content is not None
    return dict(result.structured_content)


async def poll(client: Client, handle: str) -> dict[str, Any]:
    with anyio.fail_after(10):
        while True:
            out = await call(client, "status", {"handle": handle})
            if out["state"] in {"done", "failed"}:
                return out
            await anyio.sleep(0.01)


async def query(client: Client, seen: list[str]) -> dict[str, Any]:
    started = await call(client, "confirm", {})
    seen.append(json.dumps(started))
    assert set(started) == {"handle", "state"}, started
    out = await poll(client, started["handle"])
    seen.append(json.dumps(out))
    return out


def events(run: Path, name: str) -> list[dict[str, Any]]:
    return [
        e for e in RunRecord.open(run / "record.jsonl").read() if e["event"] == name
    ]


def test_passes_is_one_sided_and_strict() -> None:
    assert not passes(5, 5, 10, 0.0)  # tie fails at margin 0
    assert passes(6, 5, 10, 0.0)
    assert not passes(7, 5, 10, 0.2)  # beats by exactly the margin: fails
    assert passes(8, 5, 10, 0.2)
    assert not passes(4, 5, 10, 0.0)
    # 0.29 * 100 is 28.999999999999996 in floats: still an exact tie
    assert not passes(29, 0, 100, 0.29)
    assert not passes(57, 0, 100, 0.57)
    assert passes(30, 0, 100, 0.29)


@pytest.mark.anyio
async def test_queries_pass_fail_and_move_the_best(tmp_path: Path) -> None:
    run = confirm_run(tmp_path)  # n = 4, margin 0.25: must beat the bar by > 1 item
    model = FakeModel(run, base=1, script=[2, 3, 3, 4])
    seen: list[str] = []
    passed = []
    async with Client(build_server(run, {"confirm": model})) as client:
        for scale in (0.01, 0.02, 0.03, 0.04):
            save_edit(run, scale)
            out = await query(client, seen)
            assert out["state"] == "done"
            assert set(out["result"]) == {"passed"}
            assert "created" not in out
            assert "updated" not in out
            passed.append(out["result"]["passed"])
        save_edit(run, 0.05)
        refused = await call(client, "confirm", {})
        seen.append(json.dumps(refused))
    # 2 vs base 1: by exactly 1 = margin, fails; 3 vs 1 passes, bar 3;
    # 3 vs 3 ties, fails; 4 vs 3 by exactly the margin, fails
    assert passed == [False, True, False, False]
    assert model.calls == ["base", "edited", "edited", "edited", "edited"]
    assert "limit" in refused["error"]
    results = events(run, "confirm_result")
    assert [(e["bar"], e["correct"], e["best"]) for e in results] == [
        (1, 2, 1),
        (1, 3, 3),
        (3, 3, 3),
        (3, 4, 3),
    ]
    counts = RunRecord.open(run / "record.jsonl").confirm_counts()
    assert counts == {
        "candidates_tried": 4,
        "candidates_saved": 0,
        "confirm_queries": 4,
        "confirm_refused": 1,
    }
    for text in seen:
        assert not [p for p in PROBES if p in text], text
        assert '"correct"' not in text
        assert '"bar"' not in text
    for path in run.rglob("*"):  # item text never; the path only in run.json
        if path.is_file() and path.name != "run.json":
            data = path.read_bytes()
            assert not [p for p in PROBES if p.encode() in data], path


@pytest.mark.anyio
async def test_same_edits_twice_count_one_candidate(tmp_path: Path) -> None:
    run = confirm_run(tmp_path, margin=0.0)
    model = FakeModel(run, base=2, script=[3, 3])
    save_edit(run, 0.01)
    async with Client(build_server(run, {"confirm": model})) as client:
        first = await query(client, [])
        second = await query(client, [])
    assert first["result"] == {"passed": True}
    assert second["result"] == {"passed": False}  # tie with itself
    counts = RunRecord.open(run / "record.jsonl").confirm_counts()
    assert counts["candidates_tried"] == 1
    assert counts["confirm_queries"] == 2


@pytest.mark.anyio
async def test_an_error_is_a_plain_fail(tmp_path: Path) -> None:
    run = confirm_run(tmp_path)

    def broken(params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(f"cannot grade {params['prompts'][0]}")

    save_edit(run, 0.01)
    async with Client(build_server(run, {"confirm": broken})) as client:
        out = await query(client, [])
    assert out["state"] == "done"
    assert out["result"] == {"passed": False}
    assert out["error"] is None
    (result,) = events(run, "confirm_result")
    assert result["error"] == "RuntimeError"
    assert "leakprompt" not in (run / "record.jsonl").read_text()


@pytest.mark.anyio
async def test_wrong_output_count_fails(tmp_path: Path) -> None:
    run = confirm_run(tmp_path)
    save_edit(run, 0.01)

    def short(params: dict[str, Any]) -> dict[str, Any]:
        return {"texts": GOLDS[:1], "candidate": "x"}

    async with Client(build_server(run, {"confirm": short})) as client:
        out = await query(client, [])
    assert out["result"] == {"passed": False}
    assert events(run, "confirm_result")[0]["error"] == "ValueError"


@pytest.mark.anyio
async def test_no_edits_is_refused_and_not_a_query(tmp_path: Path) -> None:
    run = confirm_run(tmp_path)
    async with Client(build_server(run, {"confirm": FakeModel(run, 0, [])})) as client:
        out = await call(client, "confirm", {})
    assert "no stored edits" in out["error"]
    assert RunRecord.open(run / "record.jsonl").confirm_counts()["confirm_queries"] == 0
    assert [e["category"] for e in events(run, "refusal")] == ["confirm_no_edits"]


@pytest.mark.anyio
async def test_finished_run_refuses_confirm(tmp_path: Path) -> None:
    run = confirm_run(tmp_path)
    save_edit(run, 0.01)
    model = FakeModel(run, 0, [])
    async with Client(build_server(run, {"confirm": model})) as client:
        await call(client, "finish", {"summary": "done"})
        out = await call(client, "confirm", {})
    assert out == {"error": "run already finished"}
    assert model.calls == []


@pytest.mark.anyio
async def test_unavailable_without_hf_or_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forward, "hf_available", lambda: False)
    run = confirm_run(tmp_path)
    save_edit(run, 0.01)
    async with Client(build_server(run, {"exec": lambda p: {}})) as client:
        out = await call(client, "confirm", {})
    assert out["error"].startswith("confirm is not available yet")


@pytest.mark.anyio
async def test_tool_listed_only_when_configured(tmp_path: Path) -> None:
    run = confirm_run(tmp_path, max_queries=3)
    async with Client(build_server(run, {"confirm": FakeModel(run, 0, [])})) as c:
        tools = {t.name: t.description or "" for t in (await c.list_tools()).tools}
    assert len(tools) == 8
    text = tools["confirm"]
    assert '{"passed": bool}' in text
    assert "at most 3 queries" in text
    cfg = json.loads((run / "run.json").read_text())
    del cfg["confirm"]
    other = tmp_path / "plain"
    other.mkdir()
    (other / "run.json").write_text(json.dumps({**cfg, "model_dir": None}))
    async with Client(build_server(other)) as c:
        names = {t.name for t in (await c.list_tools()).tools}
    assert "confirm" not in names


@pytest.mark.anyio
async def test_restart_keeps_the_count_and_the_best(tmp_path: Path) -> None:
    run = confirm_run(tmp_path, max_queries=3)
    save_edit(run, 0.01)
    model = FakeModel(run, base=0, script=[2])
    async with Client(build_server(run, {"confirm": model})) as client:
        assert (await query(client, []))["result"] == {"passed": True}  # bar 2
    model = FakeModel(run, base=0, script=[3, 4])
    async with Client(build_server(run, {"confirm": model})) as client:
        save_edit(run, 0.02)
        assert (await query(client, []))["result"] == {"passed": False}  # 3 vs 2
        save_edit(run, 0.03)
        assert (await query(client, []))["result"] == {"passed": True}  # 4 vs 2
        assert "limit" in (await call(client, "confirm", {}))["error"]
    assert model.calls == ["edited", "edited"]  # base not run again


@pytest.mark.anyio
async def test_restart_fails_a_running_query(tmp_path: Path) -> None:
    run = confirm_run(tmp_path)
    save_edit(run, 0.01)
    model = FakeModel(run, base=0, script=[1])
    model.gate = threading.Event()
    async with Client(build_server(run, {"confirm": model})) as client:
        started = await call(client, "confirm", {})
        build_server(run, {"confirm": model})  # a restart while it runs
        out = await call(client, "status", {"handle": started["handle"]})
        model.gate.set()
        await anyio.sleep(0.05)
    assert out["state"] == "failed"
    assert out["error"] == "server restarted"
    assert out["result"] is None


def test_changed_items_or_limit_differ_from_the_record(tmp_path: Path) -> None:
    run = confirm_run(tmp_path)
    build_server(run, {"confirm": FakeModel(run, 0, [])})
    cfg = json.loads((run / "run.json").read_text())
    cfg["confirm"]["max_queries"] = 99
    (run / "run.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="differs from the run record"):
        build_server(run, {"confirm": FakeModel(run, 0, [])})
    cfg["confirm"]["max_queries"] = 4
    (run / "run.json").write_text(json.dumps(cfg))
    Path(cfg["confirm"]["items"]).write_text(
        '{"item_id":"a","prompt":"p","gold":"g"}\n'
    )
    with pytest.raises(ValueError, match="differs from the run record"):
        build_server(run, {"confirm": FakeModel(run, 0, [])})


@pytest.mark.parametrize(
    ("confirm", "msg"),
    [
        ([], "confirm must be an object"),
        ({"items": "x", "grader": "exact", "margin": 0}, "confirm must be an object"),
        ({"items": "", "grader": "exact", "margin": 0, "max_queries": 1}, "items"),
        ({"items": "x", "grader": "fuzzy", "margin": 0, "max_queries": 1}, "grader"),
        ({"items": "x", "grader": "exact", "margin": -0.1, "max_queries": 1}, "margin"),
        ({"items": "x", "grader": "exact", "margin": 1, "max_queries": 1}, "margin"),
        ({"items": "x", "grader": "exact", "margin": True, "max_queries": 1}, "margin"),
        ({"items": "x", "grader": "exact", "margin": 0, "max_queries": 0}, "max_que"),
        ({"items": "x", "grader": "exact", "margin": 0, "max_queries": 1.0}, "max_q"),
    ],
)
def test_bad_confirm_config(tmp_path: Path, confirm: Any, msg: str) -> None:
    cfg = {"seed": 1, "editor": "e", "model": None, "limits": LIMITS}
    (tmp_path / "run.json").write_text(
        json.dumps({**cfg, "model_dir": "m", "confirm": confirm})
    )
    with pytest.raises(ValueError, match=msg):
        load_run_config(tmp_path)


def test_confirm_needs_model_dir(tmp_path: Path) -> None:
    confirm = {"items": "../x", "grader": "math", "margin": 0, "max_queries": 1}
    cfg = {"seed": 1, "editor": "e", "model": None, "limits": LIMITS}
    (tmp_path / "run.json").write_text(json.dumps({**cfg, "confirm": confirm}))
    with pytest.raises(ValueError, match="confirm needs model_dir"):
        load_run_config(tmp_path)


def test_config_resolves_items_against_the_run_dir(tmp_path: Path) -> None:
    confirm = {"items": "../i.jsonl", "grader": "choice", "margin": 0.02}
    cfg = {"seed": 1, "editor": "e", "model": None, "limits": LIMITS}
    body = {**cfg, "model_dir": "m", "confirm": {**confirm, "max_queries": 5}}
    (tmp_path / "run.json").write_text(json.dumps(body))
    got = load_run_config(tmp_path).confirm
    assert got is not None
    assert (got.items, got.grader, got.margin, got.max_queries) == (
        tmp_path / "../i.jsonl",
        "choice",
        0.02,
        5,
    )


@pytest.mark.parametrize(
    ("text", "msg"),
    [
        (None, "not readable"),
        ("", "no items"),
        ("leakprompt\n", "line 1 is not JSON"),
        ('{"item_id": "a", "prompt": "leakprompt"}\n', "line 1: needs"),
        ('\n{"item_id":"a","prompt":"p","gold":"g"}\n' * 2, "line 4: needs"),
    ],
)
def test_bad_items_never_echo_content(
    tmp_path: Path, text: str | None, msg: str
) -> None:
    path = tmp_path / "items.jsonl"
    if text is not None:
        path.write_text(text)
    with pytest.raises(ValueError, match=msg) as err:
        read_items(path)
    assert "leakprompt" not in str(err.value)


def test_bad_items_stop_the_server_before_a_record(tmp_path: Path) -> None:
    run = confirm_run(tmp_path)
    Path(json.loads((run / "run.json").read_text())["confirm"]["items"]).unlink()
    with pytest.raises(ValueError, match="not readable"):
        build_server(run)
    assert not (run / "record.jsonl").exists()


def test_counts_round_trip_through_the_record(tmp_path: Path) -> None:
    path = tmp_path / "record.jsonl"
    rec = RunRecord.start(path, seed=0, config={}, editor="e", model=None)
    rec.event("confirm_query", query=1)
    rec.event("confirm_result", query=1, candidate="a", passed=False, best=1)
    rec.event("confirm_query", query=2)
    rec.event("confirm_result", query=2, passed=False, best=1, error="OSError")
    rec.refusal(category="confirm_limit", editor="e", model=None)
    rec.refusal(category="budget", editor="e", model=None)
    rec.event("edit_saved", tensor="t", rel_frobenius=0.1)
    rec.event("edit_saved", tensor="t", rel_frobenius=0.2)
    assert RunRecord.open(path).confirm_counts() == {
        "candidates_tried": 1,
        "candidates_saved": 2,
        "confirm_queries": 2,
        "confirm_refused": 1,
    }


HF = forward.hf_available()


@pytest.mark.skipif(not HF, reason="needs the hf extra")
def test_generator_runs_the_tiny_model(tmp_path: Path) -> None:
    from mcp_tiny_hf import save_tiny_model, weight_key

    from astro_weights.mcp.confirm import generator

    tiny = save_tiny_model(tmp_path / "tiny")
    store = EditStore(tmp_path / "run", tiny)
    gen = generator(forward.ForwardBackend(tiny, store, device="cpu"), 3)
    base = gen({"arm": "base", "prompts": ["w1 w2", "w3"]})
    assert base["candidate"] == "base"
    assert len(base["texts"]) == 2
    name = weight_key(tiny, "layers.0.mlp.down_proj.weight")
    w = store.original(name)
    store.save(name, w + 0.05 * np.ones_like(w) * float(np.abs(w).mean()))
    edited = gen({"arm": "edited", "prompts": ["w1 w2"]})
    sha = json.loads(store.manifest_path.read_text())[name]["sha256"]
    assert edited["candidate"] == candidate_id({name: sha})
    assert gen({"arm": "base", "prompts": ["w1 w2"]})["texts"] == base["texts"][:1]


@pytest.mark.parametrize("items", ["items.jsonl", "kernel/items.jsonl", "../run/x"])
def test_items_inside_the_run_dir_are_refused(tmp_path: Path, items: str) -> None:
    run = tmp_path / "run"
    run.mkdir()
    confirm = {"items": items, "grader": "exact", "margin": 0, "max_queries": 1}
    cfg = {"seed": 1, "editor": "e", "model": None, "limits": LIMITS}
    body = {**cfg, "model_dir": "m", "confirm": confirm}
    (run / "run.json").write_text(json.dumps(body))
    with pytest.raises(ValueError, match="outside the run directory"):
        load_run_config(run)


def test_math_grader_without_the_extra_stops_the_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    run = confirm_run(tmp_path, grader="math")
    monkeypatch.setitem(sys.modules, "math_verify", None)  # import raises
    with pytest.raises(ValueError, match=r"astro-weights\[eval\]"):
        build_server(run, {"confirm": FakeModel(run, 0, [])})
    assert not (run / "record.jsonl").exists()


def test_math_grader_with_the_extra_starts(tmp_path: Path) -> None:
    pytest.importorskip("math_verify")
    run = confirm_run(tmp_path, grader="math")
    build_server(run, {"confirm": FakeModel(run, 0, [])})


@pytest.mark.anyio
@pytest.mark.parametrize("keep_best", [True, False])
async def test_finish_says_whether_the_export_is_the_best(
    tmp_path: Path, keep_best: bool
) -> None:
    run = confirm_run(tmp_path, margin=0.0)
    model = FakeModel(run, base=0, script=[2, 1])
    save_edit(run, 0.2)
    async with Client(build_server(run, {"confirm": model})) as client:
        assert (await query(client, []))["result"] == {"passed": True}
        save_edit(run, 0.3)
        assert (await query(client, []))["result"] == {"passed": False}
        if keep_best:
            save_edit(run, 0.2)  # back to the edits that passed
        out = await call(client, "finish", {"summary": "done"})
    assert out["export"] is not None, out
    assert out["exported_is_best"] is keep_best
    (first, _) = events(run, "confirm_result")
    assert first["best_candidate"] == first["candidate"]
    (check,) = events(run, "confirm_export")
    assert check["best_candidate"] == first["candidate"]
    assert check["exported_is_best"] is keep_best
    assert (check["exported_candidate"] == first["candidate"]) is keep_best
    assert out["exported_candidate"] == check["exported_candidate"]


@pytest.mark.anyio
async def test_finish_with_nothing_exported_is_not_the_best(tmp_path: Path) -> None:
    run = confirm_run(tmp_path)
    async with Client(build_server(run, {"confirm": FakeModel(run, 0, [])})) as c:
        out = await call(c, "finish", {"summary": "done"})
    assert out["exported_candidate"] is None
    assert out["exported_is_best"] is False


@pytest.mark.anyio
async def test_best_candidate_survives_a_restart(tmp_path: Path) -> None:
    run = confirm_run(tmp_path, margin=0.0)
    save_edit(run, 0.2)
    async with Client(build_server(run, {"confirm": FakeModel(run, 0, [1])})) as c:
        assert (await query(c, []))["result"] == {"passed": True}
    async with Client(build_server(run, {"confirm": FakeModel(run, 0, [])})) as c:
        out = await call(c, "finish", {"summary": "done"})
    assert out["exported_is_best"] is True


# --- target = "tokens": fewer output tokens, accuracy non-inferior to base ---

EX03_N = 428  # EX03: Gemma 4 E4B, grader math, margin 0.02, 428 items, base 390


def write_n_items(path: Path, n: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"item_id": f"i{i}", "prompt": f"p{i}", "gold": f"g{i}"} for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def tokens_run(tmp_path: Path, n: int = EX03_N, **confirm: Any) -> Path:
    run = tmp_path / "run"
    fake_model_dir(run / "model")
    items = write_n_items(tmp_path / "secret" / "items.jsonl", n)
    cfg = {
        "seed": 7,
        "editor": "scripted",
        "model": None,
        "limits": LIMITS,
        "model_dir": "model",
        "confirm": {
            "items": str(items),
            "grader": "exact",
            "margin": 0.02,
            "max_queries": 5,
            "target": "tokens",
            **confirm,
        },
    }
    (run / "run.json").write_text(json.dumps(cfg))
    return run


class TokenModel:
    """Base: ``base`` (correct, total tokens); edited: the next pair in ``script``.

    A third number in a tuple is that arm's cap hits (default 1).
    """

    def __init__(
        self,
        run: Path,
        n: int,
        base: tuple[int, ...],
        script: Sequence[tuple[int, ...]],
    ) -> None:
        self.run, self.n, self.base = run, n, base
        self.script = list(script)
        self.calls: list[str] = []
        self.sizes: list[int] = []

    def __call__(self, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(params["arm"])
        self.sizes.append(params.get("batch_size", 1))
        assert len(params["prompts"]) == self.n
        if params["arm"] == "base":
            got, cand = self.base, "base"
        else:
            got = self.script.pop(0)
            manifest = json.loads((self.run / "edits" / "manifest.json").read_text())
            cand = candidate_id({k: v["sha256"] for k, v in manifest.items()})
        right, total, hits = (*got, 1)[:3]
        texts = [f"g{i}" if i < right else "wrong" for i in range(self.n)]
        tokens = [total // self.n] * self.n
        tokens[0] += total - sum(tokens)
        caps = [i < hits for i in range(self.n)]
        return {"texts": texts, "tokens": tokens, "hit_cap": caps, "candidate": cand}


def test_passes_tokens_rule() -> None:
    from astro_weights.mcp.confirm import passes_tokens

    # EX03: 390 of 428 at base, margin 0.02: 8.56 items may be lost
    assert passes_tokens(390, 900, 390, 1000, EX03_N, 0.02)
    assert passes_tokens(382, 900, 390, 1000, EX03_N, 0.02)  # -8 >= -8.56
    assert not passes_tokens(381, 900, 390, 1000, EX03_N, 0.02)  # -9
    assert not passes_tokens(390, 1000, 390, 1000, EX03_N, 0.02)  # token tie
    assert not passes_tokens(400, 1001, 390, 1000, EX03_N, 0.02)
    assert passes_tokens(98, 1, 100, 2, 100, 0.02)  # exactly -2 at 0.02 * 100
    assert not passes_tokens(97, 1, 100, 2, 100, 0.02)


@pytest.mark.anyio
@pytest.mark.parametrize("target", ["accuracy", "tokens"])
async def test_ex03_terse_edit_passes_only_on_tokens(
    tmp_path: Path, target: str
) -> None:
    run = tokens_run(tmp_path, target=target)
    model = TokenModel(run, EX03_N, (390, 100_000), [(390, 60_000)])
    save_edit(run, 0.2)
    async with Client(build_server(run, {"confirm": model})) as client:
        out = await query(client, [])
        assert set(out["result"]) == {"passed"}
        finished = await call(client, "finish", {"summary": "done"})
    passed = target == "tokens"
    assert out["result"]["passed"] is passed
    assert finished["exported_is_best"] is passed
    (result,) = events(run, "confirm_result")
    assert (result["correct"], result["bar"], result["n"]) == (390, 390, EX03_N)
    if passed:
        assert result["best_candidate"] == result["candidate"]
        assert result["target"] == "tokens"
        assert (result["tokens"], result["bar_tokens"]) == (60_000, 100_000)
        assert (result["base_correct"], result["best"], result["hit_cap"]) == (
            390,
            390,
            1,
        )
        (base,) = events(run, "confirm_base")
        assert base == {
            **base,
            "n": EX03_N,
            "base_correct": 390,
            "base_tokens": 100_000,
            "base_hit_cap": 1,
        }
        first = RunRecord.open(run / "record.jsonl").read()[0]
        assert first["config"]["confirm"]["target"] == "tokens"
    else:  # the default target writes what it wrote before
        assert "target" not in result
        assert not events(run, "confirm_base")
        first = RunRecord.open(run / "record.jsonl").read()[0]
        assert "target" not in first["config"]["confirm"]


@pytest.mark.anyio
async def test_tokens_bar_moves_accuracy_bar_stays_base(tmp_path: Path) -> None:
    run = tokens_run(tmp_path, max_queries=5)
    script = [
        (382, 90_000),  # -8 vs base, fewer tokens: pass, token bar 90k
        (381, 50_000),  # -9 vs base: fail
        (382, 90_000),  # token tie with the bar: fail
        (390, 95_000),  # better accuracy, more tokens than the new bar: fail
        (383, 80_000),  # -7 vs base (not vs 382), fewer tokens: pass
    ]
    model = TokenModel(run, EX03_N, (390, 100_000), script[:2])
    async with Client(build_server(run, {"confirm": model})) as client:
        for scale in (0.1, 0.2):
            save_edit(run, scale)
            await query(client, [])
    model2 = TokenModel(run, EX03_N, (0, 0), script[2:])  # base must not run again
    async with Client(build_server(run, {"confirm": model2})) as client:
        for scale in (0.3, 0.4, 0.5):
            save_edit(run, scale)
            await query(client, [])
    assert model.calls == ["base", "edited", "edited"]
    assert model2.calls == ["edited"] * 3
    results = events(run, "confirm_result")
    assert [e["passed"] for e in results] == [True, False, False, False, True]
    assert [e["bar_tokens"] for e in results] == [
        100_000,
        90_000,
        90_000,
        90_000,
        90_000,
    ]
    assert {e["base_correct"] for e in results} == {390}
    assert [e["best"] for e in results] == [382, 382, 382, 382, 383]
    assert len(events(run, "confirm_base")) == 1


@pytest.mark.anyio
async def test_tokens_restart_after_base_only(tmp_path: Path) -> None:
    run = tokens_run(tmp_path, n=4, margin=0.0)

    def broken(params: dict[str, Any]) -> dict[str, Any]:
        if params["arm"] == "base":
            return TokenModel(run, 4, (2, 40), [])(params)
        raise RuntimeError("edited arm down")

    save_edit(run, 0.1)
    async with Client(build_server(run, {"confirm": broken})) as client:
        assert (await query(client, []))["result"] == {"passed": False}
    (first,) = events(run, "confirm_result")
    assert first["error"] == "RuntimeError"
    assert first["best"] == 2
    model = TokenModel(run, 4, (0, 0), [(2, 39)])
    async with Client(build_server(run, {"confirm": model})) as client:
        assert (await query(client, []))["result"] == {"passed": True}
    assert model.calls == ["edited"]


@pytest.mark.anyio
async def test_tokens_bad_counts_fail(tmp_path: Path) -> None:
    run = tokens_run(tmp_path, n=4)

    def no_tokens(params: dict[str, Any]) -> dict[str, Any]:
        return {"texts": ["g0"] * 4, "tokens": [1, 2], "candidate": "x"}

    save_edit(run, 0.1)
    async with Client(build_server(run, {"confirm": no_tokens})) as client:
        assert (await query(client, []))["result"] == {"passed": False}
    (result,) = events(run, "confirm_result")
    assert result["error"] == "ValueError"
    assert result["best"] is None
    assert not events(run, "confirm_base")


@pytest.mark.anyio
async def test_batch_size_reaches_both_arms(tmp_path: Path) -> None:
    run = tokens_run(tmp_path, n=4, batch_size=3)
    model = TokenModel(run, 4, (2, 40), [(2, 30)])
    save_edit(run, 0.1)
    async with Client(build_server(run, {"confirm": model})) as client:
        assert (await query(client, []))["result"] == {"passed": True}
    assert model.sizes == [3, 3]
    first = RunRecord.open(run / "record.jsonl").read()[0]
    assert first["config"]["confirm"]["batch_size"] == 3


@pytest.mark.parametrize(
    ("extra", "msg"),
    [
        ({"target": "speed"}, "confirm.target must be one of"),
        ({"batch_size": 0}, "confirm.batch_size must be an int"),
        ({"batch_size": -2}, "confirm.batch_size must be an int"),
        ({"batch_size": True}, "confirm.batch_size must be an int"),
        ({"batch_size": 2.0}, "confirm.batch_size must be an int"),
        ({"other": 1}, "optional"),
        ({"max_cap_hit": 1.5}, "confirm.max_cap_hit must be a number in"),
        ({"max_cap_hit": -0.1}, "confirm.max_cap_hit must be a number in"),
        ({"max_cap_hit": True}, "confirm.max_cap_hit must be a number in"),
        ({"max_cap_hit": "0.01"}, "confirm.max_cap_hit must be a number in"),
    ],
)
def test_bad_target_or_batch_size(tmp_path: Path, extra: Any, msg: str) -> None:
    run = tokens_run(tmp_path, n=4, **extra)
    with pytest.raises(ValueError, match=msg):
        load_run_config(run)


def test_target_and_batch_size_defaults(tmp_path: Path) -> None:
    got = load_run_config(confirm_run(tmp_path)).confirm
    assert got is not None
    assert (got.target, got.batch_size) == ("accuracy", 1)


@pytest.mark.skipif(not HF, reason="needs the hf extra")
def test_generator_batches_the_tiny_model(tmp_path: Path) -> None:
    from mcp_tiny_hf import save_tiny_model

    from astro_weights.mcp.confirm import generator

    tiny = save_tiny_model(tmp_path / "tiny")
    backend = forward.ForwardBackend(
        tiny, EditStore(tmp_path / "run", tiny), device="cpu"
    )
    gen = generator(backend, 3)
    prompts = ["w1 w2", "w3", "w1"]
    one = gen({"arm": "base", "prompts": prompts})
    assert len(one["tokens"]) == len(one["hit_cap"]) == 3
    assert all(0 < t <= 3 for t in one["tokens"])
    backend._ensure()
    if backend._tok.pad_token_id is None:
        with pytest.raises(ValueError, match="pad token"):
            gen({"arm": "base", "prompts": prompts, "batch_size": 2})
        backend._tok.pad_token = backend._tok.eos_token
    two = gen({"arm": "base", "prompts": prompts, "batch_size": 2})
    assert len(two["texts"]) == 3
    assert len(two["tokens"]) == 3


ACCURACY_TEXT = (
    "Start a confirm query of the stored edits; returns a handle to poll with "
    'status. Coarse feedback only: the job result is {"passed": bool} and '
    "nothing else. The edits pass when they beat the current best (the base "
    "model at first, then the last edits that passed) on held-out confirm items "
    "by more than the run's margin; a tie fails. Use it to select, not to tune: "
    "at most 5 queries per run, later calls are refused."
)
TOKENS_RULE = (
    "The edits pass when the edited model writes fewer output tokens over held-out "
    "confirm items than the current best (the base model at first, then the last "
    "edits that passed), while its accuracy stays within the run's margin of the "
    "base model; a token tie fails."
)


@pytest.mark.anyio
@pytest.mark.parametrize("target", ["accuracy", "tokens"])
async def test_tool_description_states_the_target_rule(
    tmp_path: Path, target: str
) -> None:
    run = tokens_run(tmp_path, n=4, target=target)
    async with Client(
        build_server(run, {"confirm": TokenModel(run, 4, (0, 0), [])})
    ) as c:
        tools = {t.name: t.description or "" for t in (await c.list_tools()).tools}
    text = tools["confirm"]
    if target == "accuracy":
        assert text == ACCURACY_TEXT
    else:
        assert text == ACCURACY_TEXT.replace(
            "The edits pass when they beat the current best (the base model at "
            "first, then the last edits that passed) on held-out confirm items by "
            "more than the run's margin; a tie fails.",
            TOKENS_RULE,
        )
        assert TOKENS_RULE in text


@pytest.mark.anyio
@pytest.mark.parametrize("target", ["accuracy", "tokens"])
async def test_cap_guard_fails_censored_edits(tmp_path: Path, target: str) -> None:
    # n 4, limit 0.25: 1 cap hit ties the limit and holds, 2 are over
    run = tokens_run(tmp_path, n=4, target=target, max_cap_hit=0.25)
    model = TokenModel(run, 4, (2, 40, 1), [(4, 30, 2), (4, 30, 1)])
    async with Client(build_server(run, {"confirm": model})) as client:
        for scale in (0.1, 0.2):
            save_edit(run, scale)
            out = await query(client, [])
            assert set(out["result"]) == {"passed"}
    first, second = events(run, "confirm_result")
    assert (first["passed"], first["cap_over"], first["hit_cap"]) == (
        False,
        ["edited"],
        2,
    )
    assert (second["passed"], second["hit_cap"]) == (True, 1)
    assert "cap_over" not in second
    assert second["best_candidate"] == second["candidate"]
    (base,) = events(run, "confirm_base")
    assert (base["base_correct"], base["base_hit_cap"]) == (2, 1)
    pinned = RunRecord.open(run / "record.jsonl").read()[0]["config"]["confirm"]
    assert pinned["max_cap_hit"] == 0.25


@pytest.mark.anyio
@pytest.mark.parametrize("target", ["accuracy", "tokens"])
async def test_cap_guard_on_a_cut_base_fails_every_query(
    tmp_path: Path, target: str
) -> None:
    run = tokens_run(tmp_path, n=4, target=target, max_cap_hit=0.25)
    model = TokenModel(run, 4, (2, 40, 2), [(4, 30, 0)])
    save_edit(run, 0.1)
    async with Client(build_server(run, {"confirm": model})) as client:
        assert (await query(client, []))["result"] == {"passed": False}
    model2 = TokenModel(run, 4, (0, 0, 0), [(4, 30, 0)])  # base must not run again
    save_edit(run, 0.2)
    async with Client(build_server(run, {"confirm": model2})) as client:
        assert (await query(client, []))["result"] == {"passed": False}
    assert model2.calls == ["edited"]
    assert [e["cap_over"] for e in events(run, "confirm_result")] == [["base"]] * 2


@pytest.mark.anyio
async def test_cap_guard_off_keeps_the_old_record(tmp_path: Path) -> None:
    run = tokens_run(tmp_path, n=4, target="accuracy")
    model = TokenModel(run, 4, (2, 40, 4), [(4, 30, 4)])
    save_edit(run, 0.1)
    async with Client(build_server(run, {"confirm": model})) as client:
        assert (await query(client, []))["result"] == {"passed": True}
        tools = {t.name: t.description or "" for t in (await client.list_tools()).tools}
    (result,) = events(run, "confirm_result")
    assert "hit_cap" not in result
    assert "cap_over" not in result
    assert not events(run, "confirm_base")
    assert "generation cap" not in tools["confirm"]


@pytest.mark.anyio
async def test_cap_guard_is_named_in_the_tool_text(tmp_path: Path) -> None:
    run = tokens_run(tmp_path, n=4, max_cap_hit=0.01)
    async with Client(
        build_server(run, {"confirm": TokenModel(run, 4, (0, 0), [])})
    ) as c:
        tools = {t.name: t.description or "" for t in (await c.list_tools()).tools}
    assert "hit the generation cap" in tools["confirm"]


@pytest.mark.anyio
@pytest.mark.parametrize("target", ["accuracy", "tokens"])
@pytest.mark.parametrize("arm", ["base", "edited"])
async def test_cap_guard_without_cap_data_fails_closed_and_recovers(
    tmp_path: Path, target: str, arm: str
) -> None:
    run = tokens_run(tmp_path, n=4, target=target, max_cap_hit=0.25)
    good = TokenModel(run, 4, (2, 40, 0), [(4, 30, 0)])

    def no_caps(params: dict[str, Any]) -> dict[str, Any]:
        out = good(params)
        if params["arm"] == arm:
            del out["hit_cap"]
        return out

    save_edit(run, 0.1)
    async with Client(build_server(run, {"confirm": no_caps})) as client:
        assert (await query(client, []))["result"] == {"passed": False}
    (bad,) = events(run, "confirm_result")
    assert bad["error"] == "CapHitsMissing"
    assert bad["passed"] is False
    assert "best_candidate" not in bad
    if arm == "base":  # nothing about the base kept: the next query reruns it
        assert bad["best"] is None
        assert not events(run, "confirm_base")
    fixed = TokenModel(run, 4, (2, 40, 0), [(4, 30, 0)])
    save_edit(run, 0.2)
    async with Client(build_server(run, {"confirm": fixed})) as client:
        assert (await query(client, []))["result"] == {"passed": True}
    assert fixed.calls == (["base", "edited"] if arm == "base" else ["edited"])
