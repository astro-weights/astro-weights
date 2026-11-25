"""The scripted editor's --edit mode: candidates made outside the run go through confirm."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mcp")

from safetensors.numpy import save_file

from astro_weights.edits import Edit, LowRankDelta
from astro_weights.export.peft import save_peft
from astro_weights.mcp.editors.scripted import (
    CANDIDATES,
    ScriptFailed,
    apply_code,
    confirm_done,
    load_candidates,
    main,
    match_weight,
    run_edits,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _delta(
    module: str, out_dim: int, in_dim: int, rank: int = 1, seed: int = 0
) -> LowRankDelta:
    rng = np.random.default_rng(seed)
    return LowRankDelta(
        module=module,
        B=rng.standard_normal((out_dim, rank)).astype(np.float32),
        A=rng.standard_normal((rank, in_dim)).astype(np.float32),
    )


def test_apply_code_sets_base_plus_delta_and_drops_old_edits() -> None:
    rng = np.random.default_rng(0)
    names = [
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    ]
    base = {n: rng.standard_normal((4, 6)).astype(np.float32) for n in names}
    store: dict[str, np.ndarray] = {names[0]: base[names[0]] + 1}
    d = _delta("base_model.model.model.layers.1.mlp.down_proj", 4, 6)
    ns: dict[str, Any] = {
        "np": np,
        "NAMES": tuple(names),
        "edited": lambda: list(store),
        "drop_edit": store.pop,
        "load_original": lambda n: base[n].copy(),
        "save_edit": store.__setitem__,
        "print": lambda *a: None,
    }
    exec(apply_code([(d.module, d.B, d.A)]), ns)  # noqa: S102
    assert list(store) == [names[1]]
    np.testing.assert_allclose(store[names[1]], base[names[1]] + d.B @ d.A, rtol=1e-6)


NAMES = (
    "model.language_model.layers.3.mlp.down_proj.weight",
    "model.vision_tower.layers.3.mlp.down_proj.weight",
    "mtp.layers.3.mlp.down_proj.weight",
)


@pytest.mark.parametrize(
    "module",
    [
        "model.layers.3.mlp.down_proj",
        "base_model.model.model.language_model.layers.3.mlp.down_proj",
    ],
)
def test_match_weight_strips_only_wrappers(module: str) -> None:
    assert match_weight(module, NAMES) == NAMES[0]


@pytest.mark.parametrize(
    ("module", "names", "msg"),
    [
        ("lm_head", NAMES, "no weight"),
        ("model.vision_tower.layers.9.mlp.down_proj", NAMES, "no weight"),
        (
            "model.layers.3.mlp.down_proj",
            (*NAMES, "model.layers.3.mlp.down_proj.weight"),
            "several",
        ),
    ],
)
def test_match_weight_refuses(module: str, names: tuple[str, ...], msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        match_weight(module, names)


def _run_dir(
    tmp_path: Path, *, rank: int = 8, with_model: bool = True
) -> tuple[Path, np.ndarray]:
    w = np.arange(12, dtype=np.float32).reshape(3, 4)
    model = tmp_path / "model"
    model.mkdir()
    save_file(
        {"model.layers.0.mlp.down_proj.weight": w}, str(model / "model.safetensors")
    )
    run = tmp_path / "run"
    run.mkdir()
    cfg: dict[str, Any] = {"export": {"rank": rank}}
    if with_model:
        cfg["model_dir"] = str(model)
    (run / "run.json").write_text(json.dumps(cfg))
    return run, w


def _peft(where: Path, d: LowRankDelta, base: dict[str, str] | None = None) -> Path:
    save_peft(Edit((d,), "base", base or {}, {}), where)
    return where


def test_load_candidates_round_trip_and_list_pinned(tmp_path: Path) -> None:
    import hashlib

    run, w = _run_dir(tmp_path)
    key = "model.layers.0.mlp.down_proj.weight"
    sha = hashlib.sha256(w.tobytes()).hexdigest()
    d = _delta("model.layers.0.mlp.down_proj", 3, 4)
    cand = _peft(tmp_path / "c", d, {key: sha})
    loaded, note = load_candidates([cand], run)
    [[(_, b, a)]] = loaded
    np.testing.assert_allclose(b @ a, d.B @ d.A, rtol=1e-5)
    assert "base not checked" not in note
    assert json.loads((run / CANDIDATES).read_text())[0]["path"] == str(cand)
    load_candidates([cand], run)  # same list on resume: fine
    other = _peft(tmp_path / "o", _delta("model.layers.0.mlp.down_proj", 3, 4, seed=1))
    with pytest.raises(ValueError, match="changed since the run started"):
        load_candidates([cand, other], run)


def test_load_candidates_refuses_other_base(tmp_path: Path) -> None:
    run, _ = _run_dir(tmp_path)
    d = _delta("model.layers.0.mlp.down_proj", 3, 4)
    cand = _peft(tmp_path / "c", d, {"model.layers.0.mlp.down_proj.weight": "0" * 64})
    with pytest.raises(ValueError, match="differs from the run model"):
        load_candidates([cand], run)
    assert not (run / CANDIDATES).exists()


def test_load_candidates_refuses_rank_above_export(tmp_path: Path) -> None:
    run, _ = _run_dir(tmp_path, rank=1)
    cand = _peft(tmp_path / "c", _delta("model.layers.0.mlp.down_proj", 3, 4, rank=2))
    with pytest.raises(ValueError, match=r"rank 2, above the run's export rank 1"):
        load_candidates([cand], run)


def test_load_candidates_notes_unpinned_base(tmp_path: Path) -> None:
    run, _ = _run_dir(tmp_path)
    cand = _peft(tmp_path / "c", _delta("model.layers.0.mlp.down_proj", 3, 4))
    _, note = load_candidates([cand], run)
    assert "1 without base sha256" in note


def test_main_refuses_a_bad_candidate_before_starting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run, _ = _run_dir(tmp_path, rank=1)
    cand = _peft(tmp_path / "c", _delta("model.layers.0.mlp.down_proj", 3, 4, rank=2))
    assert main([str(run), "--edit", str(cand)]) == 2
    assert "export rank" in capsys.readouterr().err


def test_confirm_done_reads_results_in_order(tmp_path: Path) -> None:
    lines = [
        {"event": "confirm_query", "query": 1},
        {"event": "confirm_result", "passed": True},
        {"event": "confirm_query", "query": 2},
        {"event": "confirm_result", "passed": False},
    ]
    (tmp_path / "record.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in lines) + '{"event": "conf'
    )
    assert confirm_done(tmp_path) == [True, False]
    assert confirm_done(tmp_path / "none") == []


class FakeSession:
    """Answers the tools run_edits calls; confirm results come from ``passes``."""

    def __init__(self, passes: list[bool]) -> None:
        self.passes = list(passes)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.jobs: dict[str, dict[str, Any]] = {}

    async def call_tool(self, tool: str, args: dict[str, Any]) -> Any:
        self.calls.append((tool, dict(args)))
        if tool == "status":
            out = self.jobs[args["handle"]]
        elif tool in ("exec_python", "confirm"):
            h = f"{tool}-{len(self.jobs)}"
            result = (
                {"passed": self.passes.pop(0)} if tool == "confirm" else {"ok": True}
            )
            self.jobs[h] = {"handle": h, "state": "done", "result": result}
            out = {"handle": h, "state": "queued"}
        else:
            out = {"ok": True, **args}
        return SimpleNamespace(is_error=False, structured_content=out, content=[])

    def tools(self) -> list[str]:
        return [t for t, _ in self.calls if t != "status"]

    def codes(self) -> list[str]:
        return [a["code"] for t, a in self.calls if t == "exec_python"]


C1 = [("model.layers.0.mlp.down_proj", np.ones((2, 1)), np.ones((1, 2)))]
C2 = [("model.layers.0.mlp.down_proj", np.zeros((2, 1)), np.ones((1, 2)))]


@pytest.mark.anyio
async def test_last_passing_candidate_is_reapplied_then_finished() -> None:
    s: Any = FakeSession([True, False])
    out = await run_edits(s, [C1, C2], note="candidates: x")
    assert s.tools() == [
        "exec_python",
        "confirm",
        "exec_python",
        "confirm",
        "exec_python",
        "finish",
    ]
    assert s.codes()[2] == s.codes()[0] == apply_code(C1)
    assert all(a.get("wait_s") == 50 for t, a in s.calls if t == "status")
    assert "candidate 1 exported" in out[-1]["summary"]
    assert "candidates: x" in out[-1]["summary"]


@pytest.mark.anyio
async def test_last_candidate_passing_is_not_reapplied() -> None:
    s: Any = FakeSession([False, True])
    await run_edits(s, [C1, C2])
    assert s.tools() == ["exec_python", "confirm", "exec_python", "confirm", "finish"]


@pytest.mark.anyio
async def test_no_pass_still_finishes() -> None:
    s: Any = FakeSession([False])
    out = await run_edits(s, [C1])
    assert s.tools() == ["exec_python", "confirm", "finish"]
    assert "no candidate passed" in out[-1]["summary"]


@pytest.mark.anyio
async def test_resume_skips_confirmed_candidates() -> None:
    # killed after candidate 1 passed: only candidate 2 is confirmed, then 1 is exported
    s: Any = FakeSession([False])
    await run_edits(s, [C1, C2], done=[True])
    assert s.tools() == ["exec_python", "confirm", "exec_python", "finish"]
    assert s.codes() == [apply_code(C2), apply_code(C1)]


@pytest.mark.anyio
async def test_resume_after_last_result_only_reapplies_and_finishes() -> None:
    s: Any = FakeSession([])
    await run_edits(s, [C1], done=[True])
    assert s.tools() == ["exec_python", "finish"]


@pytest.mark.anyio
async def test_more_results_than_candidates_is_refused() -> None:
    s: Any = FakeSession([])
    with pytest.raises(ValueError, match="2 confirm results for 1 candidates"):
        await run_edits(s, [C1], done=[True, False])


@pytest.mark.anyio
async def test_refused_confirm_stops_the_run() -> None:
    class Refusing(FakeSession):
        async def call_tool(self, tool: str, args: dict[str, Any]) -> Any:
            if tool == "confirm":
                self.calls.append((tool, args))
                out = {"error": "confirm limit reached"}
                return SimpleNamespace(
                    is_error=False, structured_content=out, content=[]
                )
            return await super().call_tool(tool, args)

    s: Any = Refusing([])
    with pytest.raises(ScriptFailed) as e:
        await run_edits(s, [C1])
    assert e.value.index == 1
    assert "finish" not in s.tools()


def test_edit_run_end_to_end_exports_the_passing_candidate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Real server, kernel, confirm and tiny model. Candidate 1 passes (the golds are
    its own outputs, from a first run's confirm checkpoint), candidate 2 does not;
    candidate 1 is exported and the record says so."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from mcp_tiny_hf import save_tiny_model, weight_key

    from astro_weights.mcp import sandbox
    from astro_weights.mcp.record import RunRecord

    model = save_tiny_model(tmp_path / "model")
    key = weight_key(model, "layers.1.self_attn.v_proj.weight")  # moves greedy outputs
    from safetensors import safe_open

    with safe_open(str(next(model.glob("*.safetensors"))), "np") as f:
        w = f.get_tensor(key).astype(np.float32)
    module = key.removesuffix(".weight")

    def cand(name: str, seed: int) -> Path:
        d = _delta(module, w.shape[0], w.shape[1], seed=seed)
        scale = (
            0.45 * np.linalg.norm(w) / np.linalg.norm(d.B @ d.A)
        )  # under the 0.5 cap
        return _peft(tmp_path / name, LowRankDelta(module, scale * d.B, d.A))

    c1, c2 = cand("c1", 0), cand("c2", 5)
    prompts = [f"w{i} w{i + 1} w{i + 2}" for i in range(6)]
    flags = ["--unsafe-kernel"] if sandbox.probe() is None else []

    def start(run: Path, golds: list[str], queries: int) -> Path:
        items = run.parent / f"{run.name}-items.jsonl"
        items.write_text(
            "".join(
                json.dumps({"item_id": f"i{i}", "prompt": p, "gold": g}) + "\n"
                for i, (p, g) in enumerate(zip(prompts, golds, strict=True))
            )
        )
        run.mkdir()
        cfg = {
            "seed": 7, "editor": "scripted", "model": None, "model_dir": str(model),
            "limits": {"forward_passes": 5, "code_runs": 8, "tokens": 0, "dollars": 0},
            "kernel": {"timeout_s": 60},
            "forward": {"device": "cpu", "max_new_tokens": 8},
            "confirm": {"items": str(items), "grader": "exact", "margin": 0.0, "max_queries": queries},
        }  # fmt: skip
        (run / "run.json").write_text(json.dumps(cfg))
        return run

    probe = start(tmp_path / "probe", ["?"] * 6, 1)
    assert main([str(probe), "--edit", str(c1), *flags]) == 0, capsys.readouterr().out[
        -2000:
    ]
    rows = [
        json.loads(line)
        for f in (probe / "confirm").glob("*.jsonl")
        if not f.name.startswith("base-")
        for line in f.read_text().splitlines()
    ]
    golds = [r["text"] for r in sorted(rows, key=lambda r: r["index"])]
    run = start(tmp_path / "run", golds, 2)
    assert main([str(run), "--edit", str(c1), "--edit", str(c2), *flags]) == 0
    events = RunRecord.open(run / "record.jsonl").read()
    results = [e for e in events if e["event"] == "confirm_result"]
    assert [r["passed"] for r in results] == [True, False], results
    [export] = [e for e in events if e["event"] == "confirm_export"]
    assert export["exported_is_best"] is True
    assert export["exported_candidate"] == results[0]["candidate"]
    counts = RunRecord.open(run / "record.jsonl").confirm_counts()
    assert counts["candidates_tried"] == 2
    assert counts["confirm_queries"] == 2
