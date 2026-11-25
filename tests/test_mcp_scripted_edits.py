"""The scripted editor's --edit mode: candidates made outside the run go through confirm."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mcp")

from astro_weights.edits import Edit, LowRankDelta
from astro_weights.export.peft import save_peft
from astro_weights.mcp.editors.scripted import (
    ScriptFailed,
    apply_code,
    load_candidates,
    run_edits,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_apply_code_sets_base_plus_delta_and_drops_old_edits() -> None:
    rng = np.random.default_rng(0)
    names = [
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    ]
    base = {n: rng.standard_normal((4, 6)).astype(np.float32) for n in names}
    store: dict[str, np.ndarray] = {names[0]: base[names[0]] + 1}
    b = rng.standard_normal((4, 1)).astype(np.float32)
    a = rng.standard_normal((1, 6)).astype(np.float32)
    ns: dict[str, Any] = {
        "np": np,
        "NAMES": tuple(names),
        "edited": lambda: list(store),
        "drop_edit": store.pop,
        "load_original": lambda n: base[n].copy(),
        "save_edit": store.__setitem__,
        "print": lambda *a: None,
    }
    exec(apply_code([("base_model.model.model.layers.1.mlp.down_proj", b, a)]), ns)  # noqa: S102
    assert list(store) == [names[1]]
    np.testing.assert_allclose(store[names[1]], base[names[1]] + b @ a, rtol=1e-6)


def test_apply_code_refuses_an_ambiguous_module() -> None:
    names = ("a.layers.0.mlp.down_proj.weight", "b.layers.0.mlp.down_proj.weight")
    ns: dict[str, Any] = {
        "np": np,
        "NAMES": names,
        "edited": list,
        "drop_edit": None,
    }
    z = np.zeros((1, 1), np.float32)
    with pytest.raises(AssertionError):
        exec(apply_code([("layers.0.mlp.down_proj", z, z)]), ns)  # noqa: S102


def test_load_candidates_reads_peft_dirs(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    d = LowRankDelta(
        module="model.layers.3.mlp.down_proj",
        B=rng.standard_normal((4, 2)).astype(np.float32),
        A=rng.standard_normal((2, 5)).astype(np.float32),
    )
    save_peft(Edit((d,), "base", {}, {}), tmp_path / "e")
    [[(module, b, a)]] = load_candidates([tmp_path / "e"])
    assert module.endswith("layers.3.mlp.down_proj")
    np.testing.assert_allclose(b @ a, d.B @ d.A, rtol=1e-5)


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
            out = {"ok": True}
        return SimpleNamespace(is_error=False, structured_content=out, content=[])

    def tools(self) -> list[str]:
        return [t for t, _ in self.calls if t != "status"]


C1 = [("layers.0.mlp.down_proj", np.ones((2, 1)), np.ones((1, 2)))]
C2 = [("layers.0.mlp.down_proj", np.zeros((2, 1)), np.ones((1, 2)))]


@pytest.mark.anyio
async def test_last_passing_candidate_is_reapplied_then_finished() -> None:
    s: Any = FakeSession([True, False])
    await run_edits(s, [C1, C2])
    assert s.tools() == [
        "exec_python",
        "confirm",
        "exec_python",
        "confirm",
        "exec_python",
        "finish",
    ]
    codes = [a["code"] for t, a in s.calls if t == "exec_python"]
    assert codes[2] == codes[0] == apply_code(C1)
    assert all(a.get("wait_s") == 50 for t, a in s.calls if t == "status")


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
    assert out[-1] == {"ok": True}


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


def test_edit_run_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Real server, kernel and tiny model: the PEFT edit is applied, confirmed, exported."""
    import json

    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from mcp_tiny_hf import save_tiny_model, weight_key
    from safetensors import safe_open

    from astro_weights.mcp import sandbox
    from astro_weights.mcp.editors.scripted import main
    from astro_weights.mcp.record import RunRecord

    model = save_tiny_model(tmp_path / "model")
    key = weight_key(model, "layers.0.mlp.down_proj.weight")
    with safe_open(str(next(model.glob("*.safetensors"))), "np") as f:
        out_dim, in_dim = f.get_tensor(key).shape
    rng = np.random.default_rng(0)
    d = LowRankDelta(
        module=key.removesuffix(".weight"),
        B=0.002 * rng.standard_normal((out_dim, 1)).astype(np.float32),
        A=rng.standard_normal((1, in_dim)).astype(np.float32),
    )
    save_peft(Edit((d,), "base", {}, {}), tmp_path / "cand")
    items = tmp_path / "items.jsonl"
    items.write_text(
        "".join(
            json.dumps({"item_id": f"i{i}", "prompt": f"w{i} w{i + 1}", "gold": "x"})
            + "\n"
            for i in range(3)
        )
    )
    run = tmp_path / "run"
    run.mkdir()
    cfg = {
        "seed": 7,
        "editor": "scripted",
        "model": None,
        "model_dir": str(model),
        "limits": {"forward_passes": 5, "code_runs": 5, "tokens": 0, "dollars": 0},
        "kernel": {"timeout_s": 60},
        "forward": {"device": "cpu", "max_new_tokens": 4},
        "confirm": {
            "items": str(items),
            "grader": "exact",
            "margin": 0.0,
            "max_queries": 2,
        },
    }
    (run / "run.json").write_text(json.dumps(cfg))
    flags = ["--unsafe-kernel"] if sandbox.probe() is None else []
    code = main([str(run), "--edit", str(tmp_path / "cand"), *flags])
    assert code == 0, capsys.readouterr().out[-3000:]
    events = RunRecord.open(run / "record.jsonl").read()
    kinds = [e["event"] for e in events]
    assert kinds.count("confirm_query") == 1
    assert kinds.count("confirm_result") == 1
    assert kinds[-1] == "finish"
    [export] = [e for e in events if e["event"] == "confirm_export"]
    [result] = [e for e in events if e["event"] == "confirm_result"]
    assert export["exported_candidate"] == result["candidate"]
    assert export["exported_is_best"] is result["passed"]
