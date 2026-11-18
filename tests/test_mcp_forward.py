"""forward_pass: request checks, result files, and the model run on base or edited weights.

Checks and result reads run on every install. Model tests need the ``hf`` extra
and run the tiny Qwen3.5 text model on CPU; server tests also need ``mcp``.
"""

import importlib
import importlib.util
import itertools
import json
import secrets
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mcp_fakes import fake_model_dir

from astro_weights.mcp import forward
from astro_weights.mcp.budget import Ledger, Limits
from astro_weights.mcp.edit_store import EditStore
from astro_weights.mcp.forward import (
    MAX_ATTN_ELEMENTS,
    MAX_ATTN_TEXTS,
    MAX_ATTN_TOKENS,
    MAX_PROMPTS,
    MAX_TEXT_TOKENS,
    MAX_TEXTS,
    MAX_TF_ELEMENTS,
    MAX_TOTAL_TOKENS,
    BadRequest,
    ForwardBackend,
    check_request,
    read_result,
)
from astro_weights.mcp.handles import HandleStore, Job
from astro_weights.mcp.jobs import JobRunner

HF = forward.hf_available()
needs_hf = pytest.mark.skipif(not HF, reason="needs the hf extra")
N0 = "model.layers.0.mlp.down_proj.weight"
N1 = "model.layers.0.self_attn.q_proj.weight"
CAP = 64  # max_new_tokens cap in these checks


class Stub:
    """A model for checks: one token per word ("_" has none), layers 0 and 1,
    1 full attention, ``heads`` heads."""

    def __init__(self, heads: int = 4) -> None:
        self.counted: list[list[str]] = []
        self.heads = heads

    def count_tokens(self, texts: Sequence[str], *, chat: bool) -> list[int]:
        self.counted.append(list(texts))
        return [len([w for w in t.split() if w != "_"]) for t in texts]

    def layers(self) -> tuple[int, list[int], int]:
        return 2, [1], self.heads


@pytest.fixture(scope="module")
def fake_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return fake_model_dir(tmp_path_factory.mktemp("fake"))


@pytest.fixture
def edits(tmp_path: Path, fake_model: Path) -> EditStore:
    return EditStore(tmp_path / "run", fake_model)


def save_edit(store: EditStore, name: str, scale: float = 0.05) -> None:
    w = store.original(name)
    rng = np.random.default_rng(len(name))
    noise = rng.standard_normal(w.shape).astype(np.float32) * float(np.std(w))
    store.save(name, w + scale * noise)


def tf(**over: Any) -> dict[str, Any]:
    return {"kind": "teacher_forced", "texts": ["a b"], "return": ["logprobs"]} | {
        "arm": "base",
        **over,
    }


def gen(**over: Any) -> dict[str, Any]:
    base = {"kind": "generate", "prompts": ["a"], "max_new_tokens": 4}
    return base | {"arm": "base", **over}


def check(req: Any, store: EditStore, model: Any = None) -> dict[str, Any]:
    return check_request(req, model or Stub(), store, max_new_tokens=CAP)


def words(n: int) -> str:
    return " ".join(["w"] * n)


# ---------------------------------------------------------------- checks


def test_defaults_filled(edits: EditStore) -> None:
    assert check(tf(), edits) == {
        "kind": "teacher_forced",
        "texts": ["a b"],
        "return": ["logprobs"],
        "layers": [],
        "last_tokens": 36,
        "arm": "base",
    }
    assert check(gen(), edits) == {
        "kind": "generate",
        "prompts": ["a"],
        "max_new_tokens": 4,
        "arm": "base",
        "seed": 0,
        "temperature": 0.0,
    }


@pytest.mark.parametrize(
    ("req", "msg"),
    [
        ([], "JSON object"),
        ({"kind": "train"}, '"kind"'),
        (tf(extra=1), "unknown keys"),
        (gen(texts=["a"]), "unknown keys"),
        (tf(texts=[]), '"texts" must be'),
        (tf(texts="a"), '"texts" must be'),
        (tf(texts=[1]), '"texts" must be'),
        (tf(texts=["a", ""]), "empty or whitespace-only"),
        (tf(texts=[" \n\t "]), "empty or whitespace-only"),
        (gen(prompts=[" "]), "empty or whitespace-only"),
        (tf(texts=["a", "_"]), "at least one token"),
        (gen(prompts=["_"]), "at least one token"),
        (tf(**{"return": []}), '"return"'),
        (tf(**{"return": ["weights"]}), '"return"'),
        (tf(**{"return": "hidden"}), '"return"'),
        (tf(last_tokens=0), '"last_tokens"'),
        (tf(last_tokens=True), '"last_tokens"'),
        (tf(last_tokens=MAX_TEXT_TOKENS + 1), '"last_tokens"'),
        (tf(layers=[2]), '"layers"'),
        (tf(layers=[-1]), '"layers"'),
        (tf(layers=[True]), '"layers"'),
        (tf(layers=0), '"layers"'),
        (tf(layers=[1, 1]), "repeat"),
        (tf(**{"return": ["hidden"]}), "required"),
        (tf(**{"return": ["attn"]}, layers=[0]), "no attention weights"),
        (tf(arm="both"), '"arm"'),
        (tf(arm=None), '"arm"'),
        (gen(arm="original"), '"arm"'),
        (tf(arm="edited"), "no stored edits"),
        (gen(arm="both"), "no stored edits"),
        (gen(prompts=[]), '"prompts"'),
        (gen(max_new_tokens=0), '"max_new_tokens"'),
        (gen(max_new_tokens=CAP + 1), '"max_new_tokens"'),
        ({"kind": "generate", "prompts": ["a"], "arm": "base"}, '"max_new_tokens"'),
        (gen(seed=-1), '"seed"'),
        (gen(seed=2**63), '"seed"'),
        (gen(seed=1.0), '"seed"'),
        (gen(temperature=-0.1), '"temperature"'),
        (gen(temperature=float("nan")), '"temperature"'),
        (gen(temperature=True), '"temperature"'),
        (gen(temperature="1"), '"temperature"'),
    ],
)
def test_bad_requests(edits: EditStore, req: Any, msg: str) -> None:
    with pytest.raises(BadRequest, match=msg):
        check(req, edits)


def test_unknown_keys_are_capped(edits: EditStore) -> None:
    with pytest.raises(BadRequest) as e:
        check(tf(**{"k" * 500: 1}), edits)
    assert len(str(e.value)) < 400


def test_edited_arm_needs_a_valid_store(edits: EditStore) -> None:
    save_edit(edits, N0)
    assert check(tf(arm="edited"), edits)["arm"] == "edited"
    assert check(gen(arm="both"), edits)["arm"] == "both"
    manifest = json.loads(edits.manifest_path.read_text())
    edits._path(manifest[N0]["sha256"]).write_bytes(b"tampered")
    with pytest.raises(BadRequest, match="stored edits are invalid"):
        check(tf(arm="edited"), edits)


def test_shape_checks_come_before_tokenizing(edits: EditStore) -> None:
    stub = Stub()
    with pytest.raises(BadRequest):
        check(tf(texts=["a"] * (MAX_TEXTS + 1)), edits, stub)
    with pytest.raises(BadRequest):
        check(tf(arm="edited"), edits, stub)
    assert stub.counted == []


def edge(edits: EditStore, ok: dict[str, Any], bad: dict[str, Any]) -> None:
    check(ok, edits)
    with pytest.raises(BadRequest):
        check(bad, edits)


def test_limit_edges(edits: EditStore) -> None:
    edge(edits, tf(texts=["a"] * MAX_TEXTS), tf(texts=["a"] * (MAX_TEXTS + 1)))
    edge(
        edits,
        tf(texts=[words(MAX_TEXT_TOKENS)]),
        tf(texts=[words(MAX_TEXT_TOKENS + 1)]),
    )
    per = MAX_TOTAL_TOKENS // 16
    edge(
        edits,
        tf(texts=[words(per)] * 16),
        tf(texts=[words(per)] * 15 + [words(per + 1)]),
    )
    n = 1000
    last = MAX_TF_ELEMENTS // n
    edge(
        edits,
        tf(texts=["a"] * n, last_tokens=last),
        tf(texts=["a"] * n, last_tokens=last + 1),
    )
    attn = {"return": ["attn"], "layers": [1]}
    edge(
        edits,
        tf(texts=["a"] * MAX_ATTN_TEXTS, **attn),
        tf(texts=["a"] * (MAX_ATTN_TEXTS + 1), **attn),
    )
    edge(
        edits,
        tf(texts=[words(MAX_ATTN_TOKENS)], **attn),
        tf(texts=[words(MAX_ATTN_TOKENS + 1)], **attn),
    )
    edge(
        edits, gen(prompts=["a"] * MAX_PROMPTS), gen(prompts=["a"] * (MAX_PROMPTS + 1))
    )
    # 2 texts * 1 layer * 64 heads * t**2 <= 2**28 holds up to t = 1448
    wide = Stub(heads=64)
    check(tf(texts=["a", words(1448)], **attn), edits, wide)
    with pytest.raises(BadRequest, match="64 heads"):
        check(tf(texts=["a", words(1449)], **attn), edits, wide)
    assert MAX_ATTN_ELEMENTS == 2**28
    edge(edits, gen(max_new_tokens=CAP), gen(max_new_tokens=CAP + 1))
    edge(
        edits,
        gen(prompts=[words(MAX_TEXT_TOKENS)]),
        gen(prompts=[words(MAX_TEXT_TOKENS + 1)]),
    )


def test_hf_available(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda m: None if m == "peft" else real(m)
    )
    assert not forward.hf_available()


def test_import_without_torch_names_the_extra(
    monkeypatch: pytest.MonkeyPatch, edits: EditStore, fake_model: Path
) -> None:
    for name in [
        n for n in sys.modules if n.split(".")[0] in ("torch", "transformers")
    ]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    for name in (
        "astro_weights.runners.generate",
        "astro_weights.runners.hooks",
        "astro_weights.mcp.forward",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import astro_weights.mcp as pkg

    monkeypatch.setattr(pkg, "forward", forward)  # the import below rebinds it
    mod = importlib.import_module("astro_weights.mcp.forward")
    assert mod is not forward
    backend = mod.ForwardBackend(fake_model, edits)
    for call in (
        lambda: backend.count_tokens(["a"], chat=False),
        backend.layers,
        lambda: mod._import_hf(),
    ):
        with pytest.raises(ModuleNotFoundError, match=r"astro-weights\[hf\]"):
            call()


# ---------------------------------------------------------------- results on disk


def done_job(
    run: Path,
    keys: list[str],
    *,
    kind: str = "forward",
    out: Any = None,
    state: str = "done",
) -> tuple[Job, Path]:
    jobs = HandleStore(run)
    out = secrets.token_hex(16) if out is None else out
    job = jobs.submit(kind, {"request": {}, "out": out})
    d = run / forward.RESULTS / str(out)
    if state != "queued":
        jobs.start(job.handle)
    if state == "done":
        jobs.finish(job.handle, {"keys": keys})
    elif state == "failed":
        jobs.fail(job.handle, "boom")
    return job, d


def test_read_result_array_and_texts(tmp_path: Path) -> None:
    job, d = done_job(tmp_path, ["hidden_l0", "logprobs_top_ids", "texts"])
    d.mkdir(parents=True)
    forward._save_array(
        d / "hidden_l0.npy", np.arange(6, dtype=np.float32).reshape(2, 3)
    )
    forward._save_array(d / "logprobs_top_ids.npy", np.array([[2**40]], np.int64))
    forward._save_texts(d / "texts.json", ["x", "é"])
    jobs = HandleStore(tmp_path)
    got = read_result(tmp_path, jobs, job.handle, "hidden_l0")
    assert isinstance(got, np.ndarray)
    np.testing.assert_array_equal(got, np.arange(6).reshape(2, 3))
    ids = read_result(tmp_path, jobs, job.handle, "logprobs_top_ids")
    assert isinstance(ids, np.ndarray)
    assert ids.dtype == np.int64
    assert ids[0, 0] == 2**40
    assert read_result(tmp_path, jobs, job.handle, "texts") == ["x", "é"]
    assert not list(d.glob("*.tmp"))


def test_read_result_refuses(tmp_path: Path) -> None:
    jobs = HandleStore(tmp_path)
    good, d = done_job(tmp_path, ["hidden_l0"])
    d.mkdir(parents=True)
    forward._save_array(d / "hidden_l0.npy", np.zeros(2, np.float32))
    exec_job, _ = done_job(tmp_path, ["hidden_l0"], kind="exec")
    queued, _ = done_job(tmp_path, ["hidden_l0"], state="queued")
    failed, _ = done_job(tmp_path, ["hidden_l0"], state="failed")
    no_out, _ = done_job(tmp_path, ["hidden_l0"], out="../../etc")
    cases: list[tuple[Any, Any, str]] = [
        (None, "hidden_l0", "bad handle"),
        ("../x", "hidden_l0", "bad handle"),
        (good.handle, "../hidden_l0", "bad key"),
        (good.handle, "Hidden", "bad key"),
        (good.handle, 3, "bad key"),
        (good.handle, "k" * 65, "bad key"),
        ("forward-0123456789abcdef", "hidden_l0", "unknown handle"),
        (exec_job.handle, "hidden_l0", "not a forward_pass job"),
        (queued.handle, "hidden_l0", "queued, not done"),
        (failed.handle, "hidden_l0", "failed, not done"),
        (no_out.handle, "hidden_l0", "no stored results"),
        (good.handle, "mlp_l0", "no result 'mlp_l0'"),
    ]
    for handle, key, msg in cases:
        with pytest.raises(ValueError, match=msg):
            read_result(tmp_path, jobs, handle, key)


def test_read_result_never_follows_links(tmp_path: Path) -> None:
    secret = tmp_path / "secret.npy"
    np.save(secret, np.ones(3))
    job, d = done_job(tmp_path / "run", ["hidden_l0"])
    d.mkdir(parents=True)
    (d / "hidden_l0.npy").symlink_to(secret)
    with pytest.raises(ValueError, match="not a plain file"):
        read_result(
            tmp_path / "run", HandleStore(tmp_path / "run"), job.handle, "hidden_l0"
        )
    other, d2 = done_job(tmp_path / "run", ["hidden_l0"])
    d2.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "elsewhere").mkdir()
    np.save(tmp_path / "elsewhere" / "hidden_l0.npy", np.ones(3))
    d2.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="not a plain file"):
        read_result(
            tmp_path / "run", HandleStore(tmp_path / "run"), other.handle, "hidden_l0"
        )


def test_discard_results_only_unreadable_forward_folders(tmp_path: Path) -> None:
    jobs = HandleStore(tmp_path)
    made = {
        "done": done_job(tmp_path, ["x"]),
        "failed": done_job(tmp_path, ["x"], state="failed"),
        "queued": done_job(tmp_path, ["x"], state="queued"),
        "exec": done_job(tmp_path, ["x"], kind="exec", state="failed"),
    }
    for _, d in made.values():
        d.mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.mkdir()
    bad, _ = done_job(tmp_path, ["x"], out="../victim", state="failed")
    for job, _ in [*made.values(), (bad, None)]:
        forward.discard_results(tmp_path, jobs.get(job.handle))
    kept = {k for k, (_, d) in made.items() if d.exists()}
    assert kept == {"done", "exec"}
    assert victim.exists()


def test_executor_checks_out_name(tmp_path: Path, edits: EditStore) -> None:
    run = forward.executor(ForwardBackend(tmp_path, edits), tmp_path)
    with pytest.raises(ValueError, match="bad result folder"):
        run({"request": {}, "out": "../x"})
    assert not (tmp_path / forward.RESULTS).exists()


def test_param_finds_both_layouts(tmp_path: Path, edits: EditStore) -> None:
    b = ForwardBackend(tmp_path, edits)
    b._params = {"model.layers.0.w": "short", "model.language_model.layers.1.w": "long"}
    assert b._param("model.language_model.layers.0.w") == "short"
    assert b._param("model.layers.1.w") == "long"
    with pytest.raises(KeyError, match="no parameter"):
        b._param("model.layers.2.w")


# ---------------------------------------------------------------- the model

T0 = "model.layers.0.mlp.down_proj.weight"
T1 = "model.layers.1.self_attn.q_proj.weight"
TEXTS = ["w1 w2 w3 w4 w5 w6", "w7 w8 w9"]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not HF:
        pytest.skip("needs the hf extra")
    from mcp_tiny_hf import save_tiny_model

    return save_tiny_model(tmp_path_factory.mktemp("tiny"))


@pytest.fixture
def backend(tmp_path: Path, tiny: Path) -> ForwardBackend:
    return ForwardBackend(tiny, EditStore(tmp_path / "run", tiny), device="cpu")


def run(
    b: ForwardBackend, req: dict[str, Any], tmp_path: Path
) -> tuple[dict[str, Any], Path]:
    clean = check_request(req, b, b.store, max_new_tokens=CAP)
    out = tmp_path / forward.RESULTS / secrets.token_hex(16)
    return b.run(clean, out), out


def arrays(out: Path) -> dict[str, np.ndarray]:
    return {p.stem: np.load(p) for p in sorted(out.glob("*.npy"))}


def weights(b: ForwardBackend) -> dict[str, np.ndarray]:
    return {n: p.detach().numpy().copy() for n, p in b._params.items()}


TF = tf(texts=TEXTS, **{"return": ["hidden", "logprobs"]}, layers=[0, 1], last_tokens=3)


@needs_hf
def test_tool_call_checks_with_the_real_tokenizer(backend: ForwardBackend) -> None:
    assert backend.count_tokens(TEXTS, chat=False) == [6, 3]
    assert backend.count_tokens(TEXTS, chat=True) == [6, 3]
    assert backend.count_tokens([words(MAX_TEXT_TOKENS + 50)], chat=False) == [
        MAX_TEXT_TOKENS + 1
    ]
    assert backend.layers() == (2, [1], 4)
    assert backend._model is None  # the check never loads the weights


@needs_hf
def test_teacher_forced_base_edited_restore(
    backend: ForwardBackend, tmp_path: Path
) -> None:
    summary, out = run(backend, TF, tmp_path)
    assert summary == {
        "kind": "teacher_forced",
        "arm": "base",
        "n": 2,
        "total_tokens": 9,
        "last_tokens": 3,
        "edits": {},
        "arrays": {
            "hidden_l0": [2, 32],
            "hidden_l1": [2, 32],
            "logprobs_target": [2, 3],
            "logprobs_top": [2, 3, 5],
            "logprobs_top_ids": [2, 3, 5],
        },
        "keys": [
            "hidden_l0",
            "hidden_l1",
            "logprobs_target",
            "logprobs_top",
            "logprobs_top_ids",
        ],
    }
    before = arrays(out)
    assert before["logprobs_top_ids"].dtype == np.int64
    original = weights(backend)
    save_edit(backend.store, T0, 0.2)
    summary_e, out_e = run(backend, {**TF, "arm": "edited"}, tmp_path)
    sha = backend.store.validate()[T0]["sha256"]
    assert summary_e["edits"] == {T0: sha}
    edited = arrays(out_e)
    assert not np.array_equal(edited["hidden_l0"], before["hidden_l0"])
    _, out_b = run(backend, TF, tmp_path)
    for key, value in arrays(out_b).items():
        np.testing.assert_array_equal(value, before[key], err_msg=key)
    assert all(np.array_equal(weights(backend)[n], w) for n, w in original.items())
    backend.store.drop(T0)
    _, out_d = run(backend, TF, tmp_path)
    for key, value in arrays(out_d).items():
        np.testing.assert_array_equal(value, before[key], err_msg=key)


@needs_hf
def test_drop_edit_makes_edited_equal_base(
    backend: ForwardBackend, tmp_path: Path
) -> None:
    _, out = run(backend, TF, tmp_path)
    before = arrays(out)
    save_edit(backend.store, T0, 0.2)
    save_edit(backend.store, T1, 0.2)
    run(backend, {**TF, "arm": "edited"}, tmp_path)
    backend.store.drop(T0)
    _, out_e = run(backend, {**TF, "arm": "edited"}, tmp_path)
    assert not np.array_equal(arrays(out_e)["hidden_l1"], before["hidden_l1"])
    backend.store.drop(T1)
    with pytest.raises(BadRequest, match="no stored edits"):
        run(backend, {**TF, "arm": "edited"}, tmp_path)
    _, out_b = run(backend, TF, tmp_path)
    for key, value in arrays(out_b).items():
        np.testing.assert_array_equal(value, before[key], err_msg=key)


@needs_hf
def test_edits_dropped_after_the_check_fail_the_run(
    backend: ForwardBackend, tmp_path: Path
) -> None:
    save_edit(backend.store, T0, 0.2)
    run(backend, {**TF, "arm": "edited"}, tmp_path)
    w0 = backend.store.original(T0)
    req = {**TF, "arm": "edited"}
    clean = check_request(req, backend, backend.store, max_new_tokens=CAP)
    backend.store.drop(T0)
    out = tmp_path / "late"
    with pytest.raises(ValueError, match="no stored edits"):
        backend.run(clean, out)
    assert not out.exists()
    assert backend._applied == {}
    np.testing.assert_array_equal(backend._param(T0).detach().numpy(), w0)


@needs_hf
def test_only_changed_tensors_are_copied(
    backend: ForwardBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied: list[str] = []
    real = backend._copy

    def copy(n: str, w: np.ndarray) -> None:
        copied.append(n)
        real(n, w)

    monkeypatch.setattr(backend, "_copy", copy)
    save_edit(backend.store, T0)
    run(backend, {**TF, "arm": "edited"}, tmp_path)
    run(backend, {**TF, "arm": "edited"}, tmp_path)
    assert copied == [T0]
    save_edit(backend.store, T1)
    run(backend, {**TF, "arm": "edited"}, tmp_path)
    run(backend, TF, tmp_path)
    run(backend, TF, tmp_path)
    assert copied == [T0, T1, T0, T1]


@needs_hf
def test_attention(backend: ForwardBackend, tmp_path: Path) -> None:
    req = tf(texts=TEXTS, **{"return": ["attn", "mlp"]}, layers=[1])
    summary, out = run(backend, req, tmp_path)
    assert summary["arrays"] == {
        "attn_l1_t0": [4, 6, 6],
        "attn_l1_t1": [4, 3, 3],
        "mlp_l1": [2, 64],
    }
    assert sorted(p.name for p in out.iterdir()) == [
        "attn_l1_t0.npy",
        "attn_l1_t1.npy",
        "mlp_l1.npy",
    ]


GEN = gen(prompts=["w1 w2", "w3"], max_new_tokens=6, arm="both", seed=3)


@needs_hf
def test_generate_both_arms_deterministic(
    backend: ForwardBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_edit(backend.store, T0, 0.4)
    save_edit(backend.store, T1, 0.4)
    monkeypatch.setattr(forward, "TEXT_PREVIEW", 4)
    first, out = run(backend, GEN, tmp_path)
    again, _ = run(backend, GEN, tmp_path)
    assert first == again
    assert first["keys"] == ["texts_base", "texts_edited"]
    assert (first["kind"], first["n"], first["total_tokens"]) == ("generate", 2, 3)
    full = json.loads((out / "texts_base.json").read_text())
    assert [t[:4] for t in full] == first["arms"]["base"]["texts"]
    assert any(len(t) > 4 for t in full)
    assert first["arms"]["base"]["n_tokens"] == [6, 6]
    assert first["arms"]["base"]["hit_cap"] == [True, True]
    assert first["arms"]["base"]["truncated"] == [True, True]
    sampled = {**GEN, "temperature": 1.5}
    s1, _ = run(backend, sampled, tmp_path)
    s2, _ = run(backend, sampled, tmp_path)
    s3, _ = run(backend, {**sampled, "seed": 4}, tmp_path)
    assert s1 == s2
    assert s1["arms"] != s3["arms"]
    monkeypatch.setattr(forward, "PREVIEW_TOTAL", 6)
    capped, _ = run(backend, GEN, tmp_path)
    base, edited = capped["arms"]["base"], capped["arms"]["edited"]
    assert [len(t) for t in base["texts"] + edited["texts"]] == [4, 2, 0, 0]
    assert base["truncated"] == edited["truncated"] == [True, True]
    assert sorted(edited["edits"]) == [T0, T1]
    assert base["edits"] == {}
    one, out1 = run(backend, {**GEN, "arm": "edited"}, tmp_path)
    assert one["keys"] == ["texts"]
    assert json.loads((out1 / "texts.json").read_text()) == json.loads(
        (out / "texts_edited.json").read_text()
    )


@needs_hf
def test_generate_uses_the_chat_template(
    backend: ForwardBackend, tmp_path: Path
) -> None:
    backend.count_tokens(["w1"], chat=True)
    backend._ensure()
    template = "{% for m in messages %}w9 {{ m['content'] }}{% endfor %} w10"
    for tok in (backend._counter, backend._tok):
        tok.chat_template = template
    assert backend.count_tokens(["w1 w2"], chat=True) == [4]
    summary, _ = run(backend, gen(prompts=["w1 w2"]), tmp_path)
    assert summary["total_tokens"] == 4


def boom_on(b: ForwardBackend, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    real = b._copy
    fired: list[str] = []

    def copy(n: str, w: np.ndarray) -> None:
        if n == name and not fired:  # once: the restore that follows works
            fired.append(n)
            raise OSError("disk gone")
        real(n, w)

    monkeypatch.setattr(b, "_copy", copy)


@needs_hf
def test_failure_mid_sync_restores_and_is_free(
    backend: ForwardBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out = run(backend, TF, tmp_path)
    before = arrays(out)
    original = weights(backend)
    save_edit(backend.store, T0)
    run(backend, {**TF, "arm": "edited"}, tmp_path)  # T0 is now in the weights
    save_edit(backend.store, T1)
    req = {**TF, "arm": "edited"}
    clean = check_request(req, backend, backend.store, max_new_tokens=CAP)
    boom_on(backend, T1, monkeypatch)
    run_dir = tmp_path / "jobs"
    jobs = HandleStore(run_dir)
    ledger = Ledger(run_dir / "ledger.jsonl", Limits(3, 0, 0, 0))
    runner = JobRunner(
        jobs, ledger, {"forward": forward.executor(backend, run_dir)},
        {"forward": "forward_passes"},
    )  # fmt: skip
    job = runner.submit("forward", {"request": clean, "out": secrets.token_hex(16)})
    got = wait(jobs, job.handle)
    assert got.state == "failed"
    assert "disk gone" in (got.error or "")
    assert ledger.used()["forward_passes"] == 0
    assert ledger.to_json()["streaks"]["forward_passes"] == 1
    assert not (run_dir / forward.RESULTS / job.params["out"]).exists()
    assert backend._applied == {}
    assert all(np.array_equal(weights(backend)[n], w) for n, w in original.items())
    monkeypatch.undo()
    _, out_b = run(backend, TF, tmp_path)
    for key, value in arrays(out_b).items():
        np.testing.assert_array_equal(value, before[key], err_msg=key)
    summary, _ = run(backend, {**TF, "arm": "edited"}, tmp_path)
    assert sorted(backend._applied) == [T0, T1]
    assert summary["arm"] == "edited"


@needs_hf
def test_failing_restore_leaves_tensors_marked(
    backend: ForwardBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out = run(backend, TF, tmp_path)
    before = arrays(out)
    save_edit(backend.store, T0)
    save_edit(backend.store, T1)
    run(backend, {**TF, "arm": "edited"}, tmp_path)
    real = backend.store.original

    def gone(name: str) -> np.ndarray:
        raise OSError("base weights gone")

    shas = {r["sha256"] for r in backend.store.validate().values()}
    monkeypatch.setattr(backend.store, "original", gone)
    with pytest.raises(OSError, match="base weights gone"):
        backend.run(
            check_request(TF, backend, backend.store, max_new_tokens=CAP),
            tmp_path / "o",
        )
    assert set(backend._applied.values()) <= {forward._DIRTY, *shas}
    assert backend._applied  # not known to be clean
    monkeypatch.setattr(backend.store, "original", real)
    _, out_b = run(backend, TF, tmp_path)
    for key, value in arrays(out_b).items():
        np.testing.assert_array_equal(value, before[key], err_msg=key)
    assert backend._applied == {}


def wait(jobs: HandleStore, handle: str, timeout: float = 60) -> Job:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        job = jobs.get(handle)
        if job.state in {"done", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish")


@pytest.fixture
def spans(
    backend: ForwardBackend, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[float, float]]:
    got: list[tuple[float, float]] = []
    real = backend._teacher_forced

    def slow(req: Any, out: Path) -> dict[str, Any]:
        t0 = time.monotonic()
        time.sleep(0.1)
        try:
            return real(req, out)
        finally:
            got.append((t0, time.monotonic()))

    monkeypatch.setattr(backend, "_teacher_forced", slow)
    return got


def runner_for(
    backend: ForwardBackend, run_dir: Path, limit: int
) -> tuple[JobRunner, HandleStore, Ledger]:
    jobs = HandleStore(run_dir)
    ledger = Ledger(run_dir / "ledger.jsonl", Limits(limit, 0, 0, 0))
    ex = {"forward": forward.executor(backend, run_dir)}
    return JobRunner(jobs, ledger, ex, {"forward": "forward_passes"}), jobs, ledger


@needs_hf
def test_jobs_run_one_at_a_time_and_charge_once(
    backend: ForwardBackend, tmp_path: Path, spans: list[tuple[float, float]]
) -> None:
    runner, jobs, ledger = runner_for(backend, tmp_path, 3)
    clean = check_request(TF, backend, backend.store, max_new_tokens=CAP)
    handles: list[str] = []

    def submit() -> None:
        job = runner.submit("forward", {"request": clean, "out": secrets.token_hex(16)})
        handles.append(job.handle)

    threads = [threading.Thread(target=submit) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    outs = [wait(jobs, h) for h in handles]
    assert [o.state for o in outs] == ["done"] * 3
    assert ledger.used()["forward_passes"] == 3
    got = sorted(spans)
    assert len(got) == 3
    assert all(a[1] <= b[0] for a, b in itertools.pairwise(got))


@needs_hf
def test_budget_of_one_with_two_jobs(
    backend: ForwardBackend, tmp_path: Path, spans: list[tuple[float, float]]
) -> None:
    runner, jobs, ledger = runner_for(backend, tmp_path, 1)
    clean = check_request(TF, backend, backend.store, max_new_tokens=CAP)
    a = runner.submit("forward", {"request": clean, "out": secrets.token_hex(16)})
    b = runner.submit("forward", {"request": clean, "out": secrets.token_hex(16)})
    outs = sorted((wait(jobs, a.handle), wait(jobs, b.handle)), key=lambda j: j.state)
    assert [o.state for o in outs] == ["done", "failed"]
    assert outs[1].error == "budget exhausted"
    assert ledger.used()["forward_passes"] == 1
    with pytest.raises(ValueError, match="failed, not done"):
        read_result(tmp_path, jobs, outs[1].handle, "hidden_l0")
    assert read_result(tmp_path, jobs, outs[0].handle, "hidden_l0").shape == (2, 32)  # type: ignore[union-attr]
    root = tmp_path / forward.RESULTS
    assert (root / outs[1].params["out"]).is_dir()  # computed, then refused
    for o in outs:
        forward.discard_results(tmp_path, o)
    assert sorted(p.name for p in root.iterdir()) == [outs[0].params["out"]]


@needs_hf
def test_model_load_failure_is_a_failed_run(tmp_path: Path, tiny: Path) -> None:
    b = ForwardBackend(tiny, EditStore(tmp_path / "run", tiny), device="cuda:99")
    clean = check_request(TF, b, b.store, max_new_tokens=CAP)
    out = tmp_path / "out"
    with pytest.raises(Exception):  # noqa: B017, PT011 - torch's own error type
        b.run(clean, out)
    assert not out.exists()
    assert b._model is None


@needs_hf
def test_dtype_is_applied(tmp_path: Path, tiny: Path) -> None:
    import torch

    b = ForwardBackend(tiny, EditStore(tmp_path / "run", tiny), dtype="bfloat16")
    b._ensure()
    assert next(iter(b._params.values())).dtype == torch.bfloat16


# ---------------------------------------------------------------- the server

HAS_MCP = importlib.util.find_spec("mcp") is not None
needs_server = pytest.mark.skipif(
    not (HF and HAS_MCP), reason="needs the hf and mcp extras"
)
# the anyio marker exists only where the mcp extra brings anyio
on_anyio = pytest.mark.anyio if HAS_MCP else pytest.mark.skip(reason="needs mcp")
if HAS_MCP:
    import anyio  # type: ignore[import-not-found, unused-ignore]
    from mcp import Client

    from astro_weights.mcp import sandbox
    from astro_weights.mcp.record import RunRecord
    from astro_weights.mcp.server import build_server

LIMITS = {"forward_passes": 5, "code_runs": 10, "tokens": 0, "dollars": 0}


def server_run(run_dir: Path, model_dir: Path) -> Path:
    cfg = {
        "seed": 1,
        "editor": "scripted",
        "model": None,
        "limits": LIMITS,
        "model_dir": str(model_dir),
        "kernel": {"timeout_s": 60},
        "forward": {"device": "cpu", "max_new_tokens": 8},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(cfg))
    return run_dir


async def call(client: Any, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    result = await client.call_tool(tool, args)
    assert not result.is_error
    return dict(result.structured_content)


async def poll(client: Any, handle: str) -> dict[str, Any]:
    with anyio.fail_after(120):
        while True:
            out = await call(client, "status", {"handle": handle})
            if out["state"] in {"done", "failed"}:
                return out
            await anyio.sleep(0.02)


async def job(client: Any, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    started = await call(client, tool, args)
    assert "handle" in started, started
    return await poll(client, started["handle"])


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def unsafe(monkeypatch: pytest.MonkeyPatch) -> bool:
    if sandbox.probe() is None:
        monkeypatch.setattr(sandbox, "_detect", lambda: (None, "off for this test"))
        return True
    return False


@needs_server
@on_anyio
async def test_edit_forward_and_read_back(
    tmp_path: Path, tiny: Path, unsafe: bool
) -> None:
    run_dir = server_run(tmp_path / "run", tiny)
    req = {**TF, "texts": ["w1 w2 w3 w4"]}
    async with Client(build_server(run_dir, allow_unsafe_kernel=unsafe)) as client:
        base = await job(client, "forward_pass", {"request": req})
        assert base["state"] == "done", base
        assert base["result"]["arrays"]["hidden_l0"] == [1, 32]
        saved = await job(
            client,
            "exec_python",
            {"code": f"W = load('{T0}'); W[:4] *= 1.5; save_edit('{T0}', W)"},
        )
        assert saved["result"]["ok"], saved["result"]["output"]
        edited = await job(
            client, "forward_pass", {"request": {**req, "arm": "edited"}}
        )
        assert edited["state"] == "done", edited
        cell = (
            f"a = forward_result('{base['handle']}', 'hidden_l0')\n"
            f"b = forward_result('{edited['handle']}', 'hidden_l0')\n"
            f"i = forward_result('{base['handle']}', 'logprobs_top_ids')\n"
            "print(a.shape, a.dtype, i.dtype, bool((a != b).any()))"
        )
        read = await job(client, "exec_python", {"code": cell})
        assert read["result"]["output"] == "(1, 32) float32 int64 True\n"
        g = await job(
            client,
            "forward_pass",
            {"request": gen(prompts=["w1 w2"], max_new_tokens=8, arm="both")},
        )
        assert g["state"] == "done", g
        texts = await job(
            client,
            "exec_python",
            {"code": f"forward_result('{g['handle']}', 'texts_edited')"},
        )
        shown = g["result"]["arms"]["edited"]["texts"]
        assert texts["result"]["output"] == f"{shown!r}\n"
        bad = await job(
            client,
            "exec_python",
            {"code": f"forward_result('{saved['handle']}', 'texts')"},
        )
        assert "not a forward_pass job" in bad["result"]["output"]
        assert bad["budget"]["used"]["forward_passes"] == 3


@needs_server
@on_anyio
async def test_bad_request_is_free_and_leaves_nothing(
    tmp_path: Path, tiny: Path
) -> None:
    run_dir = server_run(tmp_path / "run", tiny)
    bad = [
        {"kind": "x"},
        {**TF, "arm": "edited"},
        {**TF, "texts": [words(MAX_TEXT_TOKENS + 1)]},
        gen(max_new_tokens=9),
    ]
    async with Client(build_server(run_dir, {"exec": lambda p: {}})) as client:
        for req in bad:
            out = await call(client, "forward_pass", {"request": req})
            assert out["error"].startswith("bad request: "), out
            assert out["budget"]["used"]["forward_passes"] == 0
        await call(client, "finish", {"summary": "x"})
        after = await call(client, "forward_pass", {"request": TF})
    assert after == {"error": "run already finished"}
    events = RunRecord.open(run_dir / "record.jsonl").read()
    refusals = [e for e in events if e["event"] == "refusal"]
    assert [e["category"] for e in refusals] == ["bad_request"] * 4
    assert '"kind"' in refusals[0]["detail"]
    assert not list((run_dir / "jobs").iterdir())
    assert not (run_dir / forward.RESULTS).exists()
    assert not (run_dir / "ledger.jsonl").exists()


@needs_server
@on_anyio
async def test_check_failure_is_a_free_refusal(tmp_path: Path) -> None:
    run_dir = server_run(tmp_path / "run", fake_model_dir(tmp_path / "fake"))
    async with Client(build_server(run_dir, {"exec": lambda p: {}})) as client:
        out = await call(client, "forward_pass", {"request": TF})
    assert "Unrecognized model" in out["error"], out
    assert out["budget"]["used"]["forward_passes"] == 0
    events = RunRecord.open(run_dir / "record.jsonl").read()
    assert events[-1]["category"] == "forward_check"
    assert not list((run_dir / "jobs").iterdir())
