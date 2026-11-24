import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fake_endpoint import Reply, serve
from fake_endpoint import ok as _ok

from astro_weights.eval import (
    PINS,
    check_headers,
    compare,
    config_hash,
    header_path,
    pair,
    read_header,
    read_jsonl,
    suites,
)
from astro_weights.runners.suites_run import HFBackend, HttpBackend, run_suites

DATA = Path(__file__).parent / "data" / "suites"
FILES = {
    "gsm8k": "gsm8k.parquet",
    "ifeval": "ifeval.jsonl",
    "mmlu_pro": "mmlu_pro.parquet",
}
PATHS = {n: DATA / f for n, f in FILES.items()}
ALL = ["gsm8k", "ifeval", "mmlu_pro"]
N_ITEMS = {"gsm8k": 2, "ifeval": 2, "mmlu_pro": 3}


@pytest.fixture(autouse=True)
def fixture_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the tiny fixture files instead of the real data."""
    pins = {
        n: replace(PINS[n], sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for n, p in PATHS.items()
    }
    monkeypatch.setattr(suites, "PINS", pins)


def _need_all() -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("math_verify")


def _answer(body: dict[str, Any]) -> Reply:
    """The base model answers one way, the edit another."""
    prompt = body["messages"][0]["content"]
    edit = body["model"] == "edit"
    if "\\boxed{}" in prompt and "letter" in prompt:
        return _ok("\\boxed{D}" if edit else "\\boxed{B}")
    if "\\boxed{}" in prompt:
        return _ok("\\boxed{999}" if edit else "\\boxed{5}")
    if "commas" in prompt:
        return _ok("a, b" if edit else "no commas here")
    return _ok("- a\n- b\n- c\nDone")


def _http(model: str, **kw: Any) -> HttpBackend:
    return HttpBackend(kw.pop("url"), model, **kw)


def _props(state: Any) -> None:
    state.paths["/props"] = (200, json.dumps({"model_path": "/m/x.gguf"}))


def test_http_two_arms_on_all_suites(tmp_path: Path) -> None:
    _need_all()
    with serve(_answer) as (url, state):
        _props(state)
        got = {
            arm: run_suites(
                ALL,
                arm,
                tmp_path,
                backend=_http(arm, url=url),
                max_new_tokens=16,
                suite_paths=PATHS,
            )
            for arm in ("base", "edit")
        }
    hashes = set()
    for name in ALL:
        assert got["base"][name] == tmp_path / name / "base.jsonl"
        base = read_jsonl(got["base"][name])
        edit = read_jsonl(got["edit"][name])
        assert len(base) == N_ITEMS[name]
        assert len(pair(base, edit)) == N_ITEMS[name]
        assert compare(base, edit, n_boot=50).n_items == N_ITEMS[name]
        bh = read_header(header_path(got["base"][name]))
        eh = read_header(header_path(got["edit"][name]))
        check_headers(bh, eh)
        assert (bh["suite"], bh["limit"], bh["suite_verified"]) == (name, None, True)
        assert (
            bh["suite_sha256"] == hashlib.sha256(PATHS[name].read_bytes()).hexdigest()
        )
        assert bh["suite_source"] == str(PATHS[name])
        assert bh["grader"] == f"suite_grader:{name}"
        hashes.add(config_hash(bh))
    assert len(hashes) == 3  # config_hash differs per suite
    # real grader scores, not the generate step's placeholder
    gsm = {r.item_id: r.score for r in read_jsonl(got["base"]["gsm8k"])}
    assert gsm == {"gsm8k-0": 1.0, "gsm8k-1": 0.0}
    mmlu = [r.score for r in read_jsonl(got["edit"]["mmlu_pro"])]
    assert mmlu == [0.0, 1.0, 0.0]
    ife = {r.item_id: r.score for r in read_jsonl(got["base"]["ifeval"])}
    assert ife == {"ifeval-1000": 1.0, "ifeval-1001": 1.0}
    ife = {r.item_id: r.score for r in read_jsonl(got["edit"]["ifeval"])}
    assert ife == {"ifeval-1000": 0.0, "ifeval-1001": 1.0}


def test_http_resume_limit_and_refusals(tmp_path: Path) -> None:
    with serve(_answer) as (url, state):
        _props(state)
        be = _http("base", url=url)

        def run(**kw: Any) -> Path:
            return run_suites(
                ["ifeval"], "base", tmp_path, backend=be, max_new_tokens=16,
                suite_paths=PATHS, **kw,
            )["ifeval"]  # fmt: skip

        path = run()
        want = path.read_text()
        n_requests = len(state.requests)
        # crash after one record plus a cut last line: resume redoes only the rest
        first = want.splitlines(keepends=True)[0]
        path.write_text(first + want[len(first) :][:20])
        assert run() == path
        assert path.read_text() == want
        assert len(read_jsonl(path)) == 2
        assert len(state.requests) > n_requests
        with pytest.raises(FileExistsError):
            run(resume=False)
        with pytest.raises(ValueError, match="config_hash"):
            run(limit=1)  # limit is a config key: another run's file
        run_suites(
            ["ifeval"], "base", tmp_path / "smoke", backend=be, max_new_tokens=16,
            suite_paths=PATHS, limit=1,
        )  # fmt: skip
    smoke = tmp_path / "smoke" / "ifeval" / "base"
    assert [r.item_id for r in read_jsonl(smoke.with_suffix(".jsonl"))] == [
        "ifeval-1000"
    ]
    assert read_header(header_path(smoke.with_suffix(".jsonl")))["limit"] == 1


def test_bad_arguments_and_wrong_sha(tmp_path: Path, monkeypatch: Any) -> None:
    be = HttpBackend("http://127.0.0.1:1", "m")

    def run(names: list[str], **kw: Any) -> Any:
        return run_suites(names, "a", tmp_path, backend=be, max_new_tokens=4, **kw)

    with pytest.raises(ValueError, match="non-empty"):
        run([])
    with pytest.raises(ValueError, match="without repeats"):
        run(["ifeval", "ifeval"])
    with pytest.raises(ValueError, match="unknown suites"):
        run(["nope"])
    for bad in (0, True, 1.5):
        with pytest.raises(ValueError, match="limit"):
            run(["ifeval"], limit=bad)
    # a file whose hash is not the pin is refused before anything is generated
    monkeypatch.undo()
    with pytest.raises(ValueError, match="expected"):
        run(["ifeval"], suite_paths=PATHS)
    assert not list(tmp_path.iterdir())


def _tok() -> Any:
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2} | {f"w{i}": i + 3 for i in range(125)}
    t = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=t, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
        pad_token="</s>",
    )  # fmt: skip


@pytest.mark.parametrize("batch_size", [1, 2])
def test_hf_two_arms_resume_and_header(
    tiny_llama: Any, tmp_path: Path, batch_size: int
) -> None:
    import numpy as np

    pytest.importorskip("peft")
    from astro_weights.edits import Edit, rank1
    from astro_weights.runners.hf import apply

    _need_all()
    model = tiny_llama
    tok = _tok()

    def run(arm: str, m: Any, **kw: Any) -> dict[str, Path]:
        be = HFBackend(m, tok, batch_size=batch_size, chat=False)
        return run_suites(
            ALL, arm, tmp_path, backend=be, max_new_tokens=4, suite_paths=PATHS, **kw
        )

    base = run("base", model, resume=False)
    # interrupted run: the first record and a cut line; resume leaves no duplicates
    path = base["mmlu_pro"]
    want = path.read_text()
    first = want.splitlines(keepends=True)[0]
    path.write_text(first + want[len(first) :][:15])
    header = read_header(header_path(path))
    assert run("base", model)["mmlu_pro"] == path
    assert [r.item_id for r in read_jsonl(path)] == [
        "mmlu_pro-70", "mmlu_pro-71", "mmlu_pro-72",
    ]  # fmt: skip
    assert read_header(header_path(path))["run_id"] == header["run_id"]
    with pytest.raises(FileExistsError):
        run("base", model, resume=False)
    assert read_header(header_path(path))["run_id"] == header["run_id"]
    with pytest.raises(ValueError, match="config_hash"):
        run_suites(
            ["mmlu_pro"], "base", tmp_path,
            backend=HFBackend(model, tok, batch_size=batch_size, chat=False),
            max_new_tokens=5, suite_paths=PATHS,
        )  # fmt: skip

    module = "model.layers.1.mlp.down_proj"
    w = model.get_submodule(module).weight
    rng = np.random.default_rng(0)
    delta = rank1(
        module, 5 * rng.standard_normal(w.shape[0]), rng.standard_normal(w.shape[1])
    )
    edit = run("edit", apply(model, Edit((delta,), "tiny")))
    for name in ALL:
        b, e = read_jsonl(base[name]), read_jsonl(edit[name])
        assert len(pair(b, e)) == N_ITEMS[name]
        assert compare(b, e, n_boot=50).n_items == N_ITEMS[name]
        check_headers(
            read_header(header_path(base[name])), read_header(header_path(edit[name]))
        )


def test_http_system_arm_pairs_with_base_and_resume_checks_it(tmp_path: Path) -> None:
    sys_text = "Be brief."
    with serve(_answer) as (url, state):
        _props(state)

        def run(arm: str, **kw: Any) -> Path:
            return run_suites(
                ["ifeval"], arm, tmp_path, backend=_http("base", url=url),
                max_new_tokens=16, suite_paths=PATHS, **kw,
            )["ifeval"]  # fmt: skip

        base = run("base")
        prompt = run("prompt", system=sys_text)
        assert [
            r["body"]["messages"][0]["role"] for r in state.requests if "body" in r
        ][-2:] == ["system", "system"]
        b, p = read_jsonl(base), read_jsonl(prompt)
        assert len(pair(b, p)) == len(b)
        hb, hp = read_header(header_path(base)), read_header(header_path(prompt))
        assert hp["system"] == sys_text
        assert "system" not in hb
        check_headers(hb, hp, ignore=("system", "system_sha256"))
        assert run("prompt", system=sys_text) == prompt  # same system resumes
        with pytest.raises(ValueError, match=r"does not match|config_hash"):
            run("prompt", system="Be long.")
