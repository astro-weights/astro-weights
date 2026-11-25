"""`astro-weights gen ARM.toml`: one arm's generation, resumable, from a TOML file."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from fake_endpoint import Reply, ok, serve

from astro_weights.cli import main
from astro_weights.eval import header_path, read_header, read_jsonl, suites
from astro_weights.eval.arm_config import load_arm_config, load_items
from astro_weights.eval.splits import SplitManifest
from astro_weights.runners import bfcl, endpoint

N = 8
DATA = Path(__file__).parent / "data"
BFCL = DATA / "bfcl"


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(endpoint, "_RETRY_SLEEP", 0.0)


def _answer(body: dict[str, Any]) -> Reply:
    """Even items are answered with their gold, odd ones wrongly."""
    text = body["messages"][-1]["content"]
    if text == "Hi":  # the header's probe
        return ok("hi")
    n = int(text.removeprefix("q"))
    return ok(f"a{n}" if n % 2 == 0 else "x")


def _items(path: Path, ids: list[int] | None = None) -> Path:
    rows = [
        {"item_id": f"i{i}", "prompt": f"q{i}", "gold": f"a{i}"}
        for i in (range(N) if ids is None else ids)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _toml(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k} = {_toml(v)}" for k, v in value.items()) + "}"
    if isinstance(value, bool):
        return str(value).lower()
    return json.dumps(value)


def _write(path: Path, **keys: Any) -> Path:
    path.write_text("".join(f"{k} = {_toml(v)}\n" for k, v in keys.items()))
    return path


def _http_cfg(tmp: Path, url: str, **kw: Any) -> Path:
    keys: dict[str, Any] = {
        "kind": "http",
        "arm": "base",
        "base_url": url,
        "model": "m",
        "max_new_tokens": 16,
        "items": "items.jsonl",
        "grader": "exact",
        "out": "out/base.jsonl",
    }
    _items(tmp / "items.jsonl")
    return _write(tmp / "ARM.toml", **{**keys, **kw})


def gen(
    cfg: Path, *flags: str, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str, str]:
    code = main(["gen", str(cfg), *flags])
    out, err = capsys.readouterr()
    return code, out, err


def _asked(requests: list[dict[str, Any]]) -> Counter[str]:
    """Prompt of each chat request; the header's `Hi` probe is left out."""
    return Counter(
        r["body"]["messages"][-1]["content"]
        for r in requests
        if "body" in r and r["body"]["messages"][-1]["content"] != "Hi"
    )


def _lines(path: Path) -> int:
    return path.read_bytes().count(b"\n") if path.exists() else 0


# ---------------------------------------------------------------- the TOML file


@pytest.mark.parametrize(
    ("change", "msg"),
    [
        ({"bogus": 1}, "unknown keys \\['bogus'\\]"),
        ({"items": None}, "missing keys \\['items'\\]"),
        ({"kind": "hf"}, "kind: must be one of"),
        ({"kind": None}, "kind: must be one of"),
        ({"grader": "fuzzy"}, "grader: must be one of"),
        ({"max_new_tokens": 0}, "max_new_tokens: must be an integer >= 1"),
        ({"max_new_tokens": True}, "max_new_tokens: must be an integer >= 1"),
        ({"model": ""}, "model: must be a non-empty string"),
        ({"manifest": "m.json"}, "manifest and split go together"),
        ({"split": "test"}, "manifest and split go together"),
        ({"manifest": "m.json", "split": "all"}, "split: must be one of"),
        ({"extra_body": 3}, "extra_body: must be a table"),
        ({"serving_flags": {"--parallel": 4}}, "serving_flags: values must be strings"),
        ({"timeout": 0}, "timeout: must be a number > 0"),
        ({"concurrency": 0}, "concurrency: must be an integer >= 1"),
        ({"suites": ["gsm8k"]}, "unknown keys \\['suites'\\]"),  # not an http key
        ({"trials": 2}, "unknown keys \\['trials'\\]"),
    ],
)
def test_bad_config_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], change: dict[str, Any], msg: str
) -> None:
    cfg = _http_cfg(tmp_path, "http://127.0.0.1:1")
    keys = {
        "kind": "http", "arm": "base", "base_url": "http://127.0.0.1:1", "model": "m",
        "max_new_tokens": 16, "items": "items.jsonl", "grader": "exact", "out": "o.jsonl",
    }  # fmt: skip
    keys.update(change)
    _write(cfg, **{k: v for k, v in keys.items() if v is not None})
    with pytest.raises(ValueError, match=msg):
        load_arm_config(cfg)
    code, out, err = gen(cfg, capsys=capsys)
    assert code == 2
    assert out == ""
    assert err.startswith("error: ")
    assert str(cfg) in err


def test_toml_syntax_error_and_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("kind = = 1\n")
    assert gen(bad, capsys=capsys)[0] == 2
    assert gen(tmp_path / "none.toml", capsys=capsys)[0] == 2


def test_suites_and_bfcl_keys(tmp_path: Path) -> None:
    s = _write(
        tmp_path / "s.toml", kind="suites", arm="base", base_url="http://x", model="m",
        max_new_tokens=8, out_dir="o", suites=["ifeval"], limit=1,
        suite_paths={"ifeval": "d/ifeval.jsonl"}, cache_dir="c",
    )  # fmt: skip
    c = load_arm_config(s)
    assert c.out_dir == tmp_path / "o"
    assert c.suite_paths == {"ifeval": tmp_path / "d/ifeval.jsonl"}
    assert (c.limit, c.cache_dir, c.suites) == (1, tmp_path / "c", ("ifeval",))
    b = _write(
        tmp_path / "b.toml", kind="bfcl", arm="base", base_url="http://x", model="m",
        max_new_tokens=8, out="o.jsonl", bfcl_root="g", trials=4,
        serving_flags={"--parallel": "4"}, model_id="x",
    )  # fmt: skip
    c = load_arm_config(b)
    assert (c.trials, c.bfcl_root, c.serving_flags) == (
        4,
        tmp_path / "g",
        {"--parallel": "4"},
    )
    for bad in ([], ["a", "a"], [1]):
        _write(
            tmp_path / "s.toml", kind="suites", arm="base", base_url="http://x",
            model="m", max_new_tokens=8, out_dir="o", suites=bad,
        )  # fmt: skip
        with pytest.raises(ValueError, match="suites: must be a non-empty list"):
            load_arm_config(tmp_path / "s.toml")


# ------------------------------------------------------------------ http kind


def test_http_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with serve(_answer) as (url, state):
        state.paths["/props"] = (200, json.dumps({"model_path": "/m/x.gguf"}))
        cfg = _http_cfg(
            tmp_path, url, seed=3, concurrency=2, extra_body={"top_k": 1},
            serving_flags={"--parallel": "2"}, model_id="BASE", system="Be brief.",
        )  # fmt: skip
        code, out, err = gen(cfg, capsys=capsys)
        asked = _asked(state.requests)
    assert (code, err) == (0, "")
    assert out == f"base: {N}/{N} items, {tmp_path / 'out/base.jsonl'}\n"
    recs = read_jsonl(tmp_path / "out/base.jsonl")
    assert [r.item_id for r in recs] == [f"i{i}" for i in range(N)]
    assert [r.score for r in recs] == [1.0 if i % 2 == 0 else 0.0 for i in range(N)]
    assert {(r.arm, r.seed) for r in recs} == {("base", 3)}
    assert asked == Counter({f"q{i}": 1 for i in range(N)})
    h = read_header(header_path(tmp_path / "out/base.jsonl"))
    assert (h["max_new_tokens"], h["seed"], h["extra_body"]) == (16, 3, {"top_k": 1})
    assert (h["concurrency"], h["model_id"], h["system"]) == (2, "BASE", "Be brief.")
    assert h["serving_flags"] == {"--parallel": "2"}
    first = [r for r in state.requests if "body" in r][1]["body"]  # after the probe
    assert first["messages"][0] == {"role": "system", "content": "Be brief."}
    assert first["top_k"] == 1


def test_json_output_and_refusals(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with serve(_answer) as (url, state):
        cfg = _http_cfg(tmp_path, url)
        code, out, _ = gen(cfg, "--json", capsys=capsys)
        assert code == 0
        got = json.loads(out)
        assert got == {
            "arm": "base",
            "out": str(tmp_path / "out/base.jsonl"),
            "done": N,
            "total": N,
        }
        before = (tmp_path / "out/base.jsonl").read_bytes()
        n = len(state.requests)
        code, out, err = gen(cfg, capsys=capsys)  # records there, no --resume
        assert (code, out) == (2, "")
        assert "already has records" in err
        assert len(state.requests) == n
        code, out, _ = gen(
            cfg, "--resume", "--json", capsys=capsys
        )  # finished: asks nothing
        assert (code, json.loads(out)["done"]) == (0, N)
        assert _asked(state.requests[n:]) == Counter()
    assert (tmp_path / "out/base.jsonl").read_bytes() == before


def test_resume_with_nothing_written_starts_fresh(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with serve(_answer) as (url, _state):
        code, out, _ = gen(_http_cfg(tmp_path, url), "--resume", capsys=capsys)
    assert code == 0
    assert f"{N}/{N}" in out


def test_changed_max_new_tokens_refused_and_file_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with serve(_answer) as (url, state):
        cfg = _http_cfg(tmp_path, url)
        assert gen(cfg, capsys=capsys)[0] == 0
        out = tmp_path / "out/base.jsonl"
        lines = out.read_bytes().splitlines(keepends=True)
        out.write_bytes(b"".join(lines[:3]))  # a run cut short
        before = (out.read_bytes(), header_path(out).read_bytes())
        n = len(state.requests)
        _http_cfg(tmp_path, url, max_new_tokens=32)
        code, stdout, err = gen(cfg, "--resume", capsys=capsys)
        assert (code, stdout) == (2, "")
        assert "max_new_tokens" in err or "config_hash" in err
        assert _asked(state.requests[n:]) == Counter()
    assert (out.read_bytes(), header_path(out).read_bytes()) == before


def test_server_error_exits_1_keeps_records_then_resumes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    down = [True]

    def respond(body: dict[str, Any]) -> Reply:
        if down[0] and body["messages"][-1]["content"] == "q4":
            return 500, "boom"
        return _answer(body)

    with serve(respond) as (url, _state):
        cfg = _http_cfg(tmp_path, url)
        code, out, err = gen(cfg, "--json", capsys=capsys)
        assert code == 1
        assert "q4" in err or "i4" in err
        done = json.loads(out)
        assert (done["done"], done["total"]) == (4, N)
        assert _lines(tmp_path / "out/base.jsonl") == 4
        down[0] = False
        code, out, _ = gen(cfg, "--resume", capsys=capsys)
        assert code == 0
    assert [r.item_id for r in read_jsonl(tmp_path / "out/base.jsonl")] == [
        f"i{i}" for i in range(N)
    ]


def test_malformed_reply_mid_run_exits_1_keeps_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A reply the runner cannot read, after requests began, is a runtime error."""
    bad = ["q4"]

    def respond(body: dict[str, Any]) -> Reply:
        if body["messages"][-1]["content"] in bad:
            return 200, json.dumps({"choices": [{"message": None}]})
        return _answer(body)

    with serve(respond) as (url, _state):
        cfg = _http_cfg(tmp_path, url)
        code, out, err = gen(cfg, "--json", capsys=capsys)
        assert code == 1
        assert "no message" in err
        assert json.loads(out) == {
            "arm": "base", "out": str(tmp_path / "out/base.jsonl"), "done": 4, "total": N,
        }  # fmt: skip
        bad.clear()
        assert gen(cfg, "--resume", capsys=capsys)[0] == 0
        # the very first reply unreadable: still after the first request, still 1
        (tmp_path / "out/base.jsonl").unlink()
        bad.append("q0")
        code, out, _ = gen(cfg, capsys=capsys)
        assert (code, out.startswith("base: 0/8 items")) == (1, True)
    assert _lines(tmp_path / "out/base.jsonl") == 0


def test_failure_counts_only_the_seed_files_when_k_above_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, test = _manifest(tmp_path)
    rows = [{"item_id": f"i{i}", "prompt": f"q{i}", "gold": f"a{i}"} for i in range(40)]
    (tmp_path / "all.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    down = [False]

    def respond(body: dict[str, Any]) -> Reply:
        if down[0] and body["messages"][-1]["content"] == f"q{max(test)}":
            return 500, "boom"
        return _answer(body)

    with serve(respond) as (url, _state):
        cfg = _write(
            tmp_path / "ARM.toml", kind="http", arm="base", base_url=url, model="m",
            max_new_tokens=16, items="all.jsonl", grader="exact", out="t/base.jsonl",
            manifest="splits.json", split="test",
        )  # fmt: skip
        assert gen(cfg, capsys=capsys)[0] == 0
        s1 = tmp_path / "t/base.s1.jsonl"
        lines = s1.read_bytes().splitlines(keepends=True)
        s1.write_bytes(b"".join(lines[:-1]))  # seed 1 lost its last record
        assert (tmp_path / "t/base.jsonl").exists()  # the joined file is still there
        down[0] = True
        code, out, _ = gen(cfg, "--resume", "--json", capsys=capsys)
    assert code == 1
    got = json.loads(out)
    assert (got["done"], got["total"]) == (2 * len(test) - 1, 2 * len(test))


def test_api_key_from_environment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ARM_KEY", raising=False)
    with serve(_answer) as (url, state):
        cfg = _http_cfg(tmp_path, url, api_key_env="ARM_KEY")
        code, _, err = gen(cfg, capsys=capsys)
        assert (code, "$ARM_KEY is not set" in err) == (2, True)
        monkeypatch.setenv("ARM_KEY", "sekret")
        assert gen(cfg, capsys=capsys)[0] == 0
        auth = {r["auth"] for r in state.requests if "body" in r}
    assert auth == {"Bearer sekret"}
    assert "sekret" not in (tmp_path / "out/base.header.json").read_text()


def test_math_grader_needs_the_extra(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda n, *a: None if n == "math_verify" else real(n, *a),
    )
    cfg = _http_cfg(tmp_path, "http://127.0.0.1:1", grader="math")
    code, _, err = gen(cfg, capsys=capsys)
    assert code == 2
    assert "astro-weights[eval]" in err


def _manifest(tmp: Path, k: int = 2) -> tuple[SplitManifest, list[int]]:
    m = SplitManifest.build(
        [f"i{i}" for i in range(40)],
        dataset="d",
        revision="r",
        file_sha256="0" * 64,
        seed=0,
        k=k,
    )
    m.save(tmp / "splits.json")
    return m, [int(i[1:]) for i in m.test]


def test_manifest_split_and_k_seeds_joined(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    m, test = _manifest(tmp_path)
    items = tmp_path / "all.jsonl"
    rows = [{"item_id": f"i{i}", "prompt": f"q{i}", "gold": f"a{i}"} for i in range(40)]
    items.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with serve(_answer) as (url, state):
        cfg = _write(
            tmp_path / "ARM.toml", kind="http", arm="base", base_url=url, model="m",
            max_new_tokens=16, items="all.jsonl", grader="exact", out="test/base.jsonl",
            manifest="splits.json", split="test",
        )  # fmt: skip
        code, out, _ = gen(cfg, "--json", capsys=capsys)
        asked = _asked(state.requests)
    assert code == 0
    assert json.loads(out)["total"] == 2 * len(test)
    assert set(asked) == {f"q{i}" for i in test}
    assert set(asked.values()) == {2}  # once per seed
    seeds = [read_jsonl(tmp_path / f"test/base.s{s}.jsonl") for s in range(2)]
    joined = read_jsonl(tmp_path / "test/base.jsonl")
    assert joined == seeds[0] + seeds[1]
    assert {r.seed for r in seeds[1]} == {1}
    h = read_header(header_path(tmp_path / "test/base.jsonl"))
    assert (h["k"], h["seed"]) == (2, 0)
    assert m.check_seeds(joined) == (0, 1)
    # a finished run, resumed: the same joined bytes
    before = (tmp_path / "test/base.jsonl").read_bytes()
    assert gen(cfg, "--resume", capsys=capsys)[0] == 0
    assert (tmp_path / "test/base.jsonl").read_bytes() == before
    # one seed file left from an old run, no --resume: refused before any request
    (tmp_path / "test/base.jsonl").unlink()
    assert gen(cfg, capsys=capsys)[0] == 2


def test_load_items_errors(tmp_path: Path) -> None:
    m, test = _manifest(tmp_path, k=4)
    base: dict[str, Any] = {
        "kind": "http", "arm": "b", "base_url": "http://x", "model": "m",
        "max_new_tokens": 8, "grader": "exact", "out": "o.jsonl", "items": "i.jsonl",
    }  # fmt: skip
    (tmp_path / "i.jsonl").write_text('{"item_id": "a"}\n')
    with pytest.raises(ValueError, match=r"i\.jsonl:1: not a JSON object"):
        load_items(load_arm_config(_write(tmp_path / "a.toml", **base)))
    (tmp_path / "i.jsonl").write_text("[1]\n")
    with pytest.raises(ValueError, match="not a JSON object"):
        load_items(load_arm_config(_write(tmp_path / "a.toml", **base)))
    _items(tmp_path / "i.jsonl", test[:-1])
    split = {**base, "manifest": "splits.json", "split": "test"}
    with pytest.raises(ValueError, match="1 item"):
        load_items(load_arm_config(_write(tmp_path / "a.toml", **split)))
    _items(tmp_path / "i.jsonl", test)
    with pytest.raises(ValueError, match="manifest's test k is 4"):
        load_items(load_arm_config(_write(tmp_path / "a.toml", **split, k=2)))
    items, k = load_items(load_arm_config(_write(tmp_path / "a.toml", **split)))
    assert (len(items), k) == (len(test), 4)
    other = {**base, "manifest": "splits.json", "split": "dev", "k": 3}
    _items(tmp_path / "i.jsonl", list(range(40)))
    items, k = load_items(load_arm_config(_write(tmp_path / "a.toml", **other)))
    assert (len(items), k) == (len(m.dev), 3)  # not the test split: k stays as given


# ------------------------------------------------------------ kill and resume

_DELAY = 0.15


@pytest.mark.skipif(sys.platform == "win32", reason="needs SIGKILL")
def test_sigkill_then_resume_equals_uninterrupted(tmp_path: Path) -> None:
    delay = [0.0]

    def respond(body: dict[str, Any]) -> Reply:
        time.sleep(delay[0])
        return _answer(body)

    with serve(respond) as (url, state):
        whole = _http_cfg(tmp_path, url, out="whole.jsonl")
        assert main(["gen", str(whole)]) == 0
        whole_out = tmp_path / "whole.jsonl"

        delay[0] = _DELAY
        cfg = _http_cfg(tmp_path, url, out="r.jsonl")
        out = tmp_path / "r.jsonl"
        child = subprocess.Popen(
            [sys.executable, "-I", "-m", "astro_weights", "gen", str(cfg)],
            stdout=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 30
            while _lines(out) < 2:
                assert child.poll() is None, "child ended before it could be killed"
                assert time.monotonic() < deadline, "child wrote no records in 30 s"
                time.sleep(0.02)
            os.kill(child.pid, signal.SIGKILL)
        finally:
            child.kill()
            child.wait()
        kept = out.read_bytes().split(b"\n")[
            :-1
        ]  # a kill can leave a partial last line
        assert 2 <= len(kept) < N
        assert header_path(out).exists()
        seen = len(state.requests)

        delay[0] = 0.0
        resumed = subprocess.run(
            [sys.executable, "-I", "-m", "astro_weights", "gen", str(cfg), "--resume", "--json"],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        new = _asked(state.requests[seen:])

    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout)["done"] == N
    assert read_jsonl(out) == read_jsonl(whole_out)
    data = out.read_bytes()
    assert data.startswith(b"".join(line + b"\n" for line in kept))  # kept bytes intact
    assert data == whole_out.read_bytes()
    written = {f"q{json.loads(line)['item_id'][1:]}" for line in kept}
    assert set(new) == {f"q{i}" for i in range(N)} - written  # each other item, once
    assert set(new.values()) == {1}


# ------------------------------------------------------------------ bfcl kind


@pytest.fixture
def bfcl_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dest = tmp_path / "bfcl"
    shutil.copytree(BFCL, dest, ignore=shutil.ignore_patterns("__pycache__"))
    pins = {
        p.relative_to(dest).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in dest.rglob("*")
        if p.is_file() and p.name != "TRIGGERED"
    }
    monkeypatch.setattr(bfcl, "PINS", pins)
    return dest


def test_bfcl_kind_runs_and_resumes(
    tmp_path: Path, bfcl_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    n_tasks = len(bfcl.load_bfcl(bfcl_root))
    with serve(lambda _b: ok("done")) as (url, state):
        cfg = _write(
            tmp_path / "ARM.toml", kind="bfcl", arm="base", base_url=url, model="m",
            max_new_tokens=16, bfcl_root="bfcl", trials=2, seed=5, out="b/base.jsonl",
        )  # fmt: skip
        code, out, _ = gen(cfg, "--json", capsys=capsys)
        assert code == 0
        total = n_tasks * 2
        assert json.loads(out) == {
            "arm": "base",
            "out": str(tmp_path / "b/base.jsonl"),
            "done": total,
            "total": total,
        }
        path = tmp_path / "b/base.jsonl"
        assert _lines(path) == total
        assert read_header(header_path(path))["seed"] == 5
        assert read_header(header_path(path))["k"] == 2  # k = trials
        before = path.read_bytes()
        lines = before.splitlines(keepends=True)
        path.write_bytes(b"".join(lines[:2]))
        n = len(state.requests)
        assert gen(cfg, capsys=capsys)[0] == 2  # no --resume
        code, out, _ = gen(cfg, "--resume", capsys=capsys)
        assert code == 0
        assert len(state.requests) > n
    assert path.read_bytes() == before
    changed = _write(
        tmp_path / "ARM.toml", kind="bfcl", arm="base", base_url=url, model="m",
        max_new_tokens=32, bfcl_root="bfcl", trials=2, seed=5, out="b/base.jsonl",
    )  # fmt: skip
    code, _, _ = gen(changed, "--resume", capsys=capsys)
    assert code == 2
    assert path.read_bytes() == before


def test_bfcl_reply_without_usage_exits_1(
    tmp_path: Path, bfcl_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    n = [0]

    def respond(_b: dict[str, Any]) -> Reply:
        n[0] += 1
        return ok("done") if n[0] <= 3 else ok("done", usage={})

    with serve(respond) as (url, _state):
        cfg = _write(
            tmp_path / "ARM.toml", kind="bfcl", arm="base", base_url=url, model="m",
            max_new_tokens=16, bfcl_root="bfcl", out="b.jsonl",
        )  # fmt: skip
        code, out, err = gen(cfg, "--json", capsys=capsys)
    assert code == 1
    assert "usage" in err
    got = json.loads(out)
    assert got["done"] == _lines(tmp_path / "b.jsonl")
    assert got["total"] == len(bfcl.load_bfcl(bfcl_root))


def test_bfcl_bad_root_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _write(
        tmp_path / "ARM.toml", kind="bfcl", arm="base", base_url="http://127.0.0.1:1",
        model="m", max_new_tokens=16, bfcl_root="nope", out="b.jsonl",
    )  # fmt: skip
    code, _, err = gen(cfg, capsys=capsys)
    assert code == 2
    assert "is not a directory" in err


# ----------------------------------------------------------------- suites kind


@pytest.fixture
def ifeval_pin(monkeypatch: pytest.MonkeyPatch) -> Path:
    from dataclasses import replace

    path = DATA / "suites" / "ifeval.jsonl"
    pins = {
        **suites.PINS,
        "ifeval": replace(
            suites.PINS["ifeval"], sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        ),
    }
    monkeypatch.setattr(suites, "PINS", pins)
    return path


def test_suites_kind_runs_and_resumes(
    tmp_path: Path, ifeval_pin: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def respond(_b: dict[str, Any]) -> Reply:
        return ok("- a\n- b\n- c\nDone")

    with serve(respond) as (url, state):
        cfg = _write(
            tmp_path / "ARM.toml", kind="suites", arm="base", base_url=url, model="m",
            max_new_tokens=16, out_dir="suites", suites=["ifeval"],
            suite_paths={"ifeval": str(ifeval_pin)},
        )  # fmt: skip
        code, out, _ = gen(cfg, capsys=capsys)
        assert code == 0
        path = tmp_path / "suites/ifeval/base.jsonl"
        assert out == f"base: 2/2 items, {tmp_path / 'suites'}\n"
        assert [r.item_id for r in read_jsonl(path)] == ["ifeval-1000", "ifeval-1001"]
        before = path.read_bytes()
        assert gen(cfg, capsys=capsys)[0] == 2  # no --resume
        n = len(state.requests)
        first = before.splitlines(keepends=True)[0]
        path.write_bytes(first)  # a run cut after one record
        code, out, _ = gen(cfg, "--resume", "--json", capsys=capsys)
        assert code == 0
        assert json.loads(out) == {
            "arm": "base",
            "out": str(tmp_path / "suites"),
            "done": 2,
            "total": 2,
        }
        new = _asked(state.requests[n:])
    assert path.read_bytes() == before
    assert sum(new.values()) == 1  # only the second item
    code, _, _ = gen(
        _write(
            tmp_path / "ARM.toml", kind="suites", arm="base", base_url=url, model="m",
            max_new_tokens=16, out_dir="suites", suites=["ifeval"], limit=1,
            suite_paths={"ifeval": str(ifeval_pin)},
        ),
        "--resume",
        capsys=capsys,
    )  # fmt: skip
    assert code == 2
    assert path.read_bytes() == before
