import json
import urllib.error
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fake_endpoint import Reply, Server, serve
from fake_endpoint import ok as _ok

from astro_weights.eval import (
    CONFIG_KEYS,
    PROVENANCE_KEYS,
    Record,
    check_headers,
    compare,
    config_hash,
    grade_exact,
    pair,
    read_jsonl,
)
from astro_weights.eval.header import run_identity
from astro_weights.runners import endpoint
from astro_weights.runners import endpoint as endpoint_mod
from astro_weights.runners.endpoint import generate_records_http, run_header_http

ITEMS = [
    {"item_id": f"i{k}", "prompt": f"what is {k}+1?", "gold": str(k + 1)}
    for k in range(4)
]
KEY = "sk-secret-key-123"


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(endpoint, "_RETRY_SLEEP", 0.0)


def _answer(body: dict[str, Any]) -> Reply:
    """Base model always says 2; the edit answers right for even items."""
    prompt = body["messages"][0]["content"]
    n = int(prompt.split()[2].split("+")[0])
    return _ok("2" if body["model"] == "base" or n % 2 else str(n + 1))


def test_two_arms_pair_and_compare() -> None:
    with serve(_answer) as (url, state):
        base = generate_records_http(url, "base", ITEMS, "base", grade_exact, 16)
        edit = generate_records_http(
            url + "/v1/", "edit", ITEMS, "edit", grade_exact, 16
        )
    assert [r.item_id for r in base] == ["i0", "i1", "i2", "i3"]
    assert all(isinstance(r, Record) and r.n_output_tokens == 3 for r in base)
    assert not any(r.hit_cap for r in base + edit)
    assert [r.prompt_hash for r in base] == [r.prompt_hash for r in edit]
    assert len(pair(base, edit)) == 4
    assert compare(base, edit, n_boot=50).n_items == 4
    first = state.requests[0]
    assert first["path"] == "/v1/chat/completions"
    assert first["auth"] is None
    assert first["body"] == {
        "model": "base",
        "messages": [{"role": "user", "content": "what is 0+1?"}],
        "temperature": 0,
        "max_tokens": 16,
        "seed": 0,
    }


@pytest.mark.parametrize("suffix", ["", "/", "/v1", "/v1/"])
def test_base_url_normalised(suffix: str) -> None:
    with serve(lambda _b: _ok()) as (url, state):
        generate_records_http(url + suffix, "m", ITEMS[:1], "a", grade_exact, 4)
    assert state.requests[0]["path"] == "/v1/chat/completions"


def test_extra_body_wins_and_seed_sent() -> None:
    with serve(lambda _b: _ok()) as (url, state):
        generate_records_http(
            url,
            "m",
            ITEMS[:1],
            "a",
            grade_exact,
            4,
            seed=7,
            extra_body={"seed": 9, "top_k": 1},
        )
    body = state.requests[0]["body"]
    assert body["seed"] == 9
    assert body["top_k"] == 1
    assert body["temperature"] == 0


def test_hit_cap_and_none_content_and_reasoning() -> None:
    def respond(body: dict[str, Any]) -> Reply:
        prompt = body["messages"][0]["content"]
        if "0+1" in prompt:
            return _ok(None, finish="length")
        obj = {
            "choices": [
                {
                    "message": {"content": "2", "reasoning_content": "hmm"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 5},
        }
        return 200, json.dumps(obj)

    with serve(respond) as (url, _):
        recs = generate_records_http(url, "m", ITEMS[:2], "a", grade_exact, 4)
    assert (recs[0].output, recs[0].hit_cap) == ("", True)
    assert (recs[1].output, recs[1].hit_cap, recs[1].n_output_tokens) == ("2", False, 5)


@pytest.mark.parametrize(
    "reply",
    [
        _ok(usage={}),
        _ok(usage={"completion_tokens": "3"}),
        _ok(usage={"completion_tokens": True}),
        (200, json.dumps({"choices": [{"message": {"content": "x"}}]})),
        (200, json.dumps({"choices": [{"message": {"content": 5}}]})),
        (200, json.dumps({"choices": [{"message": None}]})),
    ],
)
def test_bad_fields_raise_value_error(reply: Reply) -> None:
    with (
        serve(lambda _b: reply) as (url, _),
        pytest.raises(ValueError, match="response"),
    ):
        generate_records_http(url, "m", ITEMS[:1], "a", grade_exact, 4)


def test_missing_usage_names_response() -> None:
    reply = (
        200,
        json.dumps({"choices": [{"message": {"content": "x"}}], "id": "abc-1"}),
    )
    with serve(lambda _b: reply) as (url, _), pytest.raises(ValueError, match="abc-1"):
        generate_records_http(url, "m", ITEMS[:1], "a", grade_exact, 4)


def test_http_500_has_status_and_truncated_body() -> None:
    with serve(lambda _b: (500, "boom" * 300)) as (url, state):
        with pytest.raises(RuntimeError, match=r"item 'i0'.*HTTP 500") as exc:
            generate_records_http(url, "m", ITEMS[:1], "a", grade_exact, 4)
        assert len(state.requests) == 3  # 5xx is retried
    assert "boom" in str(exc.value)
    assert len(str(exc.value)) < 600


@pytest.mark.parametrize(
    "reply",
    [
        (200, "not json {"),
        (200, json.dumps([1, 2])),
        (200, json.dumps({"choices": []})),
        (200, json.dumps({"choices": ["x"]})),
        (200, json.dumps({"id": 1})),
    ],
)
def test_malformed_response_raises(reply: Reply) -> None:
    with (
        serve(lambda _b: reply) as (url, _),
        pytest.raises(RuntimeError, match="HTTP 200"),
    ):
        generate_records_http(url, "m", ITEMS[:1], "a", grade_exact, 4)


def test_timeout_retries_then_fails() -> None:
    import time

    def slow(_b: dict[str, Any]) -> Reply:
        time.sleep(0.5)
        return _ok()

    with serve(slow) as (url, state), pytest.raises(RuntimeError, match="3 attempts"):
        generate_records_http(url, "m", ITEMS[:1], "a", grade_exact, 4, timeout=0.1)
    assert len(state.requests) == 3


def test_timeout_then_success() -> None:
    import time

    calls: list[int] = []

    def flaky(_b: dict[str, Any]) -> Reply:
        calls.append(1)
        if len(calls) == 1:
            time.sleep(0.5)
        return _ok()

    with serve(flaky) as (url, _):
        recs = generate_records_http(
            url, "m", ITEMS[:1], "a", grade_exact, 4, timeout=0.2
        )
    assert len(recs) == 1
    assert len(calls) == 2


def test_connection_refused_retries_and_hides_key() -> None:
    with serve(lambda _b: _ok()) as (url, _):
        pass  # server now closed
    with pytest.raises(RuntimeError, match="3 attempts") as exc:
        generate_records_http(url, "m", ITEMS[:1], "a", grade_exact, 4, api_key=KEY)
    assert KEY not in str(exc.value)


def test_api_key_sent_but_never_leaks(caplog: pytest.LogCaptureFixture) -> None:
    def echo_key(_b: dict[str, Any]) -> Reply:
        return 401, f"bad key {KEY}"

    with serve(echo_key) as (url, state), pytest.raises(RuntimeError) as exc:
        generate_records_http(url, "m", ITEMS[:1], "a", grade_exact, 4, api_key=KEY)
    assert state.requests[0]["auth"] == f"Bearer {KEY}"
    assert KEY not in str(exc.value)
    assert KEY not in repr(exc.value)
    assert KEY not in caplog.text
    with (
        serve(lambda _b: _ok(usage={})) as (url, _),
        pytest.raises(ValueError, match="usage") as exc2,
    ):
        generate_records_http(url, "m", ITEMS[:1], "a", grade_exact, 4, api_key=KEY)
    assert KEY not in str(exc2.value)


@pytest.mark.parametrize(
    "url", ["ftp://host", "file:///etc/passwd", "localhost:8080", ""]
)
def test_scheme_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="http"):
        generate_records_http(url, "m", ITEMS[:1], "a", grade_exact, 4)
    with pytest.raises(ValueError, match="http"):
        run_header_http(url, "m", 4, 0)


def test_items_validated_before_any_request() -> None:
    with serve(lambda _b: _ok()) as (url, state):
        with pytest.raises(ValueError, match="missing key 'gold'"):
            generate_records_http(
                url, "m", [{"item_id": "a", "prompt": "p"}], "a", grade_exact, 4
            )
        dup = [ITEMS[0], ITEMS[0]]
        with pytest.raises(ValueError, match="duplicate item_id"):
            generate_records_http(url, "m", dup, "a", grade_exact, 4)
    assert state.requests == []


def test_out_appends_each_record(tmp_path: Path) -> None:
    out = tmp_path / "arm.jsonl"
    seen_lines: list[int] = []

    def respond(_b: dict[str, Any]) -> Reply:
        seen_lines.append(len(out.read_text().splitlines()) if out.exists() else 0)
        return _ok()

    with serve(respond) as (url, _):
        recs = generate_records_http(url, "m", ITEMS[:3], "a", grade_exact, 4, out=out)
    assert seen_lines == [0, 1, 2]  # flushed before the next request
    assert read_jsonl(out) == recs


def test_out_existing_file(tmp_path: Path) -> None:
    out = tmp_path / "arm.jsonl"
    out.write_text("")  # empty is fine
    with serve(lambda _b: _ok()) as (url, state):
        generate_records_http(url, "m", ITEMS[:1], "a", grade_exact, 4, out=out)
        out.write_text(out.read_text() or "x\n")
        n = len(state.requests)
        with pytest.raises(FileExistsError):
            generate_records_http(url, "m", ITEMS[:1], "a", grade_exact, 4, out=out)
    assert len(state.requests) == n


def test_run_header_http() -> None:
    models = {
        "data": [{"id": "x"}, {"id": "m", "owned_by": "me", "meta": {"n": 1}, "z": 0}]
    }
    with serve(lambda _b: _ok(system_fingerprint="fp-1")) as (url, state):
        state.models = (200, json.dumps(models))
        header = run_header_http(
            url + "/v1", "m", 32, 3, {"top_k": 1}, api_key=KEY, run_id="r1"
        )
    identity = run_identity("r1", 1, 1)
    assert header == {
        "base_url": url,
        "served_model": "m",
        "max_new_tokens": 32,
        "seed": 3,
        "extra_body": {"top_k": 1},
        "server_model": {"id": "m", "owned_by": "me", "meta": {"n": 1}},
        "system_fingerprint": "fp-1",
        "backend": "unknown",
        "model_id": "m",  # no /props: the served name
        **identity,
    }
    assert KEY not in json.dumps(header)
    assert header.keys() <= CONFIG_KEYS | PROVENANCE_KEYS


def test_run_header_http_server_build_and_flags() -> None:
    with serve(lambda _b: _ok()) as (url, state):
        state.paths["/props"] = (
            200,
            json.dumps({"build_info": "b4567-abc", "model_path": "/models/Q9.gguf"}),
        )
        state.paths["/version"] = (200, json.dumps({"version": "0.11.0"}))
        header = run_header_http(
            url,
            "m",
            8,
            0,
            k=4,
            concurrency=2,
            serving_flags={"--parallel": "4", "VLLM_BATCH_INVARIANT": "1"},
        )
        with_id = run_header_http(url, "m", 8, 0, model_id="sha-1")
    assert header["llama_cpp_build"] == "b4567-abc"
    assert header["vllm_version"] == "0.11.0"
    assert header["backend"] == "llama.cpp"
    assert header["model_id"] == "Q9.gguf"
    assert with_id["model_id"] == "sha-1"
    assert header["serving_flags"] == {"--parallel": "4", "VLLM_BATCH_INVARIANT": "1"}
    assert (header["k"], header["concurrency"]) == (4, 2)
    assert [r["path"] for r in state.requests[:2]] == ["/props", "/version"]
    other = {**header, "serving_flags": {"--parallel": "1"}}
    with pytest.raises(ValueError, match="serving_flags"):
        check_headers(header, other)
    with pytest.raises(ValueError, match="llama_cpp_build"):
        check_headers(header, {**header, "llama_cpp_build": "b1"})
    with pytest.raises(ValueError, match="vllm_version"):
        check_headers(header, {**header, "vllm_version": "9"})
    check_headers(header, {**header, "run_id": "z", "base_url": "http://elsewhere"})


def test_run_header_http_vllm_and_windows_path() -> None:
    with serve(lambda _b: _ok()) as (url, state):
        state.paths["/props"] = (200, json.dumps({"model_path": "C:\\m\\x.gguf"}))
        state.paths["/version"] = (200, json.dumps({"version": "0.11.0"}))
        header = run_header_http(url, "m", 8, 0)
    assert header["backend"] == "vllm"
    assert header["model_id"] == "x.gguf"
    assert "llama_cpp_build" not in header


def test_probes_ask_once_with_short_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, float]] = []

    def refuse(req: Any, timeout: float) -> Any:
        calls.append((req.full_url, timeout))
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(endpoint_mod._OPENER, "open", refuse)
    assert endpoint_mod._server_json("http://h", "/props", None, 120.0) == {}
    assert calls == [("http://h/props", 5.0)]
    calls.clear()
    endpoint_mod._server_json("http://h", "/version", None, 2.0)
    assert calls == [("http://h/version", 2.0)]


@pytest.mark.parametrize(
    "reply",
    [(404, "nope"), (200, "not json"), (200, "[1]"), (200, '{"build_info": 3}')],
)
def test_run_header_http_server_build_absent(reply: Reply) -> None:
    with serve(lambda _b: _ok()) as (url, state):
        state.paths["/props"] = reply
        state.paths["/version"] = reply
        header = run_header_http(url, "m", 8, 0)
    assert "llama_cpp_build" not in header
    assert "vllm_version" not in header
    assert "serving_flags" not in header


def test_http_arms_one_server_arm_keys() -> None:
    with serve(lambda _b: _ok()) as (url, _):
        base = run_header_http(url, "m", 8, 0, {"top_k": 1, "lora": 0.0})
        edit = run_header_http(url + "/v1", "m", 8, 0, {"top_k": 1, "lora": 1.0})
    with pytest.raises(ValueError, match="extra_body"):
        check_headers(base, edit)
    check_headers(base, edit, arm_keys=("lora",))
    assert config_hash(base, ("lora",)) == config_hash(edit, ("lora",))


def test_http_arms_two_servers_compare() -> None:
    props = (200, json.dumps({"build_info": "b1", "model_path": "/m/Q9.gguf"}))
    with (
        serve(lambda _b: _ok()) as (url_a, state_a),
        serve(lambda _b: _ok()) as (url_b, state_b),
    ):
        state_a.models = (200, json.dumps({"data": [{"id": "base"}]}))
        state_b.models = (200, json.dumps({"data": [{"id": "edit"}]}))
        state_a.paths["/props"] = state_b.paths["/props"] = props
        base = run_header_http(url_a, "base", 8, 0)
        edit = run_header_http(url_b, "edit", 8, 0)
        # without /props the served names are the only identity: not comparable
        state_b.paths.clear()
        state_a.paths.clear()
        blind_a = run_header_http(url_a, "base", 8, 0)
        blind_b = run_header_http(url_b, "edit", 8, 0)
    assert base["server_model"] != edit["server_model"]
    check_headers(base, edit)  # no --ignore-header needed
    assert config_hash(base) == config_hash(edit)
    with pytest.raises(ValueError, match="model_id"):
        check_headers(blind_a, blind_b)


def test_http_arms_different_models_or_builds_differ() -> None:
    def header(path: str, build: str) -> dict[str, Any]:
        with serve(lambda _b: _ok()) as (url, state):
            state.paths["/props"] = (
                200,
                json.dumps({"build_info": build, "model_path": path}),
            )
            return run_header_http(url, "m", 8, 0)

    ref = header("/m/a.gguf", "b1")
    with pytest.raises(ValueError, match="model_id"):
        check_headers(ref, header("/m/b.gguf", "b1"))
    with pytest.raises(ValueError, match="llama_cpp_build"):
        check_headers(ref, header("/m/a.gguf", "b2"))


@pytest.mark.parametrize(
    "models",
    [
        (404, "nope"),
        (200, "not json"),
        (200, json.dumps([])),
        (200, json.dumps({"data": 1})),
        (200, json.dumps({"data": [{"id": "other"}, "str"]})),
    ],
)
def test_run_header_http_without_server_info(models: Reply) -> None:
    with serve(lambda _b: _ok()) as (url, state):
        state.models = models
        header = run_header_http(url, "m", 8, 0)
    assert set(header) == {
        "base_url",
        "served_model",
        "max_new_tokens",
        "seed",
        "extra_body",
        "backend",
        "model_id",
        *run_identity(None, 1, 1),
    }


def test_run_header_http_server_down() -> None:
    with serve(lambda _b: _ok()) as (url, _):
        pass
    header = run_header_http(url, "m", 8, 0)
    assert "system_fingerprint" not in header
    assert "server_model" not in header
    assert "llama_cpp_build" not in header


def _redirect_pair(path: str = "/steal") -> Iterator[tuple[str, str, Server, Server]]:
    with (
        serve(lambda _b: _ok()) as (other, other_state),
        serve(lambda _b: (302, "moved")) as (url, state),
    ):
        state.location = other + path
        state.models = (302, "moved")
        yield url, other, state, other_state


def test_redirect_refused_and_key_not_forwarded() -> None:
    for url, _, state, other_state in _redirect_pair():
        with pytest.raises(RuntimeError, match="HTTP 302") as exc:
            generate_records_http(url, "m", ITEMS[:1], "a", grade_exact, 4, api_key=KEY)
        assert len(state.requests) == 1
        assert other_state.requests == []
        assert KEY not in str(exc.value)


def test_redirect_refused_for_models_probe() -> None:
    for url, _, _, other_state in _redirect_pair("/v1/models"):
        header = run_header_http(url, "m", 4, 0, api_key=KEY)
        assert "server_model" not in header
        assert other_state.requests == []


@pytest.mark.parametrize(
    "reply",
    [
        (200, json.dumps({"choices": [{"message": {"content": "x"}}], "echo": "K"})),
        (200, json.dumps({"choices": [{"message": None}], "echo": "K"})),
        (200, json.dumps({"choices": [{"message": {"content": 5}}], "echo": "K"})),
        (
            200,
            json.dumps(
                {"choices": [{"message": {"content": "x"}}], "usage": {}, "echo": "K"}
            ),
        ),
    ],
)
def test_echoed_key_scrubbed_from_value_errors(reply: Reply) -> None:
    text = reply[1].replace('"K"', json.dumps(f"Bearer {KEY}"))
    with (
        serve(lambda _b: (200, text)) as (url, _),
        pytest.raises(ValueError, match="response") as exc,
    ):
        generate_records_http(url, "m", ITEMS[:1], "a", grade_exact, 4, api_key=KEY)
    assert KEY not in str(exc.value)
    assert KEY not in repr(exc.value)


def test_key_cut_at_truncation_boundary_not_leaked() -> None:
    pad = "p" * (endpoint._BODY_CHARS - 10)
    obj = {"choices": [{"message": {"content": "x"}}], "echo": pad + KEY}
    with (
        serve(lambda _b: (200, json.dumps(obj))) as (url, _),
        pytest.raises(ValueError, match="usage") as exc,
    ):
        generate_records_http(url, "m", ITEMS[:1], "a", grade_exact, 4, api_key=KEY)
    assert KEY[:4] not in str(exc.value)


@pytest.mark.parametrize(
    "key", ["k\n", " k", "k\x00", "a b", "k\u00e9", 'k"x', "k\\x", "=k"]
)
def test_bad_api_key_rejected_without_echo(key: str) -> None:
    with serve(lambda _b: _ok()) as (url, state):
        with pytest.raises(ValueError, match="api_key") as exc:
            generate_records_http(url, "m", ITEMS[:1], "a", grade_exact, 4, api_key=key)
        with pytest.raises(ValueError, match="api_key"):
            run_header_http(url, "m", 4, 0, api_key=key)
    assert state.requests == []
    assert key not in str(exc.value)
    assert key not in repr(exc.value)


@pytest.mark.parametrize(
    "url",
    ["http://u:p@host", "http://host/?token=abc", "http://host/v1#frag"],
)
def test_url_userinfo_query_fragment_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="base_url") as exc:
        generate_records_http(url, "m", ITEMS[:1], "a", grade_exact, 4)
    assert "abc" not in str(exc.value)
    assert "u:p" not in str(exc.value)
    with pytest.raises(ValueError, match="base_url"):
        run_header_http(url, "m", 4, 0)


def test_http_exception_retried_then_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import http.client

    calls: list[int] = []

    def boom(*_a: Any, **_k: Any) -> Any:
        calls.append(1)
        raise http.client.IncompleteRead(b"")

    monkeypatch.setattr(endpoint._OPENER, "open", boom)
    with pytest.raises(RuntimeError, match="3 attempts"):
        generate_records_http(
            "http://127.0.0.1:1", "m", ITEMS[:1], "a", grade_exact, 4, api_key=KEY
        )
    assert len(calls) == 3
