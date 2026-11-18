import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fake_endpoint import Reply, Server, serve
from fake_endpoint import ok as _ok

from astro_weights.eval import Record, compare, grade_exact, pair, read_jsonl
from astro_weights.runners import endpoint
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
        with pytest.raises(RuntimeError, match="HTTP 500") as exc:
            generate_records_http(url, "m", ITEMS[:1], "a", grade_exact, 4)
        assert len(state.requests) == 1
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
        header = run_header_http(url + "/v1", "m", 32, 3, {"top_k": 1}, api_key=KEY)
    assert header == {
        "base_url": url,
        "model": "m",
        "max_new_tokens": 32,
        "seed": 3,
        "extra_body": {"top_k": 1},
        "server_model": {"id": "m", "owned_by": "me", "meta": {"n": 1}},
        "system_fingerprint": "fp-1",
    }
    assert KEY not in json.dumps(header)


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
    assert set(header) == {"base_url", "model", "max_new_tokens", "seed", "extra_body"}


def test_run_header_http_server_down() -> None:
    with serve(lambda _b: _ok()) as (url, _):
        pass
    header = run_header_http(url, "m", 8, 0)
    assert "system_fingerprint" not in header
    assert "server_model" not in header


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
