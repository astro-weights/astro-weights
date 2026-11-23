"""self_test_http and `astro-weights self-test`."""

import json
import threading
import time
from typing import Any

import pytest
from fake_endpoint import Reply, serve
from fake_endpoint import ok as _ok

from astro_weights.cli import main
from astro_weights.runners import endpoint
from astro_weights.runners.endpoint import self_test_http

PROMPT = "Count to five."
KEY = "sk-secret-key-123"


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(endpoint, "_RETRY_SLEEP", 0.0)


def _is_target(body: dict[str, Any]) -> bool:
    return bool(body["messages"][0]["content"] == PROMPT)


def _targets(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r["body"] for r in requests if _is_target(r["body"])]


def _steady(body: dict[str, Any]) -> Reply:
    return _ok("1, 2, 3, 4, 5", system_fingerprint="fp1", model="m-served")


def test_deterministic_server() -> None:
    with serve(_steady) as (url, state):
        r = self_test_http(url, "m", PROMPT)
    assert r.deterministic
    assert r.unique_outputs == 1
    assert r.first_divergence is None
    assert (r.server_model, r.system_fingerprint) == ("m-served", "fp1")
    assert list(r.outputs) == [f"batch={b}" for b in (1, 4, 8)] + [
        f"prefix={p}" for p in (0, 256, 1024)
    ]
    # 1+4+8 batch requests, 6 targets' worth counted once each, 2 prefix fillers
    assert r.n_requests == len(state.requests) == (1 + 4 + 8) + 3 + 2
    sent = _targets(state.requests)
    assert len(sent) == 6
    assert all(b == sent[0] for b in sent)  # target bodies are byte-identical
    assert sent[0]["temperature"] == 0
    assert "seed" in sent[0]


def test_extra_body_seed_is_merged() -> None:
    with serve(_steady) as (url, state):
        self_test_http(
            url, "m", PROMPT, batch_sizes=(1,), prefix_lens=(0,), extra_body={"seed": 7}
        )
    assert {b["seed"] for b in _targets(state.requests)} == {7}


def test_output_varies_with_concurrency() -> None:
    live = peak = 0  # peak: most requests in flight since the server was last idle
    lock = threading.Lock()

    def respond(body: dict[str, Any]) -> Reply:
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.2)
        with lock:
            seen, live = peak, live - 1
            peak = peak if live else 0
        return _ok("alone" if seen == 1 else "crowded")

    with serve(respond) as (url, state):
        r = self_test_http(url, "m", PROMPT, batch_sizes=(1, 4), prefix_lens=(0,))
    assert not r.deterministic
    assert r.unique_outputs == 2
    assert r.first_divergence == ("batch=1", "batch=4", 0)
    assert all(b == _targets(state.requests)[0] for b in _targets(state.requests))


def test_output_varies_with_the_preceding_request() -> None:
    last = ""

    def respond(body: dict[str, Any]) -> Reply:
        nonlocal last
        prev, last = last, body["messages"][0]["content"]
        return _ok("1, 2, 3" if "[filler]" not in prev else "1, 2, 4")

    with serve(respond) as (url, _):
        r = self_test_http(url, "m", PROMPT, batch_sizes=(1,), prefix_lens=(0, 32))
    assert r.outputs == {
        "batch=1": "1, 2, 3",
        "prefix=0": "1, 2, 3",
        "prefix=32": "1, 2, 4",
    }
    assert r.first_divergence == ("batch=1", "prefix=32", 6)


def test_prefix_filler_shares_the_target_text() -> None:
    with serve(_steady) as (url, state):
        self_test_http(url, "m", PROMPT, batch_sizes=(1,), prefix_lens=(0, 8))
    texts = [r["body"]["messages"][0]["content"] for r in state.requests]
    filler = texts[-2]
    assert filler.startswith(PROMPT[: 8 * 4])
    assert filler.endswith("[filler]")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_sizes": ()},
        {"batch_sizes": (0,)},
        {"prefix_lens": (-1,)},
        {"prompt": ""},
    ],
)
def test_bad_input(kwargs: dict[str, Any]) -> None:
    args = {"prompt": PROMPT, **kwargs}
    with pytest.raises(ValueError, match="must be"):
        self_test_http("http://127.0.0.1:1", "m", **args)


def test_infra_error_is_scrubbed() -> None:
    with (
        serve(lambda body: (500, f"boom {KEY}")) as (url, _),
        pytest.raises(RuntimeError, match="HTTP 500") as e,
    ):
        self_test_http(url, "m", PROMPT, api_key=KEY, batch_sizes=(1,))
    assert KEY not in str(e.value)


# --- CLI ---------------------------------------------------------------------------


def test_cli_deterministic(capsys: pytest.CaptureFixture[str]) -> None:
    with serve(_steady) as (url, _):
        code = main(
            ["self-test", "--base-url", url, "--model", "m", "--prompt", PROMPT]
        )
    assert code == 0
    assert "deterministic: 1 unique" in capsys.readouterr().out


def test_cli_diverges_json(capsys: pytest.CaptureFixture[str]) -> None:
    n = 0

    def respond(body: dict[str, Any]) -> Reply:
        nonlocal n
        n += 1
        return _ok(f"out{n}")

    with serve(respond) as (url, _):
        code = main(["self-test", "--base-url", url, "--model", "m", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["deterministic"] is False
    assert out["unique_outputs"] > 1
    assert out["first_divergence"][:2] == ["batch=1", out["first_divergence"][1]]


def test_cli_infra_error_exit_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SELFTEST_KEY", KEY)
    with serve(lambda body: (500, f"boom {KEY}")) as (url, _):
        code = main(
            [
                "self-test",
                "--base-url",
                url,
                "--model",
                "m",
                "--api-key-env",
                "SELFTEST_KEY",
            ]
        )
    err = capsys.readouterr().err
    assert code == 2
    assert "HTTP 500" in err
    assert KEY not in err


def test_cli_missing_key_env_exit_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("NOPE_KEY", raising=False)
    code = main(
        [
            "self-test",
            "--base-url",
            "http://127.0.0.1:1",
            "--model",
            "m",
            "--api-key-env",
            "NOPE_KEY",
        ]
    )
    assert code == 2
    assert "$NOPE_KEY is not set" in capsys.readouterr().err


def test_cli_bad_url_exit_2() -> None:
    assert main(["self-test", "--base-url", "ftp://x", "--model", "m"]) == 2
