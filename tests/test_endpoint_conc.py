"""generate_records_http: concurrency, resume and infra-error retries."""

import json
import random
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fake_endpoint import Reply, serve
from fake_endpoint import ok as _ok

from astro_weights.eval import check_headers, config_hash, grade_exact, read_jsonl
from astro_weights.eval.header import header_path, read_header, write_header
from astro_weights.runners import endpoint
from astro_weights.runners.endpoint import generate_records_http, run_header_http

ITEMS = [
    {"item_id": f"i{k}", "prompt": f"what is {k}+1?", "gold": str(k + 1)}
    for k in range(12)
]
KEY = "sk-secret-key-123"


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(endpoint, "_RETRY_SLEEP", 0.0)


def _n(body: dict[str, Any]) -> int:
    """The item number in the prompt; -1 for the header probe's "Hi"."""
    words = body["messages"][0]["content"].split()
    return int(words[2].split("+")[0]) if len(words) == 3 else -1


def _answer(body: dict[str, Any]) -> Reply:
    """Right for even items; the answer depends only on the prompt."""
    n = _n(body)
    return _ok(str(n + 1) if n % 2 == 0 else "x", finish="length" if n == 3 else "stop")


class _Jitter:
    """Answers after a random delay and records the peak number in flight."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.now = 0
        self.peak = 0

    def __call__(self, body: dict[str, Any]) -> Reply:
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
            delay = self.rng.uniform(0.0, 0.03)
        time.sleep(delay)
        with self.lock:
            self.now -= 1
        return _answer(body)


def _run(url: str, out: Path, **kw: Any) -> list[Any]:
    return generate_records_http(url, "m", ITEMS, "a", grade_exact, 8, out=out, **kw)


def test_concurrency_writes_the_same_file_in_item_order(tmp_path: Path) -> None:
    jitter = _Jitter(0)
    with serve(jitter) as (url, _):
        one = _run(url, tmp_path / "c1.jsonl")
        four = _run(url, tmp_path / "c4.jsonl", concurrency=4)
    assert jitter.peak > 1  # requests did overlap
    assert (tmp_path / "c1.jsonl").read_bytes() == (tmp_path / "c4.jsonl").read_bytes()
    assert one == four
    assert [r.item_id for r in four] == [i["item_id"] for i in ITEMS]
    assert read_jsonl(tmp_path / "c4.jsonl") == four


class _Stop(Exception):
    pass


@pytest.mark.parametrize("concurrency", [1, 4])
def test_resume_after_a_stop_gives_the_uninterrupted_file(
    tmp_path: Path, concurrency: int
) -> None:
    whole = tmp_path / "whole.jsonl"
    part = tmp_path / "part.jsonl"

    def stop_at_i7(gold: str, out: str) -> bool:
        if gold == "8":
            raise _Stop
        return grade_exact(gold, out)

    with serve(_Jitter(1)) as (url, state):
        _run(url, whole, concurrency=concurrency)
        with pytest.raises(_Stop):
            generate_records_http(
                url, "m", ITEMS, "a", stop_at_i7, 8, out=part, concurrency=concurrency
            )
        assert [r.item_id for r in read_jsonl(part)] == [f"i{k}" for k in range(7)]
        # a crash cut the last line short
        data = part.read_bytes()
        part.write_bytes(data[: len(data) - 9])
        state.requests.clear()
        recs = _run(url, part, concurrency=concurrency, resume=True)
    asked = sorted(_n(r["body"]) for r in state.requests)
    assert asked == list(range(6, 12))  # no item asked twice, none skipped
    assert part.read_bytes() == whole.read_bytes()
    assert recs == read_jsonl(whole)


def test_resume_complete_file_and_unterminated_last_line(tmp_path: Path) -> None:
    out = tmp_path / "a.jsonl"
    with serve(_answer) as (url, state):
        recs = _run(url, out)
        full = out.read_bytes()
        out.write_bytes(full[:-1])  # valid last line, newline lost
        n = len(state.requests)
        assert _run(url, out, resume=True) == recs
        assert len(state.requests) == n
    assert out.read_bytes() == full


def test_resume_with_empty_or_missing_file_runs_all(tmp_path: Path) -> None:
    with serve(_answer) as (url, _):
        assert len(_run(url, tmp_path / "new.jsonl", resume=True)) == len(ITEMS)


def test_resume_needs_out() -> None:
    with pytest.raises(ValueError, match="needs out"):
        generate_records_http("http://h", "m", ITEMS, "a", grade_exact, 8, resume=True)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"arm": "b"}, "another run's"),
        ({"seed": 1}, "another run's"),
        ({"item_id": "zz"}, "not in items"),
        ({"prompt_hash": "h"}, "prompt of item"),
        ({}, "duplicate"),
    ],
)
def test_resume_refuses_another_runs_records(
    tmp_path: Path, change: dict[str, Any], match: str
) -> None:
    out = tmp_path / "a.jsonl"
    with serve(_answer) as (url, _):
        generate_records_http(url, "m", ITEMS[:1], "a", grade_exact, 8, out=out)
    line = json.loads(out.read_text())
    out.write_text(out.read_text() + json.dumps({**line, **change}) + "\n")
    before = out.read_bytes()
    with serve(_answer) as (url, state), pytest.raises(ValueError, match=match):
        _run(url, out, resume=True)
    assert state.requests == []
    assert out.read_bytes() == before


def _header(url: str, **kw: Any) -> dict[str, Any]:
    return run_header_http(url, "m", 8, 0, **kw)


def test_header_written_and_resume_refuses_another_config(tmp_path: Path) -> None:
    out = tmp_path / "a.jsonl"
    with serve(_answer) as (url, state):
        header = _header(url, serving_flags={"--parallel": "1"})
        generate_records_http(
            url, "m", ITEMS[:2], "a", grade_exact, 8, out=out, header=header
        )
        written = read_header(header_path(out))
        assert written["infra_errors"] == {"retries": 0, "failed": 0}
        assert config_hash(written) == config_hash(header)
        other = _header(url, serving_flags={"--parallel": "4"})
        state.requests.clear()
        with pytest.raises(ValueError, match="config_hash differs"):
            _run(url, out, resume=True, header=other)
        # the call's own settings are checked against the file without a header
        with pytest.raises(ValueError, match="max_new_tokens"):
            generate_records_http(
                url, "m", ITEMS, "a", grade_exact, 9, out=out, resume=True
            )
        with pytest.raises(ValueError, match="concurrency"):
            _run(url, out, resume=True, concurrency=4)
        assert state.requests == []
        _run(
            url,
            out,
            resume=True,
            header=_header(url, serving_flags={"--parallel": "1"}),
        )
    assert read_header(header_path(out))["run_id"] == header["run_id"]  # kept
    assert len(read_jsonl(out)) == len(ITEMS)


def test_header_must_match_the_call(tmp_path: Path) -> None:
    with serve(_answer) as (url, state):
        header = _header(url, concurrency=1)
        state.requests.clear()
        with pytest.raises(ValueError, match="concurrency: header has 1, this call 4"):
            _run(url, tmp_path / "a.jsonl", header=header, concurrency=4)
        with pytest.raises(ValueError, match="extra_body"):
            _run(url, tmp_path / "a.jsonl", header=header, extra_body={"x": 1})
        assert state.requests == []
        # a header without out is only checked
        recs = generate_records_http(
            url, "m", ITEMS[:1], "a", grade_exact, 8, header=_header(url)
        )
    assert len(recs) == 1
    assert not list(tmp_path.iterdir())


def test_concurrency_1_vs_4_fails_the_header_check() -> None:
    with serve(_answer) as (url, _):
        base = _header(url, concurrency=1)
        edit = _header(url, concurrency=4)
    with pytest.raises(ValueError, match="concurrency: base=1 edit=4"):
        check_headers(base, edit)


@pytest.mark.parametrize("bad", [0, -1, True, 2.0])
def test_bad_concurrency(bad: Any) -> None:
    with pytest.raises(ValueError, match="concurrency"):
        generate_records_http(
            "http://h", "m", ITEMS, "a", grade_exact, 8, concurrency=bad
        )


@pytest.mark.parametrize("status", [500, 502, 503, 429])
def test_infra_error_then_200_retried_and_counted(tmp_path: Path, status: int) -> None:
    out = tmp_path / "a.jsonl"
    failed_once: set[int] = set()

    def flaky(body: dict[str, Any]) -> Reply:
        n = _n(body)
        if n in (2, 5) and n not in failed_once:
            failed_once.add(n)
            return status, "busy"
        return _answer(body)

    with serve(flaky) as (url, state):
        recs = generate_records_http(
            url, "m", ITEMS[:6], "a", grade_exact, 8, out=out,
            header=_header(url, concurrency=3), concurrency=3,
        )  # fmt: skip
    assert len(recs) == 6
    assert len([r for r in state.requests if "body" in r]) == 6 + 2 + 1  # +1 probe
    assert read_header(header_path(out))["infra_errors"] == {"retries": 2, "failed": 0}


def test_client_error_not_retried() -> None:
    with (
        serve(lambda _b: (400, "bad request")) as (url, state),
        pytest.raises(RuntimeError, match=r"item 'i0': server error \(HTTP 400"),
    ):
        generate_records_http(url, "m", ITEMS[:1], "a", grade_exact, 8)
    assert len(state.requests) == 1


def test_persistent_failure_raises_and_writes_no_record(tmp_path: Path) -> None:
    out = tmp_path / "a.jsonl"

    def down_at_4(body: dict[str, Any]) -> Reply:
        return (503, "overloaded") if _n(body) == 4 else _answer(body)

    with serve(down_at_4) as (url, state):
        header = _header(url, concurrency=2)
        with pytest.raises(RuntimeError, match=r"item 'i4'.*3 attempts.*HTTP 503"):
            _run(url, out, header=header, concurrency=2)
        asked = [_n(r["body"]) for r in state.requests if "body" in r]
    assert asked.count(4) == 3
    ids = [r.item_id for r in read_jsonl(out)]
    assert ids == ["i0", "i1", "i2", "i3"]  # nothing for i4, nothing past it
    assert read_header(header_path(out))["infra_errors"] == {"retries": 2, "failed": 1}
    with serve(_answer) as (url, _):
        _run(url, out, header=header, concurrency=2, resume=True)
    assert [r.item_id for r in read_jsonl(out)] == [i["item_id"] for i in ITEMS]
    # counts add up across a resume
    assert read_header(header_path(out))["infra_errors"] == {"retries": 2, "failed": 1}


def test_connection_error_then_success_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    real = endpoint._send
    calls: list[int] = []

    def drop_first(*a: Any, **k: Any) -> tuple[int, str]:
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionResetError("reset")
        return real(*a, **k)

    monkeypatch.setattr(endpoint, "_send", drop_first)
    counts = endpoint._Counts()
    with serve(_answer) as (url, _):
        obj = endpoint._chat(
            f"{url}/v1/chat/completions", {"model": "m", "messages": [
                {"role": "user", "content": "what is 0+1?"}]}, None, 5.0, counts,
        )  # fmt: skip
    assert obj["choices"][0]["message"]["content"] == "1"
    assert (counts.retries, counts.failed) == (1, 0)


def test_start_counts_from_old_header() -> None:
    c = endpoint._start_counts({"infra_errors": {"retries": 3}})
    assert (c.retries, c.failed) == (3, 0)
    c = endpoint._start_counts({"infra_errors": "garbled"})
    assert (c.retries, c.failed) == (0, 0)


def test_redirect_and_key_scrubbing_hold_under_concurrency(tmp_path: Path) -> None:
    with (
        serve(lambda _b: _ok()) as (other, other_state),
        serve(lambda _b: (302, f"moved, saw Bearer {KEY}")) as (url, state),
    ):
        state.location = other + "/steal"
        with pytest.raises(RuntimeError, match="HTTP 302") as exc:
            generate_records_http(
                url, "m", ITEMS, "a", grade_exact, 8, api_key=KEY, concurrency=4,
                out=tmp_path / "a.jsonl",
            )  # fmt: skip
        assert other_state.requests == []
        assert {r["auth"] for r in state.requests} == {f"Bearer {KEY}"}
    assert KEY not in str(exc.value)
    assert KEY not in repr(exc.value)
    assert (tmp_path / "a.jsonl").read_text() == ""


def test_resume_refuses_mismatching_header_file_keys(tmp_path: Path) -> None:
    out = tmp_path / "a.jsonl"
    with serve(_answer) as (url, _):
        generate_records_http(url, "m", ITEMS[:1], "a", grade_exact, 8, out=out)
        write_header(header_path(out), {"seed": 0, "max_new_tokens": 8})
        with pytest.raises(ValueError, match=r"extra_body: .* has '<missing>'"):
            _run(url, out, resume=True)


def test_failure_waits_for_earlier_items_in_flight(tmp_path: Path) -> None:
    # i3 answers only after i4 has failed all 3 attempts, so i3 is in flight
    # when the error arrives: it must still be written, and nothing after it
    out = tmp_path / "a.jsonl"
    i4_tries = 0
    lock = threading.Lock()
    i4_done = threading.Event()

    def slow_3_down_4(body: dict[str, Any]) -> Reply:
        nonlocal i4_tries
        if _n(body) == 3:
            assert i4_done.wait(30)
        if _n(body) == 4:
            with lock:
                i4_tries += 1
                if i4_tries == 3:
                    i4_done.set()
            return (503, "overloaded")
        return _answer(body)

    with serve(slow_3_down_4) as (url, _):
        header = _header(url, concurrency=2)
        with pytest.raises(RuntimeError, match=r"item 'i4'.*3 attempts.*HTTP 503"):
            _run(url, out, header=header, concurrency=2)
    assert [r.item_id for r in read_jsonl(out)] == ["i0", "i1", "i2", "i3"]
