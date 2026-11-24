"""`system` on the HTTP runner: a system message, same prompt hash, recorded in the header."""

from pathlib import Path
from typing import Any

import pytest
from fake_endpoint import Reply, serve
from fake_endpoint import ok as _ok

from astro_weights.eval import (
    check_headers,
    config_hash,
    grade_exact,
    header_path,
    pair,
    read_header,
)
from astro_weights.runners import endpoint
from astro_weights.runners.endpoint import generate_records_http, run_header_http

ITEMS = [
    {"item_id": f"i{k}", "prompt": f"what is {k}+1?", "gold": "2"} for k in range(3)
]
SYS = "Answer with the number only."


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(endpoint, "_RETRY_SLEEP", 0.0)


def _answer(body: dict[str, Any]) -> Reply:
    return _ok("2")


def test_system_message_is_first_and_hash_is_unchanged() -> None:
    with serve(_answer) as (url, state):
        plain = generate_records_http(url, "m", ITEMS, "base", grade_exact, 4)
        n = len(state.requests)
        with_sys = generate_records_http(
            url, "m", ITEMS, "prompt", grade_exact, 4, system=SYS
        )
    sent = [r["body"]["messages"] for r in state.requests[:n]]
    assert all(len(m) == 1 and m[0]["role"] == "user" for m in sent)
    sent = [r["body"]["messages"] for r in state.requests[n:]]
    assert all(
        m[0] == {"role": "system", "content": SYS} and m[1]["role"] == "user"
        for m in sent
    )
    assert [m[1]["content"] for m in sent] == [i["prompt"] for i in ITEMS]
    assert [r.prompt_hash for r in plain] == [r.prompt_hash for r in with_sys]
    assert len(pair(plain, with_sys)) == len(ITEMS)  # pairs on item


def test_header_has_system_only_when_set() -> None:
    with serve(_answer) as (url, _):
        bare = run_header_http(url, "m", 4, 0)
        sysd = run_header_http(url, "m", 4, 0, system=SYS)
    assert "system" not in bare
    assert "system_sha256" not in bare
    assert sysd["system"] == SYS
    assert len(sysd["system_sha256"]) == 64
    assert config_hash(bare) != config_hash(sysd)
    with pytest.raises(ValueError, match="system"):
        check_headers(bare, sysd)


def test_empty_system_is_refused() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        generate_records_http("http://x", "m", ITEMS, "a", grade_exact, 4, system="")
    with pytest.raises(ValueError, match="non-empty"):
        run_header_http("http://x", "m", 4, 0, system="")


def test_resume_under_another_system_raises(tmp_path: Path) -> None:
    out = tmp_path / "prompt.jsonl"
    with serve(_answer) as (url, _):
        header = run_header_http(url, "m", 4, 0, system=SYS)
        generate_records_http(
            url, "m", ITEMS[:1], "prompt", grade_exact, 4, out=out,
            header=header, system=SYS,
        )  # fmt: skip
        other = run_header_http(url, "m", 4, 0, system="Be brief.")
        with pytest.raises(ValueError, match="does not match"):
            generate_records_http(
                url, "m", ITEMS, "prompt", grade_exact, 4, out=out, resume=True,
                header=other, system="Be brief.",
            )  # fmt: skip
        with pytest.raises(ValueError, match="does not match"):
            generate_records_http(
                url, "m", ITEMS, "prompt", grade_exact, 4, out=out, resume=True
            )  # no system at all
        with pytest.raises(ValueError, match="does not match"):
            generate_records_http(
                url,
                "m",
                ITEMS,
                "prompt",
                grade_exact,
                4,
                out=out,
                resume=True,
                header=header,
                system="Be brief.",
            )  # header and call disagree
        done = generate_records_http(
            url, "m", ITEMS, "prompt", grade_exact, 4, out=out, resume=True,
            header=header, system=SYS,
        )  # fmt: skip
    assert len(done) == len(ITEMS)
    assert read_header(header_path(out))["system"] == SYS
