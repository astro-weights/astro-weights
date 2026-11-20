"""Run one eval arm against an OpenAI-compatible HTTP API (llama-server, vLLM).

Standard library plus `astro_weights.eval` only: no torch, no extra dependency.
Serve the base and the edit under two model names and call this once per name.
"""

import http.client
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from astro_weights.eval import Record, prompt_hash
from astro_weights.eval.header import run_identity

_KEYS = ("item_id", "prompt", "gold")
_RETRIES = 2
_RETRY_SLEEP = 0.5
_BODY_CHARS = 500


def _root(base_url: str) -> str:
    """`base_url` without a trailing slash or `/v1`; http(s) only."""
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("base_url must start with http:// or https://")
    parts = urllib.parse.urlsplit(base_url)
    if "@" in parts.netloc or parts.query or parts.fragment:
        raise ValueError("base_url must not have userinfo, a query or a fragment")
    root = base_url.rstrip("/")
    return root.removesuffix("/v1").rstrip("/")


def _check_key(api_key: str | None) -> None:
    """Reject keys that cannot be a header value; never show the key."""
    if api_key and not re.fullmatch(r"[A-Za-z0-9\-._~+/]+=*", api_key):
        raise ValueError(
            "api_key must use only A-Z a-z 0-9 - . _ ~ + / (trailing = ok)"
        )


def _scrub(text: str, api_key: str | None) -> str:
    return text.replace(api_key, "***") if api_key else text


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects: urllib would forward the Authorization header."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _request(
    url: str, body: dict[str, Any] | None, api_key: str | None, timeout: float
) -> tuple[int, str]:
    """GET (no body) or POST JSON; return (status, text). Retries connection errors."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    last = ""
    for attempt in range(_RETRIES + 1):
        if attempt:
            time.sleep(_RETRY_SLEEP)
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with _OPENER.open(req, timeout=timeout) as resp:
                return int(resp.status), resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return int(e.code), e.read().decode("utf-8", "replace")
        except (OSError, http.client.HTTPException) as e:
            last = f"{type(e).__name__}: {_scrub(str(e), api_key)}"
    raise RuntimeError(f"cannot reach {url} after {_RETRIES + 1} attempts: {last}")


def _fail(what: str, status: int, text: str, api_key: str | None) -> RuntimeError:
    return RuntimeError(
        f"{what} (HTTP {status}): {_scrub(text, api_key)[:_BODY_CHARS]}"
    )


def _chat(
    url: str, body: dict[str, Any], api_key: str | None, timeout: float
) -> dict[str, Any]:
    """POST one chat completion and return the parsed response object."""
    status, text = _request(url, body, api_key, timeout)
    if status >= 400:
        raise _fail("server error", status, text, api_key)
    try:
        obj = json.loads(text)
    except ValueError:
        raise _fail("response is not JSON", status, text, api_key) from None
    choices = obj.get("choices") if isinstance(obj, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise _fail("response has no choices", status, text, api_key)
    return dict(obj)


def _content(obj: dict[str, Any], status_text: str) -> str:
    message = obj["choices"][0].get("message")
    if not isinstance(message, dict):
        raise ValueError(f"response has no message: {status_text}")  # noqa: TRY004
    content = message.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ValueError(f"response has no message content: {status_text}")  # noqa: TRY004
    return content


def _tokens(obj: dict[str, Any], status_text: str) -> int:
    usage = obj.get("usage")
    n = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError(f"response lacks usage.completion_tokens: {status_text}")  # noqa: TRY004
    return n


def _body(
    model: str,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    seed: int,
    extra_body: Mapping[str, Any] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_new_tokens,
        "seed": seed,
    }
    body.update(extra_body or {})
    return body


def _validate(items: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for item in items:
        for key in _KEYS:
            if key not in item:
                raise ValueError(f"item is missing key {key!r}")
        item_id = str(item["item_id"])
        if item_id in seen:
            raise ValueError(f"duplicate item_id {item_id!r}")
        seen.add(item_id)


def generate_records_http(
    base_url: str,
    model: str,
    items: Sequence[Mapping[str, Any]],
    arm: str,
    scorer: Callable[[str, str], bool],
    max_new_tokens: int,
    *,
    seed: int = 0,
    api_key: str | None = None,
    timeout: float = 120.0,
    extra_body: Mapping[str, Any] | None = None,
    out: str | Path | None = None,
) -> list[Record]:
    """Generate and score each item over `{base_url}/v1/chat/completions`.

    One request per item, temperature 0. `extra_body` keys are merged last into the
    request. The output is `message.content`; `reasoning_content` is ignored, so
    wrap `scorer` yourself if the model thinks inline. `out` appends each record to a
    JSONL file as it is made; the file must not already hold records.
    """
    _validate(items)
    root = _root(base_url)
    _check_key(api_key)
    path = Path(out) if out is not None else None
    if path is not None and path.exists() and path.stat().st_size > 0:
        raise FileExistsError(f"{path} already has records")
    url = f"{root}/v1/chat/completions"
    records: list[Record] = []
    sink = path.open("a", encoding="utf-8") if path is not None else None
    try:
        for item in items:
            messages = [{"role": "user", "content": item["prompt"]}]
            body = _body(model, messages, max_new_tokens, seed, extra_body)
            obj = _chat(url, body, api_key, timeout)
            seen = _scrub(json.dumps(obj), api_key)[:_BODY_CHARS]
            output = _content(obj, seen)
            record = Record(
                item_id=str(item["item_id"]),
                seed=seed,
                arm=arm,
                prompt_hash=prompt_hash(json.dumps(messages, sort_keys=True)),
                output=output,
                score=float(scorer(item["gold"], output)),
                n_output_tokens=_tokens(obj, seen),
                hit_cap=obj["choices"][0].get("finish_reason") == "length",
            )
            records.append(record)
            if sink is not None:
                sink.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                sink.flush()
    finally:
        if sink is not None:
            sink.close()
    return records


def run_header_http(
    base_url: str,
    model: str,
    max_new_tokens: int,
    seed: int,
    extra_body: Mapping[str, Any] | None = None,
    *,
    api_key: str | None = None,
    timeout: float = 120.0,
    run_id: str | None = None,
    k: int = 1,
    concurrency: int = 1,
    serving_flags: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Settings that decide the outputs, plus what the server reports, for the run log.

    CONFIG keys (compared, hashed by `config_hash`): `max_new_tokens`, `seed`,
    `extra_body`, `k`, `concurrency` and `serving_flags`. The client cannot read the
    server's flags or environment, so declare them in `serving_flags` (for example
    `{"--parallel": "4", "VLLM_BATCH_INVARIANT": "1"}`); it is recorded only when
    given. PROVENANCE keys (stored, not compared): `base_url`, the served name
    (`served_model`), the server's `/v1/models` entry, `system_fingerprint`, the
    llama.cpp `build_info` from `/props`, the vLLM version from `/version`, the run id
    (random when not given), the library version and commit, and the numpy version.
    The server's answers are best effort: left out when the server does not give them.
    The api key is never recorded.
    """
    root = _root(base_url)
    _check_key(api_key)
    header: dict[str, Any] = {
        "base_url": root,
        "served_model": model,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "extra_body": dict(extra_body or {}),
        **run_identity(run_id, k, concurrency),
    }
    if serving_flags is not None:
        header["serving_flags"] = dict(sorted(serving_flags.items()))
    build = _server_json(root, "/props", api_key, timeout).get("build_info")
    if isinstance(build, str):
        header["llama_cpp_build"] = build
    ver = _server_json(root, "/version", api_key, timeout).get("version")
    if isinstance(ver, str):
        header["vllm_version"] = ver
    entry = _model_entry(root, model, api_key, timeout)
    if entry is not None:
        header["server_model"] = entry
    try:
        probe = [{"role": "user", "content": "Hi"}]
        body = _body(model, probe, 1, seed, extra_body)
        obj = _chat(f"{root}/v1/chat/completions", body, api_key, timeout)
    except RuntimeError:
        return header
    if obj.get("system_fingerprint") is not None:
        header["system_fingerprint"] = obj["system_fingerprint"]
    return header


def _server_json(
    root: str, path: str, api_key: str | None, timeout: float
) -> dict[str, Any]:
    """A JSON object from GET `path`, or {} when the server does not give one."""
    try:
        status, text = _request(f"{root}{path}", None, api_key, timeout)
        obj = json.loads(text) if status < 400 else {}
    except (RuntimeError, ValueError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _model_entry(
    root: str, model: str, api_key: str | None, timeout: float
) -> dict[str, Any] | None:
    try:
        status, text = _request(f"{root}/v1/models", None, api_key, timeout)
        listed = json.loads(text).get("data", []) if status < 400 else []
    except (RuntimeError, ValueError, AttributeError):
        return None
    for entry in listed if isinstance(listed, list) else []:
        if isinstance(entry, dict) and entry.get("id") == model:
            keep = ("id", "owned_by", "meta")
            return {k: entry[k] for k in keep if k in entry}
    return None
