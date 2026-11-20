"""Run one eval arm against an OpenAI-compatible HTTP API (llama-server, vLLM).

Standard library plus `astro_weights.eval` only: no torch, no extra dependency.
Serve the base and the edit under two model names and call this once per name.
"""

import http.client
import json
import re
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import Any

from astro_weights.eval import Record, prompt_hash, read_jsonl
from astro_weights.eval.header import (
    config_hash,
    header_path,
    read_header,
    run_identity,
    write_header,
)

_KEYS = ("item_id", "prompt", "gold")
_RETRIES = 2
_RETRY_SLEEP = 0.5
_BODY_CHARS = 500
_PROBE_TIMEOUT = 5.0


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


def _send(
    url: str, body: dict[str, Any] | None, api_key: str | None, timeout: float
) -> tuple[int, str]:
    """One GET (no body) or POST JSON; (status, text). Raises on connection errors."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            return int(resp.status), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return int(e.code), e.read().decode("utf-8", "replace")


def _request(
    url: str,
    body: dict[str, Any] | None,
    api_key: str | None,
    timeout: float,
    retries: int = _RETRIES,
) -> tuple[int, str]:
    """GET (no body) or POST JSON; return (status, text). Retries connection errors."""
    last = ""
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(_RETRY_SLEEP)
        try:
            return _send(url, body, api_key, timeout)
        except (OSError, http.client.HTTPException) as e:
            last = f"{type(e).__name__}: {_scrub(str(e), api_key)}"
    raise RuntimeError(f"cannot reach {url} after {retries + 1} attempts: {last}")


class _Counts:
    """Infra error counts of one run, shared by its worker threads."""

    def __init__(self, retries: int = 0, failed: int = 0) -> None:
        self.retries = retries
        self.failed = failed
        self._lock = threading.Lock()

    def add(self, retries: int = 0, failed: int = 0) -> None:
        with self._lock:
            self.retries += retries
            self.failed += failed


def _infra(status: int) -> bool:
    """A status worth retrying: the server is overloaded or failing, not refusing."""
    return status == 429 or status >= 500


def _ask(
    url: str,
    body: dict[str, Any],
    api_key: str | None,
    timeout: float,
    counts: _Counts | None,
) -> tuple[int, str]:
    """POST `body`; retry connection errors, timeouts, 429 and 5xx with backoff.

    Each retry adds 1 to `counts.retries`; running out of attempts adds 1 to
    `counts.failed` and raises RuntimeError (connection) or returns the last 429/5xx.
    """
    counts = counts if counts is not None else _Counts()
    reply: tuple[int, str] | None = None
    last = ""
    for attempt in range(_RETRIES + 1):
        if attempt:
            counts.add(retries=1)
            time.sleep(_RETRY_SLEEP * 2 ** (attempt - 1))
        try:
            reply = _send(url, body, api_key, timeout)
        except (OSError, http.client.HTTPException) as e:
            reply, last = None, f"{type(e).__name__}: {_scrub(str(e), api_key)}"
            continue
        if not _infra(reply[0]):
            return reply
    counts.add(failed=1)
    if reply is None:
        raise RuntimeError(f"cannot reach {url} after {_RETRIES + 1} attempts: {last}")
    return reply


def _fail(what: str, status: int, text: str, api_key: str | None) -> RuntimeError:
    return RuntimeError(
        f"{what} (HTTP {status}): {_scrub(text, api_key)[:_BODY_CHARS]}"
    )


def _chat(
    url: str,
    body: dict[str, Any],
    api_key: str | None,
    timeout: float,
    counts: _Counts | None = None,
) -> dict[str, Any]:
    """POST one chat completion and return the parsed response object."""
    status, text = _ask(url, body, api_key, timeout, counts)
    if status >= 400:
        tries = f" after {_RETRIES + 1} attempts" if _infra(status) else ""
        raise _fail(f"server error{tries}", status, text, api_key)
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


def _read_file(path: Path) -> tuple[list[Record], int | None, bool]:
    """Records in `path` without changing it, and how to repair its last line.

    Returns (records, cut, whole): `cut` is the size to truncate to when a crash
    left a last line without its newline that is not valid JSON; `whole` is True
    when that line is valid and only lacks its newline.
    """
    data = path.read_bytes()
    cut: int | None = None
    whole = False
    if data and not data.endswith(b"\n"):
        start = data.rfind(b"\n") + 1
        try:
            json.loads(data[start:])
        except ValueError:  # truncated JSON or a cut UTF-8 character
            cut, data = start, data[:start]
        else:
            whole = True
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "records.jsonl"
        copy.write_bytes(data)
        return read_jsonl(copy), cut, whole


def _load_done(
    path: Path, hashes: Mapping[str, str], arm: str, seed: int
) -> dict[str, Record]:
    """Records of an earlier run of this arm and seed in `path`, keyed by item_id.

    The file is repaired (a crash's partial last line) only after it checked out.
    """
    records, cut, whole = _read_file(path)
    done: dict[str, Record] = {}
    for r in records:
        if (r.arm, r.seed) != (arm, seed):
            raise ValueError(
                f"{path}: item {r.item_id!r} is arm {r.arm!r} seed {r.seed}, "
                f"this run is arm {arm!r} seed {seed}; the file is another run's"
            )
        if r.item_id not in hashes:
            raise ValueError(f"{path}: item_id {r.item_id!r} is not in items")
        if r.prompt_hash != hashes[r.item_id]:
            raise ValueError(
                f"{path}: prompt of item {r.item_id!r} differs from the one "
                "that would be sent now; the file is another run's"
            )
        if r.item_id in done:
            raise ValueError(f"{path}: duplicate item_id {r.item_id!r}")
        done[r.item_id] = r
    if cut is not None:
        with path.open("r+b") as f:
            f.truncate(cut)
    elif whole:
        with path.open("ab") as f:
            f.write(b"\n")
    return done


def _check_call(header: Mapping[str, Any], call: Mapping[str, Any], what: str) -> None:
    """Raise ValueError when `header` disagrees with this call's own settings."""
    got = json.loads(json.dumps({"concurrency": 1, **header}))
    diffs = [
        f"{k}: {what} has {got.get(k, '<missing>')!r}, this call {v!r}"
        for k, v in call.items()
        if got.get(k) != v
    ]
    if diffs:
        raise ValueError(f"{what} does not match this run: " + "; ".join(diffs))


def _header_to_keep(
    path: Path, header: Mapping[str, Any] | None, call: Mapping[str, Any]
) -> dict[str, Any] | None:
    """The header for a resumed `path`: the earlier run's if there is one, else `header`.

    An existing header file must match this call and, when `header` is given, have
    its `config_hash`.
    """
    where = header_path(path)
    if not where.exists():
        return None if header is None else dict(header)
    old = read_header(where)
    _check_call(old, call, str(where))
    if header is not None and config_hash(old) != config_hash(header):
        raise ValueError(
            f"{where}: the records were made under another config "
            "(config_hash differs); start a new file for this one"
        )
    return old


def _start_counts(header: Mapping[str, Any] | None) -> _Counts:
    prev = header.get("infra_errors") if header is not None else None
    if not isinstance(prev, dict):
        return _Counts()
    return _Counts(int(prev.get("retries", 0)), int(prev.get("failed", 0)))


def _in_order(
    todo: Sequence[Mapping[str, Any]],
    fetch: Callable[[Mapping[str, Any]], Any],
    concurrency: int,
    emit: Callable[[Mapping[str, Any], Any], None],
) -> None:
    """`fetch` each item, at most `concurrency` at once; `emit` results in item order.

    A finished result waits until every earlier item is emitted. The first error
    stops new requests, waits for those in flight, and is raised; nothing after
    the last emitted item is emitted.
    """
    pending = iter(enumerate(todo))
    running: dict[Future[Any], int] = {}
    ready: dict[int, Any] = {}
    nxt = 0
    with ThreadPoolExecutor(concurrency) as pool:
        while True:
            for i, item in pending:
                running[pool.submit(fetch, item)] = i
                if len(running) == concurrency:
                    break
            if not running:
                return
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in finished:
                ready[running.pop(f)] = f.result()
            while nxt in ready:
                emit(todo[nxt], ready.pop(nxt))
                nxt += 1


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
    concurrency: int = 1,
    resume: bool = False,
    header: Mapping[str, Any] | None = None,
) -> list[Record]:
    """Generate and score each item over `{base_url}/v1/chat/completions`.

    One request per item, temperature 0. `extra_body` keys are merged last into the
    request. The output is `message.content`; `reasoning_content` is ignored, so
    wrap `scorer` yourself if the model thinks inline. `out` appends each record to a
    JSONL file as it is made. Returns every record in the order of `items`.

    `concurrency` requests are in flight at once (default 1). Records are still
    scored in this thread and written in item order, so the file is the one a run
    with `concurrency=1` writes, if the server answers alike (llama.cpp with
    several slots does not). An error stops the run after the requests in flight;
    the lines written stay whole.

    A non-empty `out` raises `FileExistsError` unless `resume=True`. Resuming keeps
    the records in the file and asks only for the other items; a last line cut short
    by a crash is dropped and redone. A record of another arm or seed, of an item not
    in `items`, or whose prompt hash differs raises `ValueError`.

    `header` is this run's `run_header_http` result. It must agree with this call
    (`max_new_tokens`, `seed`, `extra_body`, `concurrency`), else `ValueError`.
    With `out`, it is written to `header_path(out)`. Resuming, an existing header
    file there must agree with this call too and, when `header` is given, have the
    same `config_hash`; it is kept as it is.

    Connection errors, timeouts, HTTP 429 and 5xx are retried twice with backoff.
    An item that still fails raises `RuntimeError` naming the item; no record is
    written for it, so a resume asks again. The header file's `infra_errors`
    (provenance) counts the retries and the items that ran out of attempts, added
    up across resumes.
    """
    _validate(items)
    root = _root(base_url)
    _check_key(api_key)
    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or concurrency < 1
    ):
        raise ValueError(f"concurrency must be an int >= 1, got {concurrency!r}")
    if resume and out is None:
        raise ValueError("resume=True needs out")
    call = {
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "extra_body": json.loads(json.dumps(dict(extra_body or {}))),
        "concurrency": concurrency,
    }
    if header is not None:
        _check_call(header, call, "header")
    messages = {
        str(item["item_id"]): [{"role": "user", "content": item["prompt"]}]
        for item in items
    }
    hashes = {
        k: prompt_hash(json.dumps(m, sort_keys=True)) for k, m in messages.items()
    }
    path = Path(out) if out is not None else None
    done: dict[str, Record] = {}
    kept = None if header is None else dict(header)
    if path is not None and path.exists() and path.stat().st_size > 0:
        if not resume:
            raise FileExistsError(
                f"{path} already has records; pass resume=True to continue it "
                "or choose another path"
            )
        kept = _header_to_keep(path, header, call)
        done = _load_done(path, hashes, arm, seed)
    counts = _start_counts(kept)
    url = f"{root}/v1/chat/completions"

    def fetch(item: Mapping[str, Any]) -> tuple[str, int, bool]:
        item_id = str(item["item_id"])
        body = _body(model, messages[item_id], max_new_tokens, seed, extra_body)
        try:
            obj = _chat(url, body, api_key, timeout, counts)
            seen = _scrub(json.dumps(obj), api_key)[:_BODY_CHARS]
            output = _content(obj, seen)
            n = _tokens(obj, seen)
        except RuntimeError as e:
            raise RuntimeError(f"item {item_id!r}: {e}") from e
        except ValueError as e:  # subclasses (e.g. UnicodeDecodeError) take other args
            raise ValueError(f"item {item_id!r}: {e}") from e
        return output, n, obj["choices"][0].get("finish_reason") == "length"

    def save_header() -> None:
        if path is not None and kept is not None:
            kept["infra_errors"] = {"retries": counts.retries, "failed": counts.failed}
            write_header(header_path(path), kept)

    save_header()
    todo = [item for item in items if str(item["item_id"]) not in done]
    sink = path.open("a", encoding="utf-8") if path is not None else None

    def emit(item: Mapping[str, Any], got: tuple[str, int, bool]) -> None:
        item_id = str(item["item_id"])
        output, n, hit_cap = got
        record = Record(
            item_id=item_id,
            seed=seed,
            arm=arm,
            prompt_hash=hashes[item_id],
            output=output,
            score=float(scorer(item["gold"], output)),
            n_output_tokens=n,
            hit_cap=hit_cap,
        )
        done[item_id] = record
        if sink is not None:
            sink.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            sink.flush()

    try:
        _in_order(todo, fetch, concurrency, emit)
    finally:
        if sink is not None:
            sink.close()
        save_header()
    return [done[str(item["item_id"])] for item in items]


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
    model_id: str | None = None,
) -> dict[str, Any]:
    """Settings that decide the outputs, plus what the server reports, for the run log.

    CONFIG keys (compared, hashed by `config_hash`): `max_new_tokens`, `seed`,
    `extra_body`, `k`, `concurrency`, `serving_flags`, `backend` ("llama.cpp" when
    `/props` gives `build_info`, "vllm" when `/version` answers, else "unknown"),
    `llama_cpp_build`, `vllm_version` and `model_id`. `model_id` is the `model_id`
    argument (for example the base GGUF's sha256), else the basename of `/props`
    `model_path`, else the served name; it must be equal for two arms on one server. The client cannot read the
    server's flags or environment, so declare them in `serving_flags` (for example
    `{"--parallel": "4", "VLLM_BATCH_INVARIANT": "1"}`); it is recorded only when
    given. PROVENANCE keys (stored, not compared): `base_url`, the served name
    (`served_model`), the server's `/v1/models` entry, `system_fingerprint`, the run id
    (random when not given), the library version and commit, and the numpy version.
    The server's answers are best effort: left out when the server does not give them.
    `/props` and `/version` are asked once, with a timeout of at most 5 seconds.
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
    props = _server_json(root, "/props", api_key, timeout)
    ver = _server_json(root, "/version", api_key, timeout).get("version")
    build = props.get("build_info")
    path = props.get("model_path")
    if isinstance(build, str):
        header["llama_cpp_build"] = build
    if isinstance(ver, str):
        header["vllm_version"] = ver
    header["backend"] = (
        "llama.cpp"
        if isinstance(build, str)
        else "vllm"
        if "vllm_version" in header
        else "unknown"
    )
    header["model_id"] = (
        model_id
        if model_id is not None
        else path.replace("\\", "/").rsplit("/", 1)[-1]
        if isinstance(path, str) and path
        else model
    )
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
    """A JSON object from one GET `path` (short timeout, no retry), else {}."""
    try:
        status, text = _request(
            f"{root}{path}", None, api_key, min(timeout, _PROBE_TIMEOUT), retries=0
        )
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
