"""Run regression suites for one arm: load, generate, grade, write records and header.

`run_suites` is the step a gate needs without a user asking for it. Call it once
for the base arm and once for the edit arm with the same settings: the records
pair item by item (same items, same prompt hashes) and the headers differ only
in what the arms differ in, so `astro_weights.eval.compare` and `check_headers`
work on the two files directly. The HF backend imports torch lazily, so the
HTTP backend needs no extra.
"""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from astro_weights.eval import Record, config_hash, header_path, write_jsonl
from astro_weights.eval.header import read_header, write_header
from astro_weights.eval.suites import PINS, load_suite, suite_grader
from astro_weights.runners.endpoint import generate_records_http, run_header_http

__all__ = ["HFBackend", "HttpBackend", "run_suites"]


@dataclass(frozen=True)
class HFBackend:
    """A loaded transformers model (an applied edit included) and its tokenizer."""

    model: Any
    tokenizer: Any
    batch_size: int = 1
    chat: bool = True


@dataclass(frozen=True)
class HttpBackend:
    """An OpenAI-compatible server and the model name to ask for."""

    base_url: str
    model: str
    api_key: str | None = None
    concurrency: int = 1
    extra_body: Mapping[str, Any] | None = None


def _placeholder(gold: str, output: str) -> bool:
    """Scorer for the generate step; `run_suites` regrades with the suite grader."""
    return False


def _keep_header(path: Path, header: dict[str, Any]) -> None:
    """Write `header` beside `path`, or keep the one of the run being resumed."""
    where = header_path(path)
    if where.exists():
        if config_hash(read_header(where)) != config_hash(header):
            raise ValueError(
                f"{where}: the records were made under another config "
                "(config_hash differs); start a new out_dir for this one"
            )
        return
    write_header(where, header)


def _generate(
    backend: HFBackend | HttpBackend,
    items: Sequence[Mapping[str, Any]],
    arm: str,
    path: Path,
    header: dict[str, Any],
    max_new_tokens: int,
    seed: int,
    resume: bool,
    system: str | None,
) -> list[Record]:
    if isinstance(backend, HttpBackend):
        return generate_records_http(
            backend.base_url, backend.model, items, arm, _placeholder, max_new_tokens,
            seed=seed, api_key=backend.api_key, extra_body=backend.extra_body,
            out=path, concurrency=backend.concurrency, resume=resume, header=header,
            system=system,
        )  # fmt: skip
    from astro_weights.runners.generate import generate_records

    if resume:
        _keep_header(path, header)
    elif not (path.exists() and path.stat().st_size > 0):
        write_header(header_path(path), header)  # else generate_records refuses
    return generate_records(
        backend.model, backend.tokenizer, items, arm, _placeholder, max_new_tokens,
        seed, backend.chat, out=path, resume=resume, batch_size=backend.batch_size,
        system=system,
    )  # fmt: skip


def _header(
    backend: HFBackend | HttpBackend,
    max_new_tokens: int,
    seed: int,
    system: str | None,
) -> dict[str, Any]:
    if isinstance(backend, HttpBackend):
        return run_header_http(
            backend.base_url, backend.model, max_new_tokens, seed, backend.extra_body,
            api_key=backend.api_key, concurrency=backend.concurrency,
            system=system,
        )  # fmt: skip
    from astro_weights.runners.generate import run_header

    return run_header(
        backend.model, backend.tokenizer, max_new_tokens, seed, backend.chat,
        batch_size=backend.batch_size, system=system,
    )  # fmt: skip


def run_suites(
    names: Sequence[str],
    arm: str,
    out_dir: str | os.PathLike[str],
    *,
    backend: HFBackend | HttpBackend,
    max_new_tokens: int,
    seed: int = 0,
    limit: int | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    suite_paths: Mapping[str, str | os.PathLike[str]] | None = None,
    resume: bool = True,
    system: str | None = None,
) -> dict[str, Path]:
    """Run each suite in `names` for one arm; returns `{suite: records path}`.

    Records go to `out_dir/<suite>/<arm>.jsonl` and the run header (the settings
    that decide the outputs, see `eval.header`) to `<arm>.header.json` beside it.
    The header adds `suite`, `suite_source`, `suite_sha256`, `suite_verified`,
    `limit` and `grader` to the backend's keys, so `config_hash` differs per
    suite, and a base and an edit arm pair only when all of them match.

    Each suite is read by `load_suite` (the pinned download, cached under
    `cache_dir`, or the file in `suite_paths[name]`, whose sha256 must equal the
    pin: a wrong file raises `ValueError` before anything is generated). Decoding
    is greedy, one request or sequence per item. `limit` keeps the first N items
    in pinned order and is recorded in the header: for a smoke run, never for a
    verdict.

    Generation uses a placeholder scorer, because the generate step scores with
    `(gold, output)` and IFEval needs the whole item. Once every item has a
    record, each is regraded with `suite_grader(name)(item, output)` and the file
    is rewritten (atomically) with the real scores. A file left by a crash
    therefore holds scores of 0 until a resume finishes it. With `resume=True`
    (the default) a partly written file continues without duplicates; its header
    must have the same `config_hash`, else `ValueError`. With `resume=False` a
    non-empty file raises `FileExistsError`.

    `system` makes the arm a prompt arm: the text is sent as a system message and
    recorded in the header (`system`, `system_sha256`), so the records still pair
    with a base arm run without it (the prompt hash covers the item alone) and a
    resume under another `system` raises. Run the `prompt` arm with the base backend
    and the `edit+prompt` arm with the edited one, both with the same `system`.
    """
    if not names or len(set(names)) != len(names):
        raise ValueError("names must be a non-empty list without repeats")
    unknown = [n for n in names if n not in PINS]
    if unknown:
        raise ValueError(f"unknown suites {unknown}; expected from {list(PINS)}")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError(f"limit must be an int >= 1 or None, got {limit!r}")
    paths = suite_paths or {}
    out: dict[str, Path] = {}
    for name in names:
        data = load_suite(name, paths.get(name), cache_dir=cache_dir)
        items = data.items[:limit]
        path = Path(out_dir) / name / f"{arm}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            **_header(backend, max_new_tokens, seed, system),
            "suite": name,
            "suite_source": data.source,
            "suite_sha256": data.sha256,
            "suite_verified": data.verified,
            "limit": limit,
            "grader": f"suite_grader:{name}",
        }
        records = _generate(
            backend, items, arm, path, header, max_new_tokens, seed, resume, system
        )
        grade = suite_grader(name)
        graded = [
            replace(r, score=float(grade(item, r.output)))
            for item, r in zip(items, records, strict=True)
        ]
        tmp = path.with_name(path.name + ".tmp")
        write_jsonl(tmp, graded)
        tmp.replace(path)
        out[name] = path
    return out
