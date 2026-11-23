"""Regression suites (GSM8K, IFEval, MMLU-Pro) at a pinned revision and hash.

No suite data ships with the package. `load_suite` reads a file you give it, or
downloads the pinned file from the Hugging Face Hub (standard library only), and
checks its sha256 against the pin either way. Items are dicts with `item_id`,
`prompt` and `gold`, which is what the runners take. Standard library at import
time; parquet sources need pyarrow (`pip install 'astro-weights[eval]'`).
"""

import hashlib
import json
import os
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astro_weights.eval.grade import answer_part, grade_choice, grade_math

__all__ = ["PINS", "SUITES", "Pin", "SuiteData", "load_suite", "suite_grader"]

_HUB = "https://huggingface.co/datasets"


@dataclass(frozen=True)
class Pin:
    """One source file of a suite, fixed by repo, revision and sha256."""

    repo: str
    revision: str
    filename: str
    sha256: str
    rows: int
    fmt: str  # "parquet" or "jsonl"


# Verified on each dataset's Hub page and by hashing the downloaded file.
PINS: dict[str, Pin] = {
    # PDR 0.3 §8 pins the derived data/bench/gsm8k.jsonl (sha256 c55b65bc...), not
    # this source file; the prompt template there is not recorded, so it cannot be
    # rebuilt from here. The row count and revision match.
    "gsm8k": Pin(
        "openai/gsm8k",
        "740312add88f781978c0658806c59bc2815b9866",
        "main/test-00000-of-00001.parquet",
        "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
        1319,
        "parquet",
    ),
    "ifeval": Pin(
        "google/IFEval",
        "966cd89545d6b6acfd7638bc708b98261ca58e84",
        "ifeval_input_data.jsonl",
        "6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2",
        541,
        "jsonl",
    ),
    "mmlu_pro": Pin(
        "TIGER-Lab/MMLU-Pro",
        "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "data/test-00000-of-00001.parquet",
        "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8",
        12032,
        "parquet",
    ),
}
SUITES = tuple(PINS)

_GSM8K_ASK = "\n\nPut your final answer in \\boxed{}."
_MMLU_ASK = "\n\nAnswer with the letter of the correct option in \\boxed{}."


@dataclass(frozen=True)
class SuiteData:
    """Items plus where they came from.

    `sha256` is the hash of the file actually read. `verified` is False only for
    a user file loaded with `verify=False` whose hash differs from the pin; record
    `sha256` and `source` in the run header then.
    """

    items: list[dict[str, Any]]
    sha256: str
    source: str
    verified: bool


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _download(repo: str, revision: str, filename: str, dest: Path) -> None:
    """Fetch one file of a Hub dataset at a revision into `dest`."""
    url = f"{_HUB}/{repo}/resolve/{revision}/{filename}"
    with urllib.request.urlopen(url, timeout=60) as r, dest.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def _cache_root(cache_dir: str | os.PathLike[str] | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "astro-weights" / "suites"


def _fetch(pin: Pin, cache_dir: str | os.PathLike[str] | None) -> Path:
    """The pinned file in the cache; downloaded and hash-checked if absent."""
    dest = _cache_root(cache_dir) / pin.repo.replace("/", "--") / pin.revision
    dest = dest / pin.filename
    if dest.is_file() and _sha256(dest) == pin.sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        _download(pin.repo, pin.revision, pin.filename, part)
        actual = _sha256(part)
        if actual != pin.sha256:
            raise ValueError(
                f"{pin.repo}@{pin.revision[:12]} {pin.filename}: downloaded sha256 "
                f"{actual} != pinned {pin.sha256}"
            )
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)
    return dest


def _parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ModuleNotFoundError(
            "reading parquet suites needs: pip install 'astro-weights[eval]'",
            name=e.name,
        ) from e
    rows: list[dict[str, Any]] = pq.read_table(path).to_pylist()
    return rows


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _gsm8k(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for i, row in enumerate(rows):
        _, sep, gold = str(row["answer"]).rpartition("#### ")
        if not sep:
            raise ValueError(f"gsm8k row {i} has no '#### ' answer line")
        items.append(
            {
                "item_id": f"gsm8k-{i}",
                "prompt": str(row["question"]) + _GSM8K_ASK,
                "gold": gold.strip(),
            }
        )
    return items


def _ifeval(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": f"ifeval-{row['key']}",
            "prompt": row["prompt"],
            "gold": ",".join(row["instruction_id_list"]),
            "instruction_id_list": list(row["instruction_id_list"]),
            "kwargs": [dict(k) for k in row["kwargs"]],
        }
        for row in rows
    ]


def _mmlu_pro(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for row in rows:
        options = "\n".join(f"{chr(65 + j)}. {o}" for j, o in enumerate(row["options"]))
        items.append(
            {
                "item_id": f"mmlu_pro-{row['question_id']}",
                "prompt": f"{row['question']}\n\n{options}{_MMLU_ASK}",
                "gold": str(row["answer"]),
                "category": row["category"],
            }
        )
    return items


_BUILD: dict[str, Callable[[list[dict[str, Any]]], list[dict[str, Any]]]] = {
    "gsm8k": _gsm8k,
    "ifeval": _ifeval,
    "mmlu_pro": _mmlu_pro,
}


def load_suite(
    name: str,
    path: str | os.PathLike[str] | None = None,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    verify: bool = True,
) -> SuiteData:
    """Items of a regression suite: "gsm8k", "ifeval" or "mmlu_pro".

    With `path`, reads that file in the pinned source format (see `PINS`) and
    refuses it with a ValueError naming both hashes unless its sha256 equals the
    pin; `verify=False` accepts any hash and returns it in `SuiteData.sha256` with
    `verified=False`. Without `path`, downloads the pinned file (cached under
    `cache_dir`, default `$XDG_CACHE_HOME/astro-weights/suites`) and always
    checks it.

    Item ids are `gsm8k-<row>`, `ifeval-<key>` and `mmlu_pro-<question_id>`.
    Prompt templates are fixed:

    - gsm8k: the question, then "Put your final answer in \\boxed{}.". Gold is the
      text after `#### `; grade with `suite_grader` (math_verify).
    - mmlu_pro: the question, the options as `A. ...` lines, then "Answer with the
      letter of the correct option in \\boxed{}.". Gold is the letter.
    - ifeval: the dataset prompt unchanged. Items also carry `instruction_id_list`
      and `kwargs`, which the grader reads; gold is the ids joined by commas.
    """
    if name not in PINS:
        raise ValueError(f"unknown suite {name!r}; expected one of {list(PINS)}")
    pin = PINS[name]
    if path is None:
        file = _fetch(pin, cache_dir)
        source = f"{pin.repo}@{pin.revision}:{pin.filename}"
    else:
        file = Path(path)
        source = str(file)
    actual = _sha256(file)
    verified = actual == pin.sha256
    if not verified and (verify or path is None):
        raise ValueError(f"{name}: {source} has sha256 {actual}, expected {pin.sha256}")
    read = _parquet_rows if pin.fmt == "parquet" else _jsonl_rows
    return SuiteData(_BUILD[name](read(file)), actual, source, verified)


def suite_grader(name: str) -> Callable[[Mapping[str, Any], str], float]:
    """Grader `(item, answer) -> score in [0, 1]` for a suite.

    gsm8k uses `grade_math` (needs the `eval` extra), mmlu_pro `grade_choice`,
    ifeval strict prompt-level accuracy (`grade_ifeval`; 1.0 only when every
    instruction is followed). Text before the last `</think>` is ignored.
    """
    if name == "gsm8k":

        def gsm8k(item: Mapping[str, Any], answer: str) -> float:
            return float(
                grade_math(item["gold"], answer_part(answer), strip_commas=True)
            )

        return gsm8k
    if name == "mmlu_pro":

        def mmlu_pro(item: Mapping[str, Any], answer: str) -> float:
            return float(grade_choice(item["gold"], answer_part(answer)))

        return mmlu_pro
    if name == "ifeval":
        from astro_weights.eval.ifeval import grade_ifeval

        return grade_ifeval
    raise ValueError(f"unknown suite {name!r}; expected one of {list(PINS)}")
