"""Run headers: the settings that decide an arm's outputs, stored beside its records.

A header has two kinds of key. CONFIG keys decide the outputs (dtype, `max_new_tokens`,
seed, sampling, batch size, template hash, serving flags, `extra_body`, ...): two arms
are comparable only when they agree on them, and `config_hash` hashes them. PROVENANCE
keys say where and when the run happened (device string, GPU, library commit, run id,
served name, base URL): they are stored, never compared and never hashed. Model
identity and the backend and its build (`backend`, `llama_cpp_build`, `vllm_version`,
`torch`, `transformers`) are CONFIG: a pair of arms must share one backend and build.
A key in neither set is CONFIG, the safe side.
"""

import hashlib
import json
import subprocess
import uuid
from collections.abc import Collection, Mapping
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy

_MISSING = "<missing>"

CONFIG_KEYS = frozenset(
    {
        "model",  # HF: model class under any adapter
        "model_name_or_path",  # HF: base checkpoint, same under an adapter
        "model_id",  # HTTP: model file (/props model_path) or declared id
        "backend",  # HF: cpu / cuda / mps; HTTP: llama.cpp / vllm / unknown
        "llama_cpp_build",  # one build per pair of arms
        "vllm_version",
        "torch",
        "transformers",
        "dtype",
        "max_new_tokens",
        "seed",
        "chat",
        "batch_size",
        "training",
        "chat_template_sha256",
        "sampling",
        "system",  # system prompt text, only on an arm that was given one
        "system_sha256",
        "extra_body",  # minus the declared arm keys, see `check_headers`
        "k",  # samples per item
        "concurrency",
        "serving_flags",  # declared server flags and env
    }
)
PROVENANCE_KEYS = frozenset(
    {
        "device",  # the full string; its type is CONFIG under "backend"
        "gpu_model",
        "gpu_count",
        "peft",
        "numpy",
        "astro_weights",
        "library_commit",
        "run_id",
        "base_url",
        "served_model",
        "server_model",
        "system_fingerprint",
        "infra_errors",  # HTTP: retries and items out of attempts
    }
)
# A CONFIG key that older headers lack counts as its default, not as a difference.
_DEFAULTS: dict[str, Any] = {"k": 1, "concurrency": 1}


def header_path(records_path: str | Path) -> Path:
    """Header file for a records file: `X.jsonl` -> `X.header.json`."""
    return Path(records_path).with_suffix(".header.json")


def write_header(path: str | Path, header: dict[str, Any]) -> None:
    """Write `header` as JSON with sorted keys."""
    text = json.dumps(header, indent=1, sort_keys=True, ensure_ascii=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def read_header(path: str | Path) -> dict[str, Any]:
    """Read a header file; ValueError if it is not a JSON object."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")  # noqa: TRY004
    return data


def check_headers(
    base: dict[str, Any],
    edit: dict[str, Any],
    ignore: Collection[str] = (),
    arm_keys: Collection[str] = (),
) -> None:
    """Raise ValueError listing every CONFIG key that differs or is missing on one side.

    PROVENANCE keys are always skipped; `ignore` names more keys to skip (a union, so
    the old `ignore=("device",)` still works). A key in neither set is compared. A
    CONFIG key that old headers lack (`k`, `concurrency`) counts as its default (1).
    `arm_keys` names `extra_body` keys that may differ between the arms, for two arms
    on one server that differ only in a request field such as a `lora` scale.
    """
    skip = PROVENANCE_KEYS | frozenset(ignore)
    b = {**_DEFAULTS, **_arm_free(base, arm_keys)}
    e = {**_DEFAULTS, **_arm_free(edit, arm_keys)}
    diffs = [
        f"{k}: base={b.get(k, _MISSING)!r} edit={e.get(k, _MISSING)!r}"
        for k in sorted((b.keys() | e.keys()) - skip)
        if k not in b or k not in e or b[k] != e[k]
    ]
    if diffs:
        raise ValueError(
            "arms ran under different settings and are not comparable: "
            + "; ".join(diffs)
        )


def _arm_free(header: Mapping[str, Any], arm_keys: Collection[str]) -> dict[str, Any]:
    """`header` with `arm_keys` removed from its `extra_body`."""
    out = dict(header)
    body = out.get("extra_body")
    if arm_keys and isinstance(body, Mapping):
        out["extra_body"] = {k: v for k, v in body.items() if k not in arm_keys}
    return out


def _config(header: Mapping[str, Any], arm_keys: Collection[str]) -> dict[str, Any]:
    """The CONFIG part of `header`: every key not in PROVENANCE_KEYS, defaults filled."""
    h = {**_DEFAULTS, **_arm_free(header, arm_keys)}
    return {k: v for k, v in h.items() if k not in PROVENANCE_KEYS}


def _norm(value: Any) -> Any:
    """`value` with whole-number floats as ints, so 1 and 1.0 hash alike."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, Mapping):
        return {k: _norm(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_norm(v) for v in value]
    return value


def config_hash(header: Mapping[str, Any], arm_keys: Collection[str] = ()) -> str:
    """SHA-256 hex of the canonical JSON (sorted keys) of the header's CONFIG keys.

    Unchanged by key order, by any PROVENANCE value and by 1 versus 1.0; changes with
    any other CONFIG value.
    `arm_keys` names `extra_body` keys left out, as in `check_headers`.
    """
    text = json.dumps(
        _norm(_config(header, arm_keys)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def system_header(system: str | None) -> dict[str, str]:
    """Header keys for a system prompt: `system` and `system_sha256`; empty for None.

    Raises ValueError for an empty or non-string `system`.
    """
    if system is None:
        return {}
    if not isinstance(system, str) or not system:
        raise ValueError(f"system must be a non-empty string or None, got {system!r}")
    return {
        "system": system,
        "system_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
    }


def library_version() -> str | None:
    """Installed astro-weights version, or None in an uninstalled source tree."""
    try:
        return version("astro-weights")
    except (
        PackageNotFoundError
    ):  # pragma: no cover - only in an uninstalled source tree
        return None


def library_commit() -> str | None:
    """Git commit the installed library was built from, or None (a release wheel).

    From the install's ``direct_url.json`` (PEP 610): ``vcs_info.commit_id`` for a
    ``pip install git+...``; for an editable install of a local checkout, that
    checkout's ``HEAD``, with ``-dirty`` when tracked files have changes. Never raises.
    """
    try:
        raw = distribution("astro-weights").read_text("direct_url.json")
        info = json.loads(raw) if raw else {}
        commit = info.get("vcs_info", {}).get("commit_id")
        if isinstance(commit, str) and commit:
            return commit
        url = urlparse(info.get("url", ""))
        if url.scheme != "file" or not info.get("dir_info", {}).get("editable"):
            return None
        where = url2pathname(url.path)

        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", where, *args],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=True,
            ).stdout.strip()

        head = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (
        PackageNotFoundError,
        OSError,
        ValueError,
        AttributeError,
        subprocess.SubprocessError,
    ):
        return None
    return f"{head}-dirty" if dirty else head or None


def run_identity(
    run_id: str | None, k: int, concurrency: int
) -> dict[str, str | int | None]:
    """Header keys every runner records: run id (random when not given), K, concurrency,
    library version and commit, numpy version.
    """
    if k < 1 or concurrency < 1:
        raise ValueError("k and concurrency must be at least 1")
    return {
        "run_id": run_id if run_id is not None else uuid.uuid4().hex,
        "k": k,
        "concurrency": concurrency,
        "astro_weights": library_version(),
        "library_commit": library_commit(),
        "numpy": numpy.__version__,
    }
