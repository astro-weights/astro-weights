"""Snapshot of the public API: names and signatures, per module.

On purpose a change here needs a human: regenerate the snapshot with
    ASTRO_UPDATE_API=1 uv run --extra hf --extra eval --extra gguf --extra mcp --extra hosted pytest tests/test_public_api.py
and add a changelog fragment in changes/.
"""

import importlib
import inspect
import json
import os
import re
from pathlib import Path
from types import ModuleType

import pytest

SNAPSHOT = Path(__file__).parent / "golden" / "public_api.json"
MODULES = [
    "astro_weights",
    "astro_weights.edits",
    "astro_weights.export",
    "astro_weights.eval",
    "astro_weights.directions",
    "astro_weights.export.gguf",
    "astro_weights.runners.hf",
    "astro_weights.runners.endpoint",
    "astro_weights.runners.generate",
    "astro_weights.mcp",
    "astro_weights.mcp.budget",
    "astro_weights.mcp.forward",
    "astro_weights.mcp.jobs",
    "astro_weights.mcp.kernel",
    "astro_weights.mcp.kernel_child",
    "astro_weights.mcp.record",
    "astro_weights.mcp.run",
    "astro_weights.mcp.sandbox",
    "astro_weights.mcp.server",
    "astro_weights.mcp.editors.hosted",
    "astro_weights.mcp.editors.local",
    "astro_weights.mcp.editors.loop",
    "astro_weights.mcp.editors.prices",
    "astro_weights.mcp.editors.scripted",
]
HF = ("torch", "transformers", "peft")
EXTRAS = {
    "astro_weights.export.gguf": ("gguf",),
    "astro_weights.runners.hf": HF,
    "astro_weights.runners.generate": HF,
    "astro_weights.mcp.server": ("mcp",),
    "astro_weights.mcp.editors.hosted": ("anthropic", "mcp"),
    "astro_weights.mcp.editors.local": ("mcp",),
    "astro_weights.mcp.editors.loop": ("mcp",),
    "astro_weights.mcp.editors.scripted": ("mcp",),
}
HOWTO = (
    "The public API changed. If that is intended, update the snapshot with "
    "`ASTRO_UPDATE_API=1 uv run --extra hf --extra eval --extra gguf --extra mcp --extra hosted pytest "
    "tests/test_public_api.py` and add a towncrier fragment in changes/."
)


def _public_names(mod: ModuleType) -> list[str]:
    declared = getattr(mod, "__all__", None)
    if declared is not None:
        return sorted(declared)
    return sorted(
        n
        for n, obj in vars(mod).items()
        if not n.startswith("_") and getattr(obj, "__module__", None) == mod.__name__
    )


def _no_optional(sig: str) -> str:
    """``Optional[X]`` as ``X | None``, as Python 3.14 prints it."""
    while (start := sig.find("Optional[")) != -1:
        depth, end = 0, start + len("Optional")
        while True:  # find the matching closing bracket
            depth += {"[": 1, "]": -1}.get(sig[end], 0)
            if depth == 0:
                break
            end += 1
        inner = sig[start + len("Optional[") : end]
        sig = f"{sig[:start]}{inner} | None{sig[end + 1 :]}"
    return sig


def _describe(obj: object) -> str | None:
    if not callable(obj):
        return None
    try:
        sig = str(inspect.signature(obj))
        # Python 3.13 moved Path to pathlib._local; keep the snapshot version-independent
        # 3.14 prints `Any` inside `X[..., Any] | None` without the typing. prefix
        sig = sig.replace("pathlib._local.", "pathlib.").replace("typing.Any", "Any")
        # a default that holds a function prints its memory address
        return _no_optional(re.sub(r" at 0x[0-9a-fA-F]+>", ">", sig))
    except (TypeError, ValueError):
        return None


def _surface(name: str) -> dict[str, str | None]:
    mod = importlib.import_module(name)
    return {n: _describe(getattr(mod, n)) for n in _public_names(mod)}


@pytest.mark.parametrize("name", MODULES)
def test_public_api_snapshot(name: str) -> None:
    for dep in EXTRAS.get(name, ()):
        pytest.importorskip(dep)
    current = _surface(name)
    snap: dict[str, dict[str, str | None]] = json.loads(SNAPSHOT.read_text())
    if os.environ.get("ASTRO_UPDATE_API"):
        snap[name] = current
        snap = dict(sorted(snap.items()))
        SNAPSHOT.write_text(json.dumps(snap, indent=1) + "\n")
        return
    assert name in snap, f"{name} missing from public_api.json. {HOWTO}"
    expected = snap[name]
    assert sorted(current) == sorted(expected), f"{name} names changed. {HOWTO}"
    changed = {
        n: (expected[n], current[n]) for n in current if current[n] != expected[n]
    }
    assert not changed, f"{name} signatures changed: {changed}. {HOWTO}"


def test_optional_prints_as_a_union() -> None:
    assert _no_optional("(a: Optional[Literal['x', 'y']]) -> Optional[int]") == (
        "(a: Literal['x', 'y'] | None) -> int | None"
    )


def test_describe_drops_memory_addresses() -> None:
    def fn(x: object = len) -> None: ...

    described = _describe(fn)
    assert described is not None
    assert "0x" not in described


def test_describe_is_python_version_independent() -> None:
    def fn(directory: "pathlib._local.Path | str") -> None: ...  # type: ignore[name-defined]  # noqa: F821

    described = _describe(fn)
    assert described is not None
    assert "_local" not in described
    assert "pathlib.Path" in described
