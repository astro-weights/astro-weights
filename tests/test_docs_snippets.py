"""Run every ```python block in README.md and the tutorials / how-to pages.

Blocks of one file run in order in one namespace, with cwd set to a temp dir.
A block may be skipped only via SKIP below, with a reason; "BROKEN: ..." marks a
docs bug that should be fixed in the docs, after which the entry is deleted.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    "README.md",
    *sorted(
        p.relative_to(ROOT).as_posix() for p in (ROOT / "docs/tutorials").glob("*.md")
    ),
    *sorted(
        p.relative_to(ROOT).as_posix() for p in (ROOT / "docs/how-to").glob("*.md")
    ),
]
HF = ("torch", "transformers", "peft")
# Optional modules a file needs; the whole file is skipped when one is missing.
NEEDS: dict[str, tuple[str, ...]] = {
    "docs/tutorials/first-edit.md": HF,
    "docs/tutorials/first-agent-run.md": (*HF, "mcp"),
    "docs/how-to/apply-across-model-classes.md": HF,
    "docs/how-to/export-gguf.md": ("gguf",),
    "docs/how-to/run-eval.md": HF,
    "docs/how-to/save-merged-weights.md": HF,
}
# (file, 0-based block index) -> reason. Blocks listed here are not run.
SKIP: dict[tuple[str, int], str] = {
    ("README.md", 1): "downloads a real base model and needs an adapter built for it",
    (
        "docs/how-to/run-eval.md",
        4,
    ): "needs an OpenAI-compatible server (llama-server, vLLM)",
}

FENCE = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)


def blocks(path: str) -> list[str]:
    return FENCE.findall((ROOT / path).read_text())


@pytest.mark.parametrize("path", FILES)
def test_snippets_run(
    path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for dep in NEEDS.get(path, ()):
        pytest.importorskip(dep)
    monkeypatch.chdir(tmp_path)
    ns: dict[str, object] = {"__name__": "__docs__"}
    ran = 0
    for i, code in enumerate(blocks(path)):
        if (path, i) in SKIP:
            continue
        try:
            exec(compile(code, f"{path}[block {i}]", "exec"), ns)  # noqa: S102
        except Exception as e:
            raise AssertionError(f"{path} block {i} failed: {e!r}") from e
        ran += 1
    if not ran:
        pytest.skip("every block skipped")


def test_skip_list_is_current() -> None:
    for (path, i), reason in SKIP.items():
        assert i < len(blocks(path)), f"stale skip entry {path} block {i}"
        assert reason, f"skip entry {path} block {i} needs a reason"
