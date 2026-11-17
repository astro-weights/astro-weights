import importlib
import sys

import pytest


@pytest.mark.parametrize(
    ("module", "missing", "extra"),
    [
        ("astro_weights.runners.hf", "torch", "hf"),
        ("astro_weights.runners.generate", "torch", "hf"),
        ("astro_weights.export.gguf", "gguf", "gguf"),
        ("astro_weights.mcp.server", "mcp", "mcp"),
    ],
)
def test_missing_extra_names_the_install(
    monkeypatch: pytest.MonkeyPatch, module: str, missing: str, extra: str
) -> None:
    # block the package and any submodule an earlier test already imported
    for name in [n for n in sys.modules if n.startswith(missing + ".")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, missing, None)
    monkeypatch.delitem(sys.modules, module, raising=False)
    with pytest.raises(ModuleNotFoundError) as info:
        importlib.import_module(module)
    # without the extra installed, the first import in the block is the one missing
    assert (info.value.name or "").split(".")[0] in {
        "torch",
        "transformers",
        "peft",
        "gguf",
        "mcp",
    }
    assert str(info.value).startswith(module)
    assert str(info.value).endswith(f"pip install 'astro-weights[{extra}]'")


def test_endpoint_needs_no_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("torch", "transformers", "peft", "gguf"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, "astro_weights.runners.endpoint", raising=False)
    importlib.import_module("astro_weights.runners.endpoint")
