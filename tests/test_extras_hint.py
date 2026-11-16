import importlib
import sys

import pytest


@pytest.mark.parametrize(
    ("module", "missing", "extra"),
    [
        ("astro_weights.runners.hf", "torch", "hf"),
        ("astro_weights.runners.generate", "torch", "hf"),
        ("astro_weights.export.gguf", "gguf", "gguf"),
    ],
)
def test_missing_extra_names_the_install(
    monkeypatch: pytest.MonkeyPatch, module: str, missing: str, extra: str
) -> None:
    monkeypatch.setitem(sys.modules, missing, None)
    monkeypatch.delitem(sys.modules, module, raising=False)
    with pytest.raises(ModuleNotFoundError) as info:
        importlib.import_module(module)
    assert info.value.name == missing
    assert str(info.value).startswith(module)
    assert str(info.value).endswith(f"pip install 'astro-weights[{extra}]'")


def test_endpoint_needs_no_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("torch", "transformers", "peft", "gguf"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, "astro_weights.runners.endpoint", raising=False)
    importlib.import_module("astro_weights.runners.endpoint")
