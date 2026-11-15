from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

from astro_weights.edits import Edit, LowRankDelta
from astro_weights.export.card import model_card, write_card


def _edit(base: str = "org/base") -> Edit:
    rng = np.random.default_rng(0)
    deltas = [
        LowRankDelta(m, rng.normal(size=(o, 2)), rng.normal(size=(2, i)))
        for m, o, i in [("b.proj", 6, 4), ("a.proj", 5, 3)]
    ]
    return Edit.from_deltas(deltas, base)


def _front(text: str) -> list[str]:
    lines = text.split("\n")
    assert lines[0] == "---"
    return lines[1 : lines.index("---", 1)]


def test_front_matter_adapter() -> None:
    assert _front(model_card(_edit(), "adapter", "apache-2.0")) == [
        'base_model: "org/base"',
        'base_model_relation: "adapter"',
        'library_name: "peft"',
        'license: "apache-2.0"',
        "tags:",
        '  - "astro-weights"',
        '  - "weight-edit"',
        '  - "lora"',
    ]


def test_front_matter_gguf() -> None:
    assert _front(model_card(_edit(), "gguf", "mit")) == [
        'base_model: "org/base"',
        'base_model_relation: "quantized"',
        'license: "mit"',
        "tags:",
        '  - "astro-weights"',
        '  - "weight-edit"',
        '  - "gguf"',
    ]


def test_table_rows_match_deltas() -> None:
    edit = _edit()
    text = model_card(edit, "adapter", "mit")
    rows = [ln for ln in text.split("\n") if ln.startswith("| `")]
    assert len(rows) == 2
    for row, d in zip(rows, sorted(edit.deltas, key=lambda d: d.module), strict=True):
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert cells[0] == f"`{d.module}`"
        assert cells[1] == f"{d.shape[0]} x {d.shape[1]}"
        assert int(cells[2]) == d.rank
        assert abs(float(cells[3]) - np.linalg.norm(d.dense())) < 1e-4


def test_title_summary_and_load_section() -> None:
    edit = _edit()
    text = model_card(edit, "adapter", "mit", title="My edit", summary="Does a thing.")
    assert "\n# My edit\n\nDoes a thing.\n\n## Edit\n" in text
    assert "PeftModel.from_pretrained" in text
    assert "# org/base edit\n" in model_card(edit, "gguf", "mit")
    assert "-m model.gguf" in model_card(edit, "gguf", "mit")
    assert f"astro-weights {version('astro-weights')}" in text.splitlines()[-1]


def test_results_only_when_given() -> None:
    assert "## Results" not in model_card(_edit(), "adapter", "mit")
    text = model_card(_edit(), "adapter", "mit", results={"acc": "0.9"})
    assert "## Results" in text
    assert "| acc | 0.9 |" in text


def test_errors() -> None:
    with pytest.raises(ValueError, match="kind"):
        model_card(_edit(), "bin", "mit")
    with pytest.raises(ValueError, match="base_model"):
        model_card(_edit(""), "adapter", "mit")


def test_format_and_write(tmp_path: Path) -> None:
    text = model_card(_edit(), "adapter", "mit", summary="x", results={"a": "b"})
    assert text.endswith("\n")
    assert not text.endswith("\n\n")
    assert all(ln == ln.rstrip() for ln in text.split("\n"))
    path = write_card(tmp_path / "out", text)
    assert path == tmp_path / "out" / "README.md"
    assert path.read_text() == text
