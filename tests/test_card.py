import re
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
        'base_model_relation: "finetune"',
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


def test_load_class_follows_module_layout() -> None:
    rng = np.random.default_rng(0)

    def one(module: str) -> Edit:
        delta = LowRankDelta(module, rng.normal(size=(4, 1)), rng.normal(size=(1, 8)))
        return Edit((delta,), "org/base")

    text = model_card(
        one("model.language_model.layers.3.mlp.down_proj"), "adapter", "mit"
    )
    assert "AutoModelForImageTextToText.from_pretrained" in text
    assert "AutoModelForCausalLM" not in text
    text = model_card(one("model.layers.3.mlp.down_proj"), "adapter", "mit")
    assert "AutoModelForCausalLM.from_pretrained" in text


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


def _rows(text: str) -> list[list[str]]:
    rows = [ln for ln in text.split("\n") if ln.startswith("| ") and "---" not in ln]
    return [[c.strip() for c in re.split(r"(?<!\\)\|", r)[1:-1]] for r in rows]


def test_astral_chars_kept_literal() -> None:
    edit = _edit("org/😀")
    text = model_card(edit, "adapter", "😀\x7f\u2028")
    front = _front(text)
    assert front[0] == 'base_model: "org/😀"'
    assert front[3] == 'license: "😀\\u007f\\u2028"'
    assert 'from_pretrained("org/😀")' in text
    assert "\\ud83d" not in text


def test_table_cells_escaped() -> None:
    d = LowRankDelta("m|o`d\nel", np.ones((3, 1)), np.ones((1, 2)))
    edit = Edit.from_deltas([d], "org/base")
    text = model_card(edit, "adapter", "mit", results={"k|\n## h": "v|\nw"})
    assert "\n## h" not in text
    rows = _rows(text)
    assert ["``m\\|o`d el``", "3 x 2", "1", "2.44949"] in rows
    assert ["k\\| ## h", "v\\| w"] in rows


def test_title_and_summary_whitespace() -> None:
    text = model_card(
        _edit(), "adapter", "mit", title="T\n## injected", summary="a  \nb   \n"
    )
    assert "\n# T ## injected\n" in text
    assert "\na\nb\n" in text
    assert all(ln == ln.rstrip() for ln in text.split("\n"))
    assert "# org/base edit\n" in model_card(_edit(), "gguf", "mit", title=" \n ")


def test_empty_license() -> None:
    with pytest.raises(ValueError, match="license"):
        model_card(_edit(), "adapter", "  ")


def test_write_card_no_overwrite(tmp_path: Path) -> None:
    write_card(tmp_path, "one\n")
    with pytest.raises(FileExistsError):
        write_card(tmp_path, "two\n", overwrite=False)
    assert (tmp_path / "README.md").read_text() == "one\n"
    write_card(tmp_path, "two\n")
    assert (tmp_path / "README.md").read_text() == "two\n"


def test_gguf_card_is_not_quantized() -> None:
    # a full-model GGUF holds edited weights; "quantized" would say they are unchanged
    text = model_card(_edit(), "gguf", "mit")
    assert 'base_model_relation: "finetune"' in text
    assert "quantized" not in text


def test_modification_notice_on_every_card() -> None:
    for kind in ("adapter", "gguf"):
        text = model_card(_edit(), kind, "mit")
        assert (
            "Modification: the weights of org/base were modified by a low-rank edit "
            "of 2 modules, made with astro-weights without training.\n"
        ) in text
        assert (
            text.index("## Edit") < text.index("Modification:") < text.index("| module")
        )
    one = Edit.from_deltas(list(_edit().deltas[:1]), "org/base")
    assert "edit of 1 module," in model_card(one, "adapter", "mit")
