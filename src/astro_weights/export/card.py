"""Write a Hugging Face model card (README.md) for an edit."""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from astro_weights.edits import Edit, LowRankDelta

KINDS = ("adapter", "gguf")


def _fro_norm(d: LowRankDelta) -> float:
    """|BA|_F = sqrt(trace((B^T B)(A A^T))) without forming BA."""
    b = d.B.astype(np.float64)
    a = d.A.astype(np.float64)
    return float(np.sqrt(max(np.sum((b.T @ b) * (a @ a.T)), 0.0)))


def _front_matter(edit: Edit, kind: str, license: str) -> list[str]:
    q = json.dumps
    lines = ["---", f"base_model: {q(edit.base_model)}"]
    relation = "adapter" if kind == "adapter" else "quantized"
    lines.append(f"base_model_relation: {q(relation)}")
    if kind == "adapter":
        lines.append(f"library_name: {q('peft')}")
    lines.append(f"license: {q(license)}")
    tags = ["astro-weights", "weight-edit", "lora" if kind == "adapter" else "gguf"]
    lines.append("tags:")
    lines.extend(f"  - {q(t)}" for t in tags)
    lines.append("---")
    return lines


def _edit_table(edit: Edit) -> list[str]:
    lines = [
        "| module | shape (out x in) | rank | \\|BA\\|_F |",
        "| --- | --- | --- | --- |",
    ]
    for d in sorted(edit.deltas, key=lambda d: d.module):
        out, inp = d.shape
        lines.append(
            f"| `{d.module}` | {out} x {inp} | {d.rank} | {_fro_norm(d):.6g} |"
        )
    return lines


def _how_to_load(edit: Edit, kind: str) -> list[str]:
    if kind == "gguf":
        return ["```bash", 'llama-cli -m model.gguf -p "Hello"', "```"]
    return [
        "```python",
        "from peft import PeftModel",
        "from transformers import AutoModelForCausalLM",
        "",
        f"base = AutoModelForCausalLM.from_pretrained({json.dumps(edit.base_model)})",
        'model = PeftModel.from_pretrained(base, "path/to/this/adapter")',
        "```",
    ]


def model_card(
    edit: Edit,
    kind: str,
    license: str,
    title: str | None = None,
    summary: str | None = None,
    results: dict[str, str] | None = None,
) -> str:
    """Render a model card for `edit`; `kind` is "adapter" or "gguf"."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    if not edit.base_model:
        raise ValueError("edit.base_model is empty; a model card needs a base model")
    lines = _front_matter(edit, kind, license)
    lines += ["", f"# {title or f'{edit.base_model} edit'}", ""]
    if summary:
        lines += [summary, ""]
    lines += ["## Edit", "", *_edit_table(edit), "", "## How to load", ""]
    lines += [*_how_to_load(edit, kind), ""]
    if results:
        lines += ["## Results", "", "| metric | value |", "| --- | --- |"]
        lines += [f"| {k} | {v} |" for k, v in results.items()]
        lines.append("")
    lines.append(f"Card written by astro-weights {version('astro-weights')}.")
    return "\n".join(line.rstrip() for line in lines) + "\n"


def write_card(directory: Path | str, text: str) -> Path:
    """Write `text` as README.md in `directory` and return its path."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "README.md"
    path.write_text(text)
    return path
