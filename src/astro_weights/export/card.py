"""Write a Hugging Face model card (README.md) for an edit."""

import json
import re
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


_ESCAPES = {
    "\x7f": "\\u007f",
    "\x85": "\\u0085",
    "\u2028": "\\u2028",
    "\u2029": "\\u2029",
}


def _q(s: str) -> str:
    """Double-quoted string valid as YAML and Python; astral chars kept literal."""
    out = json.dumps(s, ensure_ascii=False)
    for ch, esc in _ESCAPES.items():
        out = out.replace(ch, esc)
    return out


def _line(s: str) -> str:
    """Collapse all whitespace, including line breaks, to single spaces."""
    return " ".join(s.split())


def _cell(s: str) -> str:
    """Escape text for a markdown table cell."""
    return _line(s).replace("\\", "\\\\").replace("|", "\\|")


def _code_cell(s: str) -> str:
    """Code span for a table cell; the fence outgrows any backtick run inside."""
    s = _line(s).replace("|", "\\|")
    fence = "`" * (_longest_run(s) + 1)
    pad = " " if s.startswith("`") or s.endswith("`") else ""
    return f"{fence}{pad}{s}{pad}{fence}"


def _longest_run(s: str) -> int:
    runs = [len(r) for r in re.findall(r"`+", s)]
    return max(runs, default=0)


def _front_matter(edit: Edit, kind: str, license: str) -> list[str]:
    q = _q
    lines = ["---", f"base_model: {q(edit.base_model)}"]
    relation = "adapter" if kind == "adapter" else "finetune"
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
            f"| {_code_cell(d.module)} | {out} x {inp} | {d.rank} | {_fro_norm(d):.6g} |"
        )
    return lines


def _how_to_load(edit: Edit, kind: str) -> list[str]:
    if kind == "gguf":
        return ["```bash", 'llama-cli -m model.gguf -p "Hello"', "```"]
    # PEFT matches keys by full module name: `language_model.layers` adapters come
    # from the multimodal class, and a CausalLM load would match none (silent no-op)
    multimodal = any(".language_model." in d.module for d in edit.deltas)
    cls = "AutoModelForImageTextToText" if multimodal else "AutoModelForCausalLM"
    return [
        "```python",
        "from peft import PeftModel",
        f"from transformers import {cls}",
        "",
        f"base = {cls}.from_pretrained({_q(edit.base_model)})",
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
    if not license.strip():
        raise ValueError("license is empty; the Hub rejects a blank license")
    lines = _front_matter(edit, kind, license)
    lines += ["", f"# {_line(title or '') or f'{edit.base_model} edit'}", ""]
    if summary and summary.strip():
        lines += ["\n".join(summary.strip().splitlines()), ""]
    n = len(edit.deltas)
    notice = (
        f"Modification: the weights of {edit.base_model} were modified by a low-rank "
        f"edit of {n} {'module' if n == 1 else 'modules'}, made with astro-weights "
        "without training."
    )
    lines += ["## Edit", "", notice, "", *_edit_table(edit), "", "## How to load", ""]
    lines += [*_how_to_load(edit, kind), ""]
    if results:
        lines += ["## Results", "", "| metric | value |", "| --- | --- |"]
        lines += [f"| {_cell(k)} | {_cell(v)} |" for k, v in results.items()]
        lines.append("")
    lines.append(f"Card written by astro-weights {version('astro-weights')}.")
    text = "\n".join(lines)
    return "\n".join(line.rstrip() for line in text.split("\n")) + "\n"


def write_card(directory: Path | str, text: str, *, overwrite: bool = True) -> Path:
    """Write `text` as README.md in `directory` and return its path.

    Replaces an existing README.md unless `overwrite=False`, which raises
    FileExistsError instead.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "README.md"
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")
    return path
