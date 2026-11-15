"""Command-line entry point."""

import argparse
import sys
from pathlib import Path

import numpy as np
from safetensors import SafetensorError

from astro_weights import __version__
from astro_weights.edits import LowRankDelta, build_edit, load_recipe
from astro_weights.eval import compare, format_report, read_jsonl
from astro_weights.export import (
    model_card,
    read_base_weight,
    read_peft,
    save_peft,
    write_card,
)
from astro_weights.export.peft import CONFIG, MANIFEST, WEIGHTS


def _norm_ba(d: LowRankDelta) -> float:
    """|BA|_F without the dense matrix: sqrt(trace((B^T B)(A A^T)))."""
    return float(np.sqrt(max(np.sum((d.B.T @ d.B) * (d.A @ d.A.T)), 0.0)))


def _inspect(directory: Path) -> int:
    if not directory.is_dir():
        print(f"error: not a directory: {directory}", file=sys.stderr)
        return 2
    missing = [n for n in (CONFIG, WEIGHTS) if not (directory / n).is_file()]
    if missing:
        print(f"error: not an adapter, missing {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        edit, problems = read_peft(directory)
    except (ValueError, KeyError, OSError, SafetensorError) as e:
        print(f"error: cannot read adapter: {e}", file=sys.stderr)
        return 2
    if problems:
        for p in problems:
            print(f"problem: {p}", file=sys.stderr)
        return 1

    print(f"base model: {edit.base_model}")
    print(f"rank: {edit.rank}")
    print(f"modules: {len(edit.deltas)}")
    deltas = sorted(edit.deltas, key=lambda d: d.module)
    w = max(len("module"), *(len(d.module) for d in deltas))
    print(
        f"{'module':<{w}}  {'out x in':<11}  {'|B|_F':>8}  {'|A|_F':>8}  {'|BA|_F':>8}"
    )
    for d in deltas:
        nb, na = np.linalg.norm(d.B), np.linalg.norm(d.A)
        print(
            f"{d.module:<{w}}  {f'{d.shape[0]}x{d.shape[1]}':<11}  "
            f"{nb:>8.3g}  {na:>8.3g}  {_norm_ba(d):>8.3g}"
        )
    print("manifest: ok" if (directory / MANIFEST).exists() else "manifest: none")
    return 0


def _eval(base: Path, edit: Path, boot: int, seed: int) -> int:
    recs = []
    for path in (base, edit):
        try:
            recs.append(read_jsonl(path))
        except (ValueError, OSError) as e:
            print(f"error: {path}: {e}", file=sys.stderr)
            return 2
    try:
        result = compare(recs[0], recs[1], n_boot=boot, seed=seed)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(format_report(result))
    return 0


def _error(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


def _build(recipe: Path, base: Path, out: Path, card: str | None, force: bool) -> int:
    if out.exists() and (not out.is_dir() or any(out.iterdir())) and not force:
        return _error(f"{out} exists and is not empty; pass --force to overwrite")
    try:
        edit = build_edit(
            load_recipe(recipe),
            lambda module: read_base_weight(base, module + ".weight"),
            recipe.parent,
        )
        text = model_card(edit, "adapter", card) if card is not None else None
        save_peft(edit, out)
        if text is not None:
            write_card(out, text)
    except (ValueError, KeyError, OSError, SafetensorError) as e:
        return _error(str(e))
    for d in sorted(edit.deltas, key=lambda d: d.module):
        print(f"{d.module}  {d.shape[0]}x{d.shape[1]}  rank {d.rank}")
    print(f"wrote {out}")
    return 0


def _export(
    directory: Path, gguf: Path | None, arch: str | None, card: str | None, force: bool
) -> int:
    if gguf is None and card is None:
        return _error("nothing to export; pass --gguf and/or --card")
    if gguf is not None and arch is None:
        return _error("--gguf needs --arch, e.g. --arch qwen35")
    if gguf is not None and gguf.exists() and not force:
        return _error(f"{gguf} exists; pass --force to overwrite")
    if card is not None and (directory / "README.md").exists() and not force:
        return _error(f"{directory / 'README.md'} exists; pass --force to overwrite")
    if not directory.is_dir():
        return _error(f"not a directory: {directory}")
    try:
        edit, problems = read_peft(directory)
    except (ValueError, KeyError, OSError, SafetensorError) as e:
        return _error(f"cannot read adapter: {e}")
    if problems:
        return _error("adapter failed verification: " + "; ".join(problems))
    try:
        if gguf is not None and arch is not None:
            try:
                from astro_weights.export.gguf import save_lora_gguf
            except ImportError:
                return _error("GGUF export needs: pip install 'astro-weights[gguf]'")
            save_lora_gguf(edit, gguf, arch, overwrite=force)
            print(f"wrote {gguf}")
        if card is not None:
            print(f"wrote {write_card(directory, model_card(edit, 'adapter', card))}")
    except (ValueError, OSError) as e:
        return _error(str(e))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="astro-weights")
    parser.add_argument(
        "--version", action="version", version=f"astro-weights {__version__}"
    )
    sub = parser.add_subparsers(dest="command", metavar="{build,eval,export,inspect}")
    p = sub.add_parser("inspect", help="summarize and verify a saved PEFT adapter")
    p.add_argument("directory", metavar="DIR", type=Path)
    p = sub.add_parser("eval", help="compare two eval arms from per-item records")
    p.add_argument("base", metavar="BASE.jsonl", type=Path)
    p.add_argument("edit", metavar="EDIT.jsonl", type=Path)
    p.add_argument("--boot", type=int, default=10000, help="bootstrap resamples")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p = sub.add_parser("build", help="build a PEFT adapter from a recipe file")
    p.add_argument("recipe", metavar="RECIPE.json", type=Path)
    p.add_argument("--base", metavar="MODEL_DIR", type=Path, required=True)
    p.add_argument("--out", metavar="DIR", type=Path, required=True)
    p.add_argument("--card", metavar="LICENSE", help="also write README.md")
    p.add_argument("--force", action="store_true", help="write into a non-empty DIR")
    p = sub.add_parser(
        "export", help="write a GGUF LoRA and/or model card for an adapter"
    )
    p.add_argument("directory", metavar="DIR", type=Path)
    p.add_argument("--gguf", metavar="OUT.gguf", type=Path)
    p.add_argument("--arch", help="GGUF architecture, e.g. qwen35")
    p.add_argument("--card", metavar="LICENSE", help="write README.md into DIR")
    p.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = parser.parse_args(argv)
    if args.command == "inspect":
        return _inspect(args.directory)
    if args.command == "eval":
        return _eval(args.base, args.edit, args.boot, args.seed)
    if args.command == "build":
        return _build(args.recipe, args.base, args.out, args.card, args.force)
    if args.command == "export":
        return _export(args.directory, args.gguf, args.arch, args.card, args.force)
    parser.print_help()
    return 0
