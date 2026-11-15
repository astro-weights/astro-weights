"""Command-line entry point."""

import argparse
import sys
from pathlib import Path

import numpy as np
from safetensors import SafetensorError

from astro_weights import __version__
from astro_weights.edits import LowRankDelta
from astro_weights.eval import compare, format_report, read_jsonl
from astro_weights.export import read_peft
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="astro-weights")
    parser.add_argument(
        "--version", action="version", version=f"astro-weights {__version__}"
    )
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("inspect", help="summarize and verify a saved PEFT adapter")
    p.add_argument("directory", metavar="DIR", type=Path)
    p = sub.add_parser("eval", help="compare two eval arms from per-item records")
    p.add_argument("base", metavar="BASE.jsonl", type=Path)
    p.add_argument("edit", metavar="EDIT.jsonl", type=Path)
    p.add_argument("--boot", type=int, default=10000, help="bootstrap resamples")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    args = parser.parse_args(argv)
    if args.command == "inspect":
        return _inspect(args.directory)
    if args.command == "eval":
        return _eval(args.base, args.edit, args.boot, args.seed)
    parser.print_help()
    return 0
