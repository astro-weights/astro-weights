"""Command-line entry point."""

import argparse
import sys
from pathlib import Path

import numpy as np
from safetensors import SafetensorError

from astro_weights import __version__
from astro_weights.export import load_peft, verify_peft
from astro_weights.export.peft import CONFIG, MANIFEST, WEIGHTS


def _inspect(directory: Path) -> int:
    if not directory.is_dir():
        print(f"error: not a directory: {directory}", file=sys.stderr)
        return 2
    missing = [n for n in (CONFIG, WEIGHTS) if not (directory / n).is_file()]
    if missing:
        print(f"error: not an adapter, missing {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        problems = verify_peft(directory)
        if problems:
            for p in problems:
                print(f"problem: {p}", file=sys.stderr)
            return 1
        edit = load_peft(directory)
    except (ValueError, KeyError, OSError, SafetensorError) as e:
        print(f"error: cannot read adapter: {e}", file=sys.stderr)
        return 2

    print(f"base model: {edit.base_model}")
    print(f"rank: {edit.rank}")
    print(f"modules: {len(edit.deltas)}")
    rows = [
        (
            d.module,
            f"{d.shape[0]}x{d.shape[1]}",
            *(f"{np.linalg.norm(m):.3g}" for m in (d.B, d.A, d.dense())),
        )
        for d in sorted(edit.deltas, key=lambda d: d.module)
    ]
    header = ("module", "out x in", "|B|_F", "|A|_F", "|BA|_F")
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(5)]
    for r in [header, *rows]:
        print(
            "  ".join(
                c.ljust(w) if i < 2 else c.rjust(w)
                for i, (c, w) in enumerate(zip(r, widths, strict=True))
            ).rstrip()
        )
    print("manifest: ok" if (directory / MANIFEST).exists() else "manifest: none")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="astro-weights")
    parser.add_argument(
        "--version", action="version", version=f"astro-weights {__version__}"
    )
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("inspect", help="summarize and verify a saved PEFT adapter")
    p.add_argument("directory", metavar="DIR", type=Path)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        return _inspect(args.directory)
    parser.print_help()
    return 0
