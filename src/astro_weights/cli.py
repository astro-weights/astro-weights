"""Command-line entry point."""

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from safetensors import SafetensorError

from astro_weights import __version__
from astro_weights.edits import LowRankDelta, build_edit, load_recipe
from astro_weights.eval import (
    Record,
    check_headers,
    compare,
    exceeds_floor,
    format_report,
    header_path,
    read_header,
    read_jsonl,
)
from astro_weights.export import (
    model_card,
    read_base_weight,
    read_peft,
    save_peft,
    write_card,
)
from astro_weights.export.peft import CONFIG, MANIFEST, WEIGHTS

_ERRORS = (
    ValueError,
    KeyError,
    OSError,
    SafetensorError,
    TypeError,
    AttributeError,
    RecursionError,
)


def _clean(x: object) -> object:
    """JSON-ready copy: tuples become lists, non-finite floats become None."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _emit(obj: dict[str, object]) -> None:
    """Print the one JSON object `--json` writes to stdout."""
    print(json.dumps(_clean(obj), allow_nan=False))


def _norm_ba(d: LowRankDelta) -> float:
    """|BA|_F without the dense matrix: sqrt(trace((B^T B)(A A^T)))."""
    return float(np.sqrt(max(np.sum((d.B.T @ d.B) * (d.A @ d.A.T)), 0.0)))


def _inspect(directory: Path, as_json: bool) -> int:
    if not directory.is_dir():
        print(f"error: not a directory: {directory}", file=sys.stderr)
        return 2
    missing = [n for n in (CONFIG, WEIGHTS) if not (directory / n).is_file()]
    if missing:
        print(f"error: not an adapter, missing {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        edit, problems = read_peft(directory)
    except _ERRORS as e:
        return _fail(directory, e)
    deltas = sorted(edit.deltas, key=lambda d: d.module)
    if as_json:
        _emit(
            {
                "base_model": edit.base_model,
                "rank": edit.rank,
                "modules": [
                    {
                        "module": d.module,
                        "shape": list(d.shape),
                        "rank": d.rank,
                        "norm_B": float(np.linalg.norm(d.B)),
                        "norm_A": float(np.linalg.norm(d.A)),
                        "norm_BA": _norm_ba(d),
                    }
                    for d in deltas
                ],
                "manifest": (directory / MANIFEST).exists(),
                "problems": list(problems),
            }
        )
    if problems:
        for p in problems:
            print(f"problem: {p}", file=sys.stderr)
        return 1
    if as_json:
        return 0

    print(f"base model: {edit.base_model}")
    print(f"rank: {edit.rank}")
    print(f"modules: {len(edit.deltas)}")
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


def _read_arm(path: Path) -> list[Record] | None:
    try:
        return read_jsonl(path)
    except (ValueError, OSError) as e:
        print(f"error: {path}: {e}", file=sys.stderr)
        return None


def _check_run_headers(base: Path, edit: Path) -> int:
    """0 when headers agree or one is missing; 1 when they differ; 2 when unreadable."""
    hb, he = header_path(base), header_path(edit)
    if not (hb.is_file() and he.is_file()):
        return 0
    try:
        headers = read_header(hb), read_header(he)
    except (ValueError, OSError) as e:
        return _error(str(e))
    try:
        check_headers(*headers)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def _eval(
    base: Path,
    edit: Path,
    boot: int,
    seed: int,
    floor: list[Path] | None,
    ignore_header: bool,
    as_json: bool,
) -> int:
    if not ignore_header and (code := _check_run_headers(base, edit)):
        return code
    arms = [_read_arm(p) for p in (base, edit, *(floor or ()))]
    if any(a is None for a in arms):
        return 2
    try:
        result = compare(arms[0] or [], arms[1] or [], n_boot=boot, seed=seed)
        floor_result = (
            compare(arms[2] or [], arms[3] or [], n_boot=boot, seed=seed)
            if floor
            else None
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if as_json:
        _emit(
            {
                **asdict(result),
                "floor": None
                if floor_result is None
                else {
                    "result": asdict(floor_result),
                    "exceeds": exceeds_floor(result, floor_result),
                },
            }
        )
    else:
        print(format_report(result, floor_result))
    return 0


def _error(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


def _fail(where: Path | str, e: Exception) -> int:
    """Report a handled error, naming `where`; KeyError text is just the key."""
    if isinstance(e, KeyError):
        return _error(f"{where}: missing key {e}")
    msg = str(e)
    return _error(msg if str(where) in msg else f"{where}: {msg}")


def _build(
    recipe: Path, base: Path, out: Path, card: str | None, force: bool, as_json: bool
) -> int:
    if out.exists() and (not out.is_dir() or any(out.iterdir())) and not force:
        return _error(f"{out} exists and is not empty; pass --force to overwrite")
    try:
        spec = load_recipe(recipe)
    except _ERRORS as e:
        return _fail(recipe, e)
    try:
        edit = build_edit(
            spec, lambda m: read_base_weight(base, m + ".weight"), recipe.parent
        )
        text = model_card(edit, "adapter", card) if card is not None else None
        save_peft(edit, out, overwrite=force)
        card_path = write_card(out, text) if text is not None else None
    except _ERRORS as e:
        return _fail(recipe, e)
    if as_json:
        _emit({"written": [str(out)] + ([str(card_path)] if card_path else [])})
        return 0
    for d in sorted(edit.deltas, key=lambda d: d.module):
        print(f"{d.module}  {d.shape[0]}x{d.shape[1]}  rank {d.rank}")
    print(f"wrote {out}")
    return 0


def _export(
    directory: Path,
    gguf: Path | None,
    arch: str | None,
    card: str | None,
    force: bool,
    as_json: bool,
) -> int:
    if gguf is None and card is None:
        return _error("nothing to export; pass --gguf and/or --card")
    if gguf is not None and not (arch and arch.strip()):
        return _error("--gguf needs a non-empty --arch, e.g. --arch qwen35")
    if gguf is not None and gguf.exists() and not force:
        return _error(f"{gguf} exists; pass --force to overwrite")
    if gguf is not None and directory.is_dir() and _inside_adapter(gguf, directory):
        return _error(
            f"--gguf {gguf} would overwrite a file of the adapter in {directory}"
        )
    if card is not None and (directory / "README.md").exists() and not force:
        return _error(f"{directory / 'README.md'} exists; pass --force to overwrite")
    if not directory.is_dir():
        return _error(f"not a directory: {directory}")
    try:
        edit, problems = read_peft(directory)
    except _ERRORS as e:
        return _fail(directory, e)
    if problems:
        return _error("adapter failed verification: " + "; ".join(problems))
    written: list[str] = []
    try:
        text = model_card(edit, "adapter", card) if card is not None else None
        if gguf is not None and arch is not None:
            try:
                from astro_weights.export.gguf import save_lora_gguf
            except ImportError:
                return _error("GGUF export needs: pip install 'astro-weights[gguf]'")
            save_lora_gguf(edit, gguf, arch, overwrite=force)
            written.append(str(gguf))
        if text is not None:
            written.append(str(write_card(directory, text)))
    except _ERRORS as e:
        return _fail(directory, e)
    if as_json:
        _emit({"written": written})
    else:
        for w in written:
            print(f"wrote {w}")
    return 0


def _inside_adapter(gguf: Path, directory: Path) -> bool:
    """True when `gguf` is one of the adapter directory's own files."""
    own = (CONFIG, WEIGHTS, MANIFEST, "README.md")
    return gguf.resolve() in {(directory / n).resolve() for n in own}


def _mcp(run_dir: Path) -> int:
    try:
        from astro_weights.mcp.server import main as serve
    except ModuleNotFoundError:
        return _error("the MCP server needs: pip install 'astro-weights[mcp]'")
    return serve([str(run_dir)])


def _json_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--json", action="store_true", help="print one JSON object instead of text"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="astro-weights",
        epilog="exit codes: 0 ok, 1 problems found or check failed, "
        "2 usage error or unreadable input",
    )
    parser.add_argument(
        "--version", action="version", version=f"astro-weights {__version__}"
    )
    sub = parser.add_subparsers(
        dest="command", metavar="{build,eval,export,inspect,mcp}"
    )
    p = sub.add_parser("inspect", help="summarize and verify a saved PEFT adapter")
    p.add_argument("directory", metavar="DIR", type=Path, help="adapter directory")
    _json_flag(p)
    p = sub.add_parser("eval", help="compare two eval arms from per-item records")
    p.add_argument(
        "base", metavar="BASE.jsonl", type=Path, help="per-item records, base arm"
    )
    p.add_argument(
        "edit", metavar="EDIT.jsonl", type=Path, help="per-item records, edited arm"
    )
    p.add_argument("--boot", type=int, default=10000, help="bootstrap resamples")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.add_argument(
        "--floor",
        nargs=2,
        metavar=("A.jsonl", "B.jsonl"),
        type=Path,
        help="two runs of the base arm, to measure the noise floor",
    )
    p.add_argument(
        "--ignore-header",
        action="store_true",
        help="do not fail when the arms' X.header.json files differ",
    )
    _json_flag(p)
    p = sub.add_parser("build", help="build a PEFT adapter from a recipe file")
    p.add_argument("recipe", metavar="RECIPE.json", type=Path, help="recipe file")
    p.add_argument(
        "--base",
        metavar="MODEL_DIR",
        type=Path,
        required=True,
        help="base model directory (safetensors)",
    )
    p.add_argument(
        "--out",
        metavar="DIR",
        type=Path,
        required=True,
        help="adapter directory to write",
    )
    p.add_argument(
        "--card", metavar="LICENSE", help="also write README.md with this license"
    )
    _json_flag(p)
    p.add_argument(
        "--force",
        action="store_true",
        help="write into a non-empty DIR (other files, e.g. an old README.md, are kept)",
    )
    p = sub.add_parser(
        "export", help="write a GGUF LoRA and/or model card for an adapter"
    )
    p.add_argument("directory", metavar="DIR", type=Path, help="adapter directory")
    p.add_argument(
        "--gguf", metavar="OUT.gguf", type=Path, help="write a GGUF LoRA to this file"
    )
    p.add_argument("--arch", help="GGUF architecture, e.g. qwen35")
    p.add_argument(
        "--card", metavar="LICENSE", help="write README.md into DIR with this license"
    )
    p.add_argument("--force", action="store_true", help="overwrite existing outputs")
    _json_flag(p)
    p = sub.add_parser("mcp", help="serve the agent tools over stdio MCP")
    p.add_argument(
        "run_dir", metavar="RUN_DIR", type=Path, help="run directory with run.json"
    )
    args = parser.parse_args(argv)
    if args.command == "inspect":
        return _inspect(args.directory, args.json)
    if args.command == "eval":
        return _eval(
            args.base,
            args.edit,
            args.boot,
            args.seed,
            args.floor,
            args.ignore_header,
            args.json,
        )
    if args.command == "build":
        return _build(
            args.recipe, args.base, args.out, args.card, args.force, args.json
        )
    if args.command == "export":
        return _export(
            args.directory, args.gguf, args.arch, args.card, args.force, args.json
        )
    if args.command == "mcp":
        return _mcp(args.run_dir)
    parser.print_help()
    return 2
