"""Command-line entry point."""

import argparse
import json
import math
import secrets
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from safetensors import SafetensorError

from astro_weights import __version__
from astro_weights.edits import Edit, LowRankDelta, build_edit, load_recipe
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
    tensor_sha256,
    write_card,
)
from astro_weights.export.base import _shard
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


def _splice_merged(
    edit: Edit, base_model: Path, splice: Path, merged: Path, out_type: str
) -> tuple[str, list[str]]:
    """Write `splice` with the edit merged in to `merged`; return (path, tensor names).

    Raises ValueError (message ready to print) before any file exists, except the
    temp file, which is removed on failure.
    """
    try:
        from astro_weights.export.gguf import (
            gguf_tensor_name,
            splice_gguf,
            verify_splice,
        )
    except ImportError as e:
        raise ValueError("--splice needs: pip install 'astro-weights[gguf]'") from e
    names: dict[str, str] = {}
    for d in edit.deltas:
        names[d.module] = gguf_tensor_name(d.module)
    if len(set(names.values())) != len(names):
        raise ValueError("two modules map to the same GGUF tensor")
    replace: dict[str, np.ndarray] = {}
    for d in edit.deltas:
        key = d.module + ".weight"
        want = edit.base_sha256.get(key)
        if want is not None and tensor_sha256(_shard(base_model, key), key) != want:
            raise ValueError(
                f"{d.module}: base weight in {base_model} does not match the "
                "adapter's recorded base hash; wrong base model?"
            )
        w0 = read_base_weight(base_model, key)
        if w0.shape != d.shape:
            raise ValueError(
                f"{d.module}: base weight shape {w0.shape} vs adapter delta {d.shape}"
            )
        replace[names[d.module]] = w0 + d.dense()
    tmp = merged.with_name(f"{merged.name}.{secrets.token_hex(4)}.tmp")
    try:
        splice_gguf(splice, tmp, replace, out_type)
        problems = verify_splice(splice, tmp, list(replace))
        if problems:
            raise ValueError("spliced GGUF failed verification: " + "; ".join(problems))
        tmp.replace(merged)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return str(merged), sorted(replace)


def _same_path(a: Path, b: Path) -> bool:
    """True when `a` and `b` are one file (samefile if both exist, else resolved)."""
    if a.exists() and b.exists():
        return a.samefile(b)
    return a.resolve() == b.resolve()


def _splice_args_error(
    directory: Path,
    gguf: Path | None,
    splice: Path | None,
    base_model: Path | None,
    merged: Path | None,
    out_type: str | None,
    force: bool,
) -> str | None:
    """First problem with the --splice flags, or None."""
    if splice is None:
        if base_model or merged or out_type:
            return "--base-model, --merged and --out-type need --splice"
        return None
    if base_model is None or merged is None:
        return "--splice needs --base-model DIR and --merged OUT.gguf"
    if not splice.is_file():
        return f"--splice {splice} is not a file"
    if _same_path(merged, splice):
        return f"--merged {merged} is the --splice base file"
    if gguf is not None and _same_path(gguf, merged):
        return f"--gguf and --merged are both {merged}"
    if gguf is not None and _same_path(gguf, splice):
        return f"--gguf {gguf} is the --splice base file"
    if not merged.resolve().parent.is_dir():
        return f"--merged {merged}: directory {merged.parent} does not exist"
    if merged.exists() and not force:
        return f"{merged} exists; pass --force to overwrite"
    if merged.resolve().is_relative_to(directory.resolve()):
        return f"--merged {merged} is inside the adapter directory {directory}"
    return None


def _export(
    directory: Path,
    gguf: Path | None,
    arch: str | None,
    card: str | None,
    force: bool,
    as_json: bool,
    splice: Path | None = None,
    base_model: Path | None = None,
    merged: Path | None = None,
    out_type: str | None = None,
) -> int:
    if gguf is None and card is None and splice is None:
        return _error("nothing to export; pass --gguf, --card and/or --splice")
    bad = _splice_args_error(
        directory, gguf, splice, base_model, merged, out_type, force
    )
    if bad is not None:
        return _error(bad)
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
    replaced: list[str] = []
    kind = out_type or "q8_0"
    try:
        text = model_card(edit, "adapter", card) if card is not None else None
        if splice is not None and base_model is not None and merged is not None:
            try:
                path, replaced = _splice_merged(edit, base_model, splice, merged, kind)
            except ValueError as e:
                return _error(str(e))
            written.append(path)
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
        extra: dict[str, object] = (
            {"replaced": replaced, "out_type": kind} if splice is not None else {}
        )
        _emit({"written": written, **extra})
    else:
        for w in written:
            print(f"wrote {w}")
    return 0


def _inside_adapter(gguf: Path, directory: Path) -> bool:
    """True when `gguf` is one of the adapter directory's own files."""
    own = (CONFIG, WEIGHTS, MANIFEST, "README.md")
    return gguf.resolve() in {(directory / n).resolve() for n in own}


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
    sub = parser.add_subparsers(dest="command", metavar="{build,eval,export,inspect}")
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
        "export",
        help="write a GGUF LoRA, a merged GGUF and/or model card for an adapter",
    )
    p.add_argument("directory", metavar="DIR", type=Path, help="adapter directory")
    p.add_argument(
        "--gguf", metavar="OUT.gguf", type=Path, help="write a GGUF LoRA to this file"
    )
    p.add_argument("--arch", help="GGUF architecture, e.g. qwen35")
    p.add_argument(
        "--card", metavar="LICENSE", help="write README.md into DIR with this license"
    )
    p.add_argument(
        "--splice",
        metavar="BASE.gguf",
        type=Path,
        help="write a merged GGUF: this base GGUF with the adapter folded in",
    )
    p.add_argument(
        "--base-model",
        metavar="DIR",
        type=Path,
        help="with --splice: the base model directory (safetensors)",
    )
    p.add_argument(
        "--merged",
        metavar="OUT.gguf",
        type=Path,
        help="with --splice: the merged GGUF to write",
    )
    p.add_argument(
        "--out-type",
        choices=("q8_0", "f16", "bf16"),
        help="with --splice: storage type of the changed tensors (default q8_0)",
    )
    p.add_argument("--force", action="store_true", help="overwrite existing outputs")
    _json_flag(p)
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
            args.directory,
            args.gguf,
            args.arch,
            args.card,
            args.force,
            args.json,
            args.splice,
            args.base_model,
            args.merged,
            args.out_type,
        )
    parser.print_help()
    return 2
