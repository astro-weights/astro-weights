"""Command-line entry point."""

import argparse
import hashlib
import json
import math
import os
import secrets
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from safetensors import SafetensorError

from astro_weights import __version__
from astro_weights.edits import Edit, LowRankDelta, build_edit, compose, load_recipe
from astro_weights.eval import (
    PairedResult,
    Record,
    check_headers,
    compare,
    exceeds_floor,
    format_report,
    header_path,
    read_header,
    read_jsonl,
)
from astro_weights.eval.seal import (
    Scoring,
    SealedError,
    check_unsealed,
    record_scoring,
    records_sha256,
    scoring_of,
    seal_salt,
)
from astro_weights.eval.splits import Split, SplitManifest
from astro_weights.export import (
    load_peft,
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


def _check_run_headers(base: Path, edit: Path, arm_keys: list[str]) -> int:
    """0 when headers agree or one is missing; 1 when they differ; 2 when unreadable."""
    hb, he = header_path(base), header_path(edit)
    if not (hb.is_file() and he.is_file()):
        return 0
    try:
        headers = read_header(hb), read_header(he)
    except (ValueError, OSError) as e:
        return _error(str(e))
    try:
        check_headers(*headers, arm_keys=arm_keys)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


@dataclass(frozen=True)
class _SplitArgs:
    """``eval --split``: which split, its manifest, and for test the seal."""

    split: Split
    manifest: Path
    seal_dir: Path | None
    adapter: Path | None
    unseal: bool


@dataclass(frozen=True)
class _Seal:
    directory: Path
    scoring: Scoring
    unseal: bool
    adapter: Path
    base: str
    shown: tuple[str, ...]


def _split_args(args: argparse.Namespace) -> _SplitArgs | str | None:
    """The split flags, None when absent, or an error message."""
    if args.split is None:
        extra = [
            f
            for f, v in (
                ("--manifest", args.manifest),
                ("--seal-dir", args.seal_dir),
                ("--adapter", args.adapter),
                ("--unseal", args.unseal),
            )
            if v
        ]
        return f"{', '.join(extra)} need --split" if extra else None
    if args.manifest is None:
        return "--split needs --manifest"
    if args.split == "test":
        if args.seal_dir is None or args.adapter is None:
            return "--split test needs --seal-dir and --adapter"
    elif args.seal_dir or args.adapter or args.unseal:
        return "--seal-dir, --adapter and --unseal apply only to --split test"
    return _SplitArgs(
        args.split, args.manifest, args.seal_dir, args.adapter, args.unseal
    )


def _check_split(
    sp: _SplitArgs, arms: list[tuple[Path, list[Record]]]
) -> int | _Seal | None:
    """Refuse records outside the split; for test, check K, arms and the seal.

    ``arms`` is base, edit, then any floor arms. On test the base and floor arms
    (runs of the base model) must have ``arm == "base"`` and must not hold the
    records of an edit in the ledger. Exit code on error.
    """
    try:
        manifest = SplitManifest.load(sp.manifest)
    except (ValueError, OSError) as e:
        return _error(str(e))
    test = sp.split == "test"
    seeds: set[tuple[int, ...]] = set()
    for path, records in arms:
        try:
            manifest.check_records(records, sp.split, complete=test)
            if test:
                seeds.add(manifest.check_seeds(records))
        except ValueError as e:
            print(f"error: {path}: {e}", file=sys.stderr)
            return 1
    if not test:
        return None
    for path, records in [arms[0], *arms[2:]]:
        if bad := sorted({r.arm for r in records} - {"base"}):
            print(
                f"error: {path}: arm {bad[0]!r}; base and floor arms must be "
                "arm 'base' on --split test",
                file=sys.stderr,
            )
            return 1
    if len(seeds) > 1:
        print(
            f"error: arms were run with different seeds: {sorted(seeds)}",
            file=sys.stderr,
        )
        return 1
    assert sp.seal_dir is not None
    assert sp.adapter is not None
    try:
        salt = seal_salt(sp.seal_dir)
    except (ValueError, OSError) as e:
        return _error(str(e))
    try:
        edit = load_peft(sp.adapter)
        scoring = scoring_of(edit, manifest, arms[1][1], salt)
    except _ERRORS as e:
        return _fail(sp.adapter, e)
    base_sha = hashlib.sha256(
        json.dumps(edit.base_sha256, sort_keys=True).encode()
    ).hexdigest()
    base = f"base_model={edit.base_model!r} base_sha256_digest={base_sha[:16]}"
    shown = tuple(records_sha256(r) for _, r in [arms[0], *arms[2:]])
    seal = _Seal(sp.seal_dir, scoring, sp.unseal, sp.adapter, base, shown)
    if not seal.unseal:
        try:
            check_unsealed(seal.directory, scoring, shown)
        except SealedError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        except (ValueError, OSError) as e:
            return _error(str(e))
    return seal


def _record_seal(seal: _Seal, base: Path, edit: Path, result: PairedResult) -> int:
    """Write the ledger line before the result is shown; exit code on refusal."""
    note = (
        f"eval base={base} edit={edit} adapter={seal.adapter} {seal.base} "
        f"n={result.n_items} diff={result.diff:+.6f}"
    )
    try:
        record_scoring(seal.directory, seal.scoring, seal.unseal, note, seal.shown)
    except SealedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (ValueError, OSError) as e:
        return _error(str(e))
    if seal.unseal:
        print(
            f"warning: --unseal: test split scored again for edit "
            f"{seal.scoring.key[:12]}; recorded in {seal.directory}",
            file=sys.stderr,
        )
    return 0


def _eval(
    base: Path,
    edit: Path,
    boot: int,
    seed: int,
    floor: list[Path] | None,
    ignore_header: bool,
    as_json: bool,
    arm_keys: list[str] | None = None,
    split: _SplitArgs | None = None,
) -> int:
    if not ignore_header and (code := _check_run_headers(base, edit, arm_keys or [])):
        return code
    arms = [_read_arm(p) for p in (base, edit, *(floor or ()))]
    if any(a is None for a in arms):
        return 2
    seal = None
    if split is not None:
        named = zip((base, edit, *(floor or ())), arms, strict=True)
        checked = _check_split(split, [(p, a or []) for p, a in named])
        if isinstance(checked, int):
            return checked
        seal = checked
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
    if seal is not None and (code := _record_seal(seal, base, edit, result)):
        return code
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


def _compose(
    adapters: list[Path], out: Path, force: bool, as_json: bool, base_model: str | None
) -> int:
    for a in adapters:
        if out.resolve().is_relative_to(a.resolve()):
            return _error(f"-o {out} is the input adapter {a} or inside it")
    if out.exists() and (not out.is_dir() or any(out.iterdir())) and not force:
        return _error(f"{out} exists and is not empty; pass --force to overwrite")
    edits = []
    for a in adapters:
        try:
            e = load_peft(a)
        except _ERRORS as err:
            return _fail(a, err)
        edits.append(e if base_model is None else replace(e, base_model=base_model))
    try:
        edit = compose(*edits)
        save_peft(edit, out, overwrite=force)
    except _ERRORS as e:
        return _fail(out, e)
    if as_json:
        _emit(
            {
                "written": str(out),
                "rank": edit.rank,
                "modules": sorted(d.module for d in edit.deltas),
                "sources": [str(a) for a in adapters],
            }
        )
        return 0
    print(
        f"composed {len(adapters)} adapters: {len(edit.deltas)} modules, rank {edit.rank}"
    )
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
        if out_type == "q8_0" and d.shape[1] % 32:
            raise ValueError(
                f"{d.module}: q8_0 needs a row length divisible by 32, got "
                f"{d.shape[1]}; use --out-type f16"
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
        if gguf is not None and arch is not None:
            try:
                from astro_weights.export.gguf import (
                    _lora_tensor_names,
                    save_lora_gguf,
                )
            except ImportError:
                return _error("GGUF export needs: pip install 'astro-weights[gguf]'")
            try:
                _lora_tensor_names(edit, arch)  # refuse before anything is written
            except ValueError as e:
                return _error(str(e))
        if splice is not None and base_model is not None and merged is not None:
            try:
                path, replaced = _splice_merged(edit, base_model, splice, merged, kind)
            except ValueError as e:
                return _error(str(e))
            written.append(path)
        if gguf is not None and arch is not None:
            save_lora_gguf(edit, gguf, arch, overwrite=force)
            written.append(str(gguf))
        hashes = {w: _file_sha256(Path(w)) for w in written if w.endswith(".gguf")}
        if text is not None:
            written.append(str(write_card(directory, text)))
    except _ERRORS as e:
        return _fail(directory, e)
    if as_json:
        extra: dict[str, object] = (
            {"replaced": replaced, "out_type": kind} if splice is not None else {}
        )
        _emit({"written": written, "sha256": hashes, **extra})
    else:
        for w in written:
            print(f"wrote {w}" + (f"  sha256 {hashes[w]}" if w in hashes else ""))
    return 0


def _file_sha256(path: Path) -> str:
    """sha256 of a written GGUF, so the file can be pinned."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _inside_adapter(gguf: Path, directory: Path) -> bool:
    """True when `gguf` is one of the adapter directory's own files."""
    own = (CONFIG, WEIGHTS, MANIFEST, "README.md")
    return gguf.resolve() in {(directory / n).resolve() for n in own}


def _mcp(run_dir: Path, unsafe_kernel: bool) -> int:
    try:
        from astro_weights.mcp.server import main as serve
    except ModuleNotFoundError:
        return _error("the MCP server needs: pip install 'astro-weights[mcp]'")
    return serve([str(run_dir), *(["--unsafe-kernel"] if unsafe_kernel else [])])


def _run(config: Path, resume: bool, show_status: bool, as_json: bool) -> int:
    from astro_weights.mcp import run

    try:
        spec = run.load_config(config)
        if not show_status:
            return run.start(spec, resume=resume)
        info = run.status(spec)
    except run.RunError as e:
        return _error(str(e))
    if as_json:
        _emit(info)
    else:
        print(run.format_status(info))
    return 0


def _self_test(
    base_url: str, model: str, prompt: str, key_env: str | None, as_json: bool
) -> int:
    from astro_weights.runners.endpoint import self_test_http

    key = os.environ.get(key_env) if key_env else None
    if key_env and not key:
        return _error(f"${key_env} is not set")
    try:
        r = self_test_http(base_url, model, prompt, api_key=key)
    except (ValueError, RuntimeError) as e:
        return _error(str(e))
    if as_json:
        _emit(asdict(r))
    else:
        for label, out in r.outputs.items():
            print(f"{label}: {out[:60]!r}")
        verdict = "deterministic" if r.deterministic else "NOT deterministic"
        print(
            f"{verdict}: {r.unique_outputs} unique output(s), {r.n_requests} requests"
        )
        if r.first_divergence:
            a, b, i = r.first_divergence
            print(f"first divergence: {a} vs {b} at character {i}")
    return 0 if r.deterministic else 1


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
        dest="command", metavar="{build,compose,eval,export,inspect,mcp,run,self-test}"
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
    p.add_argument(
        "--arm-key",
        action="append",
        default=[],
        metavar="KEY",
        dest="arm_keys",
        help="extra_body key allowed to differ between the arms (repeatable)",
    )
    p.add_argument(
        "--split",
        choices=("dev", "confirm", "test"),
        help="refuse records outside this split of --manifest; test is sealed",
    )
    p.add_argument(
        "--manifest", metavar="SPLITS.json", type=Path, help="split manifest file"
    )
    p.add_argument(
        "--seal-dir",
        metavar="DIR",
        type=Path,
        help="with --split test: directory of the seal ledger (seal.jsonl)",
    )
    p.add_argument(
        "--adapter",
        metavar="DIR",
        type=Path,
        help="with --split test: the edit's adapter directory, keyed by content",
    )
    p.add_argument(
        "--unseal",
        action="store_true",
        help="with --split test: score an edit again; the ledger records it",
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
        "compose", help="combine PEFT adapters on one base into one exact adapter"
    )
    p.add_argument(
        "adapters", metavar="ADAPTER", type=Path, nargs="+", help="adapter directory"
    )
    p.add_argument(
        "-o",
        "--out",
        metavar="OUT_DIR",
        type=Path,
        required=True,
        help="combined adapter directory to write",
    )
    p.add_argument(
        "--base-model",
        metavar="NAME",
        help="base model name to write, replacing each adapter's own "
        "(for adapters that name one base differently)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="write into a non-empty OUT_DIR (other files there are kept)",
    )
    _json_flag(p)
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
    p = sub.add_parser("mcp", help="serve the agent tools over stdio MCP")
    p.add_argument(
        "run_dir", metavar="RUN_DIR", type=Path, help="run directory with run.json"
    )
    p.add_argument(
        "--unsafe-kernel",
        action="store_true",
        help="run agent code without an OS sandbox when none works (never safe)",
    )
    p = sub.add_parser(
        "run", help="start an agent run from a TOML config: server, editor, budget"
    )
    p.add_argument("config", metavar="CONFIG.toml", type=Path, help="run config file")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="go on with the run in the config's run directory; budget used stays",
    )
    mode.add_argument(
        "--status",
        action="store_true",
        help="print the run's budget, last events and finish state; starts nothing",
    )
    _json_flag(p)
    p = sub.add_parser(
        "self-test",
        help="send one prompt under varied batch and prefix conditions; expect one output",
    )
    p.add_argument("--base-url", required=True, help="OpenAI-compatible server URL")
    p.add_argument("--model", required=True, help="served model name")
    p.add_argument(
        "--prompt",
        default="Count from 1 to 20, separated by commas.",
        help="the prompt to repeat (default: a short counting prompt)",
    )
    p.add_argument(
        "--api-key-env", metavar="VAR", help="environment variable holding the API key"
    )
    _json_flag(p)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        return _inspect(args.directory, args.json)
    if args.command == "eval":
        split = _split_args(args)
        if isinstance(split, str):
            return _error(split)
        return _eval(
            args.base,
            args.edit,
            args.boot,
            args.seed,
            args.floor,
            args.ignore_header,
            args.json,
            args.arm_keys,
            split,
        )
    if args.command == "build":
        return _build(
            args.recipe, args.base, args.out, args.card, args.force, args.json
        )
    if args.command == "compose":
        if len(args.adapters) < 2:
            parser.error("compose needs at least two adapters")
        return _compose(args.adapters, args.out, args.force, args.json, args.base_model)
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
    if args.command == "mcp":
        return _mcp(args.run_dir, args.unsafe_kernel)
    if args.command == "run":
        return _run(args.config, args.resume, args.status, args.json)
    if args.command == "self-test":
        return _self_test(
            args.base_url, args.model, args.prompt, args.api_key_env, args.json
        )
    parser.print_help()
    return 2
