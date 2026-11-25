"""Command-line entry point."""

import argparse
import hashlib
import json
import math
import os
import secrets
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import SafetensorError

from astro_weights import __version__
from astro_weights.edits import Edit, build_edit, compose, load_recipe
from astro_weights.edits.norms import delta_norm
from astro_weights.eval import (
    ArmsResult,
    ControlVerdict,
    FloorResult,
    GateInputs,
    GateResult,
    PairedResult,
    Record,
    Screen,
    Verdict,
    arms_dict,
    check_headers,
    compare,
    compare_arms,
    config_hash,
    control_rule,
    exceeds_floor,
    format_arms,
    format_gate,
    format_report,
    gate,
    header_path,
    load_floor,
    read_header,
    read_jsonl,
)
from astro_weights.eval.arm_config import ArmConfig, load_arm_config, load_items
from astro_weights.eval.gate import K_TEST, SCREENED_OUT
from astro_weights.eval.gate_config import (
    REPORT_ARMS,
    GateConfig,
    load_gate_config,
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
                        "norm_BA": delta_norm(d),
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
            f"{nb:>8.3g}  {na:>8.3g}  {delta_norm(d):>8.3g}"
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


class _Refused(Exception):
    """A helper already printed why the gate run is refused."""


# gate outcome -> exit code; refusals and bad input are 2
_GATE_EXIT = {"confirmed": 0, "not significant": 1, "regressed": 1, SCREENED_OUT: 3}


def _records(path: Path) -> list[Record]:
    try:
        return read_jsonl(path)
    except (ValueError, OSError) as e:
        raise ValueError(f"{path}: {e}") from None


def _header(path: Path) -> dict[str, object]:
    return read_header(header_path(path))


def _same_settings(
    base: Path, other: Path, arm_keys: tuple[str, ...], ignore: tuple[str, ...] = ()
) -> None:
    """ValueError unless both arms have a run header and agree on every config key."""
    try:
        check_headers(_header(base), _header(other), ignore=ignore, arm_keys=arm_keys)
    except ValueError as e:
        raise ValueError(f"{other} vs {base}: {e}") from None


_SYSTEM_ARMS = ("prompt", "edit+prompt")  # report arms made with a system prompt
_SYSTEM_KEYS = ("system", "system_sha256")


def _check_system(path: Path) -> None:
    """ValueError unless the header at `path` records a non-empty system and its hash."""
    h = _header(path)
    system = h.get("system")
    if not isinstance(system, str) or not system:
        raise ValueError(f"{path}: a prompt arm needs a system prompt in its header")
    if h.get("system_sha256") != hashlib.sha256(system.encode("utf-8")).hexdigest():
        raise ValueError(f"{path}: system_sha256 does not match system")


def _same_settings_test(cfg: GateConfig, shown: Mapping[str, Path]) -> None:
    """Test-split arms agree with base on every config key. The prompt arms differ in
    the system prompt, which they must have; base, edit and scripted must not."""
    t = cfg.test
    if "system" in _header(t.base):
        raise ValueError(f"{t.base}: the base arm must run without a system prompt")
    _same_settings(t.base, t.edit, cfg.arm_keys)
    for name, p in shown.items():
        if name in _SYSTEM_ARMS:
            _check_system(p)
            _same_settings(t.base, p, cfg.arm_keys, ignore=_SYSTEM_KEYS)
        else:
            _same_settings(t.base, p, cfg.arm_keys)


def _read_screens(path: Path) -> tuple[Screen, ...]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
    keys = {"name", "passed", "value", "limit"}
    if not isinstance(raw, list) or not all(
        isinstance(r, dict) and set(r) == keys for r in raw
    ):
        raise ValueError(f"{path}: must be a JSON list of {sorted(keys)} objects")
    return tuple(Screen(r["name"], r["passed"], r["value"], r["limit"]) for r in raw)


def _search_counts(path: Path) -> tuple[int, int]:
    """(candidates tried, confirm queries) from the agent run record.

    ValueError unless the record's last `confirm_export` says the exported edits are
    the last that passed confirm.
    """
    from astro_weights.mcp.record import RunRecord

    record = RunRecord.open(path)
    exports = [e for e in record.read() if e["event"] == "confirm_export"]
    if not exports or exports[-1].get("exported_is_best") is not True:
        raise ValueError(
            f"{path}: exported_is_best is not true (no finished run with confirm, or "
            "the exported edits are not the last that passed confirm)"
        )
    counts = record.confirm_counts()
    return counts["candidates_tried"], counts["confirm_queries"]


def _confirm_arms(
    cfg: GateConfig, base: Path, edit: Path, controls: Path
) -> ArmsResult:
    """The edit and every control in `controls` against `base` on the confirm split."""
    if not controls.is_dir():
        raise ValueError(f"{controls}: not a directory of control arm files")
    files = sorted(controls.glob("*.jsonl"))
    if bad := {"base", "edit"} & {p.stem for p in files}:
        raise ValueError(f"{controls}: a control file cannot be named {min(bad)}.jsonl")
    for p in (edit, *files):
        _same_settings(base, p, cfg.arm_keys)
    named = [(p, _records(p)) for p in (base, edit, *files)]
    split = _SplitArgs("confirm", cfg.manifest, None, None, False)
    if _check_split(split, named) is not None:
        raise _Refused
    arms = {"edit": named[1][1], **{p.stem: r for p, r in named[2:]}}
    kinds = {p.stem: "control" for p in files}
    return compare_arms(named[0][1], arms, kinds, n_boot=cfg.boot, seed=cfg.seed)


def _side_arms(cfg: GateConfig) -> dict[str, ArmsResult]:
    """Each `[confirm.side.NAME]` table's arms on the confirm split, by split name."""
    return {
        f"side.{name}": _confirm_arms(cfg, f.base, f.edit, f.controls)
        for name, f in cfg.confirm.side.items()
        if name != "accuracy"  # accuracy uses the confirm records, as _controls does
    }


def _controls(
    cfg: GateConfig, arms: ArmsResult, side: Mapping[str, ArmsResult]
) -> ControlVerdict:
    """The control rule on the confirm split, with every named side effect."""
    c = cfg.confirm
    sides = {}
    for name in c.side_effects:
        s = arms if name == "accuracy" else side[f"side.{name}"]
        sides[name] = (s.results["edit"], list(s.controls().values()))
    return control_rule(
        arms.results["edit"],
        list(arms.controls().values()),
        target=c.target,
        direction=c.direction,  # type: ignore[arg-type]
        side_effects=sides,
        q=c.q,
    )


def _infra_errors(*paths: Path) -> int:
    """Failed and retried requests the HTTP runner recorded in these arms' headers."""
    total = 0
    for p in paths:
        counts = _header(p).get("infra_errors")
        if isinstance(counts, dict):
            total += sum(
                v
                for v in (counts.get("retries"), counts.get("failed"))
                if isinstance(v, int) and not isinstance(v, bool)
            )
    return total


@dataclass(frozen=True)
class _TestStage:
    named: list[tuple[Path, list[Record]]]
    arms: ArmsResult
    floor: FloorResult
    suites: dict[str, tuple[PairedResult, float, str]]
    suite_arms: dict[str, ArmsResult]
    k: int
    config_hash: str
    infra_errors: int
    report: list[tuple[Path, list[Record]]]


def _test_stage(cfg: GateConfig, confirm_base: Path, unseal: bool) -> _TestStage:
    """Read and check everything the test stage needs; nothing is sealed yet."""
    t = cfg.test
    manifest = SplitManifest.load(cfg.manifest)
    if manifest.k != K_TEST:
        raise ValueError(
            f"{cfg.manifest}: k is {manifest.k}, the test needs K={K_TEST}"
        )
    shown = {
        name: cfg.arms[key] for key, (name, _) in REPORT_ARMS.items() if key in cfg.arms
    }
    _same_settings_test(cfg, shown)
    _same_settings(confirm_base, t.base, cfg.arm_keys, ignore=("k",))
    key = config_hash(_header(t.base))
    floor = load_floor(t.floor, key)
    if floor is None:
        raise ValueError(f"{t.floor}: no floor for config_hash {key}; measure it first")
    suites, suite_arms = {}, {}
    for name, s in cfg.suites.items():
        _same_settings(s.base, s.edit, cfg.arm_keys)
        edit_vs_base = compare_arms(
            _records(s.base), {"edit": _records(s.edit)}, n_boot=cfg.boot, seed=cfg.seed
        )
        suites[name] = (edit_vs_base.results["edit"], s.margin, s.margin_source)
        suite_arms[f"suite.{name}"] = edit_vs_base
    base, edit = _records(t.base), _records(t.edit)
    arms = {"edit": edit} | {n: _records(p) for n, p in shown.items()}
    kinds = {n: REPORT_ARMS[k][1] for k, (n, _) in REPORT_ARMS.items() if n in shown}
    compared = compare_arms(base, arms, kinds, n_boot=cfg.boot, seed=cfg.seed)
    infra = _infra_errors(t.base, t.edit)
    named = [(t.base, base), (t.edit, edit)]
    report = [(p, arms[n]) for n, p in shown.items()]
    return _TestStage(
        named, compared, floor, suites, suite_arms, manifest.k, key, infra, report
    )


def _open_seal(cfg: GateConfig, stage: _TestStage, unseal: bool) -> _Seal:
    """The seal for the test scoring: the same checks as `eval --split test`."""
    t = cfg.test
    split = _SplitArgs("test", cfg.manifest, t.seal_dir, t.adapter, unseal)
    seal = _check_split(split, stage.named)
    if not isinstance(seal, _Seal):
        raise _Refused  # printed by _check_split
    if not stage.report:
        return seal
    # Report arms print test-split results too: same split checks, and their
    # records count as shown, so a sealed edit's records can't be re-shown here.
    manifest = SplitManifest.load(cfg.manifest)
    for path, records in stage.report:
        try:
            manifest.check_records(records, "test", complete=True)
            manifest.check_seeds(records)
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from e
    extra = tuple(records_sha256(r) for _, r in stage.report)
    seal = replace(seal, shown=seal.shown + extra)
    if not seal.unseal:
        try:
            check_unsealed(seal.directory, seal.scoring, seal.shown)
        except SealedError as e:
            print(f"error: {e}", file=sys.stderr)
            raise _Refused from e
    return seal


def _screens_failed(cfg: GateConfig, screens: tuple[Screen, ...]) -> GateResult | None:
    """The screened-out result when any pre-screen failed, else None."""
    if all(s.passed for s in screens):
        return None
    return gate(GateInputs(screens, cfg.margin, cfg.margin_source))


def _cap_hits(arms: ArmsResult) -> dict[str, float]:
    """Share of records that hit the generation cap: base, then each arm in order."""
    first = next(iter(arms.results.values()))
    return {"base": first.cap_hit_base} | {
        n: r.cap_hit_edit for n, r in arms.results.items()
    }


def _cap_line(arms: ArmsResult) -> str:
    """One report line: cap-hit of base, edit and report arms; controls as their max."""
    ctl = arms.controls()
    hits = {n: v for n, v in _cap_hits(arms).items() if n not in ctl}
    line = "  ".join(f"{n}={v:.1%}" for n, v in hits.items())
    if ctl:
        worst = max(ctl, key=lambda n: ctl[n].cap_hit_edit)
        line += f"  controls max={ctl[worst].cap_hit_edit:.1%} ({worst})"
    return line


def _cap_guard(limit: float | None, splits: Mapping[str, ArmsResult]) -> None:
    """Refuse (exit 2, no verdict) when any arm's cap-hit share is above `limit`."""
    if limit is None:
        return
    over = {
        f"{split} {n}": v
        for split, arms in splits.items()
        for n, v in _cap_hits(arms).items()
        if v > limit
    }
    if over:
        shown = ", ".join(f"{n} {v:.2%}" for n, v in over.items())
        print(
            f"error: cap-hit guard: {shown} above max_cap_hit "
            f"{limit:.2%}; answers cut at the cap are censored, not wrong. Raise "
            "max_new_tokens and rerun those arms. No verdict written.",
            file=sys.stderr,
        )
        raise _Refused


def _gate_output(
    result: GateResult,
    counts: tuple[int, int],
    confirm: ArmsResult | None,
    test: ArmsResult | None,
    written: Path | None,
    as_json: bool,
    cap: int | None = None,
    cap_splits: Mapping[str, ArmsResult] | None = None,
) -> None:
    searched = {"candidates_tried": counts[0], "confirm_queries": counts[1]}
    splits = {"confirm": confirm, "test": test}
    cap_splits = cap_splits or {}
    if as_json:
        _emit(
            {
                **asdict(result),
                **searched,
                "verdict_file": None if written is None else str(written),
                "arms": {
                    k: None if a is None else arms_dict(a) for k, a in splits.items()
                },
                "max_new_tokens": cap,
                "cap_hit": {k: _cap_hits(a) for k, a in cap_splits.items()},
            }
        )
        return
    print(format_gate(result), end="")
    print(f"candidates tried: {counts[0]}  confirm queries: {counts[1]}")
    for name, arms in splits.items():
        if arms is not None:
            print(f"\narms on the {name} split, each against base:")
            print(format_arms(arms))
    if cap_splits:
        where = "cap unknown" if cap is None else f"max_new_tokens {cap}"
        print(f"\ncap-hit share ({where} in the confirm base header):")
    for name, arms in cap_splits.items():
        print(f"  {name}: {_cap_line(arms)}")
    print(f"verdict file: {written}" if written else "no verdict file written")


def _gate_run(config: Path, as_json: bool, unseal: bool, out: Path | None) -> int:
    cfg = load_gate_config(config)
    counts = _search_counts(cfg.run_record)
    screens = _read_screens(cfg.screens)
    if (screened := _screens_failed(cfg, screens)) is not None:
        _gate_output(screened, counts, None, None, None, as_json)
        return _GATE_EXIT[screened.outcome]
    c = cfg.confirm
    confirm = _confirm_arms(cfg, c.base, c.edit, c.controls)
    side = _side_arms(cfg)
    splits = {"confirm": confirm, **side}
    _cap_guard(cfg.max_cap_hit, splits)
    controls = _controls(cfg, confirm, side)
    inputs = GateInputs(
        screens,
        cfg.margin,
        cfg.margin_source,
        confirm=confirm.results["edit"],
        controls=controls,
        control_target=c.target,
    )
    stage = None
    if controls.passed:  # a failed confirm never spends the sealed test split
        stage = _test_stage(cfg, c.base, unseal)
        later = {"test": stage.arms, **stage.suite_arms}
        _cap_guard(cfg.max_cap_hit, later)
        splits |= later
        inputs = replace(
            inputs,
            test=stage.arms.results["edit"],
            floor=stage.floor,
            k_test=stage.k,
            suites=stage.suites,
        )
    result = gate(inputs)  # pure; every input is judged before the seal is touched
    key = stage.config_hash if stage else config_hash(_header(c.base))
    cap = _header(c.base).get("max_new_tokens")
    cap = cap if isinstance(cap, int) and not isinstance(cap, bool) else None
    verdict = Verdict(
        result.outcome,
        cfg.backend,
        cfg.build,
        key,
        SplitManifest.load(cfg.manifest).manifest_sha256,
        counts[0],
        counts[1],
        stage.infra_errors if stage else _infra_errors(c.base, c.edit),
        dict(cfg.footprint),
        cap,
        {k: _cap_hits(a) for k, a in splits.items()},
    )
    written = out or config.parent / "verdict.json"
    # Write first, rename after sealing: a bad --out never spends the test split.
    tmp = written.with_name(f".{written.name}.tmp")
    tmp.write_text(verdict.to_json(), encoding="utf-8")
    try:
        if stage is not None:
            seal = _open_seal(cfg, stage, unseal)
            t = cfg.test
            if _record_seal(seal, t.base, t.edit, stage.arms.results["edit"]):
                raise _Refused  # printed by _record_seal
        tmp.replace(written)
    finally:
        tmp.unlink(missing_ok=True)
    _gate_output(
        result,
        counts,
        confirm,
        stage.arms if stage else None,
        written,
        as_json,
        cap,
        splits,
    )
    return _GATE_EXIT[result.outcome]


def _gate(config: Path, as_json: bool, unseal: bool, out: Path | None) -> int:
    try:
        return _gate_run(config, as_json, unseal, out)
    except _Refused:
        return 2
    except _ERRORS as e:
        return _error(str(e))


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


class _Spy:
    """Whether the current runner call has sent a request for an item yet."""

    started = False


@dataclass(frozen=True)
class _GenPlan:
    """What one `gen` run writes (`paths`, `total` items or None) and how to run it.

    `counted` are the files that hold each record once (`paths` also has the joined
    `out` of a K > 1 arm). `spy` tells a failure before the first request (a config
    error or a refused resume) from one after it (a server's malformed reply).
    """

    paths: tuple[Path, ...]
    counted: tuple[Path, ...]
    total: int | None
    run: Callable[[], int]  # returns the records it holds at the end
    spy: _Spy


@contextmanager
def _watch(spy: _Spy) -> Iterator[None]:
    """Set `spy.started` when the runner inside sends its first item request.

    The header's `Hi` probe is not one. Wraps `endpoint._chat`, which every runner
    sends its requests through, and puts it back afterwards.
    """
    from astro_weights.runners import endpoint

    real = endpoint._chat
    probe = [{"role": "user", "content": "Hi"}]

    def chat(url: str, body: dict[str, Any], *args: Any, **kw: Any) -> Any:
        if body.get("messages") != probe:
            spy.started = True
        return real(url, body, *args, **kw)

    spy.started = False
    endpoint._chat = chat
    try:
        yield
    finally:
        endpoint._chat = real


def _whole_lines(paths: tuple[Path, ...]) -> int:
    return sum(p.read_bytes().count(b"\n") for p in paths if p.exists())


def _api_key(cfg: ArmConfig) -> str | None:
    if cfg.api_key_env is None:
        return None
    key = os.environ.get(cfg.api_key_env)
    if not key:
        raise ValueError(f"${cfg.api_key_env} is not set")
    return key


def _grader(name: str) -> Callable[[str, str], bool]:
    from astro_weights.eval import grade_choice, grade_exact, grade_math

    if name == "math":
        import importlib.util

        if importlib.util.find_spec("math_verify") is None:
            raise ValueError("grader 'math' needs: pip install 'astro-weights[eval]'")
    return {"exact": grade_exact, "math": grade_math, "choice": grade_choice}[name]


def _header_for(
    cfg: ArmConfig, key: str | None, seed: int, k: int
) -> dict[str, object]:
    from astro_weights.runners.endpoint import run_header_http

    return run_header_http(
        cfg.base_url, cfg.model, cfg.max_new_tokens, seed, cfg.extra_body,
        api_key=key, timeout=cfg.timeout, k=k, concurrency=cfg.concurrency,
        serving_flags=cfg.serving_flags, model_id=cfg.model_id, system=cfg.system,
    )  # fmt: skip


def _plan_http(cfg: ArmConfig, resume: bool) -> _GenPlan:
    from astro_weights.eval import (
        header_path,
        read_header,
        read_jsonl,
        write_header,
        write_jsonl,
    )
    from astro_weights.runners.endpoint import generate_records_http

    assert cfg.out is not None
    assert cfg.grader is not None
    items, k = load_items(cfg)
    scorer = _grader(cfg.grader)
    key = _api_key(cfg)
    out = cfg.out
    spy = _Spy()
    parts = (
        [out] if k == 1 else [out.with_name(f"{out.stem}.s{s}.jsonl") for s in range(k)]
    )

    def run() -> int:
        out.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        for s, part in enumerate(parts):
            seed = cfg.seed + s
            header = _header_for(cfg, key, seed, k)
            with _watch(spy):
                got = generate_records_http(
                    cfg.base_url, cfg.model, items, cfg.arm, scorer, cfg.max_new_tokens,
                    seed=seed, api_key=key, timeout=cfg.timeout,
                    extra_body=cfg.extra_body, out=part, concurrency=cfg.concurrency,
                    resume=resume, header=header, system=cfg.system,
                )  # fmt: skip
            n += len(got)
        if k > 1:  # the seeds joined into `out`, with seed 0's header (README run_k4)
            tmp = out.with_name(out.name + ".tmp")
            write_jsonl(tmp, [r for p in parts for r in read_jsonl(p)])
            tmp.replace(out)
            write_header(header_path(out), read_header(header_path(parts[0])))
        return n

    paths = (out, *parts[k == 1 :])
    return _GenPlan(paths, tuple(parts), len(items) * k, run, spy)


def _plan_suites(cfg: ArmConfig, resume: bool) -> _GenPlan:
    from astro_weights.runners.suites_run import HttpBackend, run_suites

    out_dir = cfg.out_dir
    assert out_dir is not None
    key = _api_key(cfg)
    backend = HttpBackend(
        cfg.base_url, cfg.model, key, cfg.concurrency, cfg.extra_body or None
    )
    paths = tuple(out_dir / n / f"{cfg.arm}.jsonl" for n in cfg.suites)
    spy = _Spy()

    def run() -> int:
        with _watch(spy):
            run_suites(
                list(cfg.suites), cfg.arm, out_dir, backend=backend,
                max_new_tokens=cfg.max_new_tokens, seed=cfg.seed, limit=cfg.limit,
                cache_dir=cfg.cache_dir, suite_paths=cfg.suite_paths or None,
                resume=resume, system=cfg.system,
            )  # fmt: skip
        return _whole_lines(paths)

    return _GenPlan(paths, paths, None, run, spy)


def _plan_bfcl(cfg: ArmConfig, resume: bool) -> _GenPlan:
    from astro_weights.runners.bfcl import load_bfcl, run_bfcl

    assert cfg.out is not None
    assert cfg.bfcl_root is not None
    out = cfg.out
    key = _api_key(cfg)
    tasks = load_bfcl(cfg.bfcl_root)
    spy = _Spy()

    def run() -> int:
        out.parent.mkdir(parents=True, exist_ok=True)
        header = _header_for(cfg, key, cfg.seed, cfg.trials)  # k = trials, as K = 4
        with _watch(spy):
            got = run_bfcl(
                tasks, cfg.base_url, cfg.model, cfg.arm, trials=cfg.trials,
                max_new_tokens=cfg.max_new_tokens,
                max_steps_per_turn=cfg.max_steps_per_turn, seed=cfg.seed,
                api_key=key, extra_body=cfg.extra_body, timeout=cfg.timeout,
                out=out, resume=resume, header=header,
            )  # fmt: skip
        return len(got)

    return _GenPlan((out,), (out,), len(tasks) * cfg.trials, run, spy)


def _gen(config: Path, resume: bool, as_json: bool) -> int:
    planners = {"http": _plan_http, "suites": _plan_suites, "bfcl": _plan_bfcl}
    try:
        cfg = load_arm_config(config)
        plan = planners[cfg.kind](cfg, resume)
        if not resume and (
            old := [p for p in plan.paths if p.exists() and p.stat().st_size]
        ):
            raise FileExistsError(
                f"{old[0]} already has records; pass --resume to continue it "
                "or choose another path"
            )
    except (*_ERRORS, ImportError) as e:
        return _fail(config, e)
    shown = cfg.out_dir if cfg.kind == "suites" else cfg.out
    code = 0
    try:
        done = plan.run()
    except (*_ERRORS, ImportError, RuntimeError) as e:
        # a server that failed past its retries, or answered something unreadable
        # after requests began: the records stay, run again with --resume. Anything
        # raised before the first request is a bad config or a refused resume.
        late = isinstance(e, RuntimeError) or (
            isinstance(e, ValueError) and plan.spy.started
        )
        if not late:
            return _fail(config, e)
        print(f"error: {e}", file=sys.stderr)
        done, code = _whole_lines(plan.counted), 1
    total = plan.total if code else (plan.total if plan.total is not None else done)
    if as_json:
        _emit({"arm": cfg.arm, "out": str(shown), "done": done, "total": total})
    else:
        print(f"{cfg.arm}: {done}/{'?' if total is None else total} items, {shown}")
    return code


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
        "2 usage error or unreadable input; gate: 1 regressed or not significant, "
        "3 screened out",
    )
    parser.add_argument(
        "--version", action="version", version=f"astro-weights {__version__}"
    )
    sub = parser.add_subparsers(
        dest="command",
        metavar="{build,compose,eval,export,gate,gen,inspect,mcp,run,self-test}",
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
    p = sub.add_parser(
        "gate",
        help="run the 0.3 gate end to end from recorded files; seals the test split",
    )
    p.add_argument("config", metavar="GATE.toml", type=Path, help="gate config file")
    p.add_argument(
        "--out",
        metavar="FILE",
        type=Path,
        help="where to write verdict.json (default: beside GATE.toml)",
    )
    p.add_argument(
        "--unseal",
        action="store_true",
        help="score an edit on the test split again; the ledger records it",
    )
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
        "gen",
        help="run one eval arm against an OpenAI-compatible server from ARM.toml",
    )
    p.add_argument("config", metavar="ARM.toml", type=Path, help="arm config file")
    p.add_argument(
        "--resume",
        action="store_true",
        help="go on with the records already written; a changed setting is refused",
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
    if args.command == "gate":
        return _gate(args.config, args.json, args.unseal, args.out)
    if args.command == "gen":
        return _gen(args.config, args.resume, args.json)
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
