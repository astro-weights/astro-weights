"""Synthetic inputs for `astro-weights gate`: records, headers, manifest, adapter,
floor, screens, run record and GATE.toml. No model runs; every number is seeded, so
the same call writes the same bytes. `build(dst, **knobs)` returns the GATE.toml path;
each knob breaks one stage (see `Knobs`).
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from astro_weights.edits import Edit, rank1
from astro_weights.eval import (
    FloorResult,
    Record,
    config_hash,
    header_path,
    prompt_hash,
    save_floor,
    write_header,
    write_jsonl,
)
from astro_weights.eval.header import system_header
from astro_weights.eval.splits import SplitManifest
from astro_weights.export import save_peft
from astro_weights.mcp.record import RunRecord

N_ITEMS = 200
N_CONTROLS = 20
N_SUITE = 120
CUT = 40  # tokens the edit saves per answer when `tokens_cut`
SYSTEM = "Answer in as few words as you can."  # the prompt arms' instruction
MODULE = "model.layers.1.mlp.down_proj"


@dataclass(frozen=True)
class Knobs:
    screens_fail: str | None = None  # name of a screen that fails
    controls_beat: bool = True  # the edit beats the control quantile on confirm
    tokens_cut: bool = True  # the edit lowers tokens on the test split
    side_drop: bool = False  # the edit loses accuracy on confirm, below every control
    acc_drop: bool = False  # the edit loses accuracy on the test split
    suite_drop: bool = False  # the edit loses accuracy on the gsm8k suite
    exported: str = "best"  # "best", "not_best" or "missing" in the run record
    floor: bool = True
    manifest_k: int = 4
    arms: bool = False  # also write scripted, prompt and edit+prompt arms


def _header(k: int, run_id: str) -> dict[str, object]:
    return {
        "model": "tiny",
        "backend": "hf",
        "dtype": "float32",
        "max_new_tokens": 256,
        "seed": 0,
        "k": k,
        "run_id": run_id,
    }


def _write_arm(path: Path, records: list[Record], k: int, **extra: object) -> None:
    write_jsonl(path, records)
    write_header(header_path(path), {**_header(k, path.stem), **extra})


def _rec(item: str, seed: int, arm: str, score: float, tokens: int) -> Record:
    return Record(
        item, seed, arm, prompt_hash(f"prompt {item}"), f"out {arm} {item} {seed}",
        score, tokens, False,
    )  # fmt: skip


def _base_rows(
    ids: list[str], seeds: int, tag: str
) -> list[tuple[str, int, float, int]]:
    rng = random.Random(f"{tag}")
    solved = {i: float(rng.random() < 0.6) for i in ids}
    return [
        (i, s, solved[i], 100 + rng.randint(-10, 10)) for i in ids for s in range(seeds)
    ]


def _arm(
    rows: list[tuple[str, int, float, int]],
    arm: str,
    drop: int = 0,
    gain: int = 0,
    cut: int = 0,
) -> list[Record]:
    """`rows` as one arm: the first `drop` solved items fail, the first `gain` unsolved succeed."""
    lose = {i for i, _, sc, _ in rows if sc == 1.0}
    win = {i for i, _, sc, _ in rows if sc == 0.0}
    lose = set(sorted(lose)[:drop])
    win = set(sorted(win)[:gain])
    return [
        _rec(i, s, arm, 0.0 if i in lose else 1.0 if i in win else sc, t - cut)
        for i, s, sc, t in rows
    ]


def build(dst: Path, **knobs: object) -> Path:
    k = Knobs(**knobs)  # type: ignore[arg-type]
    dst.mkdir(parents=True, exist_ok=True)
    ids = [f"i{n:03d}" for n in range(N_ITEMS)]
    manifest = SplitManifest.build(
        ids, dataset="synthetic", revision="r1", file_sha256="0" * 64, seed=0,
        k=k.manifest_k,
    )  # fmt: skip
    manifest.save(dst / "splits.json")

    confirm_rows = _base_rows(list(manifest.confirm), 1, "confirm")
    conf = dst / "confirm"
    conf.mkdir(exist_ok=True)
    _write_arm(conf / "base.jsonl", _arm(confirm_rows, "base"), 1)
    edit_cut = 30 if k.controls_beat else 2
    _write_arm(
        conf / "edit.jsonl",
        _arm(confirm_rows, "edit", drop=8 if k.side_drop else 0, cut=edit_cut),
        1,
    )
    (conf / "controls").mkdir(exist_ok=True)
    for j in range(N_CONTROLS):  # accuracy moves both ways around the edit's 0
        flips = {"drop": j % 3 if j < 10 else 0, "gain": j % 3 if j >= 10 else 0}
        rows = _arm(confirm_rows, f"control-{j:02d}", cut=j % 9, **flips)
        _write_arm(conf / "controls" / f"control-{j:02d}.jsonl", rows, 1)

    test_rows = _base_rows(list(manifest.test), k.manifest_k, "test")
    test = dst / "test"
    test.mkdir(exist_ok=True)
    _write_arm(test / "base.jsonl", _arm(test_rows, "base"), k.manifest_k)
    _write_arm(
        test / "edit.jsonl",
        _arm(
            test_rows,
            "edit",
            drop=12 if k.acc_drop else 0,
            cut=CUT if k.tokens_cut else 0,
        ),
        k.manifest_k,
        infra_errors={"retries": 2, "failed": 1},
    )
    if k.arms:
        for name, cut in (("scripted", 5), ("prompt", 10), ("edit_prompt", 50)):
            _write_arm(
                test / f"{name}.jsonl",
                _arm(test_rows, name, cut=cut),
                k.manifest_k,
                **(system_header(SYSTEM) if name != "scripted" else {}),
            )

    suites = {}
    for name in ("gsm8k", "ifeval"):
        srows = _base_rows([f"s{n:03d}" for n in range(N_SUITE)], 1, name)
        drop = 15 if k.suite_drop and name == "gsm8k" else 0
        _write_arm(dst / f"{name}-base.jsonl", _arm(srows, "base"), 1)
        _write_arm(dst / f"{name}-edit.jsonl", _arm(srows, "edit", drop=drop), 1)
        suites[name] = (f"{name}-base.jsonl", f"{name}-edit.jsonl")

    if k.floor:
        floor_dir = dst / "floors"
        key = config_hash(_header(k.manifest_k, "x"))
        save_floor(
            floor_dir,
            FloorResult(
                key,
                len(manifest.test),
                4,
                0.0,
                0.01,
                0.0,
                1.0,
                1.0,
                "2026-11-24T00:00:00+00:00",
                None,
            ),
        )

    rng = np.random.default_rng(0)
    delta = rank1(MODULE, rng.standard_normal(16), rng.standard_normal(48))
    save_peft(Edit.from_deltas([delta], "tiny"), dst / "adapter")

    names = ("norm", "perplexity", "kl")
    screens = [
        {"name": n, "passed": n != k.screens_fail, "value": 0.5, "limit": 1.0}
        for n in names
    ]
    (dst / "screens.json").write_text(json.dumps(screens), encoding="utf-8")

    record = RunRecord.start(
        dst / "record.jsonl", seed=0, config={}, editor="scripted", model=None
    )
    for q, cand in enumerate(("c1", "c2", "c2"), 1):
        record.event("confirm_query", query=q)
        record.event("confirm_result", query=q, candidate=cand, best_candidate="c2")
    if k.exported != "missing":
        record.event(
            "confirm_export", best_candidate="c2", exported_candidate="c2",
            exported_is_best=k.exported == "best",
        )  # fmt: skip
    record.finish(budget={}, outcome="finished")

    arms = "\n".join(
        f'{n} = "test/{n}.jsonl"' for n in ("scripted", "prompt", "edit_prompt")
    )
    suite_toml = "\n".join(
        f'[suites.{n}]\nbase = "{b}"\nedit = "{e}"\nmargin = 0.02\n'
        f'source = "pilot on dev, 2026-11-20"\n'
        for n, (b, e) in suites.items()
    )
    toml = f"""margin = 0.02
margin_source = "pilot on dev, 2026-11-20"
backend = "hf"
build = "torch-test"
screens = "screens.json"
run_record = "record.jsonl"
manifest = "splits.json"
boot = 300

[confirm]
base = "confirm/base.jsonl"
edit = "confirm/edit.jsonl"
controls = "confirm/controls"
target = "tokens_diff"
side_effects = ["accuracy"]

[test]
base = "test/base.jsonl"
edit = "test/edit.jsonl"
seal_dir = "seal"
adapter = "adapter"
floor = "floors"

[footprint]
rank = 1

{suite_toml}
"""
    if k.arms:
        toml += f"[arms]\n{arms}\n"
    path = dst / "GATE.toml"
    path.write_text(toml, encoding="utf-8")
    return path
