# 0.3 exit test: data layout and how to produce it

`tests/test_exit_0_3.py` checks exit criteria 2 to 8 of PDR 0.3 section 7 on files that
earlier runs wrote. It runs no model and needs no GPU. Criterion 1 (planted overfit)
is `tests/test_eval_gate.py`. Without `ASTRO_EXIT_DATA` the `real` variant skips and
only the `synthetic` variant (generated in a temp dir, 3 controls) runs.

```bash
ASTRO_EXIT_DATA=/path/to/exit uv run pytest -m realdata tests/test_exit_0_3.py -s
```

The tests copy the whole tree to a temp dir first. A gate run writes a seal ledger and
`verdict.json`; the copy takes them, so the data stays unsealed and can be tested
again. A seal left in the data is deleted in the copy. A `GATE.toml` whose `seal_dir`
points outside the data directory fails the copy. Do not run `astro-weights gate` in
the data directory yourself.

## Layout

Every path inside a `GATE.toml` is relative to that file, and `../` is allowed, so the
files below may be shared between gates. Every records file `X.jsonl` needs
`X.header.json` beside it.

```
exit/
  gate/
    edit/GATE.toml            the 27B edit, full gate
    edit/verdict.json         the verdict the first (real) gate run wrote and Research published
    controls/00/GATE.toml     the 20 random controls, each gated as if it were the edit
    controls/01/GATE.toml     ...
    controls/19/GATE.toml
    bad/GATE.toml             the planted bad edit
    (splits.json, confirm/, test/, floors/, adapter/, screens.json,
     suites/ ... wherever the GATE.toml files point)
  floor/
    deterministic/run1.jsonl  base-vs-base, deterministic serving (+ run1.header.json)
    deterministic/run2.jsonl
    default/run1.jsonl        base-vs-base, default serving: the config of test/base.jsonl
    default/run2.jsonl
  bfcl/base.jsonl             TaskRecords, BFCL multi_turn_base, base
  bfcl/edit.jsonl             TaskRecords, same tasks and trials, edit
  run/record.jsonl            the agent run's RunRecord; every GATE.toml's run_record points here
```

What each test reads, and what it expects:

| Test | Criterion | Reads | Expects |
|---|---|---|---|
| `test_2_edit_confirmed_every_control_not` | 2 | `gate/edit`, `gate/controls/*` | exit 0 and `outcome: confirmed` for the edit; for each of exactly 20 controls exit 1 and `outcome: not significant` or `regressed` (never `screened out`: the whole gate must have run). Each control gate's `[confirm] edit` is one of the 20 files in `[confirm] controls`, and no two gates use the same file |
| `test_3_floor_zero_flips_deterministic_default_reported` | 3 | `floor/*`, `gate/edit` | deterministic `flip_rate == 0.0` and zero deltas, from two distinct runs (different `run_id`) whose headers declare `serving_flags` that differ from the default floor's; default floor printed; its `config_hash` equals the edit's test base arm; the floor stored in `gate/edit`'s `[test] floor` directory equals it; the gate's tokens check ran |
| `test_4_second_scoring_refused_unseal_recorded` | 4 | `gate/edit` | second `gate` exits 2 with `already scored`, ledger unchanged; `--unseal` exits 0 and the ledger holds `unseal: false` then `true` |
| `test_5_report_has_mde_candidates_and_every_arm` | 5 | `gate/edit` | the report holds `MDE (test)`, `candidates tried: N`, `confirm queries: M` (equal to `verdict.json`, from the run record), the confirm and test arm tables, the controls row, and exactly the arms `prompt`, `scripted`, `edit+prompt` (a line each) |
| `test_6_bfcl_paired_run_prints_tokens_and_tool_calls` | 6 | `bfcl/*.jsonl` | the full run (200 tasks, ids `multi_turn_base_*`, not the 10-task pilot) and the `format_tasks` rows `pass rate`, `tokens/task`, `tool calls/task`, `turns/task`. No assertion on the outcome: the numbers are published whatever they show |
| `test_7_rerun_from_copied_files_gives_identical_verdict` | 7 | all of `gate/edit` (with its recorded `verdict.json`), `run/record.jsonl` | a rerun on two copies gives `verdict.json` bytes equal to each other and to the recorded one; `run_record` points at `run/record.jsonl` |
| `test_8_planted_bad_edit_screened_out_or_regressed` | 8 | `gate/bad`, `gate/edit` | suite base and edit records exist (non-empty, with headers) for both; the bad edit is `screened out` (exit 3) or `regressed` (exit 1), with the keep-the-original-weights advice |

`-s` shows the floors and the BFCL report. Expected tail: `7 passed` (plus the 2
harness tests of the synthetic variant, which are not `realdata`).

## Producing the files

Serve the base and the edit under two model names from one server (vLLM or llama.cpp;
for HF use the `runners.hf` path of the how-to `run-eval`). Both arms of a comparison
must run under identical settings; `check_headers` refuses any arm whose header
differs (`gate` runs it). Names below: `BASE_URL`, `base` (served base), `edit` (served
edit with its LoRA), `SAMPLING` (the `extra_body` of the K = 4 runs, fixed once per PDR
0.3 section 4), `BASE_MODEL_ID` (one fixed id for the base weights, for example the HF repo and revision; the edit and the controls are LoRAs on it, so they share it).

### 1. Splits, edit, controls

```python
from pathlib import Path
from astro_weights.edits import random_controls
from astro_weights.eval.splits import SplitManifest
from astro_weights.export import load_peft, save_peft

manifest = SplitManifest.build(
    item_ids, dataset="gsm8k", revision=REV, file_sha256=SHA, seed=0, k=4
)  # item_ids: every item id of the benchmark; k=4 is checked by `gate`
manifest.save("exit/gate/splits.json")

edit = load_peft("exit/gate/adapter")  # the exported edit (run RUN_DIR/export/)
for i, c in enumerate(random_controls(edit, n=20, seed=0)):
    save_peft(
        c, f"exit/gate/controls_adapters/{i:02d}"
    )  # serve each as LoRA "control-NN"
```

Serve `edit` and the 20 controls as LoRA adapters (vLLM `--lora-modules`), or as
merged weights for llama.cpp; the base is served unmodified.

### 2. Records: confirm split (K = 1) and test split (K = 4)

One call per arm; items are `{"item_id", "prompt", "gold"}` of the split. The confirm
split is run for the base, the edit and each of the 20 controls (the controls go in
`confirm/controls/control-NN.jsonl`, any file name except `edit.jsonl`); the test split
for the base and the edit only, once per seed 0 to 3, then the four files are joined
into one `test/base.jsonl` and one `test/edit.jsonl` (`header["k"] == 4`).

```python
from astro_weights.eval import (
    header_path,
    read_header,
    read_jsonl,
    write_header,
    write_jsonl,
)
from astro_weights.runners.endpoint import generate_records_http, run_header_http


def run(model, arm, items, out, seed=0, k=1, extra_body=None, serving_flags=None):
    # model_id: vLLM has no /props, so without it the header takes the served name
    # ("base", "edit", "control-NN"). model_id is a CONFIG key: arms that differ in it
    # are refused by check_headers, and the gate would refuse every pair.
    header = run_header_http(
        BASE_URL,
        model,
        MAX_NEW_TOKENS,
        seed,
        extra_body,
        k=k,
        concurrency=8,
        serving_flags=serving_flags,
        model_id=BASE_MODEL_ID,
    )
    return generate_records_http(
        BASE_URL,
        model,
        items,
        arm,
        scorer,
        MAX_NEW_TOKENS,
        seed=seed,
        extra_body=extra_body,
        out=out,
        concurrency=8,
        header=header,
    )


def run_k4(model, arm, items, stem, extra_body=SAMPLING, serving_flags=None):
    """One K = 4 arm: seeds 0 to 3, joined into `stem.jsonl` with a k = 4 header."""
    for s in range(4):
        run(model, arm, items, f"{stem}.s{s}.jsonl", s, 4, extra_body, serving_flags)
    write_jsonl(
        f"{stem}.jsonl",
        [r for s in range(4) for r in read_jsonl(f"{stem}.s{s}.jsonl")],
    )
    write_header(  # seed 0's header: k = 4, the same config for the four seeds
        header_path(f"{stem}.jsonl"), read_header(header_path(f"{stem}.s0.jsonl"))
    )


run("base", "base", confirm_items, "exit/gate/confirm/base.jsonl")
run("edit", "edit", confirm_items, "exit/gate/confirm/edit.jsonl")
for i in range(20):
    run(
        f"control-{i:02d}",
        f"control-{i:02d}",
        confirm_items,
        f"exit/gate/confirm/controls/control-{i:02d}.jsonl",
    )
run_k4("base", "base", test_items, "exit/gate/test/base")
run_k4("edit", "edit", test_items, "exit/gate/test/edit")
```

The run headers must agree on every config key across base, edit and controls
(`served_model` and `base_url` are provenance and may differ; `model_id`,
`serving_flags` and `extra_body` are config). Confirm and test may differ in `k` only.
If the server's flags are declared (`serving_flags`), declare the same on every arm
served under them.

### 3. Noise floors

Under the same server and settings as the test base arm, run the base twice more.
Each floor run is a full K = 4 run (seeds 0 to 3 joined into one file with a `k = 4`
header, like `test/base.jsonl`): `floor_from_records` takes `k` from the number of
paired seeds, and the gate refuses a floor that was not scored with K = 4.

```python
from pathlib import Path
from astro_weights.eval import floor_from_records, read_header, save_floor

# default serving (the settings of exit/gate/test/base.jsonl): two new runs, or run1
# copied from test/base.jsonl (+ header) and run2 new. Each run has its own run_id.
run_k4("base", "base", test_items, "exit/floor/default/run1")
run_k4("base", "base", test_items, "exit/floor/default/run2")

# deterministic serving (restart the server in its batch-invariant mode) and declare
# it, so the header shows that mode: two runs, headers differ from the default floor's.
BI = {"VLLM_BATCH_INVARIANT": "1"}
run_k4("base", "base", test_items, "exit/floor/deterministic/run1", serving_flags=BI)
run_k4("base", "base", test_items, "exit/floor/deterministic/run2", serving_flags=BI)

d = Path("exit/floor/default")
header = read_header(d / "run1.header.json")
floor = floor_from_records(
    read_jsonl(d / "run1.jsonl"), read_jsonl(d / "run2.jsonl"), header
)
save_floor(
    "exit/gate/floors", floor
)  # keyed by config_hash(header): the test base arm's
print(floor.to_json())
```

The default floor's runs must hash to the test base arm's header (`config_hash`),
otherwise the gate finds no floor: serve and declare them exactly like `test/base`.
The deterministic runs differ from it in `serving_flags` (and so are not the default
floor's config). Check the server's batch invariance first:
`astro-weights self-test --base-url BASE_URL --model base`.

### 4. Screens, run record, adapter

- `exit/run/record.jsonl`: `RUN_DIR/record.jsonl` of the agent run that found the edit
  (`astro-weights run CONFIG.toml`, then `finish` exports the best edit). `gate` reads
  `candidates_tried`, `confirm_queries` and `exported_is_best` from it.
- `exit/gate/screens.json`: a JSON list of `{"name", "passed", "value", "limit"}` for
  `norm`, `perplexity` and `kl`, from `runners.guards.perplexity_guard` and
  `runners.guards.kl_last_token` and the norm of `Edit` against its limit.
- `exit/gate/adapter/`: the exported PEFT directory of the edit (`save_peft`).
  Each control's `GATE.toml` uses its own control adapter directory.

### 5. BFCL multi_turn_base (criterion 6)

A pinned gorilla checkout (`BFCL_COMMIT` in `runners/bfcl.py`; `load_bfcl` verifies
every file against `PINS`):

```python
from astro_weights.runners.bfcl import load_bfcl, run_bfcl

tasks = load_bfcl("gorilla/berkeley-function-call-leaderboard")  # 200 tasks
for arm in ("base", "edit"):
    run_bfcl(
        tasks,
        BASE_URL,
        arm,
        arm,
        trials=1,
        max_new_tokens=1024,
        out=f"exit/bfcl/{arm}.jsonl",
    )
```

Pilot first with `tasks[:10]`, then the full 200. Serve base and edit with the same
settings; the files must have the same tasks and trials (`compare_tasks` refuses
otherwise).

### 6. Regression suites (criterion 8), after 7b lands

`runners.suites_run.run_suites` (branch `orca/w7-7b-suites-run`; not in this branch)
writes `out_dir/<suite>/<arm>.jsonl` and its header for one arm:

```python
from astro_weights.runners.suites_run import HttpBackend, run_suites

for arm in ("base", "edit"):
    run_suites(
        ["gsm8k", "ifeval"],
        arm,
        "exit/gate/suites",
        backend=HttpBackend(BASE_URL, arm),
        max_new_tokens=1024,
    )
```

In each `GATE.toml`: `[suites.gsm8k]` with `base = "suites/gsm8k/base.jsonl"`,
`edit = "suites/gsm8k/edit.jsonl"`, its own `margin` and `source`; same for `ifeval`.
Run the same for the planted bad edit (served as `bad`) into `exit/gate/suites_bad`.

### 7. The `GATE.toml` files

One per gate, written by hand once; the key table is in `docs/reference/cli.md`
(section gate). They differ only in `[confirm] edit` (the edit's, or control NN's
confirm records), `[test] edit`, `[test] adapter`, `[test] seal_dir`, `[confirm]
controls` (all 20 controls, the same directory for every gate) and `[arms]` (edit
gate only). `run_record = "../../run/record.jsonl"` in all of them.
Control NN's gate has `[confirm] edit = ".../controls/control-NN.jsonl"` and its own
`[test] adapter` and `seal_dir`. Controls are run on the confirm split only, so its
`[test]` base, edit and floor point at the edit's test files; a control stopped at
stage 3 never reads them. A control that passed stage 3 would be scored on those, and
test 2 would fail on its exit code. Each control is gated against the same pool of 20
controls it is itself part of, which only makes confirmation harder.

`gate/bad`: a planted bad edit (a large-norm one, or one with a known GSM8K loss),
run through the same steps; the expected outcome is `screened out` (norm screen
fails) or `regressed`.

## Without a GPU

`uv run pytest tests/test_exit_0_3.py` (no env var) runs the synthetic variant:
the same seven tests plus two harness tests, on a directory of the same layout that
`make_synthetic` generates in a temp dir from `tests/data/gate/make_fixture.py`.
