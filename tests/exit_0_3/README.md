# 0.3 exit test: data layout and how to produce it

`tests/test_exit_0_3.py` checks exit criteria 2 to 8 of PDR 0.3 section 7 on files that
earlier runs wrote. It runs no model and needs no GPU. Criterion 1 (planted overfit)
is `tests/test_eval_gate.py`. Without `ASTRO_EXIT_DATA` the `real` variant skips and
only the `synthetic` variant (generated in a temp dir, 3 control gates) runs.

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
    controls/00/GATE.toml     the 21 random controls, each gated as if it were the edit
    controls/01/GATE.toml     ...
    controls/20/GATE.toml
    bad/GATE.toml             the planted bad edit
    confirm/controls/         control-00.jsonl .. control-20.jsonl (+ headers): all 21
    pools/edit/               control-01 .. control-20: the edit's (and gate/bad's) pool
    pools/control-NN/         the other 20 of the 21 controls: control NN's pool
    (splits.json, test/, floors/, adapter/, screens.json,
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
| `test_2_edit_confirmed_every_control_not` | 2 | `gate/edit`, `gate/controls/*` | exit 0 and `outcome: confirmed` for the edit; for each of exactly 21 controls exit 1 and `outcome: not significant` or `regressed` (never `screened out`: the whole gate must have run). Every gate's pool (`[confirm] controls`) holds 20 files with distinct records. The edit's pool is `control-01` to `control-20` and holds no records of the edit. Control `i`'s pool is the other 20 of `control-00` to `control-20`: it never contains the gated control (name or records), and no two gates gate the same control |
| `test_3_floor_zero_flips_deterministic_default_reported` | 3 | `floor/*`, `gate/edit` | deterministic `flip_rate == 0.0` and zero deltas, from two distinct runs (different `run_id`) whose headers declare `serving_flags` that differ from the default floor's; default floor printed; its `config_hash` equals the edit's test base arm; the floor stored in `gate/edit`'s `[test] floor` directory equals it; the gate's tokens check ran |
| `test_4_second_scoring_refused_unseal_recorded` | 4 | `gate/edit` | second `gate` exits 2 with `already scored`, ledger unchanged; `--unseal` exits 0 and the ledger holds `unseal: false` then `true` |
| `test_5_report_has_mde_candidates_and_every_arm` | 5 | `gate/edit` | the report holds `MDE (test)`, `candidates tried: N`, `confirm queries: M` (equal to `verdict.json`, from the run record), the confirm and test arm tables, the controls row, and exactly the arms `prompt`, `scripted`, `edit+prompt` (a line each) |
| `test_6_bfcl_paired_run_prints_tokens_and_tool_calls` | 6 | `bfcl/*.jsonl` | the full run (200 tasks, ids `multi_turn_base_*`, not the 10-task pilot) and the `format_tasks` rows `pass rate`, `tokens/task`, `tool calls/task`, `turns/task`. No assertion on the outcome: the numbers are published whatever they show |
| `test_7_rerun_from_copied_files_gives_identical_verdict` | 7 | all of `gate/edit` (with its recorded `verdict.json`), `run/record.jsonl` | a rerun on two copies gives `verdict.json` bytes equal to each other and to the recorded one; `run_record` points at `run/record.jsonl` |
| `test_8_planted_bad_edit_screened_out_or_regressed` | 8 | `gate/bad`, `gate/edit` | suite base and edit records exist (non-empty, with headers) for both; the bad edit is `screened out` (exit 3) or `regressed` (exit 1), with the keep-the-original-weights advice |

`-s` shows the floors and the BFCL report. Expected tail: `7 passed` (plus the 5
harness tests of the synthetic variant, which are not `realdata`).

## Producing the files

Serve the base and the edit under two model names from one server (vLLM
`--lora-modules`; for HF use the `runners.hf` path of the how-to `run-eval`). llama-server
has no model names: load the base GGUF with every adapter as a LoRA GGUF
(`--lora X.gguf`, written by `astro-weights export DIR --gguf X.gguf --arch ARCH` or
`export.gguf.save_lora_gguf`) and pick the arm per request with `extra_body`
`{"lora": [{"id": i, "scale": 0 or 1}, ...]}`, every adapter listed, the base arm with all
scales 0 (how-to `run-eval`, "Base and edit on one llama-server"; the prompt-cache flags
there matter). Then `lora` is the one `extra_body` key that differs between arms: give every
`GATE.toml` `arm_keys = ["lora"]`, merge the `lora` list into `SAMPLING` in the calls below
and pass the same `serving_flags` on every arm. BFCL prompts are about 6.5k tokens before
the first turn: start llama-server with at least 16384 context per slot (`-c` is shared
by `-np` slots). Both arms of a comparison must run under identical settings;
`check_headers` refuses any arm whose header
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
for i, c in enumerate(random_controls(edit, n=21, seed=0)):
    save_peft(
        c, f"exit/gate/controls_adapters/{i:02d}"
    )  # serve each as LoRA "control-NN", NN = 00 to 20
```

Serve `edit` and the 21 controls as LoRA adapters (vLLM `--lora-modules`), or as
merged weights for llama.cpp; the base is served unmodified.

### 2. Records: confirm split (K = 1) and test split (K = 4)

One `ARM.toml` and one `astro-weights gen` call per arm. `items` is the benchmark as JSONL
of `{"item_id", "prompt", "gold"}`; `manifest` and `split` keep that split's items. The
confirm split is run for the base, the edit and each of the 21 controls (the controls go
in `confirm/controls/control-NN.jsonl`); the test split for the base and the edit only.
For `split = "test"` `gen` takes `k` (4) from the manifest, runs seeds 0 to 3 into
`test/base.s0.jsonl` .. `s3.jsonl` and joins them into `test/base.jsonl` with seed 0's
header (`header["k"] == 4`).

Paths in an `ARM.toml` are relative to that file. This one is the confirm base arm
(`exit/gate/arms/confirm-base.toml`); `model` is the served name (`base`, `edit`,
`control-NN`). `model_id` is a CONFIG key: arms that differ in it are refused by
`check_headers`, and the gate would refuse every pair. vLLM has no `/props`, so without it
the header takes the served name; every arm sets the one `BASE_MODEL_ID`.

```toml
kind = "http"
arm = "base"                    # the arm name in the records
base_url = "http://localhost:8000"   # BASE_URL
model = "base"
model_id = "BASE_MODEL_ID"
max_new_tokens = 1024           # MAX_NEW_TOKENS, the same on every arm
items = "../../items.jsonl"
grader = "math"                 # exact, choice or math
manifest = "../splits.json"
split = "confirm"
out = "../confirm/base.jsonl"
concurrency = 8
extra_body = { temperature = 0.7, top_p = 0.95 }   # SAMPLING, fixed once, every arm
```

The confirm base arm must carry the test base arm's settings (`gate` checks them, `k`
aside), so every arm, confirm and test, takes the same `extra_body` and
`serving_flags`. The other arms differ in `arm`, `model`, `split` and `out`:

| File | `arm` | `model` | `split` | `out` |
|---|---|---|---|---|
| `confirm-edit.toml` | `edit` | `edit` | `confirm` | `../confirm/edit.jsonl` |
| `confirm-control-NN.toml` (NN = 00 to 20) | `control-NN` | `control-NN` | `confirm` | `../confirm/controls/control-NN.jsonl` |
| `test-base.toml` | `base` | `base` | `test` | `../test/base.jsonl` |
| `test-edit.toml` | `edit` | `edit` | `test` | `../test/edit.jsonl` |

`gen` creates the output directories (`exit/gate/pools`, `exit/floor/default` and
`exit/floor/deterministic` are made in the steps below). Every run is resumable after a
kill (GPU sessions are renewed): run each command with `--resume`, which on a finished
arm asks nothing and on an arm not started yet starts it. A resume under a changed
setting (`max_new_tokens`, `extra_body`, `serving_flags`, `system`...) exits 2 and
writes nothing; delete the file to start the arm over.

```bash
cd exit/gate/arms
for t in confirm-base confirm-edit test-base test-edit; do
  astro-weights gen $t.toml --resume
done
for i in $(seq -f %02g 0 20); do
  astro-weights gen confirm-control-$i.toml --resume
done
```

Then the gate pools. The edit is gated against `control-01` to `control-20` (20 files);
control NN is gated against the other 20 of `control-00` to `control-20` and never against
a pool that holds it (the rule asks the gated arm to beat the pool's second best, so a
control in its own pool would always pass stage 3 whenever it is the pool's best).
`gate/bad` uses the edit's pool. A pool is a directory of records and headers (copies, so
`header_path` finds each header beside its records):

```python
import shutil
from pathlib import Path

src = Path("exit/gate/confirm/controls")
names = [f"control-{i:02d}" for i in range(21)]


def pool(dst, exclude):
    dst = Path(dst)
    dst.mkdir(parents=True)
    for n in names:
        if n != exclude:
            for suffix in (".jsonl", ".header.json"):
                shutil.copy(src / f"{n}{suffix}", dst / f"{n}{suffix}")


pool("exit/gate/pools/edit", "control-00")
for n in names:
    pool(f"exit/gate/pools/{n}", n)
```

The run headers must agree on every config key across base, edit and controls
(`served_model` and `base_url` are provenance and may differ; `model_id`,
`serving_flags` and `extra_body` are config). Confirm and test may differ in `k` only.
If the server's flags are declared (`serving_flags`), declare the same on every arm
served under them.

### 2b. Report arms (`[arms]`, criterion 5)

The edit gate also needs test-split records, K = 4, for the three comparison arms. They
are paired with `test/base.jsonl` item by item, so each file needs the same `item_id`s,
seeds and `prompt_hash`es as the base, and a header that agrees with the base's on every
config key (`arm_keys` aside).

- `scripted`: the closed-form baseline `runners.scripted_arm.scripted_arm` (HF model and
  the contrast texts), saved with `save_peft`, served as one more adapter, run like the
  edit: `test-scripted.toml` is `test-edit.toml` with `arm = "scripted"`,
  `model = "scripted"` and `out = "../test/scripted.jsonl"`.
- `prompt` and `edit_prompt`: the base, and the edit, run with a system prompt. Give the
  `ARM.toml` a `system` key: the header records `system` and `system_sha256`, the item
  prompts and so their `prompt_hash`es stay the base's, and `gate` refuses a prompt arm
  without it (and a base, edit or scripted arm with it).

```toml
# test-prompt.toml: test-base.toml plus
arm = "prompt"
out = "../test/prompt.jsonl"
system = "Be extremely brief: at most two short sentences of reasoning, then the answer."
# test-edit-prompt.toml: test-edit.toml plus the same system key, and
# arm = "edit_prompt", out = "../test/edit_prompt.jsonl"
```

```bash
for t in test-scripted test-prompt test-edit-prompt; do astro-weights gen $t.toml --resume; done
```

### 3. Noise floors

Under the same server and settings as the test base arm, run the base twice more.
Each floor run is a full K = 4 run (seeds 0 to 3 joined into one file with a `k = 4`
header, like `test/base.jsonl`): `floor_from_records` takes `k` from the number of
paired seeds, and the gate refuses a floor that was not scored with K = 4.

Four more `ARM.toml` files, each `test-base.toml` with another `out`, run with `gen` (each
has its own `run_id`; the default floor's run1 may instead be `test/base.jsonl` copied
with its header):

| File | `out` | other keys |
|---|---|---|
| `floor-default-1.toml`, `floor-default-2.toml` | `../../floor/default/run1.jsonl`, `run2.jsonl` | as `test-base.toml` |
| `floor-det-1.toml`, `floor-det-2.toml` | `../../floor/deterministic/run1.jsonl`, `run2.jsonl` | `serving_flags = { VLLM_BATCH_INVARIANT = "1" }` |

For the deterministic pair restart the server in its batch-invariant mode and declare it
in `serving_flags`, so the header shows that mode and differs from the default floor's.

```bash
for t in floor-default-1 floor-default-2 floor-det-1 floor-det-2; do
  astro-weights gen exit/gate/arms/$t.toml --resume
done
```

```python
from pathlib import Path
from astro_weights.eval import floor_from_records, read_header, read_jsonl, save_floor

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

On llama.cpp the two modes are slot counts: default serving is `-np 4` with
`concurrency=4` and `serving_flags={"--parallel": "4"}` (concurrent requests change the
batches, so two runs disagree), deterministic serving is `-np 1` with `concurrency=1` and
`{"--parallel": "1"}`, which gives `flip_rate == 0.0`. `concurrency` and `serving_flags`
are config keys: the deterministic floor's headers differ from the default floor's in
both, as criterion 3 asks, and the test arms, the default floor and every control use
the default values.

The default floor's runs must hash to the test base arm's header (`config_hash`),
otherwise the gate finds no floor: serve and declare them exactly like `test/base`.
The deterministic runs differ from it in `serving_flags` (and so are not the default
floor's config). Check the server's batch invariance first:
`astro-weights self-test --base-url BASE_URL --model base`.

### 4. Screens, run record, adapter

- `exit/run/record.jsonl`: `RUN_DIR/record.jsonl` of the agent run that found the edit
  (`astro-weights run CONFIG.toml`, then `finish` exports the best edit). `gate` reads
  `candidates_tried`, `confirm_queries` and `exported_is_best` from it. The run needs the
  `confirm` tool (a `[confirm]` table in the config: `items` = the confirm split as
  `{"item_id", "prompt", "gold"}` JSONL outside the run directory, `grader`, `margin`,
  `max_queries`) and an editor that calls it: the `scripted` editor's default script does
  not, so its record has no `confirm_export` and `gate` refuses it. `exported_is_best`
  is false when `finish` could not export (`export_failed` in the record): the store
  rounds every edit to the base weight's dtype, so on a bf16 model a small edit is mostly
  rounding noise and fails `[export] max_rel_residual` (0.05 by default): a rank-1 edit
  of 0.5% of each weight's norm left a rank-8 residual of 0.21 on Qwen2.5-0.5B. Raise
  the residual or make the edit larger, and check the record's `confirm_export` line
  before using the run.
- `exit/gate/screens.json`: a JSON list of `{"name", "passed", "value", "limit"}` for
  `norm`, `perplexity` and `kl`, from `runners.guards.perplexity_guard` and
  `runners.guards.kl_last_token` and the norm of `Edit` against its limit:

```python
import json
from pathlib import Path

from astro_weights.edits.norms import check_norms
from astro_weights.runners.guards import kl_last_token, perplexity_guard

# `model`: the HF base model on CPU or CUDA (the guards call `.double()`, which an MPS
# tensor refuses); `ppl_ids` token ids of texts that look like the task, `prompt_ids`
# of neutral prompts; the three limits are data you choose and publish
nc = check_norms(edit, (0.0, NORM_MAX))
pg = perplexity_guard(model, ppl_ids, edit, max_ratio=1.25)
kl = kl_last_token(model, prompt_ids, edit)  # threshold 0.1
screens = [
    {
        "name": "norm",
        "passed": nc.passed,
        "value": max(nc.norms.values()),
        "limit": NORM_MAX,
    },
    {
        "name": "perplexity",
        "passed": pg.passed,
        "value": pg.ratio,
        "limit": pg.max_ratio,
    },
    {"name": "kl", "passed": kl.passed, "value": kl.mean, "limit": kl.threshold},
]
Path("exit/gate/screens.json").write_text(json.dumps(screens))
```

  The planted bad edit gets its own list in `exit/gate/screens_bad.json` (see section 7).
- `exit/gate/adapter/`: the exported PEFT directory of the edit (`save_peft`).
  Each control's `GATE.toml` uses its own control adapter directory.

### 5. BFCL multi_turn_base (criterion 6)

A pinned gorilla checkout (`BFCL_COMMIT` in `runners/bfcl.py`; `load_bfcl` verifies
every file against `PINS`):

```toml
# exit/bfcl/base.toml (edit.toml: arm = "edit", model = "edit", out = "edit.jsonl")
kind = "bfcl"
arm = "base"
base_url = "http://localhost:8000"   # BASE_URL
model = "base"
bfcl_root = "../../gorilla/berkeley-function-call-leaderboard"   # 200 tasks
trials = 4          # K=4 per task (PDR 0.3 section 4.7); the pilot may use 1
max_new_tokens = 1024
out = "base.jsonl"
```

```bash
astro-weights gen exit/bfcl/base.toml --resume
astro-weights gen exit/bfcl/edit.toml --resume
```

`gen` appends each finished (task, trial) record to `out` (fsynced) and writes the header
beside it. After a kill or a server error (HTTP 400 when a prompt exceeds the context, a
timeout) rerun with `--resume`: only the pairs not yet written are asked. A resume under
another `max_new_tokens`, `seed`, `extra_body` or `serving_flags` is refused (exit 2) and
the file is left alone. Pilot first with `trials = 1` and a copy of the config to measure
time per task (a separate `out`), then the full 200 with `trials = 4` (200 tasks x 4
trials x 2 arms = 1600 conversations). Serve base and edit with the same settings; the
files must have the same tasks and trials (`compare_tasks` refuses otherwise).

### 6. Regression suites (criterion 8), after 7b lands

`gen` with `kind = "suites"` (`runners.suites_run.run_suites` over `HttpBackend`) writes
`out_dir/<suite>/<arm>.jsonl` and its header for one arm:

```toml
# exit/gate/suites-base.toml (suites-edit.toml: arm = "edit", model = "edit")
kind = "suites"
arm = "base"
base_url = "http://localhost:8000"   # BASE_URL
model = "base"          # llama.cpp: extra_body = the arm's lora list
suites = ["gsm8k", "ifeval"]
out_dir = "suites"
max_new_tokens = 1024
```

```bash
for arm in base edit; do astro-weights gen exit/gate/suites-$arm.toml --resume; done
```

A suite file left by a kill holds scores of 0 until a resume finishes it and regrades it;
always rerun with `--resume`.

In each `GATE.toml`: `[suites.gsm8k]` with `base = "suites/gsm8k/base.jsonl"`,
`edit = "suites/gsm8k/edit.jsonl"`, its own `margin` and `source`; same for `ifeval`.
Run the same for the planted bad edit (served as `bad`, `out_dir = "suites_bad"`).

### 7. The `GATE.toml` files

One per gate, written by hand once; the key table is in `docs/reference/cli.md`
(section gate). They differ only in `[confirm] edit` (the edit's, or control NN's
confirm records), `[confirm] controls` (the gate's pool directory, below), `[test] edit`,
`[test] adapter`, `[test] seal_dir` and `[arms]` (edit gate only).
`run_record = "../../run/record.jsonl"` in all of them.

| Gate | `[confirm] edit` | `[confirm] controls` |
|---|---|---|
| `gate/edit` and `gate/bad` | `confirm/edit.jsonl` | `../pools/edit` (control-01 to control-20) |
| `gate/controls/NN` | `confirm/controls/control-NN.jsonl` | `../../pools/control-NN` (the other 20) |

The gates, one command each (the 21 control gates in a loop):

```bash
astro-weights gate exit/gate/edit/GATE.toml
for i in $(seq -f %02g 0 20); do astro-weights gate exit/gate/controls/$i/GATE.toml; done
astro-weights gate exit/gate/bad/GATE.toml
```

Controls are run on the confirm split only, so control NN's `[test]` base, edit and floor
point at the edit's test files; a control stopped at stage 3 never reads them.

Gate every control once before you trust those paths. A control that is the best or
second best of the 21 on the target clears stage 3 by chance (about 2 in 21, and only when
its side effects stay inside the other 20's range). It then goes on to the test stage and,
scored on the edit's test files, would get the edit's verdict, so test 2 would fail on its
exit code. For each control whose gate gets past stage 3 (the gate prints stage 4 lines,
or fails with a missing test file), run its own K = 4 test arm
(a `test-control-NN.toml` like `test-edit.toml` with `arm`/`model` `control-NN` and
`out = "../test/control-NN.jsonl"`, served as above) and its own regression suites (a
`suites-control-NN.toml`), and point
that gate's `[test] edit` and `[suites.*] edit` at them. Its own random edit then has no
effect to confirm, and test 2 expects `not significant`.

`gate/bad` uses its own `screens` (`../screens_bad.json`), `[test] adapter` and `[suites.*] edit`
(`exit/gate/suites_bad`); its other paths are the edit's. `gate/bad`: a planted bad edit (a large-norm one, or one with a known GSM8K loss),
run through the same steps; the expected outcome is `screened out` (norm screen
fails) or `regressed`.

## Without a GPU

`uv run pytest tests/test_exit_0_3.py` (no env var) runs the synthetic variant:
the same seven tests plus five harness tests, on a directory of the same layout that
`make_synthetic` generates in a temp dir from `tests/data/gate/make_fixture.py`.
