# Run the 0.3 gate end to end

The gate turns recorded runs into one verdict for one edit: `confirmed`, `regressed` or `not significant`, or `screened out` before any eval. This page walks the inputs in the order the gate reads them, then runs `astro-weights gate` on made-up files. What each outcome means: [Gate verdicts](../explanation/gate-verdicts.md). Every `GATE.toml` key: [CLI reference: gate](../reference/cli.md#gate).

The gate runs no model. Each step below writes files; the gate reads them.

| Gate input | Made by | Section |
|---|---|---|
| `manifest`, `[test] seal_dir` | `SplitManifest`, seal ledger | [Splits and seal](#splits-and-seal) |
| `[test] floor` | two base runs, `save_floor` | [Serving self-test and noise floor](#serving-self-test-and-noise-floor) |
| `screens` | norm check, perplexity and KL guards | [Pre-screens](#pre-screens) |
| `[confirm] controls` | `random_controls`, then one eval per control | [Random controls](#random-controls) |
| `[arms]` | scripted arm, prompt arm | [Scripted and prompt arms](#scripted-and-prompt-arms) |
| `run_record` | agent run with the `confirm` tool | [Confirm tool for agents](#confirm-tool-for-agents) |
| `[suites.NAME]` | regression suite records | [Regression suites](#regression-suites) |

## Splits and seal

Split the benchmark once into dev, confirm and test by seeded item id, with `k = 4` seeds per test item (the gate refuses any other `k`). Keep the manifest and the seal directory outside the agent's `RUN_DIR`. How: [Run an eval: splits and the sealed test](run-eval.md#splits-and-the-sealed-test).

The agent sees dev items. The `confirm` tool and the control rule use confirm items. The test split is scored once per edit, by the gate, through the same seal as `astro-weights eval --split test`.

## Serving self-test and noise floor

Before an HTTP run, check that the server gives one output for one prompt whatever else it is serving:

```bash
astro-weights self-test --base-url http://127.0.0.1:8080 --model base
```

It sends the same prompt at temperature 0 beside 0, 3 and 7 filler requests (batch sizes 1, 4 and 8), then after a filler that shares its leading text, about 256 and 1024 tokens long, and with no filler. Exit 0 and `deterministic` when every output matches; exit 1 and the first differing character when not. `--prompt` changes the prompt, `--api-key-env VAR` names the key's environment variable, `--json` prints the `SelfTestResult`. In Python: `runners.endpoint.self_test_http`.

Then run the base arm twice on the test split under the same settings and store the noise floor: [Run an eval: noise floor](run-eval.md#noise-floor). The gate looks the floor up by the `config_hash` of the test base arm's header and refuses the run when none is stored. The edit must lower tokens by more than the floor (`tokens_beyond_floor`).

## Pre-screens

Three cheap checks run before any eval. All three are required in `screens.json`; any failure gives `screened out`.

The update norm check needs only the edit. Ranges are yours to bring (what a normal edit looks like for this model); the library ships none:

```python
import numpy as np

from astro_weights.edits import Edit, check_norms, rank1

rng = np.random.default_rng(0)
module = "model.layers.1.mlp.down_proj"
edit = Edit.from_deltas(
    [rank1(module, rng.standard_normal(16), rng.standard_normal(48))], "demo"
)
norm = check_norms(edit, {"mlp.down_proj": (0.0, 50.0)})
print(norm.passed, norm.outside)
```

Prints:

```text
True ()
```

A range key is a full module name or a dotted suffix; a module no key matches raises `ValueError`.

Perplexity and last-token KL need the base model (`hf` extra) and token ids of a neutral prompt set, which you bring:

```python
import json
from pathlib import Path

from astro_weights.runners.guards import kl_last_token, perplexity_guard

kl = kl_last_token(model, neutral_ids, edit)  # passed: mean KL <= 0.1 nats
ppl = perplexity_guard(model, neutral_ids, edit, max_ratio=1.1)
screens = [
    {
        "name": "norm",
        "passed": norm.passed,
        "value": max(norm.norms.values()),
        "limit": 50.0,
    },
    {"name": "perplexity", "passed": ppl.passed, "value": ppl.ratio, "limit": 1.1},
    {"name": "kl", "passed": kl.passed, "value": kl.mean, "limit": kl.threshold},
]
Path("screens.json").write_text(json.dumps(screens), encoding="utf-8")
```

`max_ratio` has no default: 1.1 above is an example, not a recommendation. KL 0.1 nats is the default threshold; low KL rules broken edits out and proves nothing about good ones. `value` and `limit` must be finite numbers.

## Random controls

Random edits of the same size show whether the edit's direction matters or only its size. Make at least 20 (the gate refuses fewer at `q = 0.95`), save each as an adapter, and run each on the confirm split like the edit:

```python
from pathlib import Path

from astro_weights.edits import random_controls
from astro_weights.export import save_peft

for c in random_controls(edit, n=20, seed=0):
    save_peft(c, Path("controls") / f"c{c.meta['control']:02d}")
print(len(list(Path("controls").iterdir())))
```

Prints:

```text
20
```

Each control has the same modules, shapes, rank and per-module `|B @ A|_F` as the edit. More: [Build from a recipe: random controls](build-from-recipe.md#random-controls).

Put the 20 confirm-split record files in one directory, `[confirm] controls` in `GATE.toml`. The control rule (`eval.control_rule`):

- target: the edit must be strictly better than the controls' `q`-quantile (default 0.95) on `target`: `tokens_diff` (lower is better) or `diff` (higher is better). Else `not significant`.
- side effects: each named side effect's `diff` must lie inside the controls' `[min, max]`, bounds included. Else `regressed`. With `target = "tokens_diff"`, `accuracy` is required.

## Scripted and prompt arms

Two baselines go into the report, not into the verdict. List their test-split records under `[arms]` as `scripted`, `prompt` and `edit_prompt`.

Scripted arm: `runners.scripted_arm.scripted_arm` builds a closed-form difference-of-means `project_out` edit on the modules the agent's edit touches, at zero API cost. Only `self_attn.o_proj` and `mlp.down_proj` qualify. Needs the `hf` extra and token ids of two text sets: one showing the behaviour to remove, one the contrast.

```python
from astro_weights.runners.scripted_arm import scripted_arm

scripted = scripted_arm(model, verbose_ids, terse_ids, like=edit)
save_peft(scripted, "scripted")
```

Prompt arm: the base model with one fixed short instruction added to every item, same items. `edit_prompt` is the edit plus that instruction. Generate these arms like any other: [Run an eval](run-eval.md).

## Gate an edit made without an agent

The gate reads `confirm_export` from a run record, so an edit made outside a run (a scripted arm picked on the dev split, say) goes through a run with the scripted editor first. List the candidates in the order they were tried; each is applied to the base weights and sent to `confirm`, then the last one that passed is exported. The record then counts every candidate sent to `confirm`, as for an agent. Record any search done before the run (strengths tried on the dev split) beside the config.

```toml
[run]
dir = "runs/scripted"
seed = 0
model_dir = "models/base"

[limits]
forward_passes = 0
code_runs = 4
tokens = 0

[editor]
kind = "scripted"
edit = ["scripted"]          # PEFT adapter directories, in the order tried

[confirm]
items = "data/confirm.jsonl"
grader = "math"
margin = 0.02
max_queries = 1              # one per candidate

[export]
rank = 8                     # at least the highest candidate rank, so the export is the confirmed weights
```

`astro-weights run CONFIG.toml`, then point `run_record` in `GATE.toml` at `runs/scripted/record.jsonl` and `adapter` at the exported adapter. A killed run resumes with `astro-weights run CONFIG.toml --resume`: candidates that already have a confirm result are not confirmed again, a query cut by the kill is asked again (it does not count toward `max_queries`), and the outputs it had already generated are reused.

The editor refuses before any tool call when a candidate's rank is above `[export] rank`, when an adapter records a base tensor sha256 that differs from the run's model, or, on a resume, when the list or its files changed (`RUN_DIR/scripted_edits.json` holds the list a run started with). The `finish` summary in the record names each candidate with its file sha256.

## Confirm tool for agents

An agent run selects its edit on the confirm split through the MCP `confirm` tool. Add a `confirm` object to `run.json` (or a `[confirm]` table to the `astro-weights run` config); the first four keys are required:

```json
{
  "confirm": {
    "items": "../data/confirm.jsonl",
    "grader": "math",
    "margin": 0.02,
    "max_queries": 10,
    "target": "accuracy",
    "batch_size": 1
  }
}
```

- `items`: JSONL of `{"item_id", "prompt", "gold"}`, the confirm split only. A path inside the run directory is refused, and `model_dir` is required.
- `grader`: `exact`, `choice` or `math`.
- `margin`: in items correct per item (0.02 = 2 points).
- `max_queries`: calls past it are refused.
- `target` (default `accuracy`): the selection rule. Match it to the gate's `[confirm] target`: `accuracy` for `diff`, `tokens` for `tokens_diff`.
- `batch_size` (default 1): confirm items generated at once, left-padded. Base and edit use the same size. Greedy outputs at a batch size above 1 can differ from batch size 1 (padding changes float rounding), as in `generate_records`.

A query greedy-generates every confirm item with the stored edits. With `target = "accuracy"` it passes when they beat the current best (the base model first, then the last edits that passed) by more than `margin * n` items. A tie fails.

With `target = "tokens"` it passes when both hold: the edits' total output tokens over all confirm items are below the current best's (the base model first, then the last edits that passed; a tie fails), and their correct count is at least the base model's minus `margin * n` (always against base, never against an earlier edit). This is the gate's test rule for a `tokens_diff` target: fewer tokens, accuracy non-inferior. The base model runs once per run; its counts go to a `confirm_base` event and survive a restart. Each `confirm_result` event then also carries `target`, `tokens`, `bar_tokens`, `base_correct` and `hit_cap` (outputs that hit `max_new_tokens`; capped outputs make a token cut look bigger, so check it). There `bar` is the base count and `best` the count of the current best. The agent gets `{"passed": bool}` and nothing else. At `finish` the server records whether the exported edits are the ones that last passed; the gate refuses a run record where they are not. Reference: [MCP server: tools](../reference/mcp-server.md#tools).

## Regression suites

Every confirmed edit must also hold on GSM8K, IFEval and MMLU-Pro, each within its own margin. `eval.load_suite` reads the pinned file (repo, revision and sha256 in `eval.PINS`), from a path you give or downloaded from the Hugging Face Hub, and refuses a file whose hash differs:

```python
from astro_weights.eval import load_suite, suite_grader

data = load_suite("ifeval")  # downloads once, then cached
grade = suite_grader("ifeval")
item = data.items[0]
print(item["item_id"], grade(item, "an answer"))
```

`suite_grader(name)` takes `(item, answer)` and returns a score in `[0, 1]`; IFEval needs the whole item, not only `gold`. Parquet suites (GSM8K, MMLU-Pro) need the `eval` extra. Write base and edit records for each suite (`Record`, `write_jsonl`, `write_header`) and list them as `[suites.NAME]` with `margin` and `source`. When edits are stacked, run the suites on the stack.

## BFCL multi_turn_base

The agentic check uses BFCL `multi_turn_base` (200 tasks) from a local checkout of https://github.com/ShishirPatil/gorilla at the commit the library pins, `runners.bfcl.BFCL_COMMIT`. There is no pip dependency. Clone and pin:

```bash
git clone https://github.com/ShishirPatil/gorilla
git -C gorilla checkout 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8
```

Serve base and edit under two names on an OpenAI-compatible server with tool calls, then run and compare:

```python
from astro_weights.eval import compare_tasks, format_tasks
from astro_weights.runners.bfcl import load_bfcl, run_bfcl

tasks = load_bfcl("gorilla/berkeley-function-call-leaderboard")
url = "http://127.0.0.1:8080"
base = run_bfcl(tasks, url, "base", "base", trials=4, out="bfcl-base.jsonl")
edit = run_bfcl(tasks, url, "edit", "edit", trials=4, out="bfcl-edit.jsonl")
print(format_tasks(compare_tasks(base, edit)))
```

`load_bfcl` checks every file it reads against `runners.bfcl.PINS` (sha256) and refuses one outside the table. `compare_tasks` pairs per task over pass rate, output tokens, tool calls and turns, averaging trials per task first. BFCL is not a `GATE.toml` input: report it beside the gate.

## Run the gate

The rest of this page runs as is: it writes made-up records for every input, then runs the gate. Records are 600 items split by seed, 20 controls, K = 4 on test and a 400-item suite; the edit flips 3 answers each way and saves 30 tokens per answer.

```python
import json
import random
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
from astro_weights.eval.splits import SplitManifest
from astro_weights.export import save_peft
from astro_weights.mcp.record import RunRecord

ids = [f"q{n:03d}" for n in range(600)]
manifest = SplitManifest.build(
    ids, dataset="demo", revision="r1", file_sha256="0" * 64, seed=0, k=4
)
manifest.save("splits.json")
rng = random.Random(0)
solved = {i: float(rng.random() < 0.6) for i in ids}
solved |= {f"s{n:03d}": float(rng.random() < 0.6) for n in range(400)}


def header(k):
    return {"model": "demo", "backend": "hf", "dtype": "float32",
            "max_new_tokens": 256, "seed": 0, "k": k}  # fmt: skip


def arm(path, items, k, name, cut=0, down=0, up=0):
    """One arm's records: `down` solved items fail, `up` unsolved items pass,
    every answer `cut` tokens shorter than the base's."""
    noise = random.Random(path)
    flip = set(noise.sample([i for i in items if solved[i]], down))
    flip |= set(noise.sample([i for i in items if not solved[i]], up))
    records = [
        Record(
            i,
            s,
            name,
            prompt_hash(i),
            f"{name} {i} {s}",
            1.0 - solved[i] if i in flip else solved[i],
            100 + noise.randint(-5, 5) - cut,
            False,
        )  # fmt: skip
        for i in items
        for s in range(k)
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, records)
    write_header(header_path(path), header(k))


# confirm split: base, edit and 20 random controls, one sample per item
arm("confirm/base.jsonl", manifest.confirm, 1, "base")
arm("confirm/edit.jsonl", manifest.confirm, 1, "edit", cut=30, down=3, up=3)
for j in range(20):  # random edits of the same size: small, no direction
    arm(f"confirm/controls/c{j:02d}.jsonl", manifest.confirm, 1, f"c{j:02d}",
        cut=j % 5, down=j % 5, up=(j * 3) % 5)  # fmt: skip
# test split: K = 4 samples per item
arm("test/base.jsonl", manifest.test, 4, "base")
arm("test/edit.jsonl", manifest.test, 4, "edit", cut=30, down=3, up=3)
# one regression suite on its own items
suite = tuple(f"s{n:03d}" for n in range(400))
arm("gsm8k/base.jsonl", suite, 1, "base")
arm("gsm8k/edit.jsonl", suite, 1, "edit", down=1, up=1)

# noise floor stored under the test base arm's config hash (made-up numbers here)
floor = FloorResult(config_hash(header(4)), len(manifest.test), 4, 0.0, 0.0, 0.0,
                    0.5, 0.5, "2026-11-24T00:00:00+00:00", None)  # fmt: skip
save_floor("floors", floor)
# the exported adapter; the seal ledger keys it by content
w = np.random.default_rng(0)
module = "model.layers.1.mlp.down_proj"
edit = Edit.from_deltas(
    [rank1(module, w.standard_normal(16), w.standard_normal(48))], "demo"
)
save_peft(edit, "adapter")
# pre-screen results
screens = [{"name": n, "passed": True, "value": 0.05, "limit": 0.1}
           for n in ("norm", "perplexity", "kl")]  # fmt: skip
Path("screens.json").write_text(json.dumps(screens), encoding="utf-8")
# agent run record: the exported edits are the last that passed confirm
record = RunRecord.start("record.jsonl", seed=0, config={}, editor="demo", model=None)
record.event("confirm_query", query=1)
record.event("confirm_result", query=1, candidate="c1", best_candidate="c1")
record.event("confirm_export", best_candidate="c1", exported_candidate="c1",
             exported_is_best=True)  # fmt: skip
record.finish(budget={}, outcome="finished")
```

Write `GATE.toml`. Paths are relative to its directory. `margin` is the test benchmark's non-inferiority margin, at most 0.02, fixed from a pilot before the test is scored; `margin_source` says where it came from:

```python
Path("GATE.toml").write_text(
    """\
margin = 0.02
margin_source = "pilot on dev, 2026-11-20"
backend = "hf"
build = "demo"
screens = "screens.json"
run_record = "record.jsonl"
manifest = "splits.json"
boot = 2000

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

[suites.gsm8k]
base = "gsm8k/base.jsonl"
edit = "gsm8k/edit.jsonl"
margin = 0.02
source = "pilot on dev, 2026-11-20"
""",
    encoding="utf-8",
)
```

Run it. On the command line this is `astro-weights gate GATE.toml`:

```python
from astro_weights.cli import main

code = main(["gate", "GATE.toml"])
print("exit", code, json.loads(Path("verdict.json").read_text())["verdict"])
```

Prints:

```text
outcome: confirmed
  stage 1  PASS  screen norm  value=+0.0500  limit=+0.1000
  stage 1  PASS  screen perplexity  value=+0.0500  limit=+0.1000
  stage 1  PASS  screen kl  value=+0.0500  limit=+0.1000
  stage 3  PASS  target beats control q0.95 (controls beaten >= limit)  value=+20.0000  limit=+19.0000
  stage 3  PASS  side effect accuracy inside control range  value=-  limit=-
  stage 4  PASS  accuracy non-inferior (bootstrap CI lower > -margin)  value=-0.0167  limit=-0.0200
  stage 4  PASS  tokens lower beyond floor  value=-30.0452  limit=-1.4800
  stage 5  PASS  suite gsm8k non-inferior (bootstrap CI lower > -margin)  value=-0.0075  limit=-0.0200
MDE (test): 0.0230
margin: 0.0200 (pilot on dev, 2026-11-20)
suite gsm8k margin: 0.0200 (pilot on dev, 2026-11-20)
candidates tried: 1  confirm queries: 1

arms on the confirm split, each against base:
arm   kind  n    diff     95% CI              method  MDE     tokens diff
edit  edit  182  +0.0000  [-0.0265, +0.0265]  normal  0.0378  -29.6
controls: n=20  diff min=-0.0110  median=+0.0000  max=+0.0110  MDE min=0.0000  median=0.0308  max=0.0408
cap-hit (max_new_tokens 256): base=0.0%  edit=0.0%  controls max=0.0% (c00)

arms on the test split, each against base:
arm   kind  n    diff     95% CI              method  MDE     tokens diff
edit  edit  299  +0.0000  [-0.0161, +0.0161]  normal  0.0230  -30.0
cap-hit (max_new_tokens 256): base=0.0%  edit=0.0%
verdict file: verdict.json
exit 0 confirmed
```

Stage 2 (selection on confirm) happens in the agent run, so the report has no stage 2 rows. `verdict.json` holds the verdict, backend, build, `config_hash`, manifest sha256, candidates tried, confirm queries, infra errors, footprint, `max_new_tokens` (the arms' cap) and `cap_hit` (per split, the share of each arm's records that hit the cap: base, edit, every control, every report arm), and no timestamp. `--json` prints the whole result as one object; `--out FILE` moves `verdict.json`.

### Cap-hit guard

An answer cut at `max_new_tokens` has no final answer, so it is graded wrong and its tokens stop at the cap: a cap that bites hides both accuracy and token changes. The report prints each split's cap-hit line, and `verdict.json` keeps the shares. Set `max_cap_hit` in `GATE.toml` (a share in [0, 1], for example `max_cap_hit = 0.01`) to refuse the gate when any arm on the confirm split (base, edit, controls) or the test split (base, edit, report arms, all K records) is above it. The refusal names the arms and shares, exits 2 and writes no verdict; the test check runs before the seal, so a refused run never spends the test split. Raise `max_new_tokens` and rerun those arms. Without the key nothing is refused.

The test split is now sealed for this edit. A second run is refused and seals nothing:

```python
print("exit", main(["gate", "GATE.toml"]))
```

Prints `exit 2`, with `error: already scored (ledger seal/seal.jsonl): ...` on stderr. `--unseal` scores it again and the ledger records that it did.

A failed pre-screen stops the gate before any eval, so the test split is never read:

```python
screens[2].update(passed=False, value=0.3)  # kl above its 0.1 limit
Path("screens.json").write_text(json.dumps(screens), encoding="utf-8")
print("exit", main(["gate", "GATE.toml", "--out", "screened.json"]))
print(Path("screened.json").exists())
```

Prints:

```text
outcome: screened out
  stage 1  PASS  screen norm  value=+0.0500  limit=+0.1000
  stage 1  PASS  screen perplexity  value=+0.0500  limit=+0.1000
  stage 1  FAIL  screen kl  value=+0.3000  limit=+0.1000
margin: 0.0200 (pilot on dev, 2026-11-20)
advice: keep the original weights
candidates tried: 1  confirm queries: 1
no verdict file written
exit 3
False
```

No `verdict.json` is written: the verdict schema has no screened-out value.

## Exit codes

| Exit | Outcome |
|---|---|
| 0 | `confirmed` |
| 1 | `regressed` or `not significant` |
| 3 | `screened out` |
| 2 | bad input or refused: unknown key, records outside their split, arms whose headers disagree, `k` not 4, no stored floor, fewer than 20 controls, exported edits not the last that passed confirm, a second scoring without `--unseal` |
