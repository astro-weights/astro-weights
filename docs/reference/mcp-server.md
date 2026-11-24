# MCP server

!!! note "New in 0.2"

`astro-weights mcp RUN_DIR` is a stdio MCP server for one run directory. It serves seven tools. Any MCP client can drive it: the built-in editors of [`astro-weights run`](cli.md#run), Claude Code, Codex or your own.

Related pages: [Tutorial: first agent run](../tutorials/first-agent-run.md) for a first run, [Explanation: agent budgets and sandbox](../explanation/agent-budgets-and-sandbox.md) for why the budget and sandbox work as they do.

## Start

```text
astro-weights mcp RUN_DIR [--unsafe-kernel]
python -m astro_weights.mcp RUN_DIR [--unsafe-kernel]
```

Both start the same server. Install the `mcp` extra; `forward_pass` also needs `hf`:

```bash
pip install --pre "astro-weights[mcp,hf]"
```

- The server reads `RUN_DIR/run.json` once, at start. Write it before you start the server.
- One server process per run directory. The ledger and the record have no cross-process lock.
- Exit code 2, with `error: ...` on stderr, when `run.json` is missing or bad, or when it differs from an existing run record on `seed`, `editor`, `model` or `limits`. `astro-weights mcp` also exits 2, with the install hint, when the `mcp` extra is missing.
- `--unsafe-kernel` runs agent code without an OS sandbox when none works. It changes nothing where a sandbox works. See [Explanation: agent budgets and sandbox](../explanation/agent-budgets-and-sandbox.md#the-sandbox).

## Client config

Most clients take a command and its arguments. Use an absolute `RUN_DIR`; a client may start the server from another directory.

```json
{
  "mcpServers": {
    "astro-weights": {
      "command": "astro-weights",
      "args": ["mcp", "/path/to/run"]
    }
  }
}
```

Per client: [Claude Code](../how-to/agent-claude-code.md), [Codex](../how-to/agent-codex.md), [Claude Agent SDK](../how-to/agent-sdk.md).

## Run directory

| Path | Written by | Holds |
|---|---|---|
| `run.json` | you, or `astro-weights run` | seed, labels, limits, model and settings; see [run.json](#runjson) |
| `record.jsonl` | server | run history, one JSON event per line; see [Run record](#run-record) |
| `ledger.jsonl` | server | budget: one `charge`, `refuse`, `fail` or `reset` per line. The only source of truth for spend |
| `jobs/<handle>.json` | server | one file per job: request, state, result |
| `edits/` | server | `manifest.json` plus one `<sha256>.safetensors` per stored edit |
| `forward/<id>/` | server | results of one `forward_pass` job: `<key>.npy` arrays, `texts*.json` |
| `kernel/` | agent code | the only directory `exec_python` code can write |
| `kernel.log`, `kernel.started` | server | the kernel process's log, and a marker that a kernel has run here |
| `export/` | server, on `finish` | `adapter/` and, with `export.base_gguf`, `model.gguf` |

`astro-weights run` also writes `config.toml` here; see [Reference: CLI](cli.md#run).

## run.json

| Key | Type | Default |
|---|---|---|
| `seed` | int | required |
| `editor` | non-empty string, a label for the record | required |
| `model` | string or `null`, a label for the record | required |
| `limits` | object with exactly the four keys below | required |
| `model_dir` | path to the model, relative to `RUN_DIR` or absolute | absent: `exec_python` and `forward_pass` return `... is not available yet` |
| `kernel` | object: `timeout_s` (number > 0), `mem_gib` (int >= 1, Linux only), `allow_nested_userns` (bool: sandbox with bwrap older than 0.8) | `600.0`, `16`, `false` |
| `forward` | object: `device` (`auto`, `cpu`, `cuda`, `mps`; `auto` picks `cuda`, then `mps`, then `cpu`), `dtype` (`auto`, `float32`, `bfloat16`, `float16`), `max_new_tokens` (int from 1 to 32768) | `auto`, `auto`, `4096` |
| `export` | object: `rank` (int >= 1), `max_rel_residual` (finite number >= 0), `base_gguf` (path, relative to `RUN_DIR` or absolute), `gguf_out_type` (`q8_0`, `f16`, `bf16`) | `8`, `0.05`, absent, `q8_0` |
| `confirm` | object: `items` (JSONL of `item_id`, `prompt`, `gold`; outside `RUN_DIR`), `grader` (`exact`, `choice`, `math`), `margin` (number from 0 below 1, in items correct per item), `max_queries` (int >= 1); every key required, and `model_dir` too | absent: no `confirm` tool |

An unknown key inside `kernel`, `forward` or `export` is an error.

### Limits

All four keys are required. 0 allows none.

| Key | Type | Counts | Past the limit |
|---|---|---|---|
| `forward_passes` | int >= 0 | finished `forward_pass` jobs | refused before the job starts |
| `code_runs` | int >= 0 | finished `exec_python` jobs | refused before the job starts |
| `tokens` | int >= 0 | tokens sent with `report_usage` | still recorded; listed in `over` |
| `dollars` | finite number >= 0 | dollars sent with `report_usage` | still recorded; listed in `over` |

`limits` cannot change once the run has a record: a restart with other limits exits 2.

## Tools

| Tool | Arguments | Returns | Charges | Mode |
|---|---|---|---|---|
| `exec_python` | `code`: string | `{"handle", "state": "queued"}` | 1 `code_runs` when the job finishes | submit, then poll |
| `forward_pass` | `request`: object | `{"handle", "state": "queued"}` | 1 `forward_passes` when the job finishes | submit, then poll |
| `status` | `handle`: string | the job plus `budget` | nothing | sync |
| `finish` | `summary`: string | `{"state": "finished", "budget", "export"}` | nothing | sync, works once |
| `report_usage` | `tokens`, `dollars`, `turn`, `model` | `{"budget", "over"}` | `tokens` and `dollars` | sync, editor-only |
| `check_spend` | `estimate` | `{"ok", "budget"}` | nothing | sync, editor-only |
| `report_refusal` | `category`, `model`, `detail` (optional) | `{"recorded": true}` | nothing | sync, editor-only |
| `confirm` | none | `{"handle", "state": "queued"}`; job `result` is exactly `{"passed": bool}` | nothing; at most `confirm.max_queries` per run, later calls refused | submit, then poll; only with `confirm` in `run.json` |

A refusal is an ordinary tool result with an `error` key, not an MCP protocol error. Check every result for `error`.

### Jobs and `status`

`exec_python` and `forward_pass` return at once. The work runs in a background thread. Poll `status(handle)` until `state` is `done` or `failed`.

`status` returns:

| Key | Value |
|---|---|
| `handle` | `exec-<16 hex digits>` or `forward-<16 hex digits>` |
| `kind` | `exec` or `forward` |
| `state` | `queued`, `running`, `done` or `failed` |
| `params` | the request as stored; a `forward_pass` request has its defaults filled in |
| `result` | object when `done`, else `null` |
| `error` | string when `failed`, else `null`; at most 2000 characters |
| `created`, `updated` | Unix times, seconds |
| `budget` | the [budget object](#budget-object) |

- A `done` job was charged. A `failed` job is free. A job that failed because its work raised adds 1 to that kind's failure streak.
- A restarted server marks every `queued` or `running` job `failed` with error `server restarted`.

### `exec_python`

Runs one cell in a sandboxed kernel that has numpy (`np`) and the standard library only. Variables stay between cells. The kernel's helpers (`NAMES`, `names()`, `load`, `load_original`, `save_edit`, `drop_edit`, `edited()`, `forward_result`) are described in [Explanation: agent budgets and sandbox](../explanation/agent-budgets-and-sandbox.md#the-sandbox).

Job `result`:

| Key | Value |
|---|---|
| `ok` | `false` when the cell raised, was killed or the kernel died |
| `output` | what the cell printed, plus the traceback when it raised; cut at 10000 characters |
| `restarted` | `true` when this cell started a fresh kernel after an earlier one was lost; earlier variables are gone |

A cell that raises still ends `done`, with `ok: false`, and costs 1 `code_runs`. A cell that runs longer than `kernel.timeout_s` is killed and its variables are lost.

### `forward_pass`

Runs the model in the server process, on the base or the edited weights. The request is checked in the tool call; a bad one is refused for free and leaves no job. An unknown key is refused.

`"kind": "teacher_forced"`:

| Key | Type | Default |
|---|---|---|
| `texts` | non-empty list of non-blank strings, at most 1024 | required |
| `return` | non-empty list drawn from `hidden`, `mlp`, `attn`, `logprobs` | required |
| `layers` | list of layer indices, no repeats | `[]`; required for `hidden`, `mlp` and `attn` |
| `last_tokens` | int from 1 to 32768 | `36` |
| `arm` | `base` or `edited` | required |

`"kind": "generate"`:

| Key | Type | Default |
|---|---|---|
| `prompts` | non-empty list of non-blank strings, at most 512 | required |
| `max_new_tokens` | int from 1 to `forward.max_new_tokens` of `run.json` | required |
| `arm` | `base`, `edited` or `both` | required |
| `seed` | int from 0 to 2**63 - 1 | `0` |
| `temperature` | finite number >= 0; 0 is greedy | `0.0` |

Limits on a request:

- Each text or prompt: at least 1 token, at most 32768. All together: at most 524288. A prompt is counted after the model's chat template is applied, when the tokenizer has one.
- `teacher_forced`: `len(texts) * last_tokens` at most 4000000.
- `attn`: at most 2 texts of at most 2048 tokens each, full attention layers only, and `texts * layers * heads * max_tokens**2` at most 268435456 (2**28).
- `arm` `edited` or `both` needs at least one stored edit.

Job `result` for `teacher_forced`: `kind`, `arm`, `n`, `total_tokens`, `last_tokens`, `edits` (weight name to sha256 of the edit applied), `arrays` (key to shape) and `keys`. Array keys: `hidden_l<layer>`, `mlp_l<layer>`, `attn_l<layer>_t<text>`, `logprobs_top_ids`, `logprobs_top`, `logprobs_target`.

Job `result` for `generate`: `kind`, `arm`, `n`, `total_tokens`, `max_new_tokens`, `seed`, `temperature`, `arms` and `keys`. `arms` has one entry per arm run, each with `edits`, `texts`, `truncated`, `n_tokens` and `hit_cap`. `texts` there is a preview: at most 4096 characters per text and 32768 over the whole result. `keys` is `["texts"]`, or `["texts_base", "texts_edited"]` for `"arm": "both"`.

The result holds shapes and previews only. Read an array or the full texts in `exec_python` with `forward_result(handle, key)`, using a key from `keys`.

### `finish`

Ends the run. Works once; `summary` is stored in the record, cut at 2000 characters.

When `run.json` names a `model_dir` and there are stored edits, the server exports them to `RUN_DIR/export/`: a PEFT LoRA adapter in `export/adapter` and, with `export.base_gguf`, a merged GGUF at `export/model.gguf`. Each edit is cut to `export.rank` by truncated SVD and must keep its relative residual within `export.max_rel_residual`.

The result is `{"state": "finished", "budget": {...}}` plus one of:

| Case | Keys added |
|---|---|
| exported | `"export": {"adapter", "gguf", "residuals", "rank"}`; `gguf` is `null` without `export.base_gguf` |
| no stored edits | `"export": null, "note": "no edits: nothing to export"` |
| no `model_dir` | `"export": null, "note": "no model_dir: nothing to export"` |
| export failed | `"export": null, "export_error": "<ExceptionType>: <message>"` |

A failed export does not stop the run from finishing. Inspect the adapter with [`astro-weights inspect`](../how-to/inspect-adapter.md).

### Editor-only tools

`report_usage`, `check_spend` and `report_refusal` are for the client program, not for the model. A client that shows tools to a model should hide these three; the built-in editors do. Claude Code and Codex show all seven.

| Tool | Arguments | Result |
|---|---|---|
| `report_usage` | `tokens` and `turn`: ints >= 0. `dollars`: finite number >= 0. `model`: non-empty string, at most 200 characters | `{"budget", "over"}`; `over` lists the kinds now past their limit. Bad arguments: `{"error": "bad usage report"}` |
| `check_spend` | `estimate`: finite number >= 0, in dollars | `{"ok": true, "budget"}` or `{"ok": false, "error", "budget"}`. Writes nothing. Bad argument: `{"error": "estimate must be a finite number >= 0"}` |
| `report_refusal` | `category`: non-empty string, cut at 200 characters. `model`: as above. `detail`: string, default `""`, cut at 2000 | `{"recorded": true}`. Bad arguments: `{"error": "bad refusal report"}` |

- `report_usage` is always recorded, even past a limit and after `finish`. Usage recorded after `finish` is not in the `finish` result's `budget`; read spend from `ledger.jsonl`.
- `check_spend` with estimate 0 is ok even when `dollars` is already past its limit, but not while the `dollars` failure streak blocks work.

## Budget object

`status`, `finish`, `report_usage`, `check_spend` and most refusals return `budget`:

| Key | Value |
|---|---|
| `limits` | the four limits of `run.json` |
| `used` | charged so far, per kind |
| `remaining` | limit minus used, per kind, never below 0 |
| `streaks` | failures in a row, per kind |
| `max_failures` | `3` |

## Errors and refusals

Every refusal is free.

| When | Result | Also written |
|---|---|---|
| limit reached, at submit | `{"error": "code_runs budget exhausted: used 2 of 2, requested 1", "budget"}` (or `forward_passes`) | ledger `refuse`; record `refusal`, category `budget` |
| job ends past the limit | job `failed`, `error` `budget exhausted`, result withheld | nothing |
| 3 failed jobs of one kind in a row | `{"error": "code_runs failed 3 times in a row; an operator must fix this and reset it", "budget"}` | ledger `refuse`; record `refusal`, category `failure_streak` |
| bad `forward_pass` request | `{"error": "bad request: ...", "budget"}` | record `refusal`, category `bad_request` |
| the request check itself fails | `{"error": "<ExceptionType>: ...", "budget"}` | record `refusal`, category `forward_check` |
| no working OS sandbox | `exec_python`: `{"error": "no OS sandbox: <why>; pass allow_unsafe=True (CLI: --unsafe-kernel) to run agent code without a sandbox"}` | record `sandbox_unavailable` |
| no `model_dir` | `{"error": "exec_python is not available yet"}` (or `forward_pass`) | record `tool_unavailable` |
| `hf` extra missing | `forward_pass`: `{"error": "forward_pass is not available yet: it needs torch and transformers: pip install 'astro-weights[hf]'"}` | record `tool_unavailable` |
| after `finish` | `{"error": "run already finished"}` from every tool but `status` and `report_usage` | nothing |
| `status` on a bad handle | `{"error": "unknown handle"}` | nothing |

- The sandbox error names the Python argument `allow_unsafe`. From the command line the switch is `--unsafe-kernel`.
- A failure streak is reset by the next charge of that kind. Once it reaches 3, no tool and no CLI command resets it; `astro_weights.mcp.budget.Ledger.reset_failures` does, from Python, with the server stopped.

## Run record

`record.jsonl` starts with a `start` line: `seed`, `config` (the limits), `editor`, `model`, `library_version`, `library_commit` (the git commit of a `pip install git+...` or of an editable checkout, with `-dirty` for uncommitted changes; `null` for a release wheel from PyPI, where `library_version` and its `v` tag name the commit), `python`, `platform`. Every line has `event` and `time`. Events the server writes:

| Event | Fields | When |
|---|---|---|
| `edit_saved` | `tensor`, `rel_frobenius` | `save_edit` stored an edit |
| `edit_rejected` | `tensor`, `error` | `save_edit` refused one |
| `edit_dropped` | `tensor` | `drop_edit` removed one |
| `kernel_restarted` | none | a cell started a fresh kernel after one was lost |
| `refusal` | `category`, `editor`, `model`, `detail` | a refusal from the table above, or `report_refusal` |
| `usage` | `turn`, `model`, `tokens`, `dollars` | `report_usage` |
| `tool_unavailable` | `tool`, sometimes `error` | a work tool that cannot run was called |
| `sandbox_unavailable` | `tool`, `error` | `exec_python` was called with no sandbox |
| `unsafe_kernel` | none | the server started with `--unsafe-kernel` |
| `sandbox_compat` | `lost` | the kernel runs under bwrap without `--disable-userns` (`kernel.allow_nested_userns` on bwrap older than 0.8) |
| `summary` | `text` | `finish` |
| `export` | `adapter`, `gguf`, `residuals`, `rank` | `finish` exported the edits |
| `export_skipped` | `reason` | `finish` found no edits |
| `export_failed` | `error` | the export raised |
| `finish` | `budget`, `outcome` (`finished`) | last step of `finish` |

Python API of the server modules: [Reference](index.md#agent-over-mcp).
