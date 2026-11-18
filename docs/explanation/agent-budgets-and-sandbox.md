# Agent budgets and sandbox

!!! note "New in 0.2"

The MCP server runs code an agent wrote, on your machine, against your model. Two things limit it: a budget the agent cannot raise and an OS sandbox around its code. This page says what each one counts, when it charges, and what it does not stop.

## What the budget counts

Each run has four limits in `run.json`: `forward_passes`, `code_runs`, `tokens` and `dollars`. 0 allows none.

The ledger `RUN_DIR/ledger.jsonl` is the only source of truth. Each line is a `charge`, `refuse`, `fail` or `reset`, written and fsynced before the call returns. Used budget is the sum of the `charge` lines; no other file is read, so editing or rolling back another file cannot lower it.

| Kind | Charged by | When | Past the limit |
|---|---|---|---|
| `code_runs` | `exec_python` | after the cell ran | refused before it starts; a job that finishes past the limit is failed and its result withheld |
| `forward_passes` | `forward_pass` | after the pass ran | the same |
| `tokens` | the editor's `report_usage` | after each model turn | always recorded, since the tokens are already spent; the editor then stops |
| `dollars` | the editor's `report_usage` | after each model call | always recorded; the hosted editor checks the bound before each call so it does not get there |

- Only successful work spends. A refused request is free. A bad `forward_pass` request is refused in the tool call, free, with no job.
- A failed job is free but extends that kind's failure streak. After 3 failures in a row that kind is refused until an operator resets it. A charge resets the streak.
- The scripted editor reports no usage. The local editor reports tokens and $0. The hosted editor reports tokens and dollars.
- After `finish`, `exec_python` and `forward_pass` return `run already finished`. A `report_usage` after `finish` is still recorded.

## The sandbox

`exec_python` runs each cell in a long-lived child process. The child can read the Python interpreter, its standard library and numpy, and write only `RUN_DIR/kernel`. It has no network, no GPU or accelerator devices and no view of other processes. It never opens the model or the edits: `load`, `load_original`, `save_edit`, `drop_edit`, `edited`, `names` and `forward_result` are requests the server checks and answers. Its environment holds only `PATH`, `LANG`, `LC_ALL`, `HOME`, `TMPDIR` and `CUDA_VISIBLE_DEVICES` (plus `SYSTEMROOT` on Windows), so API keys in the server's environment do not reach it.

| OS | Sandbox | Notes |
|---|---|---|
| Linux | bubblewrap | needs bwrap 0.8 or newer (`--disable-userns`); a bwrap that AppArmor blocks counts as none. With an older bwrap (RHEL 9 ships 0.6), `kernel.allow_nested_userns = true` runs the same sandbox without `--disable-userns`: agent code can then create user namespaces, which opens more of the Linux kernel to it. The run record gets a `sandbox_compat` event. Memory capped by `kernel.mem_gib` (default 16) |
| macOS | Seatbelt (`/usr/bin/sandbox-exec`) | deny by default. No memory cap |
| Windows | none | `exec_python` is refused |

Each server process tries the sandbox once for real. When none works, `exec_python` returns an error and the budget is unchanged. `--unsafe-kernel` (on `astro-weights mcp` and every editor; `unsafe = true` under `[kernel]` in a run config) then runs the code with your full user rights instead. It changes nothing where a sandbox works; the run record gets an `unsafe_kernel` event. Each cell also has a wall clock limit, `kernel.timeout_s` (default 600 s); a killed cell loses the kernel's variables.

The kernel has numpy only: no torch, safetensors or astro_weights. The model runs in the server process, through `forward_pass`.

## What it does not protect against

- A numpy forward pass. The kernel can read every editable weight with `load`, so an agent can write its own forward pass in numpy. That costs one `code_runs` per cell, not `forward_passes`.
- Disk fill. Nothing caps the size of `RUN_DIR/kernel`; a cell can fill the disk there.
- Your own MCP client. Claude Code, Codex or the Agent SDK make their model calls themselves; the server never sees those tokens or dollars, so the `tokens` and `dollars` limits do not cap them. That client also sees the editor tools `report_usage`, `check_spend` and `report_refusal`, which the built-in editors hide. Your API key there is that client's business, not the server's.
- Windows unsafe mode. With `--unsafe-kernel` on Windows the code runs with no sandbox and no memory cap.
- Unsafe mode anywhere. On a machine with no working sandbox, `--unsafe-kernel` drops every sandbox rule on this page.
