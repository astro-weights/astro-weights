# Drive the agent tools from Claude Code

!!! note "New in 0.2"

`astro-weights mcp RUN_DIR` serves the agent tools over stdio MCP. Add it to Claude Code as a local server and Claude Code becomes the editor: it calls `exec_python`, `forward_pass`, `status` and `finish` on your model.

You need the `mcp` extra, plus `hf` for `forward_pass`:

```bash
pip install "astro-weights[mcp,hf]"
```

## 1. Make a run directory

The server reads `RUN_DIR/run.json`. `model_dir` is read relative to `RUN_DIR`. Every key is listed in [Reference: CLI](../reference/cli.md#mcp).

```json
{
  "seed": 0,
  "editor": "claude-code",
  "model": null,
  "limits": {"forward_passes": 20, "code_runs": 50, "tokens": 0, "dollars": 0},
  "model_dir": "/path/to/model"
}
```

Save it as `/path/to/run/run.json`. Use one run directory per run: a restart with a changed `seed`, `editor`, `model` or `limits` is refused.

## 2. Add the server

```bash
claude mcp add --transport stdio astro-weights -- astro-weights mcp /path/to/run
```

Everything after `--` is the server command. Use an absolute `RUN_DIR`; Claude Code may start the server from another directory. `--scope project` writes the server to `.mcp.json` instead ([Claude Code MCP docs](https://code.claude.com/docs/en/mcp)).

The same as a `.mcp.json` entry:

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

## 3. Run

Start `claude` and ask it to edit the model, for example "Use the astro-weights tools to find an edit that shortens answers; call finish when done." The tools are named `mcp__astro-weights__exec_python` and so on; Claude Code asks before each call unless you allow them.

`exec_python` and `forward_pass` return a handle at once. Claude polls `status(handle)` until the job is `done` or `failed`. Read arrays in `exec_python` with `forward_result(handle, key)`.

## Limits

- The server counts `code_runs` and `forward_passes` and refuses work past them. Claude Code's own tokens and dollars are not reported to the server, so `tokens` and `dollars` limits do not cap a Claude Code session. Your own Claude Code limits and your provider's terms apply.
- Claude Code also sees the three editor tools `report_usage`, `check_spend` and `report_refusal`. The built-in editors hide them from their model; Claude Code does not.
- No OS sandbox (Windows, or a blocked bwrap): `exec_python` returns an error. `astro-weights mcp RUN_DIR --unsafe-kernel` runs the code unsandboxed; see [Explanation: agent budgets and sandbox](../explanation/agent-budgets-and-sandbox.md).

Tested once: a headless `claude -p` run on a tiny random model (0.2.0a1, macOS) used `exec_python`, `forward_pass`, `status` and `finish`, and the run finished with an exported adapter.
