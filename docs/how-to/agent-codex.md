# Drive the agent tools from Codex

!!! note "New in 0.2"

Add `astro-weights mcp RUN_DIR` to Codex as a stdio MCP server and Codex becomes the editor.

You need the `mcp` extra, plus `hf` for `forward_pass`:

```bash
pip install "astro-weights[mcp,hf]"
```

## 1. Make a run directory

Write `RUN_DIR/run.json` as in [Drive the agent tools from Claude Code](agent-claude-code.md#1-make-a-run-directory); every key is in [Reference: CLI](../reference/cli.md#mcp). Use an absolute `RUN_DIR`. Tools, result keys and refusals: [Reference: MCP server](../reference/mcp-server.md).

## 2. Add the server

```bash
codex mcp add astro-weights -- astro-weights mcp /path/to/run
```

Or in `~/.codex/config.toml`:

```toml
[mcp_servers.astro-weights]
command = "astro-weights"
args = ["mcp", "/path/to/run"]
```

Syntax from the [Codex MCP docs](https://learn.chatgpt.com/docs/extend/mcp?surface=cli). Codex stops a tool call after `tool_timeout_sec` (60 s default per those docs). `exec_python` and `forward_pass` return a handle at once and the work runs in the background, so a long job does not hit that timeout; Codex polls `status(handle)`.

## 3. Run

Start `codex` and ask it to use the astro-weights tools to edit the model and call `finish` when done.

To run without prompts, use `codex exec`. It does not ask before a tool call, so it refuses every MCP call with `MCP tool call requires approval, but approval policy is never` unless the server's tools are approved in `~/.codex/config.toml`:

```toml
[mcp_servers.astro-weights]
command = "astro-weights"
args = ["mcp", "/path/to/run"]
default_tools_approval_mode = "approve"
```

Or for one run: `codex exec -c 'mcp_servers.astro-weights.default_tools_approval_mode="approve"' "..."`. Codex's own shell stays under its sandbox (`-s read-only` is enough, since the work runs in the MCP server).

`finish` exports each edit at rank `export.rank` (default 8), so ask for low-rank changes. A dense edit fails the export with `export_error`.

## Limits

- The server counts `code_runs` and `forward_passes`. Codex's own tokens and dollars are not reported to the server, so `tokens` and `dollars` limits do not cap a Codex session. Your own Codex limits and your provider's terms apply.
- Codex also sees the editor tools `report_usage`, `check_spend` and `report_refusal`.
- With no OS sandbox, `exec_python` returns an error; see [Explanation: agent budgets and sandbox](../explanation/agent-budgets-and-sandbox.md).

Tested once: a headless `codex exec` run (codex-cli 0.171.0, signed in with ChatGPT, astro-weights 0.2.2, macOS arm64) on a tiny random model used `exec_python`, `forward_pass`, `status` and `finish`, and the run finished with an exported adapter.
