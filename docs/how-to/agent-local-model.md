# Run the agent on a local model

!!! note "New in 0.2 (pre-release)"

The local editor lets a model behind any OpenAI-compatible server with tool calls (vLLM, llama-server) drive the MCP server. It costs $0: tokens go to the ledger, dollars stay 0.

You need the `mcp` and `hf` extras:

```bash
pip install --pre "astro-weights[mcp,hf]"
```

## 1. Start a model server with tool calls on

vLLM needs `--enable-auto-tool-choice` and `--tool-call-parser` ([vLLM tool calling docs](https://github.com/vllm-project/vllm/blob/main/docs/features/tool_calling.md)); pick the parser for your model:

```bash
vllm serve MODEL --enable-auto-tool-choice --tool-call-parser PARSER --port 8000
```

llama-server needs `--jinja` ([llama.cpp function calling docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)):

```bash
llama-server --jinja -m MODEL.gguf --port 8000
```

This model is the editor. The model you edit is `model_dir` below; it may be a different one.

## 2. Write the run config

```toml
[run]
dir = "runs/local"
seed = 0
model_dir = "/path/to/model"

[limits]
forward_passes = 20
code_runs = 50
tokens = 2000000
dollars = 0

[editor]
kind = "local"
base_url = "http://127.0.0.1:8000"
model = "MODEL"
max_turns = 40
temperature = 0.0
```

- `base_url`: `http://` or `https://`, with or without a trailing `/v1`. The editor posts to `{base_url}/v1/chat/completions`.
- `model`: the name the server expects.
- `api_key_env`: optional. The name of an environment variable that holds the server's key, for example `api_key_env = "VLLM_API_KEY"`. The key is never read from the config or the command line.
- `max_turns` (default 40) and `temperature` (default 0.0).

Every key is in [Reference: CLI](../reference/cli.md#run).

## 3. Run

```bash
astro-weights run local.toml
```

The editor sends the model the system prompt, the budget and the tool list, then runs each tool call the model makes and waits for its job to end, so the model sees one result per call. It stops when the model calls `finish`, after `max_turns`, when a limit is reached (the editor calls `finish` itself), or when the model twice in a row replies with no tool call. Each tool result the model sees is capped at 16384 characters.

Exit codes: 0 the run finished; 1 the model refused (`finish_reason` `content_filter`) or the loop failed, with the refusal or error as JSON on stdout; 2 bad arguments, a bad run directory, an unset key variable or a server that did not start. The key is removed from every message.

The server's own terms apply when it serves a hosted model.

Not tested: `astro-weights run` (checked against its spec only) and real vLLM or llama-server output. The editor itself is tested against a fake OpenAI-compatible server.
