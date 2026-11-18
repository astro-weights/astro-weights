# Run the agent on Claude with your own key

!!! note "New in 0.2 (pre-release)"
    The hosted editor comes after 0.2.0a1.

The hosted editor lets Claude drive the MCP server on your own Anthropic API key. It checks the dollar budget before every call and writes every call's cost to the ledger. When Claude refuses, the same run goes on with a local model.

You need the `mcp`, `hf` and `hosted` extras, and a key in the environment:

```bash
pip install --pre "astro-weights[mcp,hf,hosted]"
export ANTHROPIC_API_KEY=...
```

The key is read from `ANTHROPIC_API_KEY` only. It is never printed or written to the run directory, and the MCP server process does not get it. Your own agreement with Anthropic applies to the calls the editor makes on your key.

## 1. Write the run config

```toml
[run]
dir = "runs/hosted"
seed = 0
model_dir = "/path/to/model"

[limits]
forward_passes = 20
code_runs = 50
tokens = 2000000
dollars = 20.0

[editor]
kind = "hosted"
model = "claude-opus-5-5"
max_output_tokens = 16000
max_turns = 40
fallback_base_url = "http://127.0.0.1:8000"
fallback_model = "MODEL"
```

- `dollars`: the run's dollar cap. Default 20.0 for the hosted editor. Set your own; 0 is refused, since a hosted run needs a dollar budget.
- `model`: a Claude model id with a price in `astro_weights.mcp.editors.prices.PRICES` ([Reference](../reference/index.md)). Any other id is refused before the run starts.
- `max_output_tokens` (default 16000) and `max_turns` (default 40).
- `fallback_base_url` and `fallback_model`: the OpenAI-compatible server and model that take over after a refusal; set it up as in [How-to: run the agent on a local model](agent-local-model.md#1-start-a-model-server-with-tool-calls-on). Or `no_fallback = true` to end the run on a refusal. One of the two is required.

Every key is in [Reference: CLI](../reference/cli.md#run).

## 2. Run

```bash
astro-weights run hosted.toml
```

## How spend is capped

- Before each call the editor counts the prompt's tokens, adds 5%, and prices that prompt plus `max_output_tokens` at the model's price. If that bound could take the run past `dollars`, the call is not made and the run finishes with outcome `dollar_cap`.
- After each call its tokens and dollars go to the ledger. A stream that breaks off mid-call is charged the full bound, since tokens may have been billed.
- Prices come from the [Claude pricing page](https://platform.claude.com/docs/en/about-claude/pricing); the read date is in the `prices` module docstring. Check your own bill; the ledger is an estimate from that table.

## Refusals

When Claude stops with a refusal, the refused turn is still charged, its partial output is dropped and the refusal category goes to `record.jsonl`. The fallback model then gets the same messages plus a note that it now drives the run, and goes on at $0. With `no_fallback = true` the run ends with outcome `refused`.

Exit codes: 0 the run finished (by the model, the turn cap, a spent budget or the dollar cap); 1 refused with no fallback, or a loop or API error (as JSON on stdout, key removed); 2 bad arguments, a bad run directory, an unknown model, a dollar limit of 0, an unset key or a server that did not start.

Not tested: never run against the real API; tested with fakes only, no key, no network. `astro-weights run` was checked against its spec only.
