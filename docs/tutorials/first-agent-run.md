# First agent run

!!! note "New in 0.2 (pre-release)"

In this tutorial you run the agent loop once on a tiny random model, with no API key and no GPU. The scripted editor stands in for a model: it sends a fixed list of tool calls to the MCP server, saves one edit, runs two forward passes and finishes. Then you read what the run left on disk.

You need the `mcp` and `hf` extras of a pre-release:

```bash
pip install --pre "astro-weights[mcp,hf]"
```

## 1. Make a model

A two-layer random Llama with a word-level tokenizer stands in for a real model, so nothing is downloaded. The server loads the model from a directory, so save both to `tiny-model/`.

```python
import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers

torch.manual_seed(0)
config = transformers.LlamaConfig(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=64,
)
transformers.LlamaForCausalLM(config).save_pretrained("tiny-model")

words = ["<unk>", "<s>", "</s>"] + [f"w{i}" for i in range(125)]
tok = Tokenizer(
    models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>")
)
tok.pre_tokenizer = pre_tokenizers.Whitespace()
transformers.PreTrainedTokenizerFast(
    tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
).save_pretrained("tiny-model")
```

## 2. Write the run config

A run config is a TOML file. It names the run directory, the model, the budget and the editor. Relative paths are read from the config file's directory.

```python
from pathlib import Path

Path("agent.toml").write_text("""\
[run]
dir = "runs/first"
seed = 0
model_dir = "tiny-model"

[limits]
forward_passes = 4
code_runs = 4
tokens = 0
dollars = 0

[editor]
kind = "scripted"

[forward]
device = "cpu"
""")
```

The scripted editor uses 2 code runs and 2 forward passes, so a limit of 4 each leaves room.

## 3. Run it

```bash
astro-weights run agent.toml
```

The same from Python:

```python
from astro_weights.cli import main

assert main(["run", "agent.toml"]) == 0
```

`run` writes `runs/first/run.json` and a copy of the config, then starts the editor. The editor starts the MCP server (`python -m astro_weights.mcp runs/first`) as its child and talks to it over stdio. Exit code 0 means the run finished.

On Linux the agent's code runs under bubblewrap, on macOS under Seatbelt. With neither, the run stops with an error; see [Explanation: agent budgets and sandbox](../explanation/agent-budgets-and-sandbox.md).

## 4. Read the ledger

`ledger.jsonl` holds one line per charge, refusal or failure. Only finished work is charged.

```python
import json

ledger = [json.loads(line) for line in open("runs/first/ledger.jsonl")]
for entry in ledger:
    print(entry["op"], entry["kind"], entry["amount"], entry["ref"])
```

Prints four charges, two `code_runs` and two `forward_passes`, each with the handle of the job that spent it, for example `charge code_runs 1 exec-7b2ba8487f2dbaf9`.

## 5. Read the record

`record.jsonl` is the run's history: a `start` line (seed, limits, editor, library version), events such as `edit_saved`, the editor's `summary` and a final `finish` line with the budget.

```python
record = [json.loads(line) for line in open("runs/first/record.jsonl")]
print([e["event"] for e in record])
print(record[-1]["outcome"], record[-1]["budget"]["used"])
```

Prints:

```text
['start', 'edit_saved', 'summary', 'finish']
finished {'forward_passes': 2, 'code_runs': 2, 'tokens': 0, 'dollars': 0.0}
```

The saved edit sits in `runs/first/edits/`, and each job's request and result in `runs/first/jobs/`.

## Next

- Let a model drive the run: [How-to: run the agent on a local model](../how-to/agent-local-model.md) or [on Claude with your own key](../how-to/agent-hosted.md).
- Drive the server from your own MCP client: [Claude Code](../how-to/agent-claude-code.md), [Codex](../how-to/agent-codex.md), [Claude Agent SDK](../how-to/agent-sdk.md).
- What the budget counts and what the sandbox does not stop: [Explanation: agent budgets and sandbox](../explanation/agent-budgets-and-sandbox.md).
