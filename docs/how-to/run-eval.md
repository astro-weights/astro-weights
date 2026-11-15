# Run eval end to end

`generate_records` runs one arm: greedy decoding, one scored `Record` per item. Run it on base model, then on edited model, write both arms as JSONL, compare them with `astro-weights eval`. Generating needs `hf` extra. `grade_math` needs `eval` extra (`pip install "astro-weights[eval]"`).

## Generate both arms

Tiny random Llama and word-level tokenizer below stand in for real model; their scores mean nothing.

```python
import numpy as np
import tokenizers
import torch
import transformers

from astro_weights.edits import Edit, rank1
from astro_weights.eval import grade_exact, write_jsonl
from astro_weights.runners.generate import generate_records, run_header
from astro_weights.runners.hf import apply

torch.manual_seed(0)
config = transformers.LlamaConfig(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
)
model = transformers.LlamaForCausalLM(config).eval()

# Word-level tokenizer: tokens "w0" ... "w124" plus three specials.
vocab = {"<unk>": 0, "<s>": 1, "</s>": 2} | {f"w{i}": i + 3 for i in range(125)}
backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
tok = transformers.PreTrainedTokenizerFast(
    tokenizer_object=backend,
    unk_token="<unk>",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="</s>",
)

items = [
    {"item_id": f"i{k}", "prompt": f"w1 w2 w{k + 3}", "gold": "w5"} for k in range(4)
]
print(run_header(model, tok, max_new_tokens=6, seed=0, chat=False)["model"])

base = generate_records(model, tok, items, "base", grade_exact, 6, chat=False)
write_jsonl("base.jsonl", base)

module = "model.layers.1.mlp.down_proj"
rng = np.random.default_rng(0)
delta = rank1(module, 5 * rng.standard_normal(32), rng.standard_normal(64))
peft_model = apply(model, Edit((delta,), "tiny-llama"))
edit = generate_records(peft_model, tok, items, "edit", grade_exact, 6, chat=False)
write_jsonl("edit.jsonl", edit)
print(len(base), len(edit))
```

Prints:

```text
LlamaForCausalLM
4 4
```

## Write as you go, resume, batch

`out` appends each record to JSONL file as soon as it exists, so crash keeps finished items. Non-empty `out` raises `FileExistsError` unless `resume=True`; resume keeps records already in file and generates only missing items. `batch_size` generates that many items at once with left padding. Continues example above:

```python
from pathlib import Path

from astro_weights.runners.hf import remove

model = remove(peft_model)

done = generate_records(
    model,
    tok,
    items[:2],
    "base",
    grade_exact,
    6,
    chat=False,
    out="base-run.jsonl",
    batch_size=2,
)
# crash here; rerun with resume=True keeps those 2 records, generates the other 2
full = generate_records(
    model,
    tok,
    items,
    "base",
    grade_exact,
    6,
    chat=False,
    out="base-run.jsonl",
    resume=True,
    batch_size=2,
)
print(len(done), len(full), len(Path("base-run.jsonl").read_text().splitlines()))
```

Prints:

```text
2 4 4
```

- Resume raises `ValueError` when file belongs to another run: other arm or seed, `item_id` not in `items`, or prompt hash differs. Last line cut short by crash is dropped and redone.
- Greedy outputs with `batch_size > 1` can differ from `batch_size=1` (padding changes float rounding; near tie can flip). Use one batch size for both arms. Tokenizer needs pad token.

## Run headers

`run_header(model, tokenizer, max_new_tokens, seed, chat, batch_size=...)` returns settings that decide outputs: model class (class under adapter for `apply` result), dtype, device, `max_new_tokens`, seed, chat, `batch_size`, chat template sha256, torch, transformers and peft versions. `write_header(header_path("X.jsonl"), header)` stores it beside records as `X.header.json`. Continues example above:

```python
from astro_weights.eval import check_headers, header_path, write_header

peft_model = apply(model, Edit((delta,), "tiny-llama"))
for arm, m in [("base", model), ("edit", peft_model)]:
    write_header(header_path(f"{arm}.jsonl"), run_header(m, tok, 6, 0, False))
print(header_path("base.jsonl"))

base_h = run_header(model, tok, 6, 0, False)
edit_h = run_header(peft_model, tok, 6, 0, False)
check_headers(base_h, edit_h)  # raises when arms are not comparable
print(base_h["model"], edit_h["model"])
```

Prints:

```text
base.header.json
LlamaForCausalLM LlamaForCausalLM
```

`check_headers(base, edit, ignore=("device",))` raises listing every key that differs or is missing on one side.

## Compare arms

```bash
astro-weights eval base.jsonl edit.jsonl
```

Prints [`format_report`](compare-arms.md) output:

```text
n=4 items  base=0.0000  edit=0.0000
score diff=+0.0000  SE=0.0000  95% CI [+0.0000, +0.0000]; not significant; smallest detectable difference at this n is 0.0000
McNemar exact p=1  (base-only right=0, edit-only right=0)
tokens: base=6.0  edit=6.0  diff=+0.0 (+0.0%)  bootstrap 95% CI [+0.0, +0.0]; not significant; smallest detectable difference at this n is 0.0
cap-hit rate: base=1.000  edit=1.000
WARNING: only 4 items (<100); intervals are unreliable
```

- When `BASE.header.json` and `EDIT.header.json` both exist, `eval` compares them (ignoring `device`) and exits 1 on any difference, with `error: arms ran under different settings ...` on stderr. `--ignore-header` skips check.
- `--floor A.jsonl B.jsonl`: two runs of base arm under same settings; report adds noise-floor lines. See [compare arms](compare-arms.md#noise-floor).
- `--json`: one JSON object on stdout instead of text (`PairedResult` fields plus `floor`; non-finite numbers as `null`).
- `--boot` sets bootstrap resamples (default 10000), `--seed` bootstrap seed (default 0). Unreadable file or arms that do not pair: message on stderr, exit code 2.

## Serve arms over HTTP

`astro_weights.runners.endpoint` runs arm against OpenAI-compatible server (llama-server, vLLM) with standard library only: no torch, no extra. Serve base and edit under two model names, call once per name.

```python
from astro_weights.eval import grade_exact
from astro_weights.runners.endpoint import generate_records_http, run_header_http

URL = "http://127.0.0.1:8080"  # server root; trailing /v1 also accepted
items = [{"item_id": "i0", "prompt": "Say w5.", "gold": "w5"}]
for arm, name in [("base", "base-model"), ("edit", "edit-model")]:
    write_header(header_path(f"{arm}-http.jsonl"), run_header_http(URL, name, 64, 0))
    generate_records_http(
        URL, name, items, arm, grade_exact, 64, out=f"{arm}-http.jsonl"
    )
```

- One `POST /v1/chat/completions` per item, temperature 0, `max_tokens` and `seed` set; `extra_body` keys merged last. Output is `message.content` (`reasoning_content` ignored); tokens from `usage.completion_tokens`; `hit_cap` when `finish_reason` is `length`.
- `out` must be empty or absent; no resume. `api_key` sent as bearer token, never recorded or shown in errors. Redirects refused.
- `run_header_http` records `base_url`, `model`, `max_new_tokens`, `seed`, `extra_body`, plus server's `/v1/models` entry and `system_fingerprint` when server gives them. Two served names give headers that differ on `model` (and `server_model`), so `astro-weights eval` needs `--ignore-header` for HTTP arms.

## What `generate_records` does

- Every item needs `item_id`, `prompt` and `gold` keys. Missing key or repeated `item_id` raises `ValueError` before any generation.
- `chat=True` (default) wraps prompt in tokenizer's chat template when tokenizer has one. Without template, prompt goes in as is.
- Model runs in eval mode; previous train/eval mode comes back afterwards.
- `n_output_tokens` counts tokens up to and including first EOS.
- Records leave `cluster_id` unset. For clustered items set it yourself before `compare`; see [compare arms](compare-arms.md#clustered-items).
- `scorer(gold, output)` sees full decoded output. For models that think first, wrap scorer with `answer_part` (text after last `</think>`), or `\boxed{}` inside thinking can get graded.
- `hit_cap` is `True` when output used all `max_new_tokens` and last token is not EOS.

## Graders

All take `(gold, answer)` and return `bool`.

| Grader | True when |
|---|---|
| `grade_exact` | stripped strings equal |
| `grade_choice` | last `\boxed{}` holds gold letter (A-J) |
| `grade_math` | last boxed value equals gold, checked by `math_verify`; `strip_commas=True` drops commas from gold |
