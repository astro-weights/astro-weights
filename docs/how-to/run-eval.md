# Run an eval end to end

`generate_records` runs one arm, greedy unless you pass `sampling`, and returns one scored `Record` per item. Run it on the base model, then on the edited model, write both arms as JSONL, and compare them with `astro-weights eval`. Generating needs the `hf` extra. `grade_math` needs the `eval` extra (`pip install "astro-weights[eval]"`).

## Generate both arms

The tiny random Llama and word-level tokenizer below stand in for a real model, so their scores mean nothing.

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

Output:

```text
LlamaForCausalLM
4 4
```

## Write as you go, resume, batch

`out` appends each record to a JSONL file as soon as it exists, so a crash keeps the finished items. A non-empty `out` raises `FileExistsError` unless `resume=True`; a resumed run keeps the records already in the file and generates only the missing items. `batch_size` generates that many items at once, with left padding. This continues the example above:

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

Output:

```text
2 4 4
```

- Resuming raises `ValueError` when the file belongs to another run: a different arm or seed, an `item_id` not in `items`, or a different prompt hash. A last line cut short by a crash is dropped and redone.
- Greedy outputs with `batch_size > 1` can differ from `batch_size=1`: padding changes float rounding, so a near tie can flip. Use one batch size for both arms. The tokenizer needs a pad token.

## Run headers

`run_header(model, tokenizer, max_new_tokens, seed, chat, batch_size=..., sampling=..., run_id=..., k=..., concurrency=...)` returns the run record of one arm. Its keys are of two kinds.

- **Config** keys decide the outputs: the model class (for an `apply` result, the class under the adapter), dtype, `max_new_tokens`, seed, chat, `batch_size`, `training`, the chat template sha256, `sampling` when it is given, `k` (samples per item, default 1) and `concurrency` (default 1). Two arms are comparable only when their config keys agree. A key that is in neither list is also config.
- **Provenance** keys say where and when the run happened and are never compared: `device`, `gpu_model` and `gpu_count` (when CUDA is available), the torch, transformers, peft and numpy versions, the astro-weights version, `library_commit` (`None` for a release wheel, `-dirty` appended when the checkout has uncommitted changes) and `run_id` (random unless you pass one). The sets are `astro_weights.eval.CONFIG_KEYS` and `PROVENANCE_KEYS`.

`config_hash(header)` is the SHA-256 of the canonical JSON of the config keys only. It does not depend on key order or on any provenance value, so two runs with the same hash ran under the same settings. `write_header(header_path("X.jsonl"), header)` stores a header beside the records as `X.header.json`. This continues the example above:

```python
from astro_weights.eval import check_headers, config_hash, header_path, write_header

peft_model = apply(model, Edit((delta,), "tiny-llama"))
for arm, m in [("base", model), ("edit", peft_model)]:
    write_header(header_path(f"{arm}.jsonl"), run_header(m, tok, 6, 0, False))
print(header_path("base.jsonl"))

base_h = run_header(model, tok, 6, 0, False)
edit_h = run_header(peft_model, tok, 6, 0, False)
check_headers(base_h, edit_h)  # raises when arms are not comparable
print(base_h["model"], edit_h["model"])
print(config_hash(base_h) == config_hash(edit_h))  # same config, different run ids
```

Output:

```text
base.header.json
LlamaForCausalLM LlamaForCausalLM
True
```

`check_headers(base, edit)` compares the config keys only and raises an error that lists every one that differs or is missing on one side. A provenance key that is missing, as in a header written by 0.1 or 0.2, is not a difference, and neither is a missing `k` or `concurrency` when the other side has the default. With an explicit `ignore=...`, exactly those keys are skipped, so `ignore=()` compares provenance too. `arm_keys=("lora",)` names `extra_body` keys that may differ between the arms, for two arms on one HTTP server that differ only in a request field.

`astro_weights.eval.Verdict` is the schema of a `verdict.json`: the verdict, the backend and build it ran on, the config hash, the split manifest sha256, the candidates tried, the confirm queries, the infra errors and the edit footprint. `to_json` and `Verdict.from_json` round-trip it, and `from_json` rejects unknown keys. No command writes one yet.

## Greedy or sampled

Decoding is greedy by default. Greedy is a stress condition: some models repeat one sentence until they hit the cap. For the headline number, use the sampling settings that the model card recommends, and use the same settings on both arms. `sampling` maps any of `temperature` (> 0), `top_p`, `top_k` and `presence_penalty` to a number and turns sampling on. Pass the same `sampling` to `run_header`, which records it under `"sampling"`, so `check_headers` rejects a greedy arm paired with a sampled one, or two arms sampled differently. This continues the example above:

```python
from astro_weights.runners.hf import remove

model = remove(peft_model)
# placeholder values: copy the ones your model's card recommends
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}

sampled = generate_records(
    model, tok, items, "base", grade_exact, 6, chat=False, sampling=SAMPLING
)
again = generate_records(
    model, tok, items, "base", grade_exact, 6, chat=False, sampling=SAMPLING
)
print([r.output for r in sampled] == [r.output for r in again])

sampled_h = run_header(model, tok, 6, 0, False, sampling=SAMPLING)
print(sampled_h["sampling"])
try:
    check_headers(base_h, sampled_h)
except ValueError as e:
    print(e)
```

Output:

```text
True
{'presence_penalty': 1.5, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}
arms ran under different settings and are not comparable: sampling: base='<missing>' edit={'presence_penalty': 1.5, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}
```

- Sampled outputs repeat for the same model, items, `seed`, `sampling` and `batch_size`, also across a resume: each batch is reseeded from `seed` and its position in `items`.
- `presence_penalty` subtracts that value from the logit of every token among the last 64 (prompt included), as llama.cpp does.
- An unknown key or a temperature that is not above 0 raises `ValueError`.

## Compare arms

```bash
astro-weights eval base.jsonl edit.jsonl
```

This prints the [`format_report`](compare-arms.md) output:

```text
n=4 items  base=0.0000  edit=0.0000
score diff=+0.0000  SE=0.0000  95% CI [+0.0000, +0.0000]; not significant; smallest detectable difference at this n is 0.0000
McNemar exact p=1  (base-only right=0, edit-only right=0)
tokens: base=6.0  edit=6.0  diff=+0.0 (+0.0%)  bootstrap 95% CI [+0.0, +0.0]; not significant; smallest detectable difference at this n is 0.0
cap-hit rate: base=1.000  edit=1.000
WARNING: only 4 items (<100); intervals are unreliable

WARNING: cap-hit censoring
  cap-hit rate: base=100.0%  edit=100.0%  (McNemar exact p=1)
  A capped answer usually has no final answer, so it is graded wrong and its token count is cut short.
  The score and token diffs over all items above are censored by the cap.
  items where neither arm hit the cap: n=0 items
  loops (capped, tail repeats one chunk): base=0  edit=1
  Raise max_new_tokens and rerun; pick it from a short base-arm pilot so that cap-hit is near 0.
```

With `max_new_tokens=6` every output hits the cap, so the report warns that the diffs are censored. Pick `max_new_tokens` from a short pilot of the base arm: raise it until almost no output hits the cap, then run both arms with that value. The warning and its fields are described in [Compare two eval arms](compare-arms.md#cap-hit-censoring).

- When `BASE.header.json` and `EDIT.header.json` both exist, `eval` compares their config keys (provenance keys are ignored) and exits 1 on any difference, with `error: arms ran under different settings ...` on stderr. `--ignore-header` skips the check.
- `--floor A.jsonl B.jsonl` takes two runs of the base arm under the same settings, and the report adds noise-floor lines. See [Compare two eval arms](compare-arms.md#noise-floor).
- `--json` prints one JSON object on stdout instead of text (`PairedResult` fields plus `floor`, with non-finite numbers as `null`).
- `--boot` sets the number of bootstrap resamples (default 10000) and `--seed` the bootstrap seed (default 0). An unreadable file or arms that do not pair give a message on stderr and exit code 2.

## Serve arms over HTTP

`astro_weights.runners.endpoint` runs an arm against an OpenAI-compatible server (llama-server, vLLM) with the standard library only: no torch and no extra. Serve the base and the edit under two model names and call it once per name.

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

- It sends one `POST /v1/chat/completions` per item, with temperature 0 and `max_tokens` and `seed` set; `extra_body` keys are merged last. The output is `message.content` (`reasoning_content` is ignored), the token count comes from `usage.completion_tokens`, and `hit_cap` is set when `finish_reason` is `length`.
- `out` must be empty or absent; there is no resume. `api_key` is sent as a bearer token and is never recorded or shown in errors. Redirects are refused.
- `run_header_http(url, name, max_new_tokens, seed, extra_body, run_id=..., k=..., concurrency=..., serving_flags=...)` records the config keys `max_new_tokens`, `seed`, `extra_body`, `k`, `concurrency` and `serving_flags`, and the provenance keys `base_url`, `served_model` (the name), the run id, the library version and commit, and the numpy version. When the server answers it adds the `/v1/models` entry (`server_model`), `system_fingerprint`, the llama.cpp `build_info` from `/props` (`llama_cpp_build`) and the vLLM version from `/version` (`vllm_version`). The client cannot read the server's flags or environment, so declare them in `serving_flags` (for example `{"--parallel": "4"}`); they are compared. Two arms under different served names or URLs but the same config pass the header check without `--ignore-header`.

### Base and edit on one llama-server

One server process can run both arms: load the base GGUF and the adapter GGUF, and pick the arm per request with the `lora` field (scale 0 = base, scale 1 = edit). Both arms then share the binary, the backend and the weights file.

```bash
astro-weights export ADAPTER_DIR --gguf adapter.gguf --arch qwen35   # prints the file's sha256
llama-server -m base.gguf --lora adapter.gguf -np 1 --no-cache-prompt --cache-reuse 0 --cache-ram 0 --no-cache-idle-slots -nocb --port 8080
```

```python
from astro_weights.eval import check_headers, grade_exact, header_path, write_header
from astro_weights.runners.endpoint import generate_records_http, run_header_http

URL, NAME = "http://127.0.0.1:8080", "base.gguf"
items = [{"item_id": "i0", "prompt": "Say w5.", "gold": "w5"}]
headers = {}
for arm, scale in [("base", 0), ("edit", 1)]:
    # always explicit, also for the base arm
    lora = {"lora": [{"id": 0, "scale": scale}]}
    headers[arm] = run_header_http(URL, NAME, 64, 0, lora)
    write_header(header_path(f"{arm}-http.jsonl"), headers[arm])
    generate_records_http(
        URL, NAME, items, arm, grade_exact, 64, extra_body=lora, out=f"{arm}-http.jsonl"
    )
check_headers(headers["base"], headers["edit"], ignore=("device", "extra_body"))
```

Then `astro-weights eval base-http.jsonl edit-http.jsonl --ignore-header`. The headers differ only in `extra_body`, which `check_headers` above already allowed for.

- Send `lora` with an explicit scale on every request, base arm included. A request without `lora` runs every adapter at the server's global scale, which is 1 after loading. `--lora-init-without-apply` does not change that in upstream llama-server: the scales stay 1.0 (upstream issue [#28674](https://github.com/ggml-org/llama.cpp/issues/28674)).
- One slot (`-np 1`): in a test on llama.cpp b12359, 4 slots changed 17 of 20 outputs compared with 1 slot, while 1 slot gave the same outputs run after run.
- Prompt caches off (`--no-cache-prompt`, `--cache-reuse 0`, `--cache-ram 0`, `--no-cache-idle-slots`): llama.cpp builds before mid-November 2026 can reuse a cached prompt computed under the other adapter setting.
- Pin the files you compare: record the sha256 that `export` prints, and the base GGUF's sha256 (`shasum -a 256 base.gguf`), with the results.
- The verdict holds for that backend and build. Compare arms only from the same server process.

## What `generate_records` does

- Every item needs `item_id`, `prompt` and `gold` keys. A missing key or a repeated `item_id` raises `ValueError` before anything is generated.
- `chat=True` (the default) wraps the prompt in the tokenizer's chat template when the tokenizer has one. Without a template, the prompt goes in as is.
- The model runs in eval mode, and its previous train/eval mode is restored afterwards.
- `n_output_tokens` counts tokens up to and including the first EOS.
- Records leave `cluster_id` unset. For clustered items, set it yourself before `compare`; see [Compare two eval arms](compare-arms.md#clustered-items).
- `scorer(gold, output)` sees the full decoded output. For models that think first, wrap the scorer with `answer_part` (the text after the last `</think>`), or a `\boxed{}` inside the thinking can get graded.
- `hit_cap` is `True` when the output used all `max_new_tokens` and the last token is not EOS.

## Graders

All graders take `(gold, answer)` and return a `bool`.

| Grader | True when |
|---|---|
| `grade_exact` | the stripped strings are equal |
| `grade_choice` | the last `\boxed{}` holds the gold letter (A-J) |
| `grade_math` | the last boxed value equals the gold answer, checked by `math_verify`; `strip_commas=True` drops commas from the gold answer. In the main thread each check has `math_verify`'s 5 s timeouts; in other threads those timeouts (built on `signal.alarm`) are off, so put your own time limit on a worker pool |
