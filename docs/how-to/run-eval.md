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

- **Config** keys decide the outputs: the model class and `model_name_or_path` (the base checkpoint; for an `apply` result, the one under the adapter), `backend` (the device type: cpu, cuda or mps), the torch and transformers versions, dtype, `max_new_tokens`, seed, chat, `batch_size`, `training`, the chat template sha256, `sampling` when it is given, `k` (samples per item, default 1) and `concurrency` (default 1). Two arms are comparable only when their config keys agree, which includes one backend and build for both. A key that is in neither list is also config.
- **Provenance** keys say where and when the run happened and are never compared: `device` (the full string), `gpu_model` and `gpu_count` (when CUDA is available), the peft and numpy versions, the astro-weights version, `library_commit` (`None` for a release wheel, `-dirty` appended when the checkout has uncommitted changes) and `run_id` (random unless you pass one). The sets are `astro_weights.eval.CONFIG_KEYS` and `PROVENANCE_KEYS`.

`config_hash(header)` is the SHA-256 of the canonical JSON of the config keys only. It does not depend on key order, on any provenance value or on `1` versus `1.0`, so two runs with the same hash ran under the same settings. `write_header(header_path("X.jsonl"), header)` stores a header beside the records as `X.header.json`. This continues the example above:

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

`check_headers(base, edit)` compares the config keys only and raises an error that lists every one that differs or is missing on one side. A provenance key that is missing, as in a header written by 0.1 or 0.2, is not a difference, and neither is a missing `k` or `concurrency` when the other side has the default. `ignore=...` names more keys to skip, on top of the provenance keys. `arm_keys=("lora",)` names `extra_body` keys that may differ between the arms, for two arms on one HTTP server that differ only in a request field.

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
    generate_records_http(
        URL,
        name,
        items,
        arm,
        grade_exact,
        64,
        out=f"{arm}-http.jsonl",
        header=run_header_http(URL, name, 64, 0),  # written to {arm}-http.header.json
        resume=True,  # after a crash, asks only for the items not in the file
    )
```

- It sends one `POST /v1/chat/completions` per item, with temperature 0 and `max_tokens` and `seed` set; `extra_body` keys are merged last. The output is `message.content` (`reasoning_content` is ignored), the token count comes from `usage.completion_tokens`, and `hit_cap` is set when `finish_reason` is `length`.
- A non-empty `out` raises `FileExistsError` unless `resume=True`. Resuming keeps the records in the file and asks only for the other items; a last line cut short by a crash is dropped and redone. Records of another arm, seed or prompt are refused.
- `header` is the `run_header_http` result for this run. It must agree with the call (`max_new_tokens`, `seed`, `extra_body`, `concurrency`) and is written beside `out`. On a resume the header file already there is kept, and is refused when it disagrees with the call or, with `header` given, its `config_hash` differs: records made under two configs never end up in one file.
- Connection errors, timeouts, HTTP 429 and 5xx are retried twice, with backoff. An item that still fails raises `RuntimeError` naming the item; no record is written for it (it is not scored as wrong), and a resume asks for it again. The header file's provenance key `infra_errors` counts `retries` and `failed` items, added up across resumes. Other 4xx answers are not retried.
- `concurrency=N` keeps N requests in flight (default 1). Records are still written in item order, so the file is the one a one-at-a-time run writes, if the server answers alike. Most servers do not: on llama.cpp several slots break byte repeatability (in a test, 4 slots changed 17 of 20 outputs compared with 1 slot), and vLLM needs batch-invariant serving (`VLLM_BATCH_INVARIANT=1`). `concurrency` is a config key, so the header check refuses a pair of arms run at different concurrency; run both arms with the same value, and declare the server's slots in `serving_flags`.
- `api_key` is sent as a bearer token and is never recorded or shown in errors. Redirects are refused.
- `run_header_http(url, name, max_new_tokens, seed, extra_body, run_id=..., k=..., concurrency=..., serving_flags=..., model_id=...)` records the config keys `max_new_tokens`, `seed`, `extra_body`, `k`, `concurrency`, `serving_flags`, `model_id`, `backend` and the server build, and the provenance keys `base_url`, `served_model` (the name), the run id, the library version and commit, and the numpy version. `model_id` is the argument you pass (for example the base GGUF's sha256), else the basename of the `model_path` that llama.cpp reports at `/props`, else the served name; it is what makes two servers with different models differ. When the server answers it adds `llama_cpp_build` (`build_info` from `/props`; `backend` is then `llama.cpp`), `vllm_version` (from `/version`; `backend` is `vllm`), and the provenance keys `server_model` (the `/v1/models` entry) and `system_fingerprint`. `/props` and `/version` are asked once each, with a timeout of at most 5 seconds. The client cannot read the server's flags or environment, so declare them in `serving_flags` (for example `{"--parallel": "4"}`); they are compared. Two arms under different served names or URLs, with the same model and build and the same config, pass the header check without `--ignore-header`; give the arms different `extra_body` fields (a `lora` scale) with `--arm-key lora`.

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

## Splits and the sealed test

Split a dataset's items into dev, confirm and test once, by seeded item id, and keep the manifest with the results. An item's split depends only on the seed and its id, so adding or removing items never moves another item. The default fractions are 20 / 30 / 50, and `k` (default 4) is the number of seeds each test item is run with.

```python
from astro_weights.eval import Record
from astro_weights.eval.seal import read_seal, record_scoring, scoring_of, seal_salt
from astro_weights.eval.splits import SplitManifest, assign_splits

ids = [f"gsm8k-{i}" for i in range(1319)]  # ids of the pinned dataset file
manifest = SplitManifest.build(
    ids,
    dataset="openai/gsm8k main test",
    revision="740312add88f781978c0658806c59bc2815b9866",
    file_sha256="c55b65bc330379188022e1d2c8180e434f828b21bd089814e575d7896d92b8c4",
    seed=0,
)
manifest.save("gsm8k.splits.json")
print(len(manifest.dev), len(manifest.confirm), len(manifest.test), manifest.k)
# fewer items: the remaining ones keep their split
assert assign_splits(ids[:10], seed=0) == {
    i: assign_splits(ids, 0)[i] for i in ids[:10]
}

test_records = [
    Record(i, seed, "edit", f"h-{i}", "out", 1.0, 5, False)
    for i in sorted(manifest.test)
    for seed in range(manifest.k)
]
manifest.check_records(test_records, "test", complete=True)
print(manifest.check_seeds(test_records))

edit = Edit((delta,), "tiny-llama")  # the edit from "Generate both arms"
salt = seal_salt("seal")  # made on first use: seal/seal.salt, 32 bytes, mode 0600
record_scoring("seal", scoring_of(edit, manifest, test_records, salt), note="first")
print(read_seal("seal")[0]["k"])
```

Output:

```text
250 428 641 4
(0, 1, 2, 3)
4
```

`SplitManifest.load` refuses unknown or missing keys, a `manifest_sha256` that does not match the contents, and an id listed under a split it does not hash to. Score dev and confirm with `--split` and the manifest; `eval` refuses records, `--floor` arms included, whose ids are outside that split:

```bash
astro-weights eval base.jsonl edit.jsonl --split confirm --manifest gsm8k.splits.json
```

The test split is sealed: each edit is scored on it once.

```bash
astro-weights eval base.jsonl edit.jsonl --split test --manifest gsm8k.splits.json \
  --seal-dir seal --adapter ADAPTER_DIR
```

- The seal guards only runs with `--split test`. `astro-weights eval` without `--split` still compares any two record files, test records included, and records nothing.
- On `--split test` every arm (base, edit and any `--floor` arms) must hold every test id, no other id, and the same `k` seeds for every item; anything else exits 1.
- On `--split test` the base arm and both `--floor` arms are runs of the base model (the floor is two base runs, the noise floor): every record must say `arm` `"base"`, and none of them may hold the records of an edit already in the ledger (same `records_sha256`). Either exits 1, with or without `--unseal`, so a scored edit's test result cannot come back as a base or floor arm. `arm` is a label the run chose, so the first rule catches mistakes, not a relabelled file; the second catches a relabelled file of a scored edit.
- `eval` writes one line to `SEAL_DIR/seal.jsonl` (fsynced) before it prints the result. The line holds the adapter's exact key (`seal_key`, a sha256 of its factors, without the base model), its fingerprint, the hash of the sorted test ids, the sorted prompt hashes scored, `k`, and `records_sha256`, a hash of the edit arm's sorted item id, seed, prompt hash and output. The base model goes in `note`.
- A scoring is refused (exit 1, no result printed) when an earlier line has the same `records_sha256`, whatever adapter is named, or is the same edit and has the same test ids or shares any prompt. The same edit means the same key, or a fingerprint distance of at most 1e-3.
- The fingerprint is `U.T @ (B @ A) @ V` per module, with 8 Gaussian probe vectors on each side seeded by the ledger's salt, the module name and its shape. It depends on `B @ A` only: a rescale `B*c, A/c`, a rotation of the factors, a renamed or copied adapter directory, a new run id, new `meta` or another base model name all count as the same edit.
- The distance is the larger of two relative distances: over all modules together (a module missing on one side counts as zero, so adding a tiny module keeps it the same edit), and the worst single module present on both sides among those carrying at least 1% of the earlier edit's fingerprint norm (so a large module changed alone counts, even when the edit as a whole barely moves). A module under 1% counts only through the first: tweaking it does not make a new edit, and a wholly different module carrying under about 0.07% of the norm counts as the same edit (refused; `--unseal` is the escape). An edit is new when that distance is above 1e-3 for every earlier one.
- The salt is `SEAL_DIR/seal.salt`: 32 random bytes, mode 0600, made on first use under the ledger lock. Without it the probes are unknown, so a change cannot be aimed at them to look new while barely changing the edit. Keep it with `seal.jsonl` and never give it to whatever makes the edits. A seal dir with ledger lines and no salt is an error (exit 2); restore the file rather than letting a new salt make every earlier fingerprint incomparable.
- `--unseal` scores it again, warns on stderr, and the line says `"unseal": true`.
- Two `eval` processes racing on one edit: one prints its result, the other exits 1. The ledger is locked with `flock` (`msvcrt.locking` on Windows), so keep the seal dir on a local disk, and use one seal dir for every scoring of a dataset.
- Each line holds the sha256 of the line before it, so an edited or deleted earlier line makes reads fail. A last line cut short by a crash is dropped; that scoring never printed a result. Deleting the whole file, or pointing `--seal-dir` somewhere new, is not detected.
- Keep manifests and the seal dir outside an agent run's `RUN_DIR`: the MCP server never reads them, and the kernel sandbox cannot open them.

What the seal stops, and what it does not: it stops an accidental second scoring, and an agent or optimiser hunting for a second look at the test split. It does not stop the machine's owner, who can delete the ledger or the salt, point `--seal-dir` somewhere new, or change the code. For that, the check is the run record and a published hash of the ledger: a reader compares the ledger behind a result with the hash published before it.

## Noise floor

Run the base arm twice under the same settings and keep what the two runs disagree on. `floor_from_records(run1, run2, header)` pairs the runs by item and seed and returns a `FloorResult`: the accuracy and token deltas with their standard errors, `flip_rate` (the share of (item, seed) pairs whose output text differs) and `k`. `save_floor` stores it as `floor-<config_hash>.json`, and `load_floor(dir, config_hash)` returns it only for that exact `config_hash` (the hash of the header's config keys, so a change to the model, sampling, `max_new_tokens`, `k` or any other config value gives `None`; provenance keys such as `run_id` do not count). A file whose stored hash is not the one in its name, or that is not a valid floor, raises `ValueError`. A deterministic server (greedy, one slot) gives a floor of exactly 0.

```python
from astro_weights.eval import floor_from_records, load_floor, save_floor
from astro_weights.eval.header import config_hash

ids = range(6)
run1 = [
    Record(f"n{i}", 0, "base", f"h{i}", f"out{i}", float(i % 2), 5, False) for i in ids
]
run2 = [
    Record(f"n{i}", 0, "base", f"h{i}", f"out{i}", float(i % 2), 5, False) for i in ids
]
header = {"model": "tiny-llama", "temperature": 0.0, "max_new_tokens": 64}
floor = floor_from_records(run1, run2, header)
save_floor("floors", floor)
print(floor.flip_rate, floor.accuracy_delta)
assert load_floor("floors", floor.config_hash) == floor
assert load_floor("floors", config_hash({**header, "max_new_tokens": 128})) is None
```

Output:

```text
0.0 0.0
```
