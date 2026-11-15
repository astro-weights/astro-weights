# Run eval end to end

`generate_records` runs one arm: greedy decoding, batch size 1, one scored `Record` per item. Run it on base model, then on edited model, write both arms as JSONL, compare them with `astro-weights eval`. Generating needs `hf` extra. `grade_math` needs `eval` extra (`pip install "astro-weights[eval]"`).

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

## Compare arms

```bash
astro-weights eval base.jsonl edit.jsonl
```

Prints [`format_report`](compare-arms.md) output:

```text
n=4 items  base=0.0000  edit=0.0000
score diff=+0.0000  SE=0.0000  95% CI [+0.0000, +0.0000]
McNemar exact p=1  (base-only right=0, edit-only right=0)
tokens: base=6.0  edit=6.0  diff=+0.0 (+0.0%)  bootstrap 95% CI [+0.0, +0.0]
cap-hit rate: base=1.000  edit=1.000
WARNING: only 4 items (<100); intervals are unreliable
```

`--boot` sets bootstrap resamples (default 10000), `--seed` bootstrap seed (default 0). Unreadable file or arms that do not pair: message on stderr, exit code 2.

## What `generate_records` does

- Every item needs `item_id`, `prompt` and `gold` keys. Missing key or repeated `item_id` raises `ValueError` before any generation.
- `chat=True` (default) wraps prompt in tokenizer's chat template when tokenizer has one. Without template, prompt goes in as is.
- Model runs in eval mode; previous train/eval mode comes back afterwards.
- `scorer(gold, output)` sees full decoded output. For models that think first, wrap scorer with `answer_part` (text after last `</think>`), or `\boxed{}` inside thinking can get graded.
- `hit_cap` is `True` when output used all `max_new_tokens` and last token is not EOS.
- `run_header(model, tokenizer, max_new_tokens, seed, chat)` returns settings that decide outputs (model class, dtype, device, chat template sha256, torch, transformers and peft versions) for run log.

## Graders

All take `(gold, answer)` and return `bool`.

| Grader | True when |
|---|---|
| `grade_exact` | stripped strings equal |
| `grade_choice` | last `\boxed{}` holds gold letter (A-J) |
| `grade_math` | last boxed value equals gold, checked by `math_verify`; `strip_commas=True` drops commas from gold |
