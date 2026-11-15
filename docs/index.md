# astro-weights

![astro-weights logo](assets/logo.svg){ width="96" }

Small course corrections for open LLMs. Agent proposes small low-rank edits to model weights; edits that hold up under paired tests get saved as plain PEFT LoRA adapters.

!!! warning "Pre-alpha"
    0.0.2 on PyPI. API changes without notice until 1.0.

Covers edit core: build rank-1 edits, save and load as LoRA adapters, apply to transformers models.

## Install

```bash
pip install astro-weights
```

Applying edits to transformers models needs `hf` extra (adds torch, transformers, peft):

```bash
pip install "astro-weights[hf]"
```

## Where to go

| Need | Page |
|---|---|
| First edit, start to finish | [Tutorial: first edit](tutorials/first-edit.md) |
| Check adapter before shipping | [How-to: inspect adapter](how-to/inspect-adapter.md) |
| Every public function and type | [Reference](reference/index.md) |
| Why output is LoRA | [Explanation: why LoRA](explanation/why-lora.md) |

Source, issues, changelog: [GitHub](https://github.com/astro-weights/astro-weights).
