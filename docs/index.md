# astro-weights

![astro-weights logo](assets/logo.svg){ width="96" }

Small course corrections for open LLMs. Agent proposes small low-rank edits to model weights; edits that hold up under paired tests get saved as plain PEFT LoRA adapters.

!!! warning "Pre-alpha"
    0.1.1 on PyPI. API changes without notice until 1.0.

Covers edit core (build low-rank edits, save and load as LoRA adapters, apply to transformers models, export as GGUF with model card), difference-of-means directions with gate, eval runner with graders, paired eval statistics.

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
| Adapter from recipe file, GGUF and card from CLI | [How-to: build adapter from recipe file](how-to/build-from-recipe.md) |
| Dense `dW` as low-rank delta, with known error | [How-to: convert dense update](how-to/dense-to-low-rank.md) |
| Check adapter before shipping | [How-to: inspect adapter](how-to/inspect-adapter.md) |
| Edited weights without adapter | [How-to: save merged weights](how-to/save-merged-weights.md) |
| llama.cpp adapter, GGUF splice, model card | [How-to: export GGUF and model card](how-to/export-gguf.md) |
| Direction from paired prompts | [How-to: find and gate direction](how-to/find-direction.md) |
| Base vs edit on same items | [How-to: compare two eval arms](how-to/compare-arms.md) |
| Generate and compare arms on model | [How-to: run eval end to end](how-to/run-eval.md) |
| Every public function and type | [Reference](reference/index.md) |
| Why output is LoRA | [Explanation: why LoRA](explanation/why-lora.md) |

Source, issues, changelog: [GitHub](https://github.com/astro-weights/astro-weights).
