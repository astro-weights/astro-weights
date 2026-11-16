# astro-weights

![astro-weights logo](assets/logo.svg){ width="96" }

Small course corrections for open LLMs. The goal is an agent that proposes small low-rank edits to a model's weights and keeps the ones that hold up under paired tests, saved as plain PEFT LoRA adapters. The agent loop is not here yet; the library today is the toolkit that loop needs.

!!! warning "Pre-alpha"
    The API will change without notice until 1.0.

The toolkit covers:

- Edits: build low-rank edits from numpy recipes, rebuild one from a recipe file, or convert a dense update with `from_dense`.
- Export: save and load edits as hash-checked LoRA adapters, write a GGUF adapter for llama.cpp, and write a model card.
- Runners: apply edits to transformers models and remove them again, and generate eval records from a local model or an OpenAI-compatible endpoint.
- Directions: difference-of-means directions with a split-half gate.
- Eval: graders and paired comparison of a base arm and an edit arm.

## Install

```bash
pip install astro-weights
```

To apply edits to transformers models, add the `hf` extra (it pulls in torch, transformers and peft):

```bash
pip install "astro-weights[hf]"
```

## Where to go

| Need | Page |
|---|---|
| Your first edit, start to finish | [Tutorial: first edit](tutorials/first-edit.md) |
| An adapter from a recipe file, plus GGUF and a card from the CLI | [How-to: build an adapter from a recipe file](how-to/build-from-recipe.md) |
| A dense `dW` as a low-rank delta, with a known error | [How-to: convert a dense update](how-to/dense-to-low-rank.md) |
| Checking an adapter before you ship it | [How-to: inspect an adapter](how-to/inspect-adapter.md) |
| One adapter on text-only and multimodal loads of a checkpoint | [How-to: apply across model classes](how-to/apply-across-model-classes.md) |
| Edited weights without an adapter | [How-to: save merged weights](how-to/save-merged-weights.md) |
| A llama.cpp adapter, a GGUF splice, a model card | [How-to: export GGUF and a model card](how-to/export-gguf.md) |
| A direction from paired prompts | [How-to: find and gate a direction](how-to/find-direction.md) |
| Base and edit compared on the same items | [How-to: compare two eval arms](how-to/compare-arms.md) |
| Generating and comparing arms on a model | [How-to: run an eval end to end](how-to/run-eval.md) |
| Every public function and type | [Reference](reference/index.md) |
| Why the output is LoRA | [Explanation: why LoRA](explanation/why-lora.md) |

Source code, issues and the changelog are on [GitHub](https://github.com/astro-weights/astro-weights).
