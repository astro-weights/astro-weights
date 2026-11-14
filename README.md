# astro-weights

Agent-driven weight-space edits for open LLMs. An agent proposes small, targeted edits to a model's weights, the edits are judged against paired statistical tests on held-out data, and the ones that hold up are exported as standard PEFT LoRA adapters (and GGUF) that existing tools already load.

> **Status: planning.** Nothing usable is released yet. The API will change without notice until 1.0.

## Install

```bash
pip install astro-weights
```

## Development

```bash
git clone https://github.com/astro-weights/astro-weights
cd astro-weights
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE).
