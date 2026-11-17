# Use with Ollama

Ollama loads one merged GGUF, not an adapter. Since v0.34.1 it rejects LoRA adapters with `LoRA adapters are no longer supported` ([`server/create.go`](https://raw.githubusercontent.com/ollama/ollama/main/server/create.go)), so the Modelfile `ADAPTER` line no longer works. `astro-weights export --splice` writes the merged file: a copy of a base GGUF with the adapter folded in.

You need:

- `ADAPTER`: an adapter directory saved by astro-weights, for example by [`astro-weights build`](build-from-recipe.md).
- `BASE.gguf`: the GGUF of the base model, the file you would run without the edit. It may be quantized.
- `HF_DIR`: the same base model as safetensors, a directory with `model.safetensors`, or with `model.safetensors.index.json` plus shards.
- The `gguf` extra.

## Write the merged GGUF

```bash
pip install 'astro-weights[gguf]'
astro-weights export ADAPTER --splice BASE.gguf --base-model HF_DIR --merged OUT.gguf
```

Output:

```text
wrote OUT.gguf
```

- Each edited tensor is the base weight read from `HF_DIR` plus the adapter's delta, stored as `--out-type`: `q8_0` (default), `f16` or `bf16`. Every other tensor is copied as raw bytes from `BASE.gguf` and keeps its type, so a quantized base stays quantized. Metadata is copied too.
- `q8_0` needs a tensor whose row length is a multiple of 32. Any other length stops with a `QuantError` traceback and writes no file; use `--out-type f16` there.
- Only `mlp.down_proj` and `self_attn.o_proj` modules map to GGUF tensors; see [How-to: export GGUF and a model card](export-gguf.md).
- When the adapter records base weight hashes, a weight in `HF_DIR` that does not match is an error (`wrong base model?`), and nothing is written.
- An existing `OUT.gguf` is refused unless `--force` is given. `OUT.gguf` cannot be `BASE.gguf`, cannot sit inside `ADAPTER`, and its directory must exist.
- The merged file is verified against `BASE.gguf` before it is moved into place. A failed check leaves no file.
- `--json` prints `written`, `replaced` (the GGUF tensor names) and `out_type`.

## Load it in Ollama

Put a `Modelfile` next to the merged file:

```text
FROM ./OUT.gguf
```

Then create and run the model:

```bash
ollama create NAME -f Modelfile
ollama run NAME
```

- Ollama does not quantize a GGUF on import ([Ollama: importing a model](https://docs.ollama.com/import)). The merged file is used as written.
- The chat template must match the base model's. The merged file carries the metadata of `BASE.gguf`, so a template stored there comes along; a wrong template in the Modelfile gives bad output.
- LM Studio loads GGUF files too, so the same `OUT.gguf` should work there.

Not tested yet: the splice is tested on small synthetic GGUFs, including a base with `Q8_0` and `Q4_K` tensors. A merged file from a `Q4_K_M` base has not been run end to end in Ollama or LM Studio.

## Licence

The merged file contains the base model's weights, so the base model's licence applies to it. Check that licence before you share the file.

## Keep the adapter small: llama.cpp

llama.cpp still loads the adapter on its own, with no merged copy of the model:

```bash
llama-server -m base.gguf --lora adapter.gguf
```

Write `adapter.gguf` with `astro-weights export ADAPTER --gguf adapter.gguf --arch ARCH`; see [How-to: export GGUF and a model card](export-gguf.md).
