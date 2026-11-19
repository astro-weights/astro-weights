# Export GGUF and a model card

`astro_weights.export.gguf` writes an edit as a llama.cpp LoRA adapter GGUF, reads it back, and splices replacement tensors into an existing base GGUF. It needs the `gguf` extra (`pip install "astro-weights[gguf]"`). `astro_weights.export.card` writes a Hugging Face model card and needs numpy only.

For an adapter already saved on disk, `astro-weights export DIR --gguf OUT.gguf --arch ARCH --card LICENSE` does both without Python; see the [CLI reference](../reference/cli.md). For Ollama, which loads a merged GGUF only, see [How-to: use with Ollama](use-with-ollama.md).

## Write and read an adapter GGUF

```python
import numpy as np

from astro_weights.edits import Edit, LowRankDelta
from astro_weights.export.gguf import gguf_tensor_name, read_lora_gguf, save_lora_gguf

rng = np.random.default_rng(0)
delta = LowRankDelta(
    "model.layers.3.mlp.down_proj",
    rng.standard_normal((32, 1)).astype(np.float32),
    rng.standard_normal((1, 96)).astype(np.float32),
)
edit = Edit(deltas=(delta,), base_model="org/model")

print(gguf_tensor_name(delta.module))
save_lora_gguf(edit, "edit.gguf", "qwen35", overwrite=True)
factors, meta = read_lora_gguf("edit.gguf")
B, A = factors["blk.3.ffn_down.weight"]
print(B.shape, A.shape, meta["alpha"])
```

Output:

```text
blk.3.ffn_down.weight
(32, 1) (1, 96) 1.0
```

- Seven module types map to GGUF names: `mlp.down_proj` (`ffn_down`), `mlp.gate_proj` (`ffn_gate`), `mlp.up_proj` (`ffn_up`), `self_attn.q_proj` (`attn_q`), `self_attn.k_proj` (`attn_k`), `self_attn.v_proj` (`attn_v`) and `self_attn.o_proj` (`attn_output`). Other modules raise `ValueError`. The last `layers.<N>` in the name gives the block number.
- `q_proj` and `k_proj` deltas export only for architectures whose converter keeps their rows in Hugging Face order: `qwen2`, `qwen3`, `qwen3moe`, `qwen35` and `qwen35moe`. For any other architecture, for example `llama`, llama.cpp reorders those rows, so the export raises `ValueError`.
- `architecture` (the third argument) is written to `general.architecture`.
- Alpha equals the edit rank, so the LoRA scale is 1. `read_lora_gguf` returns the raw factors without the scale applied: delta = `(alpha / rank) * B @ A`.
- An existing file raises `FileExistsError` unless `overwrite=True`. The writer writes a temp file next to the target, then renames it.
- `read_lora_gguf` accepts F32 and F16 tensors only.

## Splice a tensor into a base GGUF

`splice_gguf(base, out, replace, out_type="q8_0")` copies a base GGUF and replaces the named tensors; all other tensors are copied as raw bytes. `out_type` is `q8_0`, `f16` or `bf16`. Replacement arrays are HF `[out, in]` weights; an array passed transposed is transposed back. `verify_splice` returns the problems it finds, and an empty list means the splice is fine. This continues the example above, with a tiny synthetic base standing in for a real model:

```python
import gguf

from astro_weights.export.gguf import splice_gguf, verify_splice

W0 = rng.standard_normal((32, 96)).astype(np.float32)
writer = gguf.GGUFWriter("base.gguf", "qwen35")
writer.add_block_count(4)
writer.add_tensor("blk.3.ffn_down.weight", W0)
writer.add_tensor("token_embd.weight", rng.standard_normal((50, 32)).astype(np.float32))
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file(progress=False)
writer.close()

name = "blk.3.ffn_down.weight"
splice_gguf("base.gguf", "spliced.gguf", {name: W0 + B @ A})
print(verify_splice("base.gguf", "spliced.gguf", [name]))
```

Output:

```text
[]
```

- A tensor missing from the base, a NaN/Inf or non-float replacement, or a wrong shape raises `ValueError`.
- Blocks at index `block_count - nextn_predict_layers` or above (MTP/nextn blocks) raise `ValueError` and are never touched.
- An `out` that is the same file as `base` raises `ValueError`.

## Model card

`model_card(edit, kind, license, title=None, summary=None, results=None)` renders the card text; `kind` is `"adapter"` or `"gguf"`. `write_card(directory, text)` writes it as `README.md`. This continues the example above:

```python
from astro_weights.export import model_card, write_card

card = model_card(
    edit,
    "adapter",
    "apache-2.0",
    summary="Shorter answers.",
    results={"tokens": "-10%"},
)
path = write_card("my-adapter", card)
print(path)
print("\n".join(card.splitlines()[:10]))
```

Output:

```text
my-adapter/README.md
---
base_model: "org/model"
base_model_relation: "adapter"
library_name: "peft"
license: "apache-2.0"
tags:
  - "astro-weights"
  - "weight-edit"
  - "lora"
---
```

- The card holds front matter, a title (by default `<base_model> edit`), the summary, an edit section, a load snippet, an optional results table and an astro-weights version line. The edit section opens with a `Modification:` line stating that the weights of the base model were modified by a low-rank edit of N modules, made with astro-weights without training, followed by a table (module, shape, rank, Frobenius norm of `BA`).
- `kind="gguf"` sets `base_model_relation: "finetune"`, adds the tag `gguf` and uses a `llama-cli` load snippet.
- An empty `edit.base_model`, a blank license or an unknown `kind` raises `ValueError`.
- `write_card` replaces an existing `README.md`; with `overwrite=False` it raises `FileExistsError` instead.
