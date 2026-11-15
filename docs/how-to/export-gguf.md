# Export GGUF and model card

`astro_weights.export.gguf` writes edit as llama.cpp LoRA adapter GGUF, reads it back, and splices replacement tensors into existing base GGUF. Needs `gguf` extra (`pip install "astro-weights[gguf]"`). `astro_weights.export.card` writes Hugging Face model card; numpy only.

Saved adapter on disk: `astro-weights export DIR --gguf OUT.gguf --arch ARCH --card LICENSE` does both without Python; see [CLI reference](../reference/cli.md).

## Write and read adapter GGUF

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

Prints:

```text
blk.3.ffn_down.weight
(32, 1) (1, 96) 1.0
```

- Only `mlp.down_proj` (`ffn_down`) and `self_attn.o_proj` (`attn_output`) map; other modules raise `ValueError`. Last `layers.<N>` in name gives block number.
- `architecture` (third argument) goes into `general.architecture`.
- Alpha = edit rank, so LoRA scale is 1. `read_lora_gguf` returns raw factors, scale not applied: delta = `(alpha / rank) * B @ A`.
- Existing file raises `FileExistsError` unless `overwrite=True`. Writes temp file next to target, then renames.
- `read_lora_gguf` accepts F32 and F16 tensors only.

## Splice tensor into base GGUF

`splice_gguf(base, out, replace, out_type="q8_0")` copies base GGUF, replacing named tensors; other tensors copied as raw bytes. `out_type`: `q8_0`, `f16` or `bf16`. Replacement arrays are HF `[out, in]` weights; transposed array gets transposed back. `verify_splice` returns problems found; empty list means ok. Continues example above; tiny synthetic base stands in for real model:

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

Prints:

```text
[]
```

- Tensor missing from base, NaN/Inf or non-float replacement, or wrong shape: `ValueError`.
- Blocks at index `block_count - nextn_predict_layers` or above (MTP/nextn blocks) raise `ValueError`; never touched.
- `out` same file as `base`: `ValueError`.

## Model card

`model_card(edit, kind, license, title=None, summary=None, results=None)` renders card text; `kind` is `"adapter"` or `"gguf"`. `write_card(directory, text)` writes it as `README.md`. Continues example above:

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

Prints:

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

- Card holds front matter, title (default `<base_model> edit`), summary, edit table (module, shape, rank, Frobenius norm of `BA`), load snippet, optional results table, astro-weights version line.
- `kind="gguf"` sets `base_model_relation: "quantized"`, tag `gguf`, `llama-cli` load snippet.
- Empty `edit.base_model`, blank license or unknown `kind`: `ValueError`.
- `write_card` replaces existing `README.md`; `overwrite=False` raises `FileExistsError` instead.
