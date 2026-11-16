# Why the output is LoRA

Every serving stack already loads LoRA adapters: transformers, PEFT, vLLM, llama.cpp and others. Saving an edit as LoRA means it needs no custom runtime. You ship the adapter, load it like any other, and drop it to get the base model back.

Edits in astro-weights are low rank by construction. `LowRankDelta` stores `dW = B @ A`, with `B` of shape `[out, r]` and `A` of shape `[r, in]`, which is the LoRA factor layout. `save_peft` writes the factors with scaling 1 (`lora_alpha = r`), so nothing is rescaled on load.

Each adapter carries a hash manifest (`astro_weights_version`, the base model, and a SHA-256 per factor). `verify_peft` uses it to catch corrupted or altered factors.
