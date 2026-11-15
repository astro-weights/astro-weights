# Why LoRA output

Every serving stack already loads LoRA adapters: transformers, PEFT, vLLM, llama.cpp and others. Saving edit as LoRA means no custom runtime. Ship adapter, load like any other, drop it to get base model back.

Edits here are low rank by construction. `LowRankDelta` stores `dW = B @ A`, `B` shape `[out, r]`, `A` shape `[r, in]`, which is LoRA factor layout. `save_peft` writes factors with scaling 1 (`lora_alpha = r`), so no rescaling on load.

Adapter carries hash manifest (`astro_weights_version`, base model, SHA-256 per factor). `verify_peft` uses it to catch corrupted or altered factors.
