GGUF export and splice accept `q_proj` and `k_proj` deltas for Gemma 3 and Gemma 4 models (`gemma3`, `gemma4`), whose llama.cpp converter keeps those rows in Hugging Face order.
