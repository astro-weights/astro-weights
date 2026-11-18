"""A tiny Qwen3.5 text model with a tokenizer, saved offline for MCP tests."""

from importlib import import_module
from pathlib import Path
from typing import Any

QWEN_TEXT: dict[str, Any] = {
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "linear_key_head_dim": 8,
    "linear_value_head_dim": 8,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "max_position_embeddings": 64,
    "layer_types": ["linear_attention", "full_attention"],
}
_SPECIALS = ["<unk>", "<s>", "</s>"]


def save_tiny_model(directory: Path) -> Path:
    """Save the model (safetensors, seed 0) and a word-level tokenizer to `directory`."""
    torch: Any = import_module("torch")
    transformers: Any = import_module("transformers")
    tk: Any = import_module("tokenizers")

    directory = Path(directory)
    torch.manual_seed(0)
    config = transformers.Qwen3_5TextConfig(**QWEN_TEXT)
    model = transformers.Qwen3_5ForCausalLM(config).eval()
    model.save_pretrained(directory, safe_serialization=True)

    words = [f"w{i}" for i in range(QWEN_TEXT["vocab_size"] - len(_SPECIALS))]
    vocab = {w: i for i, w in enumerate(_SPECIALS + words)}
    tok = tk.Tokenizer(tk.models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = tk.pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )
    fast.save_pretrained(directory)
    return directory
