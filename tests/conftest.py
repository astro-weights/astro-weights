from typing import Any

import pytest


@pytest.fixture
def tiny_llama() -> Any:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return transformers.LlamaForCausalLM(config).eval()


@pytest.fixture
def input_ids() -> Any:
    torch = pytest.importorskip("torch")
    return torch.tensor([[1, 5, 9, 33, 100, 7, 64, 2]])


_QWEN_TEXT = {
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
_QWEN_VISION = {
    "depth": 1,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_heads": 2,
    "out_hidden_size": 32,
    "num_position_embeddings": 16,
}


@pytest.fixture
def tiny_qwen_text() -> Any:
    """Text-only Qwen3.5: layers at `model.layers.N`."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.Qwen3_5TextConfig(**_QWEN_TEXT)
    return transformers.Qwen3_5ForCausalLM(config).eval()


@pytest.fixture
def tiny_qwen_vl() -> Any:
    """Multimodal Qwen3.5: layers at `model.language_model.layers.N`."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.Qwen3_5Config(
        text_config=_QWEN_TEXT, vision_config=_QWEN_VISION
    )
    return transformers.Qwen3_5ForConditionalGeneration(config).eval()
