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
