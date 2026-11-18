"""The card's loader snippet must load every adapter weight (PEFT matches full names)."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from astro_weights.edits import Edit, LowRankDelta
from astro_weights.export import save_peft
from astro_weights.export.card import model_card

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")


def _unloaded(model: Any, adapter: Path) -> int:
    """LoRA keys in the adapter file that PEFT could not place in `model`."""
    from safetensors.torch import load_file

    pm = peft.get_peft_model(model, peft.PeftConfig.from_pretrained(str(adapter)))
    result = peft.utils.set_peft_model_state_dict(
        pm, load_file(str(adapter / "adapter_model.safetensors"))
    )
    return sum("lora" in k for k in result.unexpected_keys)


def test_card_class_loads_language_model_adapter(
    tiny_qwen_vl: Any, tmp_path: Path
) -> None:
    module = "model.language_model.layers.0.mlp.down_proj"
    w = dict(tiny_qwen_vl.named_modules())[module].weight
    rng = np.random.default_rng(0)
    delta = LowRankDelta(
        module, rng.normal(size=(w.shape[0], 1)), rng.normal(size=(1, w.shape[1]))
    )
    save_peft(Edit((delta,), "org/base"), tmp_path / "a")
    card = model_card(Edit((delta,), "org/base"), "adapter", "mit")
    cls = getattr(transformers, card.split("from transformers import ")[1].split()[0])
    assert cls is transformers.AutoModelForImageTextToText
    assert _unloaded(tiny_qwen_vl, tmp_path / "a") == 0
    # the old snippet's class names layers model.layers.N: nothing loads (PEFT
    # raises for full-name targets, loads 0 keys for suffix targets)
    text = transformers.Qwen3_5ForCausalLM(tiny_qwen_vl.config.text_config)
    try:
        assert _unloaded(text, tmp_path / "a") > 0
    except peft.utils.error.NoMatchingPeftModuleError:
        pass
