"""Apply and remove an Edit on a transformers model through PEFT."""

import torch
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from torch import nn

from astro_weights.edits import Edit
from astro_weights.export.peft import PREFIX, SUFFIX_A, SUFFIX_B


def apply(model: nn.Module, edit: Edit, adapter_name: str = "astro") -> PeftModel:
    """Wrap only the edit's modules in LoRA layers carrying the edit's factors."""
    modules = dict(model.named_modules())
    state: dict[str, torch.Tensor] = {}
    for d in edit.deltas:
        layer = modules.get(d.module)
        if not isinstance(layer, nn.Linear):
            # ValueError (not TypeError): the module may simply be absent
            raise ValueError(f"{d.module}: not an nn.Linear in the model")  # noqa: TRY004
        if layer.weight.shape != d.shape:
            raise ValueError(
                f"{d.module}: edit shape {d.shape} != weight {tuple(layer.weight.shape)}"
            )
        dtype = layer.weight.dtype
        state[f"{PREFIX}{d.module}{SUFFIX_A}"] = torch.from_numpy(d.A).to(dtype)
        state[f"{PREFIX}{d.module}{SUFFIX_B}"] = torch.from_numpy(d.B).to(dtype)

    cfg = LoraConfig(
        r=edit.rank,
        lora_alpha=edit.rank,
        lora_dropout=0.0,
        bias="none",
        target_modules=[d.module for d in edit.deltas],
    )
    peft_model = get_peft_model(model, cfg, adapter_name=adapter_name)
    set_peft_model_state_dict(peft_model, state, adapter_name=adapter_name)
    return peft_model


def remove(peft_model: PeftModel) -> nn.Module:
    """Drop the LoRA layers (no merge); base weights were never modified."""
    base: nn.Module = peft_model.unload()
    return base
