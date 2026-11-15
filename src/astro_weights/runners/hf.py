"""Apply and remove an Edit on a transformers model through PEFT."""

from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from torch import nn

from astro_weights.edits import Edit
from astro_weights.export.peft import PREFIX, SUFFIX_A, SUFFIX_B

# ponytail: one anchor name; add "h"/"blocks" when a real adapter needs them
_ANCHOR = "layers"


def _tail(name: str) -> tuple[str, ...] | None:
    """Dot components from the last `layers.<int>` onward, or None without one."""
    parts = name.split(".")
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == _ANCHOR and parts[i + 1].isdigit():
            return tuple(parts[i:])
    return None


def resolve_modules(model: nn.Module, edit: Edit) -> dict[str, str]:
    """Map each edit module name to the model's name for the same linear layer.

    The same checkpoint is named `model.layers.N...` or
    `model.language_model.layers.N...` depending on the class that loads it.
    An exact name wins. Otherwise the name is matched from its last
    `layers.<int>` onward, among linear layers of the edit's shape. No match,
    or more than one, raises: a wrong guess would edit the wrong layer.
    """
    # named_modules() yields a module registered under two names only once
    live = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    return _resolve(live, edit)


def _base_linears(peft_model: PeftModel) -> dict[str, nn.Linear]:
    """The linear layers of the base model under `peft_model`, by base-model name.

    LoRA wrappers count as the linear they wrap; the LoRA factors do not count.
    """
    prefix = "base_model.model."
    wrappers = {
        n: m.base_layer
        for n, m in peft_model.named_modules()
        if isinstance(getattr(m, "base_layer", None), nn.Linear)
    }
    live: dict[str, nn.Linear] = {}
    for n, m in peft_model.named_modules():
        if any(n.startswith(w + ".") for w in wrappers):
            continue
        linear = wrappers.get(n, m if isinstance(m, nn.Linear) else None)
        if linear is not None:
            live[n.removeprefix(prefix)] = linear
    return live


def _resolve(live: dict[str, nn.Linear], edit: Edit) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for d in edit.deltas:
        if d.module in live:
            name = d.module
        else:
            tail = _tail(d.module)
            found = [
                n
                for n, m in live.items()
                if tail is not None
                and tuple(n.split(".")[-len(tail) :]) == tail
                and tuple(m.weight.shape) == d.shape
            ]
            if not found:
                raise ValueError(
                    f"{d.module}: no nn.Linear matches in the model (shape {d.shape})"
                )
            if len(found) > 1:
                raise ValueError(
                    f"{d.module}: ambiguous, matches {sorted(found)}; "
                    "use the model's exact module name"
                )
            name = found[0]
        if tuple(live[name].weight.shape) != d.shape:
            raise ValueError(
                f"{d.module}: edit shape {d.shape} != weight "
                f"{tuple(live[name].weight.shape)}"
            )
        resolved[d.module] = name
    if len(set(resolved.values())) != len(resolved):
        raise ValueError(f"several edit modules map to one layer: {resolved}")
    return resolved


def apply(model: nn.Module, edit: Edit, adapter_name: str = "astro") -> PeftModel:
    """Wrap only the edit's modules in LoRA layers carrying the edit's factors.

    Edit module names map to model names through `resolve_modules`. Raises
    `RuntimeError` if the adapter did not wrap every edit module.
    """
    names = resolve_modules(model, edit)
    modules = dict(model.named_modules())
    state: dict[str, torch.Tensor] = {}
    for d in edit.deltas:
        name = names[d.module]
        dtype = modules[name].weight.dtype
        state[f"{PREFIX}{name}{SUFFIX_A}"] = torch.from_numpy(d.A).to(dtype)
        state[f"{PREFIX}{name}{SUFFIX_B}"] = torch.from_numpy(d.B).to(dtype)

    cfg = LoraConfig(
        r=edit.rank,
        lora_alpha=edit.rank,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(names.values()),
    )
    peft_model = get_peft_model(model, cfg, adapter_name=adapter_name)
    result: Any = set_peft_model_state_dict(
        peft_model, state, adapter_name=adapter_name
    )
    # An adapter that "loads" but changes nothing is the common silent failure.
    wrapped = sum(
        n.endswith(f".lora_A.{adapter_name}") for n, _ in peft_model.named_modules()
    )
    if result.unexpected_keys or wrapped != len(edit.deltas):
        raise RuntimeError(
            f"adapter did not load cleanly: {wrapped} of {len(edit.deltas)} modules "
            f"wrapped, unexpected keys {list(result.unexpected_keys)}"
        )
    return peft_model


def remove(peft_model: PeftModel) -> nn.Module:
    """Drop the LoRA layers (no merge); base weights were never modified."""
    base: nn.Module = peft_model.unload()
    return base


def swap(peft_model: PeftModel, edit: Edit, adapter_name: str = "astro") -> PeftModel:
    """Replace the applied edit with another: remove, then apply on the base.

    The new edit's modules are resolved first, so a bad edit raises before
    anything is removed and `peft_model` stays usable.
    """
    _resolve(_base_linears(peft_model), edit)
    return apply(remove(peft_model), edit, adapter_name)


def merged_weights(model: nn.Module, edit: Edit) -> dict[str, torch.Tensor]:
    """Edited weights `W0 + B @ A` per module, keyed by the model's module name.

    `model` must be the plain base model. The sum is done in at least float32
    (float64 weights stay float64) and cast back to the weight's dtype; results
    live on CPU. The model is not modified.
    """
    if isinstance(model, PeftModel):
        msg = "merged_weights needs the plain base model, not a PeftModel"
        raise ValueError(msg)  # noqa: TRY004 (spec: ValueError)
    names = resolve_modules(model, edit)
    modules = dict(model.named_modules())
    merged: dict[str, torch.Tensor] = {}
    for d in edit.deltas:
        name = names[d.module]
        w0 = modules[name].weight.detach().cpu()
        dense = torch.from_numpy(d.B) @ torch.from_numpy(d.A)
        acc = torch.promote_types(w0.dtype, torch.float32)
        merged[name] = (w0.to(acc) + dense.to(acc)).to(w0.dtype)
    return merged
