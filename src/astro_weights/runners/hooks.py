"""Teacher-forced capture of activations and logprobs on a transformers model.

One text at a time, no gradients. Hooks are removed and the attention
implementation is restored even when the forward pass raises.
"""

import re
from collections import defaultdict
from collections.abc import Collection, Sequence
from typing import Any, Literal

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"{__name__} needs torch and transformers: pip install 'astro-weights[hf]'",
        name=e.name,
    ) from e

Want = Literal["hidden", "mlp", "attn", "logprobs"]
_WANTS = ("hidden", "mlp", "attn", "logprobs")
_LAYER = re.compile(r"(.*)\blayers\.(\d+)$")
_TOP = 5


def _layers(model: nn.Module) -> list[nn.Module]:
    """The decoder layers in order, whether named `model.layers.N` or
    `model.language_model.layers.N` (the largest `layers` group wins)."""
    groups: dict[str, dict[int, nn.Module]] = defaultdict(dict)
    for name, module in model.named_modules():
        m = _LAYER.fullmatch(name)
        if m:
            groups[m.group(1)][int(m.group(2))] = module
    if not groups:
        raise ValueError("no decoder layers (`layers.<int>`) found in the model")
    best = max(groups.values(), key=len)
    return [best[i] for i in sorted(best)]


def layer_count(model: nn.Module) -> int:
    """Number of decoder layers."""
    return len(_layers(model))


def full_attention_layers(model: nn.Module) -> list[int]:
    """Indices of layers whose `self_attn` can return attention weights."""
    return [i for i, layer in enumerate(_layers(model)) if hasattr(layer, "self_attn")]


def _check(
    model: nn.Module,
    input_ids: Sequence[Sequence[int]],
    want: Collection[str],
    layers: Sequence[int],
) -> None:
    if not input_ids or any(len(t) == 0 for t in input_ids):
        raise ValueError("input_ids must hold at least one non-empty text")
    unknown = sorted(set(want) - set(_WANTS))
    if unknown:
        raise ValueError(f"unknown want {unknown}; choose from {list(_WANTS)}")
    n = layer_count(model)
    bad = [i for i in layers if not 0 <= i < n]
    if bad:
        raise ValueError(f"layers {bad} out of range for a model with {n} layers")
    if "attn" in want:
        full = full_attention_layers(model)
        no = [i for i in layers if i not in full]
        if no:
            raise ValueError(
                f"layers {no} have no attention weights; full attention layers: {full}"
            )


def _mean_last(x: Any, last: int) -> np.ndarray:
    out: np.ndarray = x[0, -last:].float().mean(0).cpu().numpy()
    return out


def capture(
    model: nn.Module,
    input_ids: Sequence[Sequence[int]],
    *,
    want: Collection[Want],
    layers: Sequence[int] = (),
    last_tokens: int = 36,
) -> dict[str, np.ndarray]:
    """Run each text through `model` and return what `want` asks for.

    Keys: `hidden_L{l}` and `mlp_L{l}` (layer output; input of `down_proj`):
    float32 `[n_texts, d]`, the mean over the last `last_tokens` positions.
    `attn_L{l}_t{i}`: float32 `[heads, q, k]` for text `i` (full attention
    layers only; the model runs with eager attention meanwhile).
    `logprobs_top_ids` int64 `[n, last, 5]`, `logprobs_top` float32
    `[n, last, 5]` and `logprobs_target` float32 `[n, last]`, aligned to the
    start; positions a short text lacks hold -1 or NaN, and so does the final
    position's target (it has no next token).
    """
    _check(model, input_ids, want, layers)
    wanted = set(want)
    blocks: list[Any] = _layers(model)
    last = last_tokens
    store: dict[str, list[np.ndarray]] = defaultdict(list)
    attn: dict[str, np.ndarray] = {}
    cur = {"i": 0}

    hooks = []
    for li in layers:
        layer = blocks[li]
        if "hidden" in wanted:

            def on_layer(_m: nn.Module, _a: Any, out: Any, li: int = li) -> None:
                h = out[0] if isinstance(out, tuple) else out
                store[f"hidden_L{li}"].append(_mean_last(h, last))

            hooks.append(layer.register_forward_hook(on_layer))
        if "mlp" in wanted:

            def on_mlp(_m: nn.Module, args: Any, li: int = li) -> None:
                store[f"mlp_L{li}"].append(_mean_last(args[0], last))

            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(on_mlp))
        if "attn" in wanted:

            def on_attn(_m: nn.Module, _a: Any, out: Any, li: int = li) -> None:
                w = out[1] if isinstance(out, tuple) and len(out) > 1 else None
                if w is None:
                    raise RuntimeError(f"layer {li} returned no attention weights")
                attn[f"attn_L{li}_t{cur['i']}"] = w[0].float().cpu().numpy()

            hooks.append(layer.self_attn.register_forward_hook(on_attn))

    n = len(input_ids)
    top_ids = np.full((n, last, _TOP), -1, np.int64)
    top_vals = np.full((n, last, _TOP), np.nan, np.float32)
    target = np.full((n, last), np.nan, np.float32)
    impl = getattr(model.config, "_attn_implementation", None)
    swap = "attn" in wanted
    api: Any = model
    device = next(model.parameters()).device
    try:
        if swap:
            api.set_attn_implementation("eager")
        with torch.no_grad():
            for i, text in enumerate(input_ids):
                cur["i"] = i
                ids = torch.tensor([list(text)], dtype=torch.long, device=device)
                keep = last if "logprobs" in wanted else 1
                logits = model(input_ids=ids, use_cache=False, logits_to_keep=keep)
                if "logprobs" in wanted:
                    lp = torch.log_softmax(logits.logits[0, -last:].float(), -1)
                    v, t = lp.topk(_TOP, -1)
                    m = lp.shape[0]
                    top_ids[i, :m] = t.cpu().numpy()
                    top_vals[i, :m] = v.cpu().numpy()
                    if m > 1:  # the last position has no next token
                        rows = torch.arange(m - 1, device=device)
                        target[i, : m - 1] = lp[rows, ids[0, -m + 1 :]].cpu().numpy()
    finally:
        for h in hooks:
            h.remove()
        if swap and impl is not None:
            api.set_attn_implementation(impl)

    out: dict[str, np.ndarray] = {k: np.stack(v) for k, v in store.items()}
    out.update(attn)
    if "logprobs" in wanted:
        out["logprobs_top_ids"] = top_ids
        out["logprobs_top"] = top_vals
        out["logprobs_target"] = target
    return out
