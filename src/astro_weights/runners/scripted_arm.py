"""Scripted closed-form baseline arm: a difference-of-means `project_out`."""

import re
from collections.abc import Sequence

from astro_weights.directions import direction_from_means
from astro_weights.edits import Edit, project_out
from astro_weights.runners.hf import resolve_modules
from astro_weights.runners.hooks import capture

try:
    from torch import nn
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"{__name__} needs torch and transformers: pip install 'astro-weights[hf]'",
        name=e.name,
    ) from e

_WRITER = re.compile(r"(?:^|\.)layers\.(\d+)\.(?:self_attn\.o_proj|mlp\.down_proj)$")


def scripted_arm(
    model: nn.Module,
    input_ids_a: Sequence[Sequence[int]],
    input_ids_b: Sequence[Sequence[int]],
    like: Edit,
    *,
    strength: float = 1.0,
    last_tokens: int = 36,
) -> Edit:
    """A closed-form edit at the places `like` edits: the cheap baseline arm.

    `input_ids_a` are texts showing the behaviour to remove (say verbose
    answers), `input_ids_b` the contrast (terse ones). For each layer `L` that
    `like` touches, the direction is `direction_from_means(mean_a, mean_b)` of
    the layer-`L` residual output (`runners.hooks.capture`, mean over the last
    `last_tokens` positions, then over texts). Each delta of `like` becomes
    `project_out(module, W0, v_L, strength)` with `W0` the model's weight of
    that module, in the same module order as `like`. `strength` 1.0 removes the
    direction from the module's output entirely.

    Only modules that write the residual stream qualify: `self_attn.o_proj` and
    `mlp.down_proj`. Any other module in `like` raises `ValueError`. The result
    is deterministic. The model is not modified.
    """
    for d in like.deltas:
        if not _WRITER.search(d.module):
            raise ValueError(
                f"{d.module}: output is not in the residual stream; only "
                "self_attn.o_proj and mlp.down_proj qualify"
            )
    names = resolve_modules(model, like)
    layer_of = {}
    for d in like.deltas:
        m = _WRITER.search(names[d.module])
        if m is None:
            raise ValueError(f"{d.module}: resolves to {names[d.module]}, not a writer")
        layer_of[d.module] = int(m.group(1))
    layers = sorted(set(layer_of.values()))
    means = []
    for ids in (input_ids_a, input_ids_b):
        hid = capture(
            model, ids, want={"hidden"}, layers=layers, last_tokens=last_tokens
        )
        means.append({li: hid[f"hidden_L{li}"].mean(0) for li in layers})
    direction = {li: direction_from_means(means[0][li], means[1][li]) for li in layers}
    modules = dict(model.named_modules())
    deltas = [
        project_out(
            d.module,
            modules[names[d.module]].weight.detach().float().cpu().numpy(),
            direction[layer_of[d.module]],
            strength,
        )
        for d in like.deltas
    ]
    meta: dict[str, object] = {"recipe": "scripted_arm", "strength": strength}
    return Edit(tuple(deltas), like.base_model, dict(like.base_sha256), meta)
