"""Cheap pre-screens run before any eval: perplexity and last-token KL.

No neutral dataset ships with the library: the caller passes token ids.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from astro_weights.edits import Edit
from astro_weights.runners.hf import apply, remove

try:
    import torch
    from torch import nn
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"{__name__} needs torch and transformers: pip install 'astro-weights[hf]'",
        name=e.name,
    ) from e


@dataclass(frozen=True)
class KLResult:
    """KL(base || edited) at the last position, per prompt, in nats."""

    per_prompt: tuple[float, ...]
    mean: float
    max: float
    threshold: float
    passed: bool


@dataclass(frozen=True)
class PerplexityResult:
    """Base and edited perplexity on the same texts; passed: ratio <= max_ratio."""

    base: float
    edited: float
    ratio: float
    max_ratio: float
    passed: bool


def _device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _logits(model: nn.Module, ids: Sequence[int]) -> torch.Tensor:
    x = torch.tensor([list(ids)], device=_device(model))
    out: Any = model(input_ids=x)
    logits: torch.Tensor = out.logits[0]
    return logits


def _nll(model: nn.Module, input_ids: Sequence[Sequence[int]]) -> tuple[float, int]:
    """Summed teacher-forced NLL and the number of predicted tokens."""
    total, count = 0.0, 0
    with torch.no_grad():
        for ids in input_ids:
            if len(ids) < 2:
                raise ValueError("each text needs at least 2 tokens")
            logp = torch.log_softmax(_logits(model, ids)[:-1].double(), dim=-1)
            tgt = torch.tensor(list(ids)[1:], device=logp.device)
            total -= float(logp.gather(1, tgt[:, None]).sum())
            count += len(ids) - 1
    return total, count


def perplexity(model: nn.Module, input_ids: Sequence[Sequence[int]]) -> float:
    """exp of the token-weighted mean teacher-forced NLL over all texts.

    Every text needs at least 2 tokens (`ValueError` otherwise), and there must
    be at least one text. The model is run as given; pass a neutral set of
    token ids, none ships with the library.
    """
    if not input_ids:
        raise ValueError("input_ids is empty")
    total, count = _nll(model, input_ids)
    return math.exp(total / count)


def _with_edit(model: nn.Module, edit: Edit, fn: Callable[[nn.Module], Any]) -> Any:
    """`fn(edited model)`, then the edit is removed even if `fn` raises."""
    peft_model = apply(model, edit)
    try:
        return fn(peft_model)
    finally:
        remove(peft_model)


def kl_last_token(
    model: nn.Module,
    input_ids: Sequence[Sequence[int]],
    edit: Edit,
    *,
    threshold: float = 0.1,
) -> KLResult:
    """KL(base || edited) of the next-token distribution at each prompt's last token.

    Arditi et al. keep the KL divergence at the last token below 0.1 between
    the base and the edited model; `threshold` defaults to that and `passed` is
    `mean <= threshold`. Low KL is necessary, not sufficient: an edit can keep
    the next token distribution and still change behaviour further out, so
    this screens out broken edits and proves nothing about good ones.

    One model stays in memory: base logits first, then `runners.hf.apply`, the
    edited logits, and `runners.hf.remove`, also when a forward raises. KL is
    taken in float64 from `log_softmax` over the full vocabulary. Pass token
    ids of a neutral prompt set; none ships with the library.
    """
    if not input_ids:
        raise ValueError("input_ids is empty")
    if not all(len(ids) for ids in input_ids):
        raise ValueError("each prompt needs at least 1 token")

    def last(m: nn.Module) -> list[torch.Tensor]:
        with torch.no_grad():
            return [
                torch.log_softmax(_logits(m, ids)[-1].double(), dim=-1)
                for ids in input_ids
            ]

    base = last(model)
    edited = _with_edit(model, edit, last)
    kls = tuple(
        float((b.exp() * (b - e)).sum()) for b, e in zip(base, edited, strict=True)
    )
    mean = sum(kls) / len(kls)
    return KLResult(kls, mean, max(kls), threshold, mean <= threshold)


def perplexity_guard(
    model: nn.Module,
    input_ids: Sequence[Sequence[int]],
    edit: Edit,
    *,
    max_ratio: float,
) -> PerplexityResult:
    """Perplexity of base and edited model on the same texts; passed: ratio <= `max_ratio`.

    The edit is applied and removed as in `kl_last_token`. `max_ratio` is data
    the caller brings (say 1.1 for 10% worse); no default is guessed here.
    """
    if not input_ids:
        raise ValueError("input_ids is empty")
    if not (math.isfinite(max_ratio) and max_ratio > 0):
        raise ValueError(f"max_ratio must be finite and > 0, got {max_ratio}")
    base = _nll(model, input_ids)
    edited = _with_edit(model, edit, lambda m: _nll(m, input_ids))
    base_ppl = math.exp(base[0] / base[1])
    edit_ppl = math.exp(edited[0] / edited[1])
    ratio = edit_ppl / base_ppl
    return PerplexityResult(base_ppl, edit_ppl, ratio, max_ratio, ratio <= max_ratio)
