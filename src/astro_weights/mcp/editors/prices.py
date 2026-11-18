"""Claude API token prices and the dollar cost of one turn, for the hosted editor.

Prices are US dollars per million tokens on the first-party Claude API at standard
speed, global inference. Source: https://platform.claude.com/docs/en/about-claude/pricing
(model pricing, prompt caching, long context and data residency sections), read
2026-11-17. A model not listed here has no price, and the hosted editor refuses to
start with it. Standard library only.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["PRICES", "US_GEO", "Price", "max_cost", "price", "turn_cost"]


@dataclass(frozen=True)
class Price:
    """Dollars per million tokens: base input, 5-minute and 1-hour cache writes,
    cache reads, output."""

    input: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float
    output: float


PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(4, 5, 8, 0.20, 20),
    "claude-opus-5": Price(5, 6.25, 10, 0.50, 25),
    "claude-opus-4-8": Price(5, 6.25, 10, 0.50, 25),
    "claude-opus-4-7": Price(5, 6.25, 10, 0.50, 25),
    "claude-opus-4-6": Price(5, 6.25, 10, 0.50, 25),
    "claude-sonnet-5-5": Price(2, 2.50, 4, 0.10, 10),
    "claude-sonnet-5": Price(2, 2.50, 4, 0.20, 10),
    "claude-sonnet-4-6": Price(3, 3.75, 6, 0.30, 15),
    "claude-haiku-5-5": Price(0.10, 0.125, 0.20, 0.01, 0.50),
    "claude-haiku-4-5": Price(1, 1.25, 2, 0.10, 5),
}

# Priced by prompt length: a prompt (input + cache reads + cache writes) over the
# threshold pays the second price for the whole request.
_LONG: dict[str, tuple[int, Price]] = {
    "claude-haiku-5-5": (100_000, Price(0.50, 0.625, 1, 0.05, 2.50)),
}

US_GEO = 1.1  # multiplier on every token price when inference runs US-only
_MILLION = 1_000_000


def price(model: str, prompt_tokens: int = 0) -> Price:
    """The price for ``model`` at this prompt length; ValueError for an unknown model."""
    if model not in PRICES:
        raise ValueError(
            f"no price for model {model!r}; known: {', '.join(sorted(PRICES))}"
        )
    long = _LONG.get(model)
    if long is not None and prompt_tokens > long[0]:
        return long[1]
    return PRICES[model]


def _count(usage: Mapping[str, Any], key: str, *, required: bool = False) -> int:
    value = usage.get(key)
    if value is None and not required:
        return 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"usage {key} must be an int >= 0, got {value!r}")
    return value


def turn_cost(model: str, usage: Mapping[str, Any]) -> float:
    """Dollars one Messages API response cost, from its ``usage`` as a dict.

    Cache writes use the 5-minute and 1-hour split in ``cache_creation`` when given;
    the rest of ``cache_creation_input_tokens`` is priced as 5-minute writes (the only
    kind this package asks for). ``inference_geo`` other than ``global`` costs 1.1x.
    """
    fresh = _count(usage, "input_tokens", required=True)
    output = _count(usage, "output_tokens", required=True)
    read = _count(usage, "cache_read_input_tokens")
    written = _count(usage, "cache_creation_input_tokens")
    split = usage.get("cache_creation")
    if split is None:
        split = {}
    if not isinstance(split, Mapping):
        raise ValueError(f"usage cache_creation must be an object, got {split!r}")  # noqa: TRY004
    w1h = _count(split, "ephemeral_1h_input_tokens")
    w5m = max(written - w1h, _count(split, "ephemeral_5m_input_tokens"))
    p = price(model, fresh + read + w5m + w1h)
    total = (
        fresh * p.input
        + w5m * p.cache_write_5m
        + w1h * p.cache_write_1h
        + read * p.cache_read
        + output * p.output
    ) / _MILLION
    geo = usage.get("inference_geo")
    return total * US_GEO if geo not in (None, "", "global") else total


def max_cost(model: str, prompt_tokens: int, max_output_tokens: int) -> float:
    """An upper bound on one call's dollars before it is made.

    Every prompt token at the higher of the base input and 5-minute cache write
    price, ``max_output_tokens`` at the output price, all times the US-only
    multiplier (the account's inference region is not known in advance).
    """
    if prompt_tokens < 0 or max_output_tokens < 0:
        raise ValueError("token counts must be >= 0")
    p = price(model, prompt_tokens)
    per_prompt = max(p.input, p.cache_write_5m)
    total = (prompt_tokens * per_prompt + max_output_tokens * p.output) / _MILLION
    return total * US_GEO
