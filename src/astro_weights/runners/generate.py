"""Run one eval arm: greedy, batch size 1, settings recorded."""

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import peft
import torch
import transformers

from astro_weights.eval import Record, prompt_hash

_KEYS = ("item_id", "prompt", "gold")


def _prompt_text(tokenizer: Any, prompt: str, chat: bool) -> str:
    if chat and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return str(text)
    return prompt


def generate_records(
    model: Any,
    tokenizer: Any,
    items: Sequence[Mapping[str, Any]],
    arm: str,
    scorer: Callable[[str, str], bool],
    max_new_tokens: int,
    seed: int = 0,
    chat: bool = True,
) -> list[Record]:
    """Generate and score each item; items carry `item_id`, `prompt`, `gold`."""
    for item in items:
        for key in _KEYS:
            if key not in item:
                raise ValueError(f"item is missing key {key!r}")
    torch.manual_seed(seed)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
    records: list[Record] = []
    for item in items:
        text = _prompt_text(tokenizer, item["prompt"], chat)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=not chat)
        enc = enc.to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or 0,
            )
        new = out[0, enc["input_ids"].shape[1] :]
        n = int(new.shape[0])
        output = tokenizer.decode(new, skip_special_tokens=True)
        hit_cap = n > 0 and n == max_new_tokens and int(new[-1]) not in eos_ids
        records.append(
            Record(
                item_id=str(item["item_id"]),
                seed=seed,
                arm=arm,
                prompt_hash=prompt_hash(text),
                output=output,
                score=float(scorer(item["gold"], output)),
                n_output_tokens=n,
                hit_cap=hit_cap,
            )
        )
    return records


def run_header(
    model: Any, tokenizer: Any, max_new_tokens: int, seed: int, chat: bool
) -> dict[str, Any]:
    """Settings that decide the outputs, for the run log."""
    template = getattr(tokenizer, "chat_template", None)
    return {
        "model": type(model).__name__,
        "dtype": str(model.dtype),
        "device": str(model.device),
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "chat": chat,
        "chat_template_sha256": (
            hashlib.sha256(str(template).encode("utf-8")).hexdigest()
            if template
            else None
        ),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }
