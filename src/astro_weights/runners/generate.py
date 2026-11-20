"""Run one eval arm: greedy or sampled, records written as made, settings recorded."""

import hashlib
import json
import random
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

from astro_weights.eval import Record, prompt_hash, read_jsonl
from astro_weights.eval.header import run_identity

try:
    import peft
    import torch
    import transformers
    from transformers import LogitsProcessor, LogitsProcessorList
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"{__name__} needs torch, transformers and peft: "
        "pip install 'astro-weights[hf]'",
        name=e.name,
    ) from e

_KEYS = ("item_id", "prompt", "gold")
_SAMPLING_KEYS = frozenset({"temperature", "top_p", "top_k", "presence_penalty"})
_WINDOW = 64  # llama.cpp's default penalty window


class _PresencePenalty(LogitsProcessor):  # type: ignore[misc, unused-ignore]
    """Subtract a flat `penalty` from the logit of each token in the last 64.

    The window covers prompt and generated tokens. `lead[i]` is the number of
    left-padding positions of row `i`; those are not counted as seen.
    """

    def __init__(self, penalty: float, lead: Any) -> None:
        self.penalty = penalty
        self.lead = lead

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        n, vocab = scores.shape
        end = input_ids.shape[1]
        start = max(end - _WINDOW, 0)
        pos = torch.arange(start, end, device=input_ids.device)
        real = pos[None, :] >= self.lead.to(input_ids.device)[:, None]
        # padding goes to an extra column that is dropped, so writes never clash
        ids = torch.where(real, input_ids[:, start:], vocab)
        seen = torch.zeros(n, vocab + 1, dtype=torch.bool, device=scores.device)
        seen.scatter_(1, ids.to(scores.device), True)
        return scores - self.penalty * seen[:, :vocab].to(scores.dtype)


def _check_sampling(sampling: Mapping[str, float] | None) -> None:
    if sampling is None:
        return
    unknown = sorted(set(sampling) - _SAMPLING_KEYS)
    if unknown:
        raise ValueError(
            f"unknown sampling keys {unknown}; allowed: {sorted(_SAMPLING_KEYS)}"
        )
    if "temperature" in sampling and not sampling["temperature"] > 0:
        raise ValueError(f"temperature must be > 0, got {sampling['temperature']!r}")


def _gen_kwargs(sampling: Mapping[str, float] | None, lead: Any) -> dict[str, Any]:
    """`model.generate` arguments for greedy (None) or sampled decoding."""
    if sampling is None:
        return {"do_sample": False}
    kwargs: dict[str, Any] = {"do_sample": True}
    kwargs.update({k: v for k, v in sampling.items() if k != "presence_penalty"})
    if "top_k" in kwargs:
        kwargs["top_k"] = int(kwargs["top_k"])  # transformers rejects 20.0
    if sampling.get("presence_penalty"):
        kwargs["logits_processor"] = LogitsProcessorList(
            [_PresencePenalty(sampling["presence_penalty"], lead)]
        )
    return kwargs


def _prompt_text(tokenizer: Any, prompt: str, chat: bool) -> tuple[str, bool]:
    """Prompt text and whether a chat template (which carries its own specials) ran."""
    if chat and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return str(text), True
    return prompt, False


def _check_items(items: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for item in items:
        for key in _KEYS:
            if key not in item:
                raise ValueError(f"item is missing key {key!r}")
        item_id = str(item["item_id"])
        if item_id in seen:
            raise ValueError(f"duplicate item_id {item_id!r}")
        seen.add(item_id)


def _read_file(path: Path) -> tuple[list[Record], int | None, bool]:
    """Records in `path` without changing it, and how to repair its last line.

    Returns (records, cut, whole): `cut` is the size to truncate to when a crash
    left a last line without its newline that is not valid JSON; `whole` is True
    when that line is valid and only lacks its newline.
    """
    data = path.read_bytes()
    cut: int | None = None
    whole = False
    if data and not data.endswith(b"\n"):
        start = data.rfind(b"\n") + 1
        try:
            json.loads(data[start:])
        except ValueError:  # truncated JSON or a cut UTF-8 character
            cut, data = start, data[:start]
        else:
            whole = True
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "records.jsonl"
        copy.write_bytes(data)
        return read_jsonl(copy), cut, whole


def _load_done(
    path: Path | None, resume: bool, hashes: Mapping[str, str], arm: str, seed: int
) -> dict[str, Record]:
    """Records of an earlier run in `path`, checked against this run.

    The file is only repaired (a crash's partial last line) after it checked out.
    """
    if path is None or not path.exists() or path.stat().st_size == 0:
        return {}
    if not resume:
        raise FileExistsError(
            f"{path} already has records; pass resume=True to continue it "
            "or choose another path"
        )
    records, cut, whole = _read_file(path)
    done: dict[str, Record] = {}
    for r in records:
        if (r.arm, r.seed) != (arm, seed):
            raise ValueError(
                f"{path}: item {r.item_id!r} is arm {r.arm!r} seed {r.seed}, "
                f"this run is arm {arm!r} seed {seed}; the file is another run's"
            )
        if r.item_id not in hashes:
            raise ValueError(f"{path}: item_id {r.item_id!r} is not in items")
        if r.prompt_hash != hashes[r.item_id]:
            raise ValueError(
                f"{path}: prompt of item {r.item_id!r} differs from the one "
                "that would be sent now; the file is another run's"
            )
        if r.item_id in done:
            raise ValueError(f"{path}: duplicate item_id {r.item_id!r}")
        done[r.item_id] = r
    if cut is not None:
        with path.open("r+b") as f:
            f.truncate(cut)
    elif whole:
        with path.open("ab") as f:
            f.write(b"\n")
    return done


def generate_records(
    model: Any,
    tokenizer: Any,
    items: Sequence[Mapping[str, Any]],
    arm: str,
    scorer: Callable[[str, str], bool],
    max_new_tokens: int,
    seed: int = 0,
    chat: bool = True,
    *,
    out: Path | str | None = None,
    resume: bool = False,
    batch_size: int = 1,
    sampling: Mapping[str, float] | None = None,
) -> list[Record]:
    """Generate and score each item; items carry `item_id`, `prompt`, `gold`.

    `scorer(gold, output)` sees the full decoded text. If the model emits a think
    block, wrap the scorer yourself (e.g. with `answer_part`), or a `\\boxed{}`
    inside the thinking can be graded. The model runs in eval mode (restored after).

    With `out`, each record is appended to that JSONL file (the `write_jsonl`
    line format) and flushed as soon as it exists, so a crash keeps what was done.
    A non-empty `out` raises `FileExistsError` unless `resume=True`. Resuming
    keeps the records already in the file and skips their items; a last line cut
    short by a crash is dropped and redone. A record whose arm or seed differs,
    whose `item_id` is not in `items`, or whose prompt hash differs from the
    prompt sent now raises `ValueError`: the file belongs to another run. The
    returned list holds every record in the order of `items`, resumed or new.

    `batch_size > 1` generates that many items at once with left padding (the
    tokenizer needs a pad token; its `padding_side` is restored after).
    `n_output_tokens` counts tokens up to and including the first EOS. Greedy
    outputs can differ from `batch_size=1`: padding can change float rounding,
    and a near tie may then resolve the other way. Use one batch size for both
    arms of a paired comparison. Batches sit at fixed positions of `items`, and a
    batch is scored before any of its records is written, so a resume with the
    same `batch_size` rebuilds the batches of an uninterrupted run. (Records an
    earlier run wrote with another `batch_size` are kept as they are.) The default `batch_size=1` runs each prompt
    alone, unpadded.

    Decoding is greedy by default. Greedy is a stress condition: some models
    repeat one sentence until the cap. For the headline number use the model
    card's recommended sampling settings. `sampling` maps any of `temperature`
    (> 0), `top_p`, `top_k` and `presence_penalty` to a number and turns
    sampling on; an unknown key or a non-positive temperature raises
    `ValueError`. `presence_penalty` subtracts that value from the logit of
    every token among the last 64 (prompt included), as llama.cpp does.
    Sampled outputs repeat for the same model, items, `seed`, `sampling` and
    `batch_size`, also across a resume: each batch is reseeded from `seed` and
    its position in `items`. Both arms of a paired comparison need the same
    `sampling`.
    """
    _check_sampling(sampling)
    _check_items(items)
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError(f"batch_size must be an int >= 1, got {batch_size!r}")
    if resume and out is None:
        raise ValueError("resume=True needs out")
    if batch_size > 1 and tokenizer.pad_token_id is None:
        raise ValueError(
            "batch_size > 1 needs a pad token and the tokenizer has none. "
            "Set one, e.g. tokenizer.pad_token = tokenizer.eos_token"
        )
    prompts = [_prompt_text(tokenizer, item["prompt"], chat) for item in items]
    hashes = {
        str(item["item_id"]): prompt_hash(text)
        for item, (text, _) in zip(items, prompts, strict=True)
    }
    path = None if out is None else Path(out)
    done = _load_done(path, resume, hashes, arm, seed)
    entries = [
        (item, text, templated)
        for item, (text, templated) in zip(items, prompts, strict=True)
    ]
    was_training = bool(model.training)
    model.eval()
    try:
        with path.open("a", encoding="utf-8") if path else nullcontext() as sink:
            for rec in _generate(
                model, tokenizer, entries, set(done), arm, scorer,
                max_new_tokens, seed, batch_size, sampling,
            ):  # fmt: skip
                done[rec.item_id] = rec
                if sink is not None:
                    sink.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                    sink.flush()
    finally:
        model.train(was_training)
    return [done[str(item["item_id"])] for item in items]


def _generate(
    model: Any,
    tokenizer: Any,
    entries: Sequence[tuple[Mapping[str, Any], str, bool]],
    done: set[str],
    arm: str,
    scorer: Callable[[str, str], bool],
    max_new_tokens: int,
    seed: int,
    batch_size: int,
    sampling: Mapping[str, float] | None,
) -> Iterator[Record]:
    torch.manual_seed(seed)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
    # Batches sit at fixed positions of `entries`, so a resumed run batches the
    # remaining items as an uninterrupted one would (minus the items done).
    for start in range(0, len(entries), batch_size):
        chunk = [
            e
            for e in entries[start : start + batch_size]
            if str(e[0]["item_id"]) not in done
        ]
        if not chunk:
            continue
        if sampling is not None:  # a resume reaches this batch with the same state
            torch.manual_seed(random.Random(f"{seed}:{start}").getrandbits(63))
        if batch_size == 1:
            _, text, templated = chunk[0]
            rows = [
                _run_one(
                    model, tokenizer, text, templated, max_new_tokens, eos_ids,
                    sampling,
                )
            ]  # fmt: skip
        else:
            texts = [text for _, text, _ in chunk]
            rows = _run_batch(
                model, tokenizer, texts, chunk[0][2], max_new_tokens, eos_ids,
                sampling,
            )  # fmt: skip
        # Score the whole batch before yielding any of it: a crash while scoring
        # then leaves no half-written batch, and a resume restarts at a boundary.
        records = [
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
            for (item, text, _), (output, n, hit_cap) in zip(chunk, rows, strict=True)
        ]
        yield from records


def _run_one(
    model: Any,
    tokenizer: Any,
    text: str,
    templated: bool,
    max_new_tokens: int,
    eos_ids: set[int],
    sampling: Mapping[str, float] | None = None,
) -> tuple[str, int, bool]:
    """(output, n_output_tokens, hit_cap) for one unpadded prompt."""
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=not templated)
    enc = enc.to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            **_gen_kwargs(sampling, torch.zeros(1, dtype=torch.long)),
            pad_token_id=tokenizer.pad_token_id or 0,
        )
    new = out[0, enc["input_ids"].shape[1] :]
    n = int(new.shape[0])
    output = tokenizer.decode(new, skip_special_tokens=True)
    return output, n, n > 0 and n == max_new_tokens and int(new[-1]) not in eos_ids


def _run_batch(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    templated: bool,
    max_new_tokens: int,
    eos_ids: set[int],
    sampling: Mapping[str, float] | None = None,
) -> list[tuple[str, int, bool]]:
    """Like `_run_one` for several prompts, left-padded; rows cut at the first EOS."""
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            add_special_tokens=not templated,
        )
    finally:
        tokenizer.padding_side = side
    enc = enc.to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            **_gen_kwargs(sampling, (enc["attention_mask"] == 0).sum(1)),
            pad_token_id=tokenizer.pad_token_id,
        )
    rows: list[tuple[str, int, bool]] = []
    for row in out[:, enc["input_ids"].shape[1] :]:
        ids = row.tolist()
        stop = next((k + 1 for k, t in enumerate(ids) if t in eos_ids), None)
        n = len(ids) if stop is None else stop
        output = tokenizer.decode(row[:n], skip_special_tokens=True)
        rows.append((output, n, stop is None and n > 0 and n == max_new_tokens))
    return rows


def run_header(
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    seed: int,
    chat: bool,
    *,
    batch_size: int = 1,
    sampling: Mapping[str, float] | None = None,
    run_id: str | None = None,
    k: int = 1,
    concurrency: int = 1,
) -> dict[str, Any]:
    """Settings that decide the outputs, plus where the run happened, for the run log.

    Keys are CONFIG (compared, hashed by `config_hash`) or PROVENANCE (stored only):
    see `astro_weights.eval.header`. `backend` is the device type (cpu, cuda, mps);
    the torch and transformers versions are CONFIG, the device string is provenance. `run_id` is random when not given; `k` is samples
    per item and `concurrency` requests in flight (both recorded as given). GPU model
    and count are recorded when CUDA is available. The astro-weights version and
    library commit (`-dirty` when the checkout has changes) and the numpy version are
    provenance.

    Pass the same `sampling` as to `generate_records`. It is recorded under
    "sampling" (sorted keys) only when given, so a greedy header has no such key
    and `check_headers` rejects a greedy arm paired with a sampled one, or two
    arms sampled differently. Greedy is the default and a stress condition; for
    the headline number use the model card's recommended sampling settings, on
    both arms.
    """
    _check_sampling(sampling)
    template = getattr(tokenizer, "chat_template", None)
    # the model under an applied adapter, so a base arm and an edit arm compare equal
    inner = getattr(model, "get_base_model", lambda: model)()
    header: dict[str, Any] = {
        "model": type(inner).__name__,
        "model_name_or_path": str(
            getattr(getattr(inner, "config", None), "_name_or_path", "") or ""
        ),
        "backend": model.device.type,
        "dtype": str(model.dtype),
        "device": str(model.device),
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "chat": chat,
        "batch_size": batch_size,
        "training": False,  # generate_records always runs the model in eval mode
        "chat_template_sha256": (
            hashlib.sha256(str(template).encode("utf-8")).hexdigest()
            if template
            else None
        ),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        **run_identity(run_id, k, concurrency),
    }
    if torch.cuda.is_available():
        header["gpu_model"] = torch.cuda.get_device_name(0)
        header["gpu_count"] = torch.cuda.device_count()
    if sampling is not None:
        header["sampling"] = dict(sorted(sampling.items()))
    return header
