"""GPU suite on a real model; run by hand before a release.

    ASTRO_GPU_MODEL=/path/to/hf-model ASTRO_GPU_ADAPTER=/path/to/adapter \\
        uv run --extra hf pytest -m gpu tests/test_gpu.py

Each test body is a function of (model, tokenizer, edit). `test_gpu` runs the
bodies on the real model (skipped without the env vars or CUDA); `test_cpu_smoke`
runs the same bodies on the tiny model from conftest so CI covers the code path.
"""

import contextlib
import gc
import hashlib
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from astro_weights.edits import Edit, LowRankDelta, masked_rank1, top_k_mask
from astro_weights.eval import compare, grade_exact
from astro_weights.export import load_peft

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from peft.tuners.tuners_utils import BaseTunerLayer

from astro_weights.runners.generate import generate_records
from astro_weights.runners.hf import (
    apply,
    merged_weights,
    remove,
    resolve_modules,
    rounding_error,
    swap,
)

Body = Callable[[Any, Any, Edit], None]

# Plain lowercase words: the smoke tokenizer's vocabulary is built from these.
PROMPTS = [
    "the capital of france is",
    "two plus two equals",
    "water boils at one hundred degrees",
]
ITEMS = [
    {"item_id": f"i{k}", "prompt": p, "gold": "x"}
    for k, p in enumerate(
        [
            "the sky is",
            "one plus one is",
            "the sun rises in the",
            "cats like to",
        ]
    )
]
MIB = 1 << 20


def _logits(model: Any, tok: Any) -> list[Any]:
    """Float32 CPU logits [tokens, vocab] for each fixed prompt."""
    out = []
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(model.device)
        with torch.no_grad():
            out.append(model(input_ids=ids).logits[0].float().cpu())
    return out


def _equal(a: list[Any], b: list[Any]) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def _state_sha(model: Any) -> str:
    """SHA-256 over the names and raw bytes of every tensor in the state dict."""
    h = hashlib.sha256()
    for name, t in model.state_dict().items():
        h.update(name.encode())
        raw = t.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
        h.update(raw.numpy().tobytes())
    return h.hexdigest()


def _has_adapter(model: Any) -> bool:
    return any(isinstance(m, BaseTunerLayer) for m in model.modules())


@contextlib.contextmanager
def _applied(model: Any, edit: Edit, **kw: Any) -> Iterator[Any]:
    """`apply`, then always leave the (session-shared) model plain again."""
    peft_model = apply(model, edit, **kw)
    try:
        yield peft_model
    finally:
        remove(peft_model)


def _scaled(edit: Edit, factor: float) -> Edit:
    deltas = tuple(
        LowRankDelta(d.module, d.B * np.float32(factor), d.A) for d in edit.deltas
    )
    return Edit(deltas, edit.base_model)


def _footprint(model: Any) -> int:
    """Bytes held: CUDA allocator for a CUDA model, else parameters and buffers."""
    gc.collect()
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
        return int(torch.cuda.memory_allocated(model.device))
    tensors = [*model.parameters(), *model.buffers()]
    return sum(t.numel() * t.element_size() for t in tensors)


# --- bodies ---------------------------------------------------------------


def body_apply_remove(model: Any, tok: Any, edit: Edit) -> None:
    base = _logits(model, tok)
    with _applied(model, edit) as peft_model:
        edited = _logits(peft_model, tok)
        assert all(not torch.equal(b, e) for b, e in zip(base, edited, strict=True))
    assert not _has_adapter(model)
    assert _equal(_logits(model, tok), base)


def body_merged_weights(model: Any, tok: Any, edit: Edit) -> None:
    before = _state_sha(model)
    names = resolve_modules(model, edit)
    merged = merged_weights(model, edit)
    assert set(merged) == set(names.values())
    for d in edit.deltas:
        name = names[d.module]
        w0 = model.get_submodule(name).weight.detach().cpu()
        assert merged[name].dtype == w0.dtype
        dense = torch.from_numpy(d.B).double() @ torch.from_numpy(d.A).double()
        expected = w0.double() + dense
        # bf16 keeps 8 significant bits: half an ulp is 2**-9 of the value
        assert torch.allclose(merged[name].double(), expected, rtol=1e-2, atol=1e-6)
        assert not torch.equal(merged[name], w0)
    assert _state_sha(model) == before


def body_apply_vs_merged(model: Any, tok: Any, edit: Edit) -> None:
    names = resolve_modules(model, edit)
    merged = merged_weights(model, edit)
    with _applied(model, edit) as peft_model:
        lora = _logits(peft_model, tok)
    weights = {n: model.get_submodule(n).weight for n in names.values()}
    saved = {n: w.detach().clone() for n, w in weights.items()}
    try:
        with torch.no_grad():
            for n, w in weights.items():
                w.copy_(merged[n])
        dense = _logits(model, tok)
    finally:
        with torch.no_grad():
            for n, w in weights.items():
                w.copy_(saved[n])
    assert all(torch.equal(weights[n].detach(), saved[n]) for n in saved)

    max_diff = max(float((a - b).abs().max()) for a, b in zip(lora, dense, strict=True))
    agree = float(
        torch.cat(
            [a.argmax(-1) == b.argmax(-1) for a, b in zip(lora, dense, strict=True)]
        )
        .float()
        .mean()
    )
    assert max_diff < 1.0, max_diff
    assert agree > 0.97, agree


def body_swap(model: Any, tok: Any, edit: Edit) -> None:
    half = _scaled(edit, 0.5)
    with _applied(model, edit) as peft_model:
        full = _logits(peft_model, tok)
        peft_model = swap(peft_model, half)
        halved = _logits(peft_model, tok)
        assert not _equal(halved, full)
        peft_model = swap(peft_model, edit)
        swapped = _logits(peft_model, tok)
    with _applied(model, edit) as fresh:
        again = _logits(fresh, tok)
    assert _equal(swapped, again)
    assert _equal(full, again)
    assert not _has_adapter(model)


def body_rounding(model: Any, tok: Any, edit: Edit) -> None:
    before = _state_sha(model)
    errors = rounding_error(model, edit)
    assert set(errors) == {d.module for d in edit.deltas}
    assert all(np.isfinite(e) and 0.0 <= e < 1e-2 for e in errors.values())
    with pytest.raises(ValueError, match="rounding error"):
        apply(model, edit, max_rounding_error=1e-9)
    assert not _has_adapter(model)
    assert _state_sha(model) == before


def body_generate(model: Any, tok: Any, edit: Edit) -> None:
    def run(arm: str) -> list[Any]:
        return generate_records(model, tok, ITEMS, arm, grade_exact, 32)

    base = run("base")
    with _applied(model, edit) as peft_model:
        edited = generate_records(peft_model, tok, ITEMS, "edit", grade_exact, 32)
    assert [r.item_id for r in base] == [i["item_id"] for i in ITEMS]
    assert [r.item_id for r in edited] == [i["item_id"] for i in ITEMS]
    assert {r.arm for r in base} == {"base"}
    assert {r.arm for r in edited} == {"edit"}
    assert all(0 < r.n_output_tokens <= 32 for r in [*base, *edited])
    result = compare(base, edited, n_boot=200)
    assert result.n_items == len(ITEMS)


def body_memory(model: Any, tok: Any, edit: Edit) -> None:
    baseline = _footprint(model)
    for _ in range(3):
        peft_model = apply(model, edit)
        _logits(peft_model, tok)
        remove(peft_model)
        del peft_model
        assert not _has_adapter(model)
        assert _footprint(model) - baseline <= 64 * MIB


BODIES: list[Body] = [
    body_apply_remove,
    body_merged_weights,
    body_apply_vs_merged,
    body_swap,
    body_generate,
    body_memory,
]


def _name(body: Body) -> str:
    return body.__name__.removeprefix("body_")


# --- real model -----------------------------------------------------------


def _skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "no CUDA device"
    for var in ("ASTRO_GPU_MODEL", "ASTRO_GPU_ADAPTER"):
        if not os.environ.get(var):
            return f"{var} is not set"
    return None


@pytest.fixture(scope="session")
def gpu_case() -> tuple[Any, Any, Edit]:
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    path = os.environ["ASTRO_GPU_MODEL"]
    tok = transformers.AutoTokenizer.from_pretrained(path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map={"": 0}
    ).eval()
    return model, tok, load_peft(Path(os.environ["ASTRO_GPU_ADAPTER"]))


@pytest.mark.gpu
@pytest.mark.parametrize("body", [*BODIES, body_rounding], ids=_name)
def test_gpu(body: Body, gpu_case: tuple[Any, Any, Edit]) -> None:
    body(*gpu_case)


# --- CPU smoke: same bodies on the tiny model --------------------------------

MODULES = ["model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj"]


@pytest.fixture
def tok() -> Any:
    words = sorted(
        {w for p in [*PROMPTS, *(i["prompt"] for i in ITEMS)] for w in p.split()}
    )
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2} | {w: i + 3 for i, w in enumerate(words)}
    assert len(vocab) <= 128
    t = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=t,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="</s>",
    )


@pytest.fixture
def tiny_edit(tiny_llama: Any) -> Edit:
    rng = np.random.default_rng(0)
    deltas = []
    for module in MODULES:
        w0 = tiny_llama.get_submodule(module).weight.detach().numpy()
        u = rng.standard_normal(w0.shape[0]).astype(np.float32)
        mask = top_k_mask(rng.standard_normal(w0.shape[1]), 8)
        deltas.append(masked_rank1(module, w0, u, mask, 1.0))
    return Edit(tuple(deltas), "tiny-llama")


@pytest.mark.parametrize("body", BODIES, ids=_name)
def test_cpu_smoke(body: Body, tiny_llama: Any, tok: Any, tiny_edit: Edit) -> None:
    body(tiny_llama, tok, tiny_edit)


def test_cpu_smoke_rounding_bf16(tiny_llama: Any, tok: Any, tiny_edit: Edit) -> None:
    # a float32 model rounds nothing, so the max_rounding_error guard needs bf16
    body_rounding(tiny_llama.to(torch.bfloat16), tok, tiny_edit)


def test_skip_reason_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("ASTRO_GPU_MODEL", raising=False)
    assert _skip_reason() == "ASTRO_GPU_MODEL is not set"
    monkeypatch.setenv("ASTRO_GPU_MODEL", "m")
    monkeypatch.delenv("ASTRO_GPU_ADAPTER", raising=False)
    assert _skip_reason() == "ASTRO_GPU_ADAPTER is not set"
    monkeypatch.setenv("ASTRO_GPU_ADAPTER", "a")
    assert _skip_reason() is None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _skip_reason() == "no CUDA device"
