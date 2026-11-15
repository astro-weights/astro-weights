from typing import Any

import numpy as np
import pytest

from astro_weights.edits import Edit, rank1
from astro_weights.eval import PairedResult, Record, compare, grade_exact, pair

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from astro_weights.runners.generate import generate_records, run_header
from astro_weights.runners.hf import apply

MODULE = "model.layers.1.mlp.down_proj"
ITEMS = [
    {"item_id": f"i{k}", "prompt": " ".join(["w1", "w2", f"w{k + 3}"]), "gold": "w5"}
    for k in range(4)
]


@pytest.fixture
def tok() -> Any:
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2} | {f"w{i}": i + 3 for i in range(125)}
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
def model(tiny_llama: Any) -> Any:
    return tiny_llama


def _run(
    model: Any, tok: Any, arm: str = "base", cap: int = 6, **kw: Any
) -> list[Record]:
    return generate_records(model, tok, ITEMS, arm, grade_exact, cap, chat=False, **kw)


def test_record_fields(model: Any, tok: Any) -> None:
    recs = _run(model, tok)
    assert [r.item_id for r in recs] == [i["item_id"] for i in ITEMS]
    for r in recs:
        assert r.arm == "base"
        assert r.seed == 0
        assert len(r.prompt_hash) == 64
        assert 0 < r.n_output_tokens <= 6
        assert r.score in (0.0, 1.0)


def test_hit_cap(model: Any, tok: Any) -> None:
    model.generation_config.eos_token_id = None
    recs = _run(model, tok, cap=3)
    assert all(r.n_output_tokens == 3 and r.hit_cap for r in recs)


def test_stopped_at_eos_not_cap(model: Any, tok: Any) -> None:
    first = _run(model, tok, cap=1)
    # make the first greedy token the eos token for each item: stop right there
    for item, r in zip(ITEMS, first, strict=True):
        ids = tok(item["prompt"], return_tensors="pt")
        with torch.no_grad():
            nxt = int(model(**ids).logits[0, -1].argmax())
        model.generation_config.eos_token_id = nxt
        rec = generate_records(model, tok, [item], "base", grade_exact, 1, chat=False)[
            0
        ]
        assert rec.n_output_tokens == 1
        assert not rec.hit_cap
        assert r.item_id == rec.item_id


def test_same_seed_same_records(model: Any, tok: Any) -> None:
    assert _run(model, tok, seed=3) == _run(model, tok, seed=3)


def test_missing_key(model: Any, tok: Any) -> None:
    with pytest.raises(ValueError, match="gold"):
        generate_records(
            model, tok, [{"item_id": "a", "prompt": "w1"}], "b", grade_exact, 2
        )


def test_chat_template_used(model: Any, tok: Any) -> None:
    tok.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %} w9"
    chat = generate_records(model, tok, ITEMS[:1], "base", grade_exact, 2, chat=True)
    raw = generate_records(model, tok, ITEMS[:1], "base", grade_exact, 2, chat=False)
    assert chat[0].prompt_hash != raw[0].prompt_hash


def test_arms_pair(model: Any, tok: Any) -> None:
    base = _run(model, tok, "base")
    rng = np.random.default_rng(0)
    w = model.get_submodule(MODULE).weight
    delta = rank1(
        MODULE, 5 * rng.standard_normal(w.shape[0]), rng.standard_normal(w.shape[1])
    )
    apply(model, Edit((delta,), "tiny"))
    edit = _run(model, tok, "edit")
    assert len(pair(base, edit)) == len(ITEMS)
    assert isinstance(compare(base, edit, n_boot=50), PairedResult)


def test_run_header(model: Any, tok: Any) -> None:
    h = run_header(model, tok, 6, 0, False)
    assert h["model"] == "LlamaForCausalLM"
    assert h["chat_template_sha256"] is None
    assert {
        "dtype",
        "device",
        "torch",
        "transformers",
        "peft",
        "seed",
        "chat",
    } <= h.keys()
    tok.chat_template = "x"
    assert len(run_header(model, tok, 6, 0, True)["chat_template_sha256"]) == 64
