from pathlib import Path
from typing import Any

import pytest

from astro_weights.eval import check_headers, grade_exact, read_jsonl

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from astro_weights.runners.generate import (
    _PresencePenalty,
    generate_records,
    run_header,
)

ITEMS = [
    {"item_id": f"i{k}", "prompt": f"w1 w2 w{k + 3}", "gold": "w5"} for k in range(5)
]
SAMPLING = {"temperature": 1.5, "top_p": 0.95, "top_k": 20.0, "presence_penalty": 1.0}


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


def _run(model: Any, tok: Any, **kw: Any) -> list[Any]:
    kw.setdefault("sampling", SAMPLING)
    return generate_records(model, tok, ITEMS, "a", grade_exact, 8, chat=False, **kw)


def test_presence_penalty_on_hand_made_tensor() -> None:
    ids = torch.tensor([[0, 0, 2, 3], [1, 1, 1, 5]])
    scores = torch.zeros(2, 8)
    # row 0 has two left-pad positions (ids 0 there are padding), row 1 none
    out = _PresencePenalty(1.5, torch.tensor([2, 0]))(ids, scores)
    assert out[0].tolist() == [0, 0, -1.5, -1.5, 0, 0, 0, 0]
    assert out[1].tolist() == [0, -1.5, 0, 0, 0, -1.5, 0, 0]
    assert scores.abs().sum() == 0  # input untouched


def test_presence_penalty_counts_once_and_keeps_dtype() -> None:
    ids = torch.tensor([[4, 4, 4]])
    out = _PresencePenalty(0.5, torch.tensor([0]))(
        ids, torch.ones(1, 6, dtype=torch.bfloat16)
    )
    assert out.dtype == torch.bfloat16
    assert out[0].tolist() == [1, 1, 1, 1, 0.5, 1]


def test_presence_penalty_window_is_last_64() -> None:
    ids = torch.cat([torch.full((1, 1), 7), torch.full((1, 64), 3)], dim=1)
    out = _PresencePenalty(1.0, torch.tensor([0]))(ids, torch.zeros(1, 10))
    assert out[0, 7] == 0  # fell out of the window
    assert out[0, 3] == -1


@pytest.mark.parametrize(
    "bad", [{"temp": 1.0}, {"temperature": 0}, {"temperature": -1.0}]
)
def test_bad_sampling_raises_before_generation(model: Any, tok: Any, bad: Any) -> None:
    called = []
    model.generate = lambda *a, **k: called.append(1)
    with pytest.raises(ValueError, match=r"sampling keys|temperature"):
        _run(model, tok, sampling=bad)
    with pytest.raises(ValueError, match=r"sampling keys|temperature"):
        run_header(model, tok, 8, 0, False, sampling=bad)
    assert not called


@pytest.fixture
def model(tiny_llama: Any) -> Any:
    return tiny_llama


def test_greedy_unchanged_and_sampling_changes_output(model: Any, tok: Any) -> None:
    greedy = _run(model, tok, sampling=None)
    assert greedy == _run(model, tok, sampling=None)
    sampled = _run(model, tok)
    assert [r.output for r in sampled] != [r.output for r in greedy]


def test_empty_sampling_still_samples(model: Any, tok: Any) -> None:
    assert len(_run(model, tok, sampling={})) == len(ITEMS)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_sampled_rerun_same(model: Any, tok: Any, batch_size: int) -> None:
    assert _run(model, tok, seed=4, batch_size=batch_size) == _run(
        model, tok, seed=4, batch_size=batch_size
    )
    other = _run(model, tok, seed=5, batch_size=batch_size)
    assert [r.output for r in other] != [
        r.output for r in _run(model, tok, seed=4, batch_size=batch_size)
    ]


@pytest.mark.parametrize("batch_size", [1, 2])
def test_sampled_resume_equals_uninterrupted(
    model: Any, tok: Any, tmp_path: Path, batch_size: int
) -> None:
    whole = _run(model, tok, out=tmp_path / "w.jsonl", batch_size=batch_size)

    def crash(gold: str, output: str) -> bool:
        if len(seen) == 3:
            raise RuntimeError("boom")
        seen.append(output)
        return False

    seen: list[str] = []
    out = tmp_path / "r.jsonl"
    with pytest.raises(RuntimeError):
        generate_records(
            model, tok, ITEMS, "a", crash, 8, chat=False, out=out,
            batch_size=batch_size, sampling=SAMPLING,
        )  # fmt: skip
    assert 0 < len(read_jsonl(out)) < len(ITEMS)
    resumed = _run(model, tok, out=out, resume=True, batch_size=batch_size)
    assert [r.output for r in resumed] == [r.output for r in whole]
    assert [r.output for r in read_jsonl(out)] == [r.output for r in whole]


def test_header_sampling_key(model: Any, tok: Any) -> None:
    greedy = run_header(model, tok, 8, 0, False)
    assert "sampling" not in greedy
    sampled = run_header(model, tok, 8, 0, False, sampling=SAMPLING)
    assert sampled["sampling"] == dict(sorted(SAMPLING.items()))
    assert list(sampled["sampling"]) == sorted(SAMPLING)
    with pytest.raises(ValueError, match="sampling"):
        check_headers(greedy, sampled)
    check_headers(sampled, run_header(model, tok, 8, 0, False, sampling=SAMPLING))
    other = run_header(model, tok, 8, 0, False, sampling={**SAMPLING, "top_k": 5})
    with pytest.raises(ValueError, match="sampling"):
        check_headers(sampled, other)
