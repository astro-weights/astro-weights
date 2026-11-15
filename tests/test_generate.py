from pathlib import Path
from typing import Any

import numpy as np
import pytest

from astro_weights.edits import Edit, rank1
from astro_weights.eval import (
    PairedResult,
    Record,
    compare,
    grade_exact,
    pair,
    prompt_hash,
    read_jsonl,
    write_jsonl,
)

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from astro_weights.runners import generate as generate_mod
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


@pytest.fixture
def bos_tok(tok: Any) -> Any:
    tok.backend_tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", 1)]
    )
    return tok


def _captured_ids(model: Any, tok: Any, monkeypatch: pytest.MonkeyPatch, chat: bool):  # type: ignore[no-untyped-def]
    seen: list[list[int]] = []
    real = model.generate

    def spy(*args: Any, **kw: Any) -> Any:
        seen.append(kw["input_ids"][0].tolist())
        return real(*args, **kw)

    monkeypatch.setattr(model, "generate", spy)
    generate_records(model, tok, ITEMS[:1], "base", grade_exact, 2, chat=chat)
    return seen[0]


def test_chat_without_template_keeps_bos(
    model: Any, bos_tok: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert bos_tok.chat_template is None
    chat = _captured_ids(model, bos_tok, monkeypatch, True)
    raw = _captured_ids(model, bos_tok, monkeypatch, False)
    assert chat[0] == 1
    assert chat == raw


def test_chat_template_no_extra_bos(
    model: Any, bos_tok: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    bos_tok.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    ids = _captured_ids(model, bos_tok, monkeypatch, True)
    assert ids[0] != 1


def test_eval_mode_set_and_restored(model: Any, tok: Any) -> None:
    model.train()
    modes: list[bool] = []
    real = model.generate

    def spy(*args: Any, **kw: Any) -> Any:
        modes.append(model.training)
        return real(*args, **kw)

    model.generate = spy
    _run(model, tok)
    assert modes
    assert not any(modes)
    assert model.training
    assert run_header(model, tok, 6, 0, False)["training"] is False


def test_duplicate_item_id(model: Any, tok: Any) -> None:
    dup = [ITEMS[0], ITEMS[0]]
    with pytest.raises(ValueError, match="duplicate item_id 'i0'"):
        generate_records(model, tok, dup, "b", grade_exact, 2, chat=False)


# --- out / resume / batch_size ---

MIXED = [
    {
        "item_id": f"m{k}",
        "prompt": " ".join(f"w{j + k}" for j in range(1, k + 2)),
        "gold": "w5",
    }
    for k in range(6)
]  # prompt lengths 1..6: batches need real padding


def _old_generate(model: Any, tok: Any, items: Any, arm: str, cap: int) -> list[Record]:
    """The loop as it was before out/resume/batch_size existed."""
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
    recs = []
    for item in items:
        enc = tok(item["prompt"], return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=cap,
                do_sample=False,
                pad_token_id=tok.pad_token_id or 0,
            )
        new = out[0, enc["input_ids"].shape[1] :]
        n = int(new.shape[0])
        text = tok.decode(new, skip_special_tokens=True)
        recs.append(
            Record(
                item["item_id"],
                0,
                arm,
                prompt_hash(item["prompt"]),
                text,
                float(grade_exact(item["gold"], text)),
                n,
                n > 0 and n == cap and int(new[-1]) not in eos_ids,
            )
        )
    return recs


def test_default_path_unchanged(model: Any, tok: Any, tmp_path: Path) -> None:
    for eos in (None, 7, [7, 9]):
        model.generation_config.eos_token_id = eos
        new = generate_records(model, tok, MIXED, "base", grade_exact, 6, chat=False)
        assert new == _old_generate(model, tok, MIXED, "base", 6)
    out = tmp_path / "r.jsonl"
    assert _run(model, tok, out=out) == _run(model, tok)


def test_out_matches_write_jsonl(model: Any, tok: Any, tmp_path: Path) -> None:
    recs = _run(model, tok, out=tmp_path / "a.jsonl")
    write_jsonl(tmp_path / "b.jsonl", recs)
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert read_jsonl(tmp_path / "a.jsonl") == recs
    assert _run(model, tok, out=str(tmp_path / "c.jsonl")) == recs


def test_out_exists_needs_resume(model: Any, tok: Any, tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    out.write_text("")
    _run(model, tok, out=out)  # an empty file is fine
    with pytest.raises(FileExistsError, match="resume=True"):
        _run(model, tok, out=out)
    assert len(read_jsonl(out)) == len(ITEMS)


def test_resume_needs_out(model: Any, tok: Any) -> None:
    with pytest.raises(ValueError, match="resume=True needs out"):
        _run(model, tok, resume=True)


def test_resume_fresh_and_complete(model: Any, tok: Any, tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    first = _run(model, tok, out=out, resume=True)  # no file yet
    again = _run(model, tok, out=out, resume=True)  # all done: nothing generated
    assert first == again
    assert len(read_jsonl(out)) == len(ITEMS)


def _crash_on(n: int, exc: type[BaseException]) -> Any:
    calls = 0

    def scorer(gold: str, output: str) -> bool:
        nonlocal calls
        calls += 1
        if calls == n:
            raise exc("boom")
        return grade_exact(gold, output)

    return scorer


@pytest.mark.parametrize("exc", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_crash_then_resume(
    model: Any, tok: Any, tmp_path: Path, exc: type[BaseException], batch_size: int
) -> None:
    whole = _run(model, tok, out=tmp_path / "whole.jsonl", batch_size=batch_size)
    out = tmp_path / "r.jsonl"
    with pytest.raises(exc):
        generate_records(
            model, tok, ITEMS, "base", _crash_on(3, exc), 6, chat=False,
            out=out, batch_size=batch_size,
        )  # fmt: skip
    assert read_jsonl(out) == whole[:2]
    model.train()
    resumed = _run(model, tok, out=out, resume=True, batch_size=batch_size)
    assert model.training  # restored even after the crash
    assert resumed == whole
    assert out.read_bytes() == (tmp_path / "whole.jsonl").read_bytes()


def test_resume_drops_truncated_line(model: Any, tok: Any, tmp_path: Path) -> None:
    whole = _run(model, tok, out=tmp_path / "whole.jsonl")
    out = tmp_path / "r.jsonl"
    text = (tmp_path / "whole.jsonl").read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    out.write_text("".join(lines[:2]) + lines[2][:30], encoding="utf-8")
    assert _run(model, tok, out=out, resume=True) == whole
    assert out.read_text(encoding="utf-8") == text


def test_resume_cut_inside_a_character(model: Any, tok: Any, tmp_path: Path) -> None:
    whole = _run(model, tok, out=tmp_path / "whole.jsonl")
    out = tmp_path / "r.jsonl"
    raw = (tmp_path / "whole.jsonl").read_bytes()
    first = raw.index(b"\n") + 1
    out.write_bytes(raw[:first] + b'{"item_id": "\xc3')
    assert _run(model, tok, out=out, resume=True) == whole


def test_resume_keeps_whole_unterminated_line(
    model: Any, tok: Any, tmp_path: Path
) -> None:
    whole = _run(model, tok, out=tmp_path / "whole.jsonl")
    out = tmp_path / "r.jsonl"
    text = (tmp_path / "whole.jsonl").read_text(encoding="utf-8")
    out.write_text("".join(text.splitlines(keepends=True)[:2]).rstrip("\n"))
    assert _run(model, tok, out=out, resume=True) == whole
    assert out.read_text(encoding="utf-8") == text


def test_resume_only_partial_line(model: Any, tok: Any, tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    out.write_text('{"item_id": "i0", "se')
    assert _run(model, tok, out=out, resume=True) == _run(model, tok)


def _seed_file(model: Any, tok: Any, path: Path, **kw: Any) -> list[Record]:
    return _run(model, tok, out=path, **kw)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"arm": "other"}, "another run"),
        ({"seed": 1}, "another run"),
    ],
)
def test_resume_other_run(
    model: Any, tok: Any, tmp_path: Path, kw: dict[str, Any], match: str
) -> None:
    out = tmp_path / "r.jsonl"
    _seed_file(model, tok, out)
    with pytest.raises(ValueError, match=match):
        _run(model, tok, out=out, resume=True, **kw)


def test_resume_unknown_item(model: Any, tok: Any, tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    _seed_file(model, tok, out)
    with pytest.raises(ValueError, match="'i0' is not in items"):
        generate_records(
            model, tok, ITEMS[1:], "base", grade_exact, 6, chat=False,
            out=out, resume=True,
        )  # fmt: skip


def test_resume_prompt_changed(model: Any, tok: Any, tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    _seed_file(model, tok, out)
    changed = [{**ITEMS[0], "prompt": "w9 w9"}, *ITEMS[1:]]
    with pytest.raises(ValueError, match="prompt of item 'i0' differs"):
        generate_records(
            model, tok, changed, "base", grade_exact, 6, chat=False,
            out=out, resume=True,
        )  # fmt: skip
    tok.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %} w9"
    with pytest.raises(ValueError, match="differs"):
        generate_records(
            model, tok, ITEMS, "base", grade_exact, 6, chat=True, out=out, resume=True
        )


def test_resume_duplicate_in_file(model: Any, tok: Any, tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    recs = _seed_file(model, tok, out)
    write_jsonl(out, [*recs, recs[0]])
    with pytest.raises(ValueError, match="duplicate item_id 'i0'"):
        _run(model, tok, out=out, resume=True)


def test_resume_mismatch_leaves_file_alone(
    model: Any, tok: Any, tmp_path: Path
) -> None:
    out = tmp_path / "r.jsonl"
    _seed_file(model, tok, out)
    before = out.read_bytes()
    with pytest.raises(ValueError, match="another run"):
        _run(model, tok, "other", out=out, resume=True)
    assert out.read_bytes() == before


def _spy_batches(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    seen: list[list[str]] = []
    real = generate_mod._run_batch

    def spy(model: Any, tok: Any, texts: Any, *a: Any) -> Any:
        seen.append(list(texts))
        return real(model, tok, texts, *a)

    monkeypatch.setattr(generate_mod, "_run_batch", spy)
    return seen


@pytest.mark.parametrize("done", [1, 2, 3, 4, 5])
def test_resume_batches_at_fixed_positions(
    model: Any, tok: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, done: int
) -> None:
    seen = _spy_batches(monkeypatch)
    kw: dict[str, Any] = {"chat": False, "batch_size": 4}
    whole = generate_records(model, tok, MIXED, "base", grade_exact, 8, **kw)
    plain = list(seen)
    assert [len(b) for b in plain] == [4, 2]
    seen.clear()
    out = tmp_path / "r.jsonl"
    write_jsonl(out, whole[:done])  # as if a run died after `done` records
    resumed = generate_records(
        model, tok, MIXED, "base", grade_exact, 8, out=out, resume=True, **kw
    )
    assert resumed == whole
    # every batch is the uninterrupted one minus the items already done
    expect = [b[max(0, done - 4 * k) :] for k, b in enumerate(plain)]
    assert seen == [b for b in expect if b]


def test_crash_leaves_no_half_batch(model: Any, tok: Any, tmp_path: Path) -> None:
    whole = generate_records(
        model, tok, MIXED, "base", grade_exact, 8, chat=False, batch_size=4
    )
    out = tmp_path / "r.jsonl"
    with pytest.raises(RuntimeError):
        generate_records(
            model, tok, MIXED, "base", _crash_on(2, RuntimeError), 8, chat=False,
            batch_size=4, out=out,
        )  # fmt: skip
    assert read_jsonl(out) == []  # the batch died while being scored: nothing written
    resumed = generate_records(
        model, tok, MIXED, "base", grade_exact, 8, chat=False, batch_size=4,
        out=out, resume=True,
    )  # fmt: skip
    assert resumed == whole
    assert read_jsonl(out) == whole


@pytest.mark.parametrize("tail", [b'{"item_id": "m0", "se', b'{"item_id": "\xc3'])
def test_resume_mismatch_with_partial_tail_leaves_file_alone(
    model: Any, tok: Any, tmp_path: Path, tail: bytes
) -> None:
    out = tmp_path / "r.jsonl"
    _seed_file(model, tok, out)
    raw = out.read_bytes()
    first = raw.index(b"\n") + 1
    out.write_bytes(raw[:first] + tail)
    before = out.read_bytes()
    with pytest.raises(ValueError, match="another run"):
        _run(model, tok, "other", out=out, resume=True)
    assert out.read_bytes() == before


def test_resume_mismatch_with_whole_unterminated_line_leaves_file_alone(
    model: Any, tok: Any, tmp_path: Path
) -> None:
    out = tmp_path / "r.jsonl"
    _seed_file(model, tok, out)
    out.write_bytes(out.read_bytes().rstrip(b"\n"))
    before = out.read_bytes()
    with pytest.raises(ValueError, match="another run"):
        _run(model, tok, "other", out=out, resume=True)
    assert out.read_bytes() == before


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "2"])
def test_batch_size_invalid(model: Any, tok: Any, bad: Any) -> None:
    with pytest.raises(ValueError, match="batch_size must be an int >= 1"):
        _run(model, tok, batch_size=bad)


def test_batch_needs_pad_token(model: Any, tok: Any) -> None:
    tok.pad_token = None
    with pytest.raises(
        ValueError, match=r"tokenizer\.pad_token = tokenizer\.eos_token"
    ):
        _run(model, tok, batch_size=2)
    assert len(_run(model, tok)) == len(ITEMS)  # batch_size 1 needs no pad token


@pytest.mark.parametrize("batch_size", [2, 4, 6, 10])
@pytest.mark.parametrize("eos", [None, 7, [7, 9, 11]])
def test_batch_matches_single(model: Any, tok: Any, batch_size: int, eos: Any) -> None:
    model.generation_config.eos_token_id = eos
    one = generate_records(model, tok, MIXED, "base", grade_exact, 8, chat=False)
    many = generate_records(
        model, tok, MIXED, "base", grade_exact, 8, chat=False, batch_size=batch_size
    )
    assert many == one


def test_batch_counts_tokens_to_first_eos(model: Any, tok: Any) -> None:
    # make each row's 2nd greedy token the stop token for some rows, none for others
    probe = generate_records(model, tok, MIXED, "base", grade_exact, 4, chat=False)
    stop = int(tok(probe[0].output.split()[1])["input_ids"][-1])
    model.generation_config.eos_token_id = stop
    one = generate_records(model, tok, MIXED, "base", grade_exact, 4, chat=False)
    many = generate_records(
        model, tok, MIXED, "base", grade_exact, 4, chat=False, batch_size=3
    )
    assert [r.n_output_tokens for r in many] == [r.n_output_tokens for r in one]
    assert [r.hit_cap for r in many] == [r.hit_cap for r in one]
    assert len({r.n_output_tokens for r in many}) > 1  # rows really differ


def test_batch_pads_left_and_restores_side(
    model: Any, tok: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    tok.padding_side = "right"
    seen: list[Any] = []
    real = model.generate

    def spy(*args: Any, **kw: Any) -> Any:
        seen.append((kw["input_ids"].clone(), kw["attention_mask"].clone()))
        return real(*args, **kw)

    monkeypatch.setattr(model, "generate", spy)
    generate_records(
        model, tok, MIXED[:3], "base", grade_exact, 2, chat=False, batch_size=3
    )
    ids, mask = seen[0]
    assert ids.shape == (3, 3)
    assert mask[0].tolist() == [0, 0, 1]  # shortest prompt is padded on the left
    assert mask[2].tolist() == [1, 1, 1]
    assert tok.padding_side == "right"


def test_batch_with_chat_template(model: Any, tok: Any) -> None:
    tok.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %} w9"
    one = generate_records(model, tok, MIXED, "base", grade_exact, 4, chat=True)
    many = generate_records(
        model, tok, MIXED, "base", grade_exact, 4, chat=True, batch_size=4
    )
    assert many == one


def test_run_header_batch_size(model: Any, tok: Any) -> None:
    assert run_header(model, tok, 6, 0, False)["batch_size"] == 1
    assert run_header(model, tok, 6, 0, False, batch_size=8)["batch_size"] == 8
