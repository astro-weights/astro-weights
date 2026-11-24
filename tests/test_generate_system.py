"""`system` on the HF runner: rendered through the chat template, same prompt hash."""

from typing import Any

import pytest

from astro_weights.eval import Record, grade_exact, pair

pytest.importorskip("torch")
pytest.importorskip("peft")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from astro_weights.runners.generate import (
    _prompt_text,
    generate_records,
    run_header,
)

ITEMS = [{"item_id": f"i{k}", "prompt": f"w1 w{k + 3}", "gold": "w5"} for k in range(3)]
SYS = "w9 w9"
TEMPLATE = "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}w2"


@pytest.fixture
def tok() -> Any:
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2} | {f"w{i}": i + 3 for i in range(123)}
    vocab |= {"user": 126, "system": 127}  # tiny_llama's vocab has 128 ids
    t = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=t, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
        pad_token="</s>", chat_template=TEMPLATE,
    )  # fmt: skip


def _spy(tok: Any, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    seen: list[Any] = []
    real = tok.apply_chat_template

    def spy(messages: Any, **kw: Any) -> Any:
        seen.append(messages)
        return real(messages, **kw)

    monkeypatch.setattr(tok, "apply_chat_template", spy)
    return seen


@pytest.mark.parametrize("batch_size", [1, 2])
def test_system_is_rendered_and_hash_is_unchanged(
    tiny_llama: Any, tok: Any, monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    seen = _spy(tok, monkeypatch)

    def run(arm: str, **kw: Any) -> list[Record]:
        return generate_records(
            tiny_llama, tok, ITEMS, arm, grade_exact, 3, batch_size=batch_size, **kw
        )

    plain = run("base")
    assert all(m[0]["role"] == "user" for m in seen)
    seen.clear()
    with_sys = run("prompt", system=SYS)
    assert [m[0] for m in seen if len(m) == 2] == [
        {"role": "system", "content": SYS}
    ] * 3
    assert [r.prompt_hash for r in plain] == [r.prompt_hash for r in with_sys]
    assert len(pair(plain, with_sys)) == len(ITEMS)


def test_system_changes_the_prompt_the_model_sees(tiny_llama: Any, tok: Any) -> None:
    bare, _ = _prompt_text(tok, "w1", True)
    full, templated = _prompt_text(tok, "w1", True, SYS)
    assert templated
    assert full == "system w9 w9 " + bare


@pytest.mark.parametrize("chat", [False, True])
def test_system_without_a_chat_template_is_refused(
    tiny_llama: Any, tok: Any, chat: bool
) -> None:
    if chat:
        tok.chat_template = None  # chat=True but nothing to render with
    # else chat=False: a template exists but is not used
    with pytest.raises(ValueError, match="chat template"):
        generate_records(
            tiny_llama,
            tok,
            ITEMS,
            "p",
            grade_exact,
            2,
            chat=chat,
            system=SYS,
        )


def test_empty_system_is_refused(tiny_llama: Any, tok: Any) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        generate_records(tiny_llama, tok, ITEMS, "p", grade_exact, 2, system="")


def test_header_has_system_only_when_set(tiny_llama: Any, tok: Any) -> None:
    bare = run_header(tiny_llama, tok, 3, 0, True)
    assert "system" not in bare
    assert "system_sha256" not in bare
    h = run_header(tiny_llama, tok, 3, 0, True, system=SYS)
    assert h["system"] == SYS
    assert {k for k in h if k not in bare} == {"system", "system_sha256"}
