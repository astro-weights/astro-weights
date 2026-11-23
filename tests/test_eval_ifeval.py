import builtins
from typing import Any

import pytest

from astro_weights.eval.ifeval import IFEVAL_IDS, check_ifeval, grade_ifeval

EN = "The quick brown fox jumps over the lazy dog near the river bank today."
FR = "Bonjour, comment allez-vous aujourd'hui? Je voudrais manger du pain."

# (instruction id, kwargs, response that follows it, response that does not)
CASES: list[tuple[str, dict[str, Any], str, str]] = [
    ("keywords:existence", {"keywords": ["Cat", "dog"]}, "a CAT and a dog", "a cat"),
    (
        "keywords:frequency",
        {"keyword": "tree", "frequency": 2, "relation": "at least"},
        "tree Tree",
        "tree",
    ),
    (
        "keywords:frequency",
        {"keyword": "tree", "frequency": 2, "relation": "less than"},
        "tree",
        "tree tree",
    ),
    ("keywords:forbidden_words", {"forbidden_words": ["bad"]}, "a badge", "so BAD."),
    (
        "keywords:letter_frequency",
        {"letter": "a", "let_frequency": 3, "let_relation": "at least"},
        "banana",
        "bonbon",
    ),
    (
        "length_constraints:number_sentences",
        {"num_sentences": 3, "relation": "less than"},
        "One. Two.",
        "One. Two. Three. Dr. Who? Yes!",
    ),
    (
        "length_constraints:number_paragraphs",
        {"num_paragraphs": 2},
        "a\n***\nb",
        "a\n***\n\n***\nb",
    ),
    (
        "length_constraints:number_words",
        {"num_words": 4, "relation": "at least"},
        "one two three four",
        "one two",
    ),
    (
        "length_constraints:nth_paragraph_first_word",
        {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "beta"},
        "alpha one\n\n'Beta, two",
        "alpha one\n\ngamma two",
    ),
    (
        "detectable_content:number_placeholders",
        {"num_placeholders": 2},
        "[a] and [b]",
        "[a] only",
    ),
    (
        "detectable_content:postscript",
        {"postscript_marker": "P.S."},
        "text\nP.S. more",
        "text only",
    ),
    (
        "detectable_format:number_bullet_lists",
        {"num_bullets": 2},
        "* a\n- b",
        "* a\n* b\n* c",
    ),
    ("detectable_format:constrained_response", {}, "My answer is no.", "maybe"),
    (
        "detectable_format:number_highlighted_sections",
        {"num_highlights": 2},
        "*a* and **b**",
        "*a* only",
    ),
    (
        "detectable_format:multiple_sections",
        {"section_spliter": "Section", "num_sections": 2},
        "Section 1\nx\nSection 2\ny",
        "Section 1\nx",
    ),
    ("detectable_format:json_format", {}, '```json\n{"a": 1}\n```', "{oops"),
    ("detectable_format:title", {}, "<<A title>>\nbody", "<<>>\nbody"),
    ("combination:two_responses", {}, "one******two", "one******one"),
    (
        "combination:repeat_prompt",
        {"prompt_to_repeat": "Say hi"},
        "say hi. Hi!",
        "Hi! Say hi",
    ),
    ("startend:end_checker", {"end_phrase": "Bye now"}, '"Hello. bye now"', "Bye now."),
    ("startend:quotation", {}, '"quoted"', "plain"),
    (
        "change_case:capital_word_frequency",
        {"capital_frequency": 2, "capital_relation": "at least"},
        "NASA and UK, ok",
        "NASA only",
    ),
    ("change_case:english_capital", {}, EN.upper(), EN),
    ("change_case:english_lowercase", {}, EN.lower(), EN),
    ("punctuation:no_comma", {}, "no commas here", "one, two"),
    ("language:response_language", {"language": "fr"}, FR, EN),
]
NEEDS_LANGDETECT = {
    "language:response_language",
    "change_case:english_capital",
    "change_case:english_lowercase",
}


def _item(instruction_id: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    return {"instruction_id_list": [instruction_id], "kwargs": [kwargs]}


@pytest.mark.parametrize(
    ("instruction_id", "kwargs", "good", "bad"),
    CASES,
    ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)],
)
def test_instruction_pass_and_fail(
    instruction_id: str, kwargs: dict[str, Any], good: str, bad: str
) -> None:
    if instruction_id in NEEDS_LANGDETECT:
        pytest.importorskip("langdetect")
    item = _item(instruction_id, kwargs)
    assert check_ifeval(item, good) == [True]
    assert check_ifeval(item, bad) == [False]


def test_cases_cover_every_instruction() -> None:
    assert {c[0] for c in CASES} == set(IFEVAL_IDS)
    assert len(IFEVAL_IDS) == 25


def test_prompt_level_is_all_or_nothing() -> None:
    item = {
        "instruction_id_list": ["punctuation:no_comma", "startend:quotation"],
        "kwargs": [{}, {}],
    }
    assert check_ifeval(item, '"fine"') == [True, True]
    assert check_ifeval(item, "fine") == [True, False]
    assert check_ifeval(item, '"a, b"') == [False, True]
    assert grade_ifeval(item, '"fine"') == 1.0
    assert grade_ifeval(item, "fine") == 0.0
    # reasoning before the last </think> is not graded
    assert grade_ifeval(item, 'x, y</think>"fine"') == 1.0


def test_empty_answer_follows_nothing() -> None:
    assert check_ifeval(_item("punctuation:no_comma", {}), "  \n") == [False]


def test_bad_items() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        check_ifeval({"instruction_id_list": ["startend:quotation"], "kwargs": []}, "a")
    with pytest.raises(ValueError, match="unknown IFEval instruction"):
        check_ifeval(_item("nope:nope", {}), "a")
    with pytest.raises(ValueError, match="unknown relation"):
        check_ifeval(
            _item("length_constraints:number_words", {"num_words": 1, "relation": "x"}),
            "a",
        )


def test_undetectable_language_counts_as_followed() -> None:
    pytest.importorskip("langdetect")
    assert check_ifeval(
        _item("language:response_language", {"language": "fr"}), "12 34"
    ) == [True]


def test_language_checks_need_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    real = builtins.__import__

    def fake(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langdetect":
            raise ImportError("blocked", name="langdetect")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(
        ModuleNotFoundError, match=r"pip install 'astro-weights\[eval\]'"
    ):
        check_ifeval(_item("change_case:english_capital", {}), "HELLO")
    # checks that need no extra still run
    assert check_ifeval(_item("punctuation:no_comma", {}), "ok") == [True]


@pytest.mark.parametrize(
    ("instruction_id", "kwargs", "answer", "ok"),
    [
        # empty first/last paragraph around *** is dropped; an empty middle one fails
        (
            "length_constraints:number_paragraphs",
            {"num_paragraphs": 1},
            "***\na\n***",
            True,
        ),
        (
            "length_constraints:number_paragraphs",
            {"num_paragraphs": 2},
            "a\n***\n***\nb",
            False,
        ),
        # nth paragraph out of range, and a blank nth paragraph
        (
            "length_constraints:nth_paragraph_first_word",
            {"num_paragraphs": 1, "nth_paragraph": 3, "first_word": "a"},
            "a",
            False,
        ),
        (
            "length_constraints:nth_paragraph_first_word",
            {"num_paragraphs": 2, "nth_paragraph": 1, "first_word": "a"},
            "\n\n\n\nb",
            False,
        ),
        (
            "detectable_content:postscript",
            {"postscript_marker": "P.P.S"},
            "x\nP. P. S later",
            True,
        ),
        (
            "detectable_content:postscript",
            {"postscript_marker": "Note:"},
            "x\nnote: y",
            True,
        ),
        ("combination:two_responses", {}, "a******  ******b", False),
    ],
)
def test_edge_branches(
    instruction_id: str, kwargs: dict[str, Any], answer: str, ok: bool
) -> None:
    assert check_ifeval(_item(instruction_id, kwargs), answer) == [ok]
