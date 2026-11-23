"""IFEval strict grader: a port of Google's instruction checkers (Apache-2.0).

Ported from google-research/google-research, instruction_following_eval
(instructions.py, instructions_util.py; see NOTICE). It covers the 25 instruction
types of the IFEval dataset and the strict rule: an instruction counts only if
the response is non-empty and the check passes. Standard library at import time.

Sentence and capital-word counts use nltk (punkt sentence model and
`word_tokenize`) as the original does. The NLTK data is not downloaded here; if it
is missing, grading raises a LookupError that names the command to run once:
`python -m nltk.downloader punkt_tab`. The language checks use `langdetect`.
All of this needs `pip install 'astro-weights[eval]'`; the other checks do not.

Deliberate differences from the original:

- `keywords:letter_frequency`: the original swaps a `letter` that is not an ASCII
  letter for a random one (IFEval keys 1122 and 1129 use "#" and "!"). Here the
  given character is counted, so the score is deterministic.
- Language checks set `langdetect.DetectorFactory.seed = 0` (langdetect is random
  otherwise). That is a process-wide setting and affects other langdetect users.
- Kwargs come straight from the dataset; the original's evaluation_lib drops
  falsy kwarg values before building an instruction, which no dataset item needs.
"""

import functools
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

from astro_weights.eval.grade import answer_part

__all__ = ["IFEVAL_IDS", "check_ifeval", "grade_ifeval"]

_LESS = "less than"


def _compare(count: int, relation: str, limit: int) -> bool:
    if relation == _LESS:
        return count < limit
    if relation == "at least":
        return count >= limit
    raise ValueError(f"unknown relation {relation!r}")


def _detect(text: str) -> str | None:
    """Language code of text, or None when langdetect cannot tell."""
    try:
        import langdetect
    except ImportError as e:
        raise ModuleNotFoundError(
            "this IFEval check needs: pip install 'astro-weights[eval]'", name=e.name
        ) from e
    langdetect.DetectorFactory.seed = 0
    try:
        return str(langdetect.detect(text))
    except langdetect.LangDetectException:
        return None  # the original counts an undetectable text as followed


_NLTK_DATA_HINT = "python -m nltk.downloader punkt_tab"


@functools.lru_cache(maxsize=1)
def _punkt() -> Any:
    from nltk.tokenize import PunktTokenizer

    return PunktTokenizer("english")


def _nltk(fn: Callable[[], Any]) -> Any:
    """Run an nltk call, turning a missing package or data into an actionable error."""
    try:
        return fn()
    except ImportError as e:
        raise ModuleNotFoundError(
            "this IFEval check needs: pip install 'astro-weights[eval]'", name=e.name
        ) from e
    except LookupError as e:
        raise LookupError(
            f"this IFEval check needs NLTK data. Run once: {_NLTK_DATA_HINT}"
        ) from e


def _count_sentences(text: str) -> int:
    return len(_nltk(lambda: _punkt().tokenize(text)))


def _words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _response_language(v: str, *, language: str) -> bool:
    got = _detect(v)
    return got is None or got == language


def _english(v: str, *, upper: bool) -> bool:
    if not (v.isupper() if upper else v.islower()):
        return False
    got = _detect(v)
    return got is None or got == "en"


def _paragraphs_sep(v: str, num_paragraphs: int) -> bool:
    paragraphs = re.split(r"\s?\*\*\*\s?", v)
    n = len(paragraphs)
    for i, p in enumerate(paragraphs):
        if not p.strip():
            if i in (0, len(paragraphs) - 1):
                n -= 1
            else:
                return False
    return n == num_paragraphs


def _nth_paragraph_first_word(
    v: str, *, num_paragraphs: int, nth_paragraph: int, first_word: str
) -> bool:
    paragraphs = re.split(r"\n\n", v)
    n = sum(1 for p in paragraphs if p.strip())
    if not 1 <= nth_paragraph <= n:  # the original indexes before the bound check
        return False
    paragraph = paragraphs[nth_paragraph - 1].strip()
    if not paragraph:
        return False
    word = paragraph.split()[0].strip().lstrip("'").lstrip('"')
    got = ""
    for ch in word:
        if ch in {".", ",", "?", "!", "'", '"'}:
            break
        got += ch.lower()
    return n == num_paragraphs and got == first_word.lower()


def _postscript(v: str, *, postscript_marker: str) -> bool:
    marker = postscript_marker.strip()
    if marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + marker.lower() + r".*$"
    return bool(re.findall(pattern, v.lower(), flags=re.MULTILINE))


def _highlights(v: str, *, num_highlights: int) -> bool:
    n = sum(1 for h in re.findall(r"\*[^\n\*]*\*", v) if h.strip("*").strip())
    n += sum(
        1
        for h in re.findall(r"\*\*[^\n\*]*\*\*", v)
        if h.removeprefix("**").removesuffix("**").strip()
    )
    return n >= num_highlights


def _json_format(v: str) -> bool:
    v = v.strip()
    for prefix in ("```json", "```Json", "```JSON", "```"):
        v = v.removeprefix(prefix)
    try:
        json.loads(v.removesuffix("```").strip())
    except ValueError:
        return False
    return True


def _two_responses(v: str) -> bool:
    parts = v.split("******")
    valid = []
    for i, p in enumerate(parts):
        if p.strip():
            valid.append(p)
        elif i not in (0, len(parts) - 1):
            return False
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


def _word_tokenize(text: str) -> list[str]:
    import nltk

    tokens: list[str] = nltk.word_tokenize(text)
    return tokens


def _capital_words(v: str, *, capital_frequency: int, capital_relation: str) -> bool:
    n = sum(1 for w in _nltk(lambda: _word_tokenize(v)) if w.isupper())
    return _compare(n, capital_relation, capital_frequency)


# instruction id -> check(response, **kwargs from the dataset) -> followed
_CHECKS: dict[str, Callable[..., bool]] = {
    "keywords:existence": lambda v, *, keywords: all(
        re.search(k, v, flags=re.IGNORECASE) for k in keywords
    ),
    "keywords:frequency": lambda v, *, keyword, frequency, relation: _compare(
        len(re.findall(keyword.strip(), v, flags=re.IGNORECASE)), relation, frequency
    ),
    "keywords:forbidden_words": lambda v, *, forbidden_words: (
        not any(
            re.search(r"\b" + w + r"\b", v, flags=re.IGNORECASE)
            for w in forbidden_words
        )
    ),
    "keywords:letter_frequency": lambda v, *, letter, let_frequency, let_relation: (
        _compare(
            Counter(v.lower())[letter.strip().lower()], let_relation, let_frequency
        )
    ),
    "language:response_language": _response_language,
    "length_constraints:number_sentences": lambda v, *, num_sentences, relation: (
        _compare(_count_sentences(v), relation, num_sentences)
    ),
    "length_constraints:number_paragraphs": lambda v, *, num_paragraphs: (
        _paragraphs_sep(v, num_paragraphs)
    ),
    "length_constraints:number_words": lambda v, *, num_words, relation: _compare(
        _words(v), relation, num_words
    ),
    "length_constraints:nth_paragraph_first_word": _nth_paragraph_first_word,
    "detectable_content:number_placeholders": lambda v, *, num_placeholders: (
        len(re.findall(r"\[.*?\]", v)) >= num_placeholders
    ),
    "detectable_content:postscript": _postscript,
    "detectable_format:number_bullet_lists": lambda v, *, num_bullets: (
        len(re.findall(r"^\s*\*[^\*].*$", v, flags=re.MULTILINE))
        + len(re.findall(r"^\s*-.*$", v, flags=re.MULTILINE))
        == num_bullets
    ),
    "detectable_format:constrained_response": lambda v: any(
        o in v.strip()
        for o in ("My answer is yes.", "My answer is no.", "My answer is maybe.")
    ),
    "detectable_format:number_highlighted_sections": _highlights,
    "detectable_format:multiple_sections": lambda v, *, section_spliter, num_sections: (
        len(re.split(r"\s?" + section_spliter.strip() + r"\s?\d+\s?", v)) - 1
        >= num_sections
    ),
    "detectable_format:json_format": _json_format,
    "detectable_format:title": lambda v: any(
        t.lstrip("<").rstrip(">").strip() for t in re.findall(r"<<[^\n]+>>", v)
    ),
    "combination:two_responses": _two_responses,
    "combination:repeat_prompt": lambda v, *, prompt_to_repeat: (
        v.strip().lower().startswith(prompt_to_repeat.strip().lower())
    ),
    "startend:end_checker": lambda v, *, end_phrase: (
        v.strip().strip('"').lower().endswith(end_phrase.strip().lower())
    ),
    "startend:quotation": lambda v: (
        len(v.strip()) > 1 and v.strip()[0] == '"' and v.strip()[-1] == '"'
    ),
    "change_case:capital_word_frequency": _capital_words,
    "change_case:english_capital": lambda v: _english(v, upper=True),
    "change_case:english_lowercase": lambda v: _english(v, upper=False),
    "punctuation:no_comma": lambda v: "," not in v,
}
IFEVAL_IDS = tuple(_CHECKS)


def check_ifeval(item: Mapping[str, Any], answer: str) -> list[bool]:
    """Per instruction, whether `answer` follows it (strict).

    `item` needs `instruction_id_list` and `kwargs` (one dict per id), as
    `load_suite("ifeval")` items carry. An empty answer follows nothing.
    """
    ids = item["instruction_id_list"]
    kwargs = item["kwargs"]
    if len(ids) != len(kwargs):
        raise ValueError("instruction_id_list and kwargs differ in length")
    out = []
    for instruction_id, kw in zip(ids, kwargs, strict=True):
        if instruction_id not in _CHECKS:
            raise ValueError(f"unknown IFEval instruction {instruction_id!r}")
        out.append(bool(answer.strip()) and bool(_CHECKS[instruction_id](answer, **kw)))
    return out


def grade_ifeval(item: Mapping[str, Any], answer: str) -> float:
    """Strict prompt-level score: 1.0 when every instruction is followed, else 0.0.

    Text before the last "</think>" is ignored. For the per-instruction detail
    call `check_ifeval`.
    """
    return float(all(check_ifeval(item, answer_part(answer))))
