"""Answer graders. Standard library only at import time."""

import re
import threading

_BOXED = "\\boxed{"
_CHOICE = re.compile(r"\(?([A-J])[).:]*(?:$|(?<=[).:])\s)")
_TEXT_CMD = re.compile(r"\\(?:text|textbf|mathrm|mathbf)\{(.*?)\}")


def last_boxed(s: str) -> str | None:
    """Content of the last \\boxed{...} in s (brace matched), or None."""
    i = s.rfind(_BOXED)
    if i < 0:
        return None
    start = i + len(_BOXED)
    depth = 1
    for j in range(start, len(s)):
        depth += (s[j] == "{") - (s[j] == "}")
        if depth == 0:
            return s[start:j]
    return None


def answer_part(text: str) -> str:
    """Text after the last "</think>", or all of it."""
    return text.rsplit("</think>", 1)[-1]


def grade_choice(gold: str, answer: str) -> bool:
    """True if the last \\boxed{} holds the gold letter (A-J)."""
    b = last_boxed(answer)
    if b is None:
        return False
    m = _CHOICE.match(_TEXT_CMD.sub(r"\1", b).strip())
    return m is not None and m.group(1) == gold


def grade_math(gold: str, answer: str, strip_commas: bool = False) -> bool:
    """True if the answer's last boxed value equals gold, checked by math_verify.

    math_verify's 5 s timeouts use signal.alarm, which works only in the main
    thread; elsewhere they are turned off, so a pathological answer can run long.
    """
    try:
        from math_verify import LatexExtractionConfig, parse, verify
    except ImportError as e:
        raise ImportError("grade_math needs: pip install 'astro-weights[eval]'") from e
    if strip_commas:
        gold = gold.replace(",", "")
    try:
        # ponytail: off the main thread no timeout; a caller pool needs its own
        t = 5 if threading.current_thread() is threading.main_thread() else None
        cfg = [LatexExtractionConfig(boxed_match_priority=0)]
        g = parse(f"\\boxed{{{gold}}}", parsing_timeout=t)
        a = parse(answer, extraction_config=cfg, parsing_timeout=t)
        return bool(verify(g, a, timeout_seconds=t))
    except Exception:  # noqa: BLE001  math_verify can time out or choke on odd latex
        return False


def grade_exact(gold: str, answer: str) -> bool:
    """True if the stripped strings are equal."""
    return gold.strip() == answer.strip()
