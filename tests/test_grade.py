import builtins
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from astro_weights.eval import (
    answer_part,
    grade_choice,
    grade_exact,
    grade_math,
    last_boxed,
)


def test_last_boxed_nested_and_last() -> None:
    assert last_boxed(r"\boxed{1} then \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert last_boxed("no box") is None
    assert last_boxed(r"\boxed{unclosed") is None


def test_answer_part() -> None:
    assert answer_part("think</think>a</think>b") == "b"
    assert answer_part("plain") == "plain"


@pytest.mark.parametrize(
    ("answer", "ok"),
    [
        (r"so \boxed{B}", True),
        (r"\boxed{(B)}", True),
        (r"\boxed{B.}", True),
        (r"\boxed{\text{B}}", True),
        (r"\boxed{B: foo}", True),
        (r"\boxed{C}", False),
        (r"\boxed{BC}", False),
        ("B", False),
        (r"\boxed{A} \boxed{B}", True),
    ],
)
def test_grade_choice(answer: str, ok: bool) -> None:
    assert grade_choice("B", answer) is ok


def test_grade_exact() -> None:
    assert grade_exact(" yes ", "yes\n")
    assert not grade_exact("yes", "Yes")


def test_grade_math() -> None:
    pytest.importorskip("math_verify")
    assert grade_math("1/2", r"\boxed{\frac{1}{2}}")
    assert not grade_math("3", r"\boxed{4}")
    assert not grade_math("3", "no box")
    assert grade_math("1,000", r"\boxed{1000}", strip_commas=True)


def test_grade_math_in_worker_threads() -> None:
    # math_verify's signal timeouts raise off the main thread; that once read as wrong
    pytest.importorskip("math_verify")
    with ThreadPoolExecutor(2) as pool:
        ok = pool.submit(grade_math, "1/2", r"\boxed{\frac{1}{2}}")
        bad = pool.submit(grade_math, "3", r"\boxed{4}")
        assert ok.result()
        assert not bad.result()


def test_grade_math_missing_dep(monkeypatch: pytest.MonkeyPatch) -> None:
    real = builtins.__import__

    def fake(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "math_verify":
            raise ImportError("blocked")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match=r"pip install 'astro-weights\[eval\]'"):
        grade_math("1", r"\boxed{1}")
