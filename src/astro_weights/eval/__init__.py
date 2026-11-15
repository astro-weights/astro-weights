"""Paired eval statistics from per-item records. Never runs a model."""

from astro_weights.eval.grade import (
    answer_part,
    grade_choice,
    grade_exact,
    grade_math,
    last_boxed,
)
from astro_weights.eval.paired import PairedResult, compare, format_report, pair
from astro_weights.eval.records import Record, prompt_hash, read_jsonl, write_jsonl

__all__ = [
    "PairedResult",
    "Record",
    "answer_part",
    "compare",
    "format_report",
    "grade_choice",
    "grade_exact",
    "grade_math",
    "last_boxed",
    "pair",
    "prompt_hash",
    "read_jsonl",
    "write_jsonl",
]
