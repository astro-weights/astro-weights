"""Paired eval statistics from per-item records. Never runs a model."""

from astro_weights.eval.grade import (
    answer_part,
    grade_choice,
    grade_exact,
    grade_math,
    last_boxed,
)
from astro_weights.eval.header import (
    check_headers,
    header_path,
    read_header,
    write_header,
)
from astro_weights.eval.paired import (
    PairedResult,
    compare,
    exceeds_floor,
    format_report,
    pair,
)
from astro_weights.eval.records import Record, prompt_hash, read_jsonl, write_jsonl

__all__ = [
    "PairedResult",
    "Record",
    "answer_part",
    "check_headers",
    "compare",
    "exceeds_floor",
    "format_report",
    "grade_choice",
    "grade_exact",
    "grade_math",
    "header_path",
    "last_boxed",
    "pair",
    "prompt_hash",
    "read_header",
    "read_jsonl",
    "write_header",
    "write_jsonl",
]
