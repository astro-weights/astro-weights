"""Paired eval statistics from per-item records. Never runs a model."""

from astro_weights.eval.arms import ArmsResult, arms_dict, compare_arms, format_arms
from astro_weights.eval.floor import (
    FloorResult,
    floor_from_records,
    load_floor,
    save_floor,
)
from astro_weights.eval.grade import (
    answer_part,
    grade_choice,
    grade_exact,
    grade_math,
    last_boxed,
)
from astro_weights.eval.header import (
    CONFIG_KEYS,
    PROVENANCE_KEYS,
    check_headers,
    config_hash,
    header_path,
    read_header,
    write_header,
)
from astro_weights.eval.paired import (
    PairedResult,
    StratumResult,
    compare,
    exceeds_floor,
    format_report,
    is_loop,
    pair,
)
from astro_weights.eval.records import Record, prompt_hash, read_jsonl, write_jsonl
from astro_weights.eval.verdict import Verdict

__all__ = [
    "CONFIG_KEYS",
    "PROVENANCE_KEYS",
    "ArmsResult",
    "FloorResult",
    "PairedResult",
    "Record",
    "StratumResult",
    "Verdict",
    "answer_part",
    "arms_dict",
    "check_headers",
    "compare",
    "compare_arms",
    "config_hash",
    "exceeds_floor",
    "floor_from_records",
    "format_arms",
    "format_report",
    "grade_choice",
    "grade_exact",
    "grade_math",
    "header_path",
    "is_loop",
    "last_boxed",
    "load_floor",
    "pair",
    "prompt_hash",
    "read_header",
    "read_jsonl",
    "save_floor",
    "write_header",
    "write_jsonl",
]
