"""Paired eval statistics from per-item records. Never runs a model."""

from astro_weights.eval.paired import PairedResult, compare, format_report, pair
from astro_weights.eval.records import Record, prompt_hash, read_jsonl, write_jsonl

__all__ = [
    "PairedResult",
    "Record",
    "compare",
    "format_report",
    "pair",
    "prompt_hash",
    "read_jsonl",
    "write_jsonl",
]
