`read_jsonl` now raises `ValueError` (not `OverflowError` or `RecursionError`) for an out-of-range score or deeply nested JSON.
