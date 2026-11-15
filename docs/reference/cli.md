# CLI

```text
astro-weights [--version] {inspect,eval} ...
```

| Command | Does |
|---|---|
| `inspect DIR` | summarize and verify saved PEFT adapter; see [How-to: inspect adapter](../how-to/inspect-adapter.md) |
| `eval BASE.jsonl EDIT.jsonl [--boot N] [--seed N]` | compare two arms from per-item records, print [`format_report`](../how-to/compare-arms.md); `--boot` bootstrap resamples (default 10000), `--seed` bootstrap seed (default 0); exit code 2 on unreadable file or arms that do not pair; see [How-to: run eval end to end](../how-to/run-eval.md) |
