# Run one eval arm from a TOML file

`astro-weights gen ARM.toml` runs one eval arm against OpenAI-compatible server (vLLM, llama.cpp). One file, one arm, one runner call. Long run can be killed and continued with `--resume`. All keys, kinds and outputs: [CLI reference, `gen`](../reference/cli.md#gen).

## Write ARM.toml

`kind = "http"` asks server once per item and grades each answer. Relative paths are read from ARM.toml's directory.

```toml
kind = "http"
arm = "base"
base_url = "http://127.0.0.1:8000"
model = "base"
max_new_tokens = 512
items = "items.jsonl"
grader = "exact"
out = "out/base.jsonl"
```

`items.jsonl` holds one JSON object per line with `item_id`, `prompt` and `gold`:

```json
{"item_id": "i0", "prompt": "What is 2 + 2?", "gold": "4"}
```

`grader` is `exact`, `choice` or `math` (`math` needs `pip install "astro-weights[eval]"`). Other kinds: `suites` (regression suites) and `bfcl` (BFCL multi-turn tasks). Unknown, missing or mistyped key exits 2 and names key, so misspelt setting never falls back to default.

## Run

```bash
astro-weights gen ARM.toml
```

Prints `ARM: done/total items, OUT`:

```text
base: 8/8 items, out/base.jsonl
```

`--json` prints `{"arm", "out", "done", "total"}` instead. Records go to `out`; header (settings of run) goes beside it as `out/base.header.json`. Records file that is not empty is refused (exit 2) before any request; pass `--resume` or choose another `out`.

## Resume after kill

Each record is written and fsynced as soon as it is made, so kill loses only requests in flight. Run same file again with `--resume`:

```bash
astro-weights gen ARM.toml --resume
```

- Records already written stay byte for byte. Last line cut short by kill is dropped and asked again. Only items without record are asked.
- Header of existing file must match ARM.toml. Changed setting (`max_new_tokens`, `seed`, `extra_body`, `serving_flags`, `system`, `model_id`, `concurrency`, `k`, suite `limit`...) is refused with differences, exit 2, and nothing is written. Change config → new `out`.
- `--resume` with no records yet starts fresh. `--resume` on finished arm asks nothing.

`gen` keeps no state beyond records and header files, so batch job can run same command with `--resume` every time.

## Exit codes

| Code | Meaning | Next step |
|---|---|---|
| 0 | Arm done. | — |
| 2 | Found before first request: bad config, missing extra or key variable (`api_key_env`), records file not empty without `--resume`, or resume refused. | Fix config or `out`. |
| 1 | After first request: server error that outlasted runner's retries, or reply runner cannot read (no message, no `usage`). Records so far stay; `done` counts them; error on stderr. | Run again with `--resume`. |
