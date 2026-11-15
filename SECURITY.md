# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub: open the repository's **Security** tab and choose **Report a vulnerability** ([direct link](https://github.com/astro-weights/astro-weights/security/advisories/new)). Do not open a public issue.

You should get a reply within 7 days. Once a fix is ready we will publish an advisory and credit you unless you ask us not to.

## Supported versions

Only the latest release receives security fixes while the project is at 0.x.

## Untrusted input

This section says what the library checks when it reads a file or a reply it did not make, and what it leaves to you.

### What counts as untrusted

- Adapter directories: `adapter_config.json`, `astro_manifest.json` and the safetensors weights (`read_peft`, `verify_peft`, `load_peft`).
- Base-model safetensors files and shard indexes (`read_tensor`, `tensor_sha256`, `read_base_weight`).
- GGUF files, adapter or base (`read_lora_gguf`, `splice_gguf`, `verify_splice`).
- Recipe files and the `.npy` files they name (`load_recipe`, `build_edit`).
- Eval JSONL (`read_jsonl`).
- HTTP replies from an eval server (`generate_records_http`, `run_header_http`).

### What the library guarantees

- Bad content raises `ValueError`, and a missing or unreadable file raises `OSError`. Nothing else escapes. `read_peft` may raise either one; `verify_peft` catches both and returns the message as a string. `read_tensor`, `read_jsonl`, `load_recipe`, `build_edit` and the GGUF readers raise `ValueError` only. The GGUF readers wrap `OSError` too (`_open`).
- No pickle is loaded. `.npy` files are read with `np.load(..., allow_pickle=False)`, object arrays are refused before numpy sees them (`_check_npy_header`), and weights go through safetensors. Tests: `test_pickled_array_refused`.
- No code from a file is run. Recipe names pick from a fixed table (`_RECIPES`); a recipe cannot name a function or a module.
- A recipe's `.npy` files must sit in the recipe directory. Names must be plain file names, and a symlink that points out is refused (`_plain_name`, `_load_array`). Test: `test_symlink_escape_rejected`. A shard name in `model.safetensors.index.json` gets the same check in `_shard`.
- Size and count caps for parsing:
  - `MAX_NPY_BYTES` = 256 MiB per `.npy` file, checked before reading (`_read_capped`). Only regular files are opened, so a FIFO cannot block.
  - `MAX_N_IN` = 2^20 for `column_shift`.
  - `MAX_HEADER` = 100,000,000 bytes for a safetensors header in `read_tensor`. Offsets and byte counts are checked against the shape and the file size.
  - `_MAX_ARRAY_ITEMS` = 2^21 total GGUF kv array items, and `_MAX_DEPTH` = 8 levels of nested arrays. The GGUF header counts and every length must fit in the file (`_check_layout`). Test: `test_fifo_is_refused_without_blocking`.
  - Deeply nested JSON is turned into `ValueError` (`_load_object`, `_parse`).
- The eval API key is never put in an error, a log line or a run header. It must match a safe character set (`_check_key`), error text goes through `_scrub`, and redirects are refused so `Authorization` is not forwarded (`_NoRedirect`). Tests: `test_api_key_sent_but_never_leaks`, `test_redirect_refused_and_key_not_forwarded`, `test_bad_api_key_rejected_without_echo`.

The `ValueError` rule covers file parsing. `generate_records_http` also raises `RuntimeError` when the server is unreachable or returns an HTTP error. A reply with the wrong shape raises `ValueError`.

### What is not guaranteed

- Loading a model is outside these checks. `transformers` and `peft` can run code (`trust_remote_code`) or unpickle old `.bin` checkpoints. Use safetensors, leave `trust_remote_code` off, and read the base model's source yourself.
- A file that parses is not a safe file. An adapter can parse cleanly and still change a model in harmful ways. `verify_peft` checks the files against the manifest. It does not check safety or where the adapter came from.
- The manifest hash is not a signature. Anyone who can edit the weights can edit the manifest to match. Use your own signing or a trusted channel if provenance matters.
- `save_peft` is atomic against a crashed process (temp directory, then rename or ordered replace). It does not promise anything after power loss, because it does not call `fsync`. Tests: `test_failed_write_leaves_existing_adapter_intact`, `test_failure_between_replaces_is_never_a_silent_mix`.
- Resource use is bounded for parsing only. Applying an edit to a large model needs memory in proportion to the model. Some reads have no cap of their own: `read_jsonl` and HTTP reply bodies are read whole, and the safetensors reader in `read_peft` loads every tensor in the file.

### Fuzz tests

Property tests in `tests/test_fuzz.py` feed random and mutated files to these parsers on every PR (`test`, and `test-hf` for the GGUF cases).
