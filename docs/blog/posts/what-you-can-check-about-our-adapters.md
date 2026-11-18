---
date: 2026-11-18
slug: what-you-can-check-about-our-adapters
categories:
  - security
---
# What you can check about our adapters, and what you can't

Reddit commenter raised two points about our adapters. First: published evals prove nothing, because backdoor fires only on its trigger and benchmark never contains it. Second: you're asked to load weights from stranger.

Both points hold. Below: what you can check yourself before running one of our adapters, and what stays unproven after all of it.

<!-- more -->

## Evals can't rule out backdoor

Our evals compare edited model with base on same problems. They show how edit behaves on those problems. Trigger that never appears in test set leaves no mark in numbers. BadEdit planted backdoors by model editing with 15 examples and kept performance on normal inputs; LoRA adapters have been backdoored same way with clean accuracy kept.

Small doesn't mean safe either. Our adapters are rank one in each layer they touch, and so are Concept-ROT trojans, backdoor method built on rank-one model editing. Sleeper Agents showed trained-in backdoor can survive later safety training. So nothing below proves our adapters clean. It narrows what they could be.

## What files can and can't do

Both adapters are plain safetensors: weights, small JSON config, manifest. No pickle files, which matters because loading pickle can run arbitrary code. Loading adapter file changes model weights. Safetensors format has no code-execution path, and its audit by Trail of Bits found no flaw giving one.

GGUF differs. 9B repo also ships merged GGUF, and llama.cpp's GGUF parser has had memory bugs where crafted file could lead to code execution (CVE-2024-21836, and CVE-2026-27940 earlier this year, fixed in build b8146). Using GGUF: keep llama.cpp current. Not trusting our GGUF at all: merge adapter into base GGUF yourself with `astro-weights export --splice`.

Two more checks, a minute each:
- Load base model from Qwen repo yourself (for example `Qwen/Qwen3.8-27B`), not from us. Neither adapter repo ships tokenizer, config or chat template files, so all of those come from wherever you get base.
- GGUF carries own chat template, which decides how your prompt gets formatted, so edited template is one place a change could hide. Our 9B merged GGUF was built from Unsloth's `Qwen3.5-9B-Q4_K_M.gguf` (`unsloth/Qwen3.5-9B-GGUF`). All 46 metadata keys in our file, chat template included, are byte-identical to that Unsloth file. They are not identical to Qwen's own `chat_template.jinja`, because Unsloth's build of template differs from Qwen's. Using our GGUF means trusting Unsloth's template, same as downloading their file. Check yourself with [`gguf_meta_diff.py`](https://github.com/astro-weights/astro-weights/blob/23532d2921f91918f9c57b82141bb5c81a822bc5/tools/gguf_meta_diff.py), standard-library script reading both GGUF headers over HTTP range requests, so no 6 GB download:

```
curl -LO https://raw.githubusercontent.com/astro-weights/astro-weights/23532d2921f91918f9c57b82141bb5c81a822bc5/tools/gguf_meta_diff.py
python gguf_meta_diff.py https://huggingface.co/glitch401/qwen3.5-9B-terse/resolve/main/qwen3.5-9B-terse-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf
```

Tensors too: our file has 427 tensors, and 422 are byte-identical to the Unsloth file. The 5 that differ are exactly `blk.{18,20,22,24,26}.ffn_down.weight`, the edited layers (stored as F16 in ours, quantised in theirs). To check this yourself, download both files (about 12 GB in all) and run the newer version of the script with `--tensors`:

```bash
curl -LO https://raw.githubusercontent.com/astro-weights/astro-weights/366062389616c86e469a6ebdf4c0b0c6276b49cf/tools/gguf_meta_diff.py
curl -LO https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf
curl -LO https://huggingface.co/glitch401/qwen3.5-9B-terse/resolve/main/qwen3.5-9B-terse-Q4_K_M.gguf
python gguf_meta_diff.py Qwen3.5-9B-Q4_K_M.gguf qwen3.5-9B-terse-Q4_K_M.gguf --tensors
```

Expect exit code 1. The script exits 1 when any tensor differs, and ours differ on purpose: the output shows 422 identical and 5 different, exactly the `ffn_down` names above. Any other differing tensor would be a real finding.

## Rebuild adapter from published edit

9B repo publishes five full edited weight matrices adapter was fitted from, in `edits/`. Rebuild adapter from those plus stock Qwen base with released library, on CPU:

```
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir base
hf download glitch401/qwen3.5-9B-terse --local-dir pub --exclude "*.gguf"
curl -LO https://raw.githubusercontent.com/astro-weights/astro-weights/23532d2921f91918f9c57b82141bb5c81a822bc5/tools/rebuild_adapter.py
uv run --no-project --with astro-weights==0.2.0 python rebuild_adapter.py pub/edits base out pub
```

Download about 20 GB. On our 8-core CPU rebuild took 194 seconds and 2.2 GB of RAM. Size of each layer's change, and how much rank-one fit leaves out, came out equal to published manifest at every printed digit. Change rebuilt adapter applies matches published one within 9.88e-07 (6.7e-07 relative).

Not bit-exact: adapter file's sha256 won't match ours. Rank-one LoRA stores change as two factors, and same product splits between them more than one way. Our published file splits it evenly across both factors; 0.2.0 library puts it all in one. Product, which is what changes the model, is same to about one part in a million.

This check proves one narrow thing: adapter is rank-one fit of edit matrices we published (fit leaves 0.1% to 0.6% of each layer's edit out). It doesn't show where those edits came from. AI agent made them by working on model's activations on GPU, and code it ran wasn't saved, so nobody can re-derive the edits, us included. Running agent again would most likely give different edit.

27B adapter: file hashes and `inspect` today (rank one, 49 down-projection matrices). Its edit matrices aren't published yet, so rebuild check doesn't cover it.

## Look at change itself

`uvx astro-weights==0.2.0 inspect DIR` prints what adapter touches: which layers, rank, and size of change in each. Both adapters: MLP down-projection matrices only, rank one each. Published weight-space detectors (for example PEFTGuard, and 2026 detector reading spectral statistics of LoRA updates) are research tools trained on their own sets of benign and poisoned adapters. One more signal, no more. Clean result from one doesn't prove anything either.

## What stays unproven

- That no trigger exists. No test we know of can show that.
- Why agent's edit works. We can show what it changes, not full account of what that change does inside model.
- How edits were made. Agent's code wasn't saved.
- That 27B adapter matches a published edit, until those edit matrices are up.

## What we're adding next

- Release provenance. Library's PyPI releases already carry PyPI attestations (0.2.0 does).
- Adapter signatures, later. Adapters on Hugging Face aren't signed yet; we're looking at Sigstore's model-signing for them.
- Raw outputs. We're publishing per-problem outputs for every eval (about 30 MB compressed): each problem's benchmark ID with what base model and edited model wrote. Problem text not included, so match IDs to benchmark yourself, then read what edited model actually wrote instead of trusting summary.
- 27B edit matrices, so rebuild check covers both adapters.

Found something wrong here, or want check we haven't listed: open issue or reply here.

## Sources

- BadEdit: https://arxiv.org/abs/2403.13355
- LoRA adapter backdoors with clean accuracy: https://arxiv.org/abs/2605.30189
- Concept-ROT: https://arxiv.org/abs/2412.13341
- Sleeper Agents: https://arxiv.org/abs/2401.05566
- Hugging Face on pickle: https://huggingface.co/docs/hub/security-pickle
- Safetensors audit: https://huggingface.co/blog/safetensors-security-audit
- CVE-2024-21836: https://talosintelligence.com/vulnerability_reports/TALOS-2024-1915
- CVE-2026-27940: https://github.com/ggml-org/llama.cpp/security/advisories/GHSA-3p4r-fq3f-q74v
- PEFTGuard: https://arxiv.org/abs/2411.17453
- Weight-space detection of LoRA backdoors: https://arxiv.org/abs/2602.15195
- Sigstore model signing: https://blog.sigstore.dev/model-transparency-v1.0
