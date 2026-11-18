"""Rebuild a published rank-1 adapter from its dense edited tensors and the stock base.

usage: python rebuild_adapter.py EDITS_DIR BASE_DIR OUT_DIR [PUBLISHED_ADAPTER_DIR]

EDITS_DIR holds one <module>.weight.safetensors per edited matrix (the edited weight).
dW = edited - base, then astro-weights' from_dense (truncated SVD, rank 1) and save_peft.
With PUBLISHED_ADAPTER_DIR it prints max abs differences against the published file.
Runs on CPU; needs only astro-weights (numpy, safetensors). Example (9B, ~20 GB download):

    hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir base
    hf download glitch401/qwen3.5-9B-terse --local-dir pub --exclude "*.gguf"
    uv run --no-project --with astro-weights==0.2.0 python rebuild_adapter.py pub/edits base out pub

The factor files differ from the published ones (SVD sign and how the singular value is
split between A and B), so compare the products B @ A, which are what changes the model.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from astro_weights.edits import Edit, from_dense
from astro_weights.export import read_base_weight, read_tensor, save_peft


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


edits, base, out = (Path(a) for a in sys.argv[1:4])
deltas = []
for f in sorted(edits.glob("*.weight.safetensors")):
    key = f.name.removesuffix(".safetensors")
    dW = read_tensor(f, key) - read_base_weight(base, key)
    delta, rel = from_dense(key.removesuffix(".weight"), dW, rank=1)
    print(f"{key}  |dW|_F {np.linalg.norm(dW):.6g}  rel_residual {rel:.6g}")
    deltas.append(delta)
save_peft(Edit.from_deltas(deltas, base.name), out, overwrite=True)
new = out / "adapter_model.safetensors"
print(f"rebuilt  {new}  sha256 {sha256(new)}")

if len(sys.argv) > 4:
    old_p = Path(sys.argv[4]) / "adapter_model.safetensors"
    print(f"published  {old_p}  sha256 {sha256(old_p)}")
    old, got = load_file(str(old_p)), load_file(str(new))
    assert old.keys() == got.keys(), sorted(old.keys() ^ got.keys())
    raw = aligned = dense = rel_dense = 0.0
    for k in sorted(old):
        if not k.endswith(".lora_A.weight"):
            continue
        kb = k.replace("lora_A", "lora_B")
        a0, b0, a1, b1 = old[k], old[kb], got[k], got[kb]
        raw = max(raw, abs(a0 - a1).max(), abs(b0 - b1).max())
        s = np.sign((a0 * a1).sum())  # SVD sign is arbitrary: flip both factors
        aligned = max(aligned, abs(a0 - s * a1).max(), abs(b0 - s * b1).max())
        d0 = b0.astype(np.float64) @ a0.astype(np.float64)
        d1 = b1.astype(np.float64) @ a1.astype(np.float64)
        dense = max(dense, abs(d0 - d1).max())
        rel_dense = max(rel_dense, np.linalg.norm(d0 - d1) / np.linalg.norm(d0))
    print(f"tensors {len(old)}  max|diff| raw factors {raw:.3g}")
    print(f"max|diff| factors after sign align {aligned:.3g}")
    print(f"max|diff| dense B@A {dense:.3g}  max rel Frobenius {rel_dense:.3g}")
