import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from astro_weights.edits import Edit, masked_rank1, top_k_mask
from astro_weights.export import load_peft, save_peft

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from astro_weights.runners.hf import apply, remove

MODULE = "model.layers.1.mlp.down_proj"


def make_edit(model: Any) -> Edit:
    w0 = model.get_submodule(MODULE).weight.detach().numpy()
    rng = np.random.default_rng(0)
    u = rng.standard_normal(w0.shape[0]).astype(np.float32)
    mask = top_k_mask(rng.standard_normal(w0.shape[1]), 8)
    return Edit((masked_rank1(MODULE, w0, u, mask, 1.0),), "tiny-llama")


def logits(model: Any, input_ids: Any) -> Any:
    with torch.no_grad():
        return model(input_ids).logits


def lora_modules(model: Any) -> list[str]:
    return [n for n, _ in model.named_modules() if n.endswith(".lora_A.astro")]


def test_apply_matches_dense_edit(tiny_llama: Any, input_ids: Any) -> None:
    edit = make_edit(tiny_llama)
    ref = copy.deepcopy(tiny_llama)
    with torch.no_grad():
        ref.get_submodule(MODULE).weight += torch.from_numpy(edit.deltas[0].dense())
    expected = logits(ref, input_ids)
    assert not torch.allclose(expected, logits(tiny_llama, input_ids), atol=1e-5)

    got = logits(apply(tiny_llama, edit), input_ids)
    assert torch.allclose(got, expected, atol=1e-5)


def test_only_edit_module_wrapped(tiny_llama: Any) -> None:
    peft_model = apply(tiny_llama, make_edit(tiny_llama))
    names = lora_modules(peft_model)
    assert len(names) == 1
    assert names[0].endswith(f"{MODULE}.lora_A.astro")


def test_remove_restores_base_logits(tiny_llama: Any, input_ids: Any) -> None:
    base = logits(tiny_llama, input_ids)
    peft_model = apply(tiny_llama, make_edit(tiny_llama))
    model = remove(peft_model)
    assert not lora_modules(model)
    assert torch.equal(logits(model, input_ids), base)


def test_save_load_apply_roundtrip(
    tiny_llama: Any, input_ids: Any, tmp_path: Path
) -> None:
    edit = make_edit(tiny_llama)
    ref = copy.deepcopy(tiny_llama)
    direct = logits(apply(ref, edit), input_ids)

    save_peft(edit, tmp_path)
    got = logits(apply(tiny_llama, load_peft(tmp_path)), input_ids)
    assert torch.allclose(got, direct, atol=1e-5)


def test_unknown_module_raises(tiny_llama: Any) -> None:
    edit = make_edit(tiny_llama)
    bad = Edit(
        (
            type(edit.deltas[0])(
                "model.layers.9.mlp.down_proj", edit.deltas[0].B, edit.deltas[0].A
            ),
        ),
        "x",
    )
    with pytest.raises(ValueError, match=r"not an nn\.Linear"):
        apply(tiny_llama, bad)
