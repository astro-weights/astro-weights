import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from astro_weights.edits import Edit, LowRankDelta, masked_rank1, top_k_mask
from astro_weights.export import load_peft, save_peft

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from astro_weights.runners.hf import _tail, apply, remove, resolve_modules

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
            LowRankDelta(
                "model.layers.9.mlp.down_proj", edit.deltas[0].B, edit.deltas[0].A
            ),
        ),
        "x",
    )
    with pytest.raises(ValueError, match=r"no nn\.Linear matches"):
        apply(tiny_llama, bad)


TEXT = "model.layers.1.mlp.down_proj"
VL = "model.language_model.layers.1.mlp.down_proj"


def renamed(edit: Edit, module: str) -> Edit:
    d = edit.deltas[0]
    return Edit((LowRankDelta(module, d.B, d.A),), edit.base_model)


def edit_for(model: Any, module: str) -> Edit:
    w0 = model.get_submodule(module).weight.detach().numpy()
    rng = np.random.default_rng(0)
    u = rng.standard_normal(w0.shape[0]).astype(np.float32)
    mask = top_k_mask(rng.standard_normal(w0.shape[1]), 8)
    return Edit((masked_rank1(module, w0, u, mask, 1.0),), "tiny")


def test_tail() -> None:
    assert _tail(VL) == ("layers", "1", "mlp", "down_proj")
    assert _tail("a.layers.2.b.layers.7.c") == ("layers", "7", "c")
    assert _tail("lm_head") is None
    assert _tail("model.layers.mlp") is None


@pytest.mark.parametrize(
    ("fixture", "live", "stored"),
    [("tiny_qwen_text", TEXT, VL), ("tiny_qwen_vl", VL, TEXT)],
)
def test_names_resolve_across_qwen_layouts(
    fixture: str, live: str, stored: str, input_ids: Any, request: Any
) -> None:
    model = request.getfixturevalue(fixture)
    edit = edit_for(model, live)
    ref = copy.deepcopy(model)
    with torch.no_grad():
        ref.get_submodule(live).weight += torch.from_numpy(edit.deltas[0].dense())
    expected = logits(ref, input_ids)
    assert not torch.allclose(expected, logits(model, input_ids), atol=1e-5)

    other = renamed(edit, stored)
    assert resolve_modules(model, other) == {stored: live}
    peft_model = apply(model, other)
    assert torch.allclose(logits(peft_model, input_ids), expected, atol=1e-5)
    assert len(lora_modules(peft_model)) == 1


def test_exact_name_wins(tiny_qwen_vl: Any) -> None:
    edit = edit_for(tiny_qwen_vl, VL)
    assert resolve_modules(tiny_qwen_vl, edit) == {VL: VL}


def test_layer_index_matches_whole_component(tiny_llama: Any) -> None:
    # "layers.1" must not match a stored "layers.11" (or the reverse)
    edit = renamed(make_edit(tiny_llama), "x.layers.11.mlp.down_proj")
    with pytest.raises(ValueError, match=r"no nn\.Linear matches"):
        resolve_modules(tiny_llama, edit)


nn: Any = torch.nn


class TwoTowers(nn.Module):  # type: ignore[misc]
    def __init__(self, vision_out: int) -> None:
        super().__init__()
        self.text = nn.ModuleDict({"layers": nn.ModuleList([nn.Linear(4, 6)])})
        self.vision = nn.ModuleDict(
            {"layers": nn.ModuleList([nn.Linear(4, vision_out)])}
        )


def tower_edit(module: str) -> Edit:
    delta = LowRankDelta(
        module, np.ones((6, 1), np.float32), np.ones((1, 4), np.float32)
    )
    return Edit((delta,), "towers")


def test_ambiguous_match_raises() -> None:
    with pytest.raises(ValueError, match=r"ambiguous.*text\.layers\.0.*vision"):
        resolve_modules(TwoTowers(6), tower_edit("lm.layers.0"))


def test_shape_breaks_tie() -> None:
    got = resolve_modules(TwoTowers(8), tower_edit("lm.layers.0"))
    assert got == {"lm.layers.0": "text.layers.0"}


def test_two_edits_on_one_layer_raise(tiny_llama: Any) -> None:
    d = make_edit(tiny_llama).deltas[0]
    edit = Edit((d, LowRankDelta("other.layers.1.mlp.down_proj", d.B, d.A)), "x")
    with pytest.raises(ValueError, match="map to one layer"):
        resolve_modules(tiny_llama, edit)


def test_unloaded_adapter_raises(tiny_llama: Any, monkeypatch: Any) -> None:
    from astro_weights.runners import hf

    class Result:
        unexpected_keys = ("base_model.model.x.lora_A.weight",)

    monkeypatch.setattr(hf, "set_peft_model_state_dict", lambda *a, **k: Result())
    with pytest.raises(RuntimeError, match="did not load cleanly"):
        apply(tiny_llama, make_edit(tiny_llama))


def test_from_deltas_mixed_rank_apply(tiny_llama: Any, input_ids: Any) -> None:
    m1, m0 = MODULE, "model.layers.0.mlp.down_proj"
    rng = np.random.default_rng(1)
    deltas = []
    for module, rank in ((m1, 1), (m0, 2)):
        out, inn = tiny_llama.get_submodule(module).weight.shape
        deltas.append(
            LowRankDelta(
                module,
                0.1 * rng.standard_normal((out, rank)),
                0.1 * rng.standard_normal((rank, inn)),
            )
        )
    edit = Edit.from_deltas(deltas, "tiny-llama")
    assert edit.rank == 2

    ref = copy.deepcopy(tiny_llama)
    with torch.no_grad():
        for d in deltas:
            ref.get_submodule(d.module).weight += torch.from_numpy(d.dense())
    expected = logits(ref, input_ids)
    assert not torch.allclose(expected, logits(tiny_llama, input_ids), atol=1e-5)

    peft_model = apply(tiny_llama, edit)
    assert torch.allclose(logits(peft_model, input_ids), expected, atol=1e-5)
    assert len(lora_modules(peft_model)) == 2


def test_head_edit_with_rank1_edit(tiny_llama: Any, input_ids: Any) -> None:
    from astro_weights.edits import zero_head

    o_proj = "model.layers.0.self_attn.o_proj"
    w0 = tiny_llama.get_submodule(o_proj).weight.detach().numpy()
    head = zero_head(o_proj, w0, 1, 4)
    edit = Edit.from_deltas([head, make_edit(tiny_llama).deltas[0]], "tiny-llama")
    assert edit.rank == w0.shape[1] // 4

    ref = copy.deepcopy(tiny_llama)
    with torch.no_grad():
        for d in edit.deltas:
            ref.get_submodule(d.module).weight += torch.from_numpy(d.dense())
        block = ref.get_submodule(o_proj).weight[:, 8:16]
    assert torch.count_nonzero(block) == 0
    got = logits(apply(tiny_llama, edit), input_ids)
    assert torch.allclose(got, logits(ref, input_ids), atol=1e-5)


def second_edit(model: Any, module: str = MODULE) -> Edit:
    w0 = model.get_submodule(module).weight.detach().numpy()
    rng = np.random.default_rng(5)
    u = rng.standard_normal(w0.shape[0]).astype(np.float32)
    mask = top_k_mask(rng.standard_normal(w0.shape[1]), 8)
    return Edit((masked_rank1(module, w0, u, mask, 1.0),), "tiny-llama")


def test_swap_equals_fresh_apply(tiny_llama: Any, input_ids: Any) -> None:
    from astro_weights.runners.hf import swap

    a = make_edit(tiny_llama)
    other = "model.layers.0.mlp.down_proj"
    b = second_edit(tiny_llama, other)
    base = logits(tiny_llama, input_ids)
    fresh = logits(apply(copy.deepcopy(tiny_llama), b), input_ids)

    peft_model = swap(apply(tiny_llama, a), b)
    assert torch.allclose(logits(peft_model, input_ids), fresh, atol=1e-6)
    names = lora_modules(peft_model)
    assert len(names) == 1
    assert names[0].endswith(f"{other}.lora_A.astro")

    model = remove(peft_model)
    assert not lora_modules(model)
    assert torch.equal(logits(model, input_ids), base)


def test_merged_weights_match_dense(tiny_llama: Any) -> None:
    from astro_weights.runners.hf import merged_weights

    edit = make_edit(tiny_llama)
    before = {k: v.clone() for k, v in tiny_llama.state_dict().items()}
    merged = merged_weights(tiny_llama, edit)
    w0 = tiny_llama.get_submodule(MODULE).weight
    assert list(merged) == [MODULE]
    assert torch.allclose(
        merged[MODULE], w0 + torch.from_numpy(edit.deltas[0].dense()), atol=1e-6
    )
    after = tiny_llama.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_merged_weights_other_layout(tiny_qwen_text: Any) -> None:
    from astro_weights.runners.hf import merged_weights

    edit = renamed(edit_for(tiny_qwen_text, TEXT), VL)
    merged = merged_weights(tiny_qwen_text, edit)
    w0 = tiny_qwen_text.get_submodule(TEXT).weight
    assert list(merged) == [TEXT]
    assert torch.allclose(
        merged[TEXT], w0 + torch.from_numpy(edit.deltas[0].dense()), atol=1e-6
    )


def test_merged_weights_rejects_peft_model(tiny_llama: Any) -> None:
    from astro_weights.runners.hf import merged_weights

    edit = make_edit(tiny_llama)
    with pytest.raises(ValueError, match="plain base model"):
        merged_weights(apply(tiny_llama, edit), edit)
