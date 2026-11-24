import copy
import math
from typing import Any

import numpy as np
import pytest

from astro_weights.edits import Edit, LowRankDelta, masked_rank1, top_k_mask

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from astro_weights.runners.guards import kl_last_token, perplexity, perplexity_guard

MODULE = "model.layers.1.mlp.down_proj"
IDS = [[1, 5, 9, 33, 100, 7, 64, 2], [3, 8, 8, 21, 90, 4]]


def real_edit(model: Any) -> Edit:
    w0 = model.get_submodule(MODULE).weight.detach().numpy()
    rng = np.random.default_rng(0)
    u = rng.standard_normal(w0.shape[0]).astype(np.float32)
    mask = top_k_mask(rng.standard_normal(w0.shape[1]), 8)
    return Edit((masked_rank1(MODULE, w0, u, mask, 3.0),), "tiny-llama")


def zero_edit(model: Any) -> Edit:
    w0 = model.get_submodule(MODULE).weight
    r = 1
    d = LowRankDelta(
        MODULE,
        np.zeros((w0.shape[0], r), np.float32),
        np.ones((r, w0.shape[1]), np.float32),
    )
    return Edit((d,), "tiny-llama")


def fwd(model: Any) -> Any:
    with torch.no_grad():
        return [model(input_ids=torch.tensor([i])).logits for i in IDS]


def test_perplexity_matches_hand(tiny_llama: Any) -> None:
    total, n = 0.0, 0
    for ids in IDS:
        with torch.no_grad():
            lg = tiny_llama(input_ids=torch.tensor([ids])).logits[0]
        for t in range(len(ids) - 1):
            total -= float(torch.log_softmax(lg[t].double(), -1)[ids[t + 1]])
            n += 1
    assert math.isclose(perplexity(tiny_llama, IDS), math.exp(total / n), rel_tol=1e-9)


def test_perplexity_refuses_bad_input(tiny_llama: Any) -> None:
    with pytest.raises(ValueError, match="2 tokens"):
        perplexity(tiny_llama, [[1, 2], [3]])
    with pytest.raises(ValueError, match="empty"):
        perplexity(tiny_llama, [])


def test_zero_edit_is_neutral(tiny_llama: Any) -> None:
    kl = kl_last_token(tiny_llama, IDS, zero_edit(tiny_llama))
    assert kl.max == pytest.approx(0.0, abs=1e-9)
    assert kl.passed
    g = perplexity_guard(tiny_llama, IDS, zero_edit(tiny_llama), max_ratio=1.01)
    assert g.ratio == pytest.approx(1.0, abs=1e-6)
    assert g.passed


def test_real_edit_has_kl_and_matches_torch(tiny_llama: Any) -> None:
    edit = real_edit(tiny_llama)
    kl = kl_last_token(tiny_llama, IDS, edit, threshold=1e-12)
    assert kl.mean > 0
    assert not kl.passed
    assert kl.max >= kl.mean
    assert len(kl.per_prompt) == 2
    assert kl.threshold == 1e-12

    ref = copy.deepcopy(tiny_llama)
    with torch.no_grad():
        ref.get_submodule(MODULE).weight += torch.from_numpy(edit.deltas[0].dense())
    ids = IDS[0]
    p = torch.log_softmax(fwd(tiny_llama)[0][0, -1].double(), -1)
    q = torch.log_softmax(
        ref(input_ids=torch.tensor([ids])).logits[0, -1].detach().double(), -1
    )
    hand = torch.nn.functional.kl_div(q, p, log_target=True, reduction="sum")
    assert kl.per_prompt[0] == pytest.approx(float(hand), rel=1e-4)
    assert kl_last_token(tiny_llama, IDS, edit, threshold=1e9).passed


def test_perplexity_guard_changes(tiny_llama: Any) -> None:
    edit = real_edit(tiny_llama)
    g = perplexity_guard(tiny_llama, IDS, edit, max_ratio=1.0)
    assert g.base == pytest.approx(perplexity(tiny_llama, IDS))
    assert g.ratio == pytest.approx(g.edited / g.base)
    assert g.passed == (g.ratio <= 1.0)
    assert g.ratio != pytest.approx(1.0, abs=1e-6)
    for bad in (0.0, float("nan"), -1.0):
        with pytest.raises(ValueError, match="max_ratio"):
            perplexity_guard(tiny_llama, IDS, edit, max_ratio=bad)
    with pytest.raises(ValueError, match="empty"):
        perplexity_guard(tiny_llama, [], edit, max_ratio=1.1)


def test_model_restored(tiny_llama: Any) -> None:
    edit = real_edit(tiny_llama)
    before_w = {k: v.clone() for k, v in tiny_llama.state_dict().items()}
    before = fwd(tiny_llama)
    kl_last_token(tiny_llama, IDS, edit)
    perplexity_guard(tiny_llama, IDS, edit, max_ratio=2.0)
    assert not [n for n, _ in tiny_llama.named_modules() if "lora" in n]
    assert tiny_llama.state_dict().keys() == before_w.keys()
    for k, v in tiny_llama.state_dict().items():
        assert torch.equal(v, before_w[k])
    for a, b in zip(fwd(tiny_llama), before, strict=True):
        assert torch.equal(a, b)


def test_model_restored_when_forward_raises(
    tiny_llama: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import astro_weights.runners.guards as g

    edit = real_edit(tiny_llama)
    before = fwd(tiny_llama)
    real = g._logits
    calls = {"n": 0}

    def boom(model: Any, ids: Any) -> Any:
        calls["n"] += 1
        if calls["n"] > len(IDS):  # base pass done; fail on the edited model
            raise RuntimeError("boom")
        return real(model, ids)

    monkeypatch.setattr(g, "_logits", boom)
    with pytest.raises(RuntimeError, match="boom"):
        kl_last_token(tiny_llama, IDS, edit)
    calls["n"] = 0
    with pytest.raises(RuntimeError, match="boom"):
        perplexity_guard(tiny_llama, IDS, edit, max_ratio=2.0)
    monkeypatch.undo()
    assert not [n for n, _ in tiny_llama.named_modules() if "lora" in n]
    for a, b in zip(fwd(tiny_llama), before, strict=True):
        assert torch.equal(a, b)


def test_kl_refuses_bad_prompts(tiny_llama: Any) -> None:
    e = zero_edit(tiny_llama)
    with pytest.raises(ValueError, match="empty"):
        kl_last_token(tiny_llama, [], e)
    with pytest.raises(ValueError, match="1 token"):
        kl_last_token(tiny_llama, [[1], []], e)
