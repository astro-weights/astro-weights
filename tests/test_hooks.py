from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mcp_tiny_hf import save_tiny_model

from astro_weights.runners.hooks import (
    capture,
    full_attention_layers,
    layer_count,
)

IDS = [[1, 5, 9, 33, 100, 7, 64, 2], [3, 4, 5]]
LAST = 4


def check(a: Any, dtype: Any, shape: tuple[int, ...]) -> None:
    assert a.dtype == dtype
    assert a.shape == shape


def count_hooks(model: Any) -> int:
    return sum(
        len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()
    )


def test_layout_helpers(
    tiny_llama: Any, tiny_qwen_text: Any, tiny_qwen_vl: Any
) -> None:
    assert layer_count(tiny_llama) == 2
    assert full_attention_layers(tiny_llama) == [0, 1]
    assert layer_count(tiny_qwen_text) == 2
    assert full_attention_layers(tiny_qwen_text) == [1]
    assert layer_count(tiny_qwen_vl) == 2
    assert full_attention_layers(tiny_qwen_vl) == [1]
    with pytest.raises(ValueError, match="no decoder layers"):
        layer_count(torch.nn.Linear(2, 2))


@pytest.mark.parametrize("fixture", ["tiny_llama", "tiny_qwen_text", "tiny_qwen_vl"])
def test_hidden_matches_manual_forward(
    fixture: str, request: pytest.FixtureRequest
) -> None:
    model = request.getfixturevalue(fixture)
    out = capture(model, IDS, want=["hidden"], layers=[0, 1], last_tokens=LAST)
    assert set(out) == {"hidden_L0", "hidden_L1"}
    for li in (0, 1):
        a = out[f"hidden_L{li}"]
        check(a, np.float32, (2, 32))
    with torch.no_grad():
        for i, text in enumerate(IDS):
            hs = model(
                input_ids=torch.tensor([text]), output_hidden_states=True
            ).hidden_states
            # hidden_states[l + 1] is layer l's output; the last entry is post-norm
            ref = hs[1][0, -LAST:].float().mean(0).numpy()
            np.testing.assert_allclose(out["hidden_L0"][i], ref, atol=1e-5)


def test_mlp_matches_manual_hook(tiny_qwen_text: Any) -> None:
    got: list[Any] = []
    h = tiny_qwen_text.model.layers[1].mlp.down_proj.register_forward_pre_hook(
        lambda _m, a: got.append(a[0][0, -LAST:].float().mean(0).numpy())
    )
    with torch.no_grad():
        for text in IDS:
            tiny_qwen_text(input_ids=torch.tensor([text]))
    h.remove()
    out = capture(tiny_qwen_text, IDS, want=["mlp"], layers=[1], last_tokens=LAST)
    check(out["mlp_L1"], np.float32, (2, 64))
    np.testing.assert_allclose(out["mlp_L1"], np.stack(got), atol=1e-6)


def test_logprobs(tiny_llama: Any) -> None:
    out = capture(tiny_llama, IDS, want=["logprobs"], last_tokens=LAST)
    assert set(out) == {"logprobs_top_ids", "logprobs_top", "logprobs_target"}
    ids, top, tgt = out["logprobs_top_ids"], out["logprobs_top"], out["logprobs_target"]
    check(ids, np.int64, (2, LAST, 5))
    check(top, np.float32, (2, LAST, 5))
    check(tgt, np.float32, (2, LAST))
    with torch.no_grad():
        full = torch.log_softmax(
            tiny_llama(input_ids=torch.tensor([IDS[0]])).logits[0].float(), -1
        )
    ref = full[-LAST:]
    np.testing.assert_allclose(top[0], ref.topk(5, -1).values.numpy(), atol=1e-5)
    assert (ids[0] == ref.topk(5, -1).indices.numpy()).all()
    assert (np.diff(top[0], axis=-1) <= 0).all()
    want = [
        full[len(IDS[0]) - LAST + k, IDS[0][len(IDS[0]) - LAST + k + 1]]
        for k in range(LAST - 1)
    ]
    np.testing.assert_allclose(tgt[0, : LAST - 1], np.array(want), atol=1e-5)
    assert np.isnan(tgt[0, LAST - 1])


def test_short_text_padding(tiny_llama: Any) -> None:
    out = capture(tiny_llama, [[3, 4, 5]], want=["logprobs"], last_tokens=6)
    ids, top, tgt = (
        out["logprobs_top_ids"][0],
        out["logprobs_top"][0],
        out["logprobs_target"][0],
    )
    assert (ids[:3] >= 0).all()
    assert (ids[3:] == -1).all()
    assert np.isfinite(top[:3]).all()
    assert np.isnan(top[3:]).all()
    assert np.isfinite(tgt[:2]).all()
    assert np.isnan(tgt[2:]).all()


def test_single_token_text(tiny_llama: Any) -> None:
    out = capture(tiny_llama, [[3]], want=["logprobs"], last_tokens=3)
    assert np.isnan(out["logprobs_target"]).all()
    assert out["logprobs_top_ids"][0, 0, 0] >= 0


def test_attn_full_layer(tiny_qwen_text: Any) -> None:
    out = capture(tiny_qwen_text, IDS, want=["attn"], layers=[1])
    assert set(out) == {"attn_L1_t0", "attn_L1_t1"}
    w = out["attn_L1_t0"]
    check(w, np.float32, (4, 8, 8))
    np.testing.assert_allclose(w.sum(-1), 1.0, atol=1e-5)
    assert out["attn_L1_t1"].shape == (4, 3, 3)


def test_attn_linear_layer_raises(tiny_qwen_text: Any) -> None:
    with pytest.raises(ValueError, match="no attention weights"):
        capture(tiny_qwen_text, IDS, want=["attn"], layers=[0])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"input_ids": [], "want": ["hidden"]}, "non-empty"),
        ({"input_ids": [[]], "want": ["hidden"]}, "non-empty"),
        ({"input_ids": IDS, "want": ["hidden"], "layers": [2]}, "out of range"),
        ({"input_ids": IDS, "want": ["hidden"], "layers": [-1]}, "out of range"),
        ({"input_ids": IDS, "want": ["bogus"], "layers": [0]}, "unknown want"),
    ],
)
def test_value_errors(tiny_llama: Any, kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        capture(tiny_llama, **kwargs)


@contextmanager
def fail_on_forward(model: Any) -> Iterator[None]:
    def boom(*_a: Any) -> None:
        raise RuntimeError("boom")

    h = model.lm_head.register_forward_hook(boom)
    try:
        yield
    finally:
        h.remove()


def test_cleanup_after_exception(tiny_qwen_text: Any) -> None:
    model = tiny_qwen_text
    before = count_hooks(model)
    impl = model.config._attn_implementation
    training = model.training
    with fail_on_forward(model):
        base = count_hooks(model)
        with pytest.raises(RuntimeError, match="boom"):
            capture(model, IDS, want=["hidden", "mlp", "attn"], layers=[1])
        assert count_hooks(model) == base
    assert count_hooks(model) == before
    assert model.config._attn_implementation == impl
    assert model.training == training
    # and a later capture still works
    capture(model, IDS, want=["attn"], layers=[1])
    assert model.config._attn_implementation == impl


def test_save_tiny_model_round_trip(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    from astro_weights.mcp.edit_store import EditStore

    d = save_tiny_model(tmp_path / "model")
    model = transformers.AutoModelForCausalLM.from_pretrained(d)
    tok = transformers.AutoTokenizer.from_pretrained(d)
    ids = tok("w1 w2 w3", add_special_tokens=False).input_ids
    assert ids == [4, 5, 6]
    out = capture(model.eval(), [ids], want=["hidden"], layers=[0])
    assert out["hidden_L0"].shape == (1, 32)
    store = EditStore(tmp_path / "run", d)
    assert store.names
    assert all(n.endswith(".weight") for n in store.names)
