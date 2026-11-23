from typing import Any

import numpy as np
import pytest

from astro_weights.directions import direction_from_means
from astro_weights.edits import Edit, project_out

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from astro_weights.runners.hf import apply
from astro_weights.runners.hooks import capture
from astro_weights.runners.scripted_arm import scripted_arm

A = [[1, 5, 9, 33, 100, 7, 64, 2], [3, 8, 8, 21, 90, 4, 60, 2]]
B = [[2, 2, 40, 41, 42, 43, 44, 45], [9, 70, 71, 72, 73, 74, 6, 1]]
MODS = ["model.layers.0.mlp.down_proj", "model.layers.1.self_attn.o_proj"]


def weight(model: Any, name: str) -> np.ndarray:
    w: np.ndarray = model.get_submodule(name).weight.detach().float().numpy()
    return w


def like_of(model: Any, mods: list[str]) -> Edit:
    rng = np.random.default_rng(0)
    return Edit(
        tuple(
            project_out(m, weight(model, m), rng.standard_normal(32).astype(np.float32))
            for m in mods
        ),
        "tiny",
        {"k": "v"},
    )


def direction(model: Any, layer: int) -> np.ndarray:
    ma, mb = (
        capture(model, ids, want={"hidden"}, layers=[layer])[f"hidden_L{layer}"].mean(0)
        for ids in (A, B)
    )
    return direction_from_means(ma, mb)


@pytest.mark.parametrize("fixture", ["tiny_llama", "tiny_qwen_text"])
def test_matches_hand_computation(fixture: str, request: pytest.FixtureRequest) -> None:
    model = request.getfixturevalue(fixture)
    like = like_of(model, [MODS[0], "model.layers.1.mlp.down_proj"])
    out = scripted_arm(model, A, B, like, strength=0.7)
    assert [d.module for d in out.deltas] == [d.module for d in like.deltas]
    assert (out.base_model, out.base_sha256) == ("tiny", {"k": "v"})
    for d in out.deltas:
        layer = int(d.module.split(".")[2])
        want = project_out(
            d.module, weight(model, d.module), direction(model, layer), 0.7
        )
        np.testing.assert_allclose(d.B @ d.A, want.B @ want.A, atol=1e-6)


def test_output_orthogonal_to_direction(tiny_llama: Any) -> None:
    like = like_of(tiny_llama, MODS)
    edit = scripted_arm(tiny_llama, A, B, like)
    dirs = {m: direction(tiny_llama, int(m.split(".")[2])) for m in MODS}
    peft = apply(tiny_llama, edit)
    seen: dict[str, float] = {}

    def hook(m: str) -> Any:
        def f(_mod: Any, _args: Any, out: Any) -> None:
            comp = out[0].detach().numpy() @ dirs[m]
            seen[m] = max(seen.get(m, 0.0), float(np.abs(comp).max()))

        return f

    mods = {n.removeprefix("base_model.model."): x for n, x in peft.named_modules()}
    hooks = [mods[m].register_forward_hook(hook(m)) for m in MODS]
    with torch.no_grad():
        peft(torch.tensor(A))
    for h in hooks:
        h.remove()
    assert set(seen) == set(MODS)
    assert max(seen.values()) < 1e-4  # float32 round-off


def test_refuses_non_writer(tiny_llama: Any) -> None:
    like = like_of(tiny_llama, ["model.layers.0.self_attn.q_proj"])
    with pytest.raises(ValueError, match=r"layers\.0\.self_attn\.q_proj.*residual"):
        scripted_arm(tiny_llama, A, B, like)


def test_refuses_layer_out_of_range_and_empty(tiny_llama: Any) -> None:
    w = np.zeros((32, 64), np.float32)
    far = Edit(
        (project_out("model.layers.5.mlp.down_proj", w, np.ones(32, np.float32)),), "t"
    )
    with pytest.raises(ValueError, match=r"no nn\.Linear"):
        scripted_arm(tiny_llama, A, B, far)
    like = like_of(tiny_llama, MODS[:1])
    for a, b in (([], B), (A, []), ([[]], B)):
        with pytest.raises(ValueError, match=r"non-empty|at least one"):
            scripted_arm(tiny_llama, a, b, like)


def test_deterministic(tiny_llama: Any) -> None:
    like = like_of(tiny_llama, MODS)
    x, y = (scripted_arm(tiny_llama, A, B, like) for _ in range(2))
    for p, q in zip(x.deltas, y.deltas, strict=True):
        assert np.array_equal(p.B, q.B)
        assert np.array_equal(p.A, q.A)
