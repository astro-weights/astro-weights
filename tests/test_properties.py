"""Property tests for the numeric core: each compares against a direct float64 formula."""

import copy
import itertools
import math
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from astro_weights.directions import direction_from_means
from astro_weights.edits import (
    Edit,
    LowRankDelta,
    column_shift,
    from_dense,
    masked_rank1,
    project_out,
    rank1,
    scale_column,
    scale_head,
    top_k_mask,
    zero_head,
)
from astro_weights.eval import PairedResult, Record, compare
from astro_weights.export import load_peft, save_peft, verify_peft

PROP = settings(max_examples=200, deadline=None)
MODULE = "model.layers.1.mlp.down_proj"
BOUND = 1e3
RTOL = 1e-5  # float32 factors and a float32 matmul, with headroom

dims = st.integers(1, 12)


def vec(n: int, bound: float = BOUND) -> st.SearchStrategy[np.ndarray]:
    return hnp.arrays(
        np.float32,
        n,
        elements=st.floats(
            -bound, bound, width=32, allow_nan=False, allow_subnormal=False
        ),
    )


def mat(m: int, n: int, bound: float = BOUND) -> st.SearchStrategy[np.ndarray]:
    return hnp.arrays(
        np.float32,
        (m, n),
        elements=st.floats(
            -bound, bound, width=32, allow_nan=False, allow_subnormal=False
        ),
    )


scales = st.floats(-10, 10, allow_nan=False, allow_subnormal=False)


@st.composite
def w0_and_dims(draw: st.DrawFn) -> np.ndarray:
    return draw(mat(draw(dims), draw(dims)))


def assert_close(got: np.ndarray, want: np.ndarray, scale: float) -> None:
    """Elementwise match within RTOL of the largest magnitude involved."""
    np.testing.assert_allclose(got, want, rtol=0, atol=RTOL * max(scale, 1e-30))


def check_recipe(w0: np.ndarray, delta: LowRankDelta, want: np.ndarray) -> None:
    """W0 + dense equals the docstring formula `want` (float64)."""
    got = w0.astype(np.float64) + delta.dense().astype(np.float64)
    scale = max(float(np.abs(w0).max()), float(np.abs(want).max()))
    assert_close(got, want, scale)


# --- 1. recipes ------------------------------------------------------------


@PROP
@given(data=st.data())
def test_rank1_formula(data: st.DataObject) -> None:
    m, n = data.draw(dims), data.draw(dims)
    w0, b, a = data.draw(mat(m, n)), data.draw(vec(m, 30)), data.draw(vec(n, 30))
    check_recipe(
        w0,
        rank1(MODULE, b, a),
        w0.astype(np.float64) + np.outer(b.astype(np.float64), a.astype(np.float64)),
    )


@PROP
@given(data=st.data(), alpha=st.floats(-2, 2, allow_nan=False, allow_subnormal=False))
def test_masked_rank1_formula(data: st.DataObject, alpha: float) -> None:
    m, n = data.draw(dims), data.draw(dims)
    w0, u = data.draw(mat(m, n, 30)), data.draw(vec(m, 30))
    mask = top_k_mask(data.draw(vec(n)), data.draw(st.integers(0, n)))
    w = w0.astype(np.float64) - alpha * np.outer(
        u.astype(np.float64), mask * (u.astype(np.float64) @ w0.astype(np.float64))
    )
    check_recipe(w0, masked_rank1(MODULE, w0, u, mask, alpha), w)


@PROP
@given(data=st.data())
def test_column_shift_formula(data: st.DataObject) -> None:
    m, n = data.draw(dims), data.draw(dims)
    w0, shift = data.draw(mat(m, n, 30)), data.draw(vec(m, 30))
    idx = np.array(
        sorted(data.draw(st.sets(st.integers(0, n - 1), max_size=n))), dtype=np.intp
    )
    coef = data.draw(vec(len(idx), 30))
    c = np.zeros(n)
    c[idx] = coef
    w = w0.astype(np.float64) + np.outer(shift.astype(np.float64), c)
    check_recipe(w0, column_shift(MODULE, shift, idx, coef, n), w)


@PROP
@given(
    data=st.data(), strength=st.floats(-2, 2, allow_nan=False, allow_subnormal=False)
)
def test_project_out_formula(data: st.DataObject, strength: float) -> None:
    m, n = data.draw(dims), data.draw(dims)
    w0, v = data.draw(mat(m, n, 30)), data.draw(vec(m, 30))
    norm = float(np.linalg.norm(v.astype(np.float64)))
    assume(norm > 1e-3)
    vhat = v.astype(np.float64) / norm
    w = w0.astype(np.float64) - strength * np.outer(vhat, vhat @ w0.astype(np.float64))
    check_recipe(w0, project_out(MODULE, w0, v, strength), w)


@PROP
@given(data=st.data())
def test_project_out_removes_direction_and_is_idempotent(data: st.DataObject) -> None:
    m, n = data.draw(dims), data.draw(dims)
    w0, v = data.draw(mat(m, n, 30)), data.draw(vec(m, 30))
    norm = float(np.linalg.norm(v.astype(np.float64)))
    assume(norm > 1e-3)
    vhat = v.astype(np.float64) / norm
    delta = project_out(MODULE, w0, v, 1.0)
    w1 = w0 + delta.dense()  # float32, as the edited layer would hold it
    tol = RTOL * max(float(np.abs(w0).max()), 1.0) * m
    assert np.abs(vhat @ w1.astype(np.float64)).max() <= tol
    again = project_out(MODULE, w1, v, 1.0)
    w2 = w1 + again.dense()
    assert np.abs(w2.astype(np.float64) - w1.astype(np.float64)).max() <= tol


@PROP
@given(data=st.data(), s=scales)
def test_scale_column_formula(data: st.DataObject, s: float) -> None:
    m, n = data.draw(dims), data.draw(dims)
    w0, j = data.draw(mat(m, n)), data.draw(st.integers(0, n - 1))
    w = w0.astype(np.float64)
    w[:, j] *= s
    check_recipe(w0, scale_column(MODULE, w0, j, s), w)


@st.composite
def head_case(draw: st.DrawFn) -> tuple[np.ndarray, int, int]:
    n_heads, size, m = draw(st.integers(1, 4)), draw(st.integers(1, 4)), draw(dims)
    return draw(mat(m, n_heads * size)), draw(st.integers(0, n_heads - 1)), n_heads


@PROP
@given(case=head_case(), s=scales)
def test_scale_head_formula(case: tuple[np.ndarray, int, int], s: float) -> None:
    w0, head, n_heads = case
    size = w0.shape[1] // n_heads
    w = w0.astype(np.float64)
    w[:, head * size : (head + 1) * size] *= s
    check_recipe(w0, scale_head(MODULE, w0, head, n_heads, s), w)


@PROP
@given(case=head_case())
def test_zero_head_formula(case: tuple[np.ndarray, int, int]) -> None:
    w0, head, n_heads = case
    size = w0.shape[1] // n_heads
    w = w0.astype(np.float64)
    w[:, head * size : (head + 1) * size] = 0.0
    check_recipe(w0, zero_head(MODULE, w0, head, n_heads), w)


# --- 2. padding and Edit.from_deltas ---------------------------------------


@st.composite
def low_rank(draw: st.DrawFn, name: str = MODULE) -> LowRankDelta:
    m, n, r = draw(dims), draw(dims), draw(st.integers(1, 4))
    return LowRankDelta(name, draw(mat(m, r, 30)), draw(mat(r, n, 30)))


def dense_close(got: np.ndarray, want: np.ndarray) -> None:
    scale = float(np.abs(want).max())
    assert_close(got.astype(np.float64), want.astype(np.float64), scale)


@PROP
@given(delta=low_rank(), extra=st.integers(0, 4))
def test_padded_keeps_dense(delta: LowRankDelta, extra: int) -> None:
    padded = delta.padded(delta.rank + extra)
    assert padded.rank == delta.rank + extra
    assert padded.shape == delta.shape
    dense_close(padded.dense(), delta.dense())


@PROP
@given(
    deltas=st.lists(low_rank(), min_size=1, max_size=4).map(
        lambda ds: [
            LowRankDelta(f"model.layers.{i}.mlp.down_proj", d.B, d.A)
            for i, d in enumerate(ds)
        ]
    )
)
def test_from_deltas_keeps_every_dense(deltas: list[LowRankDelta]) -> None:
    edit = Edit.from_deltas(deltas, "org/base")
    assert edit.rank == max(d.rank for d in deltas)
    assert [d.module for d in edit.deltas] == [d.module for d in deltas]
    for got, want in zip(edit.deltas, deltas, strict=True):
        dense_close(got.dense(), want.dense())


# --- 3. from_dense ---------------------------------------------------------


@PROP
@given(data=st.data())
def test_from_dense_residual_monotone_and_bounded(data: st.DataObject) -> None:
    m, n = data.draw(dims), data.draw(dims)
    dw = data.draw(mat(m, n, 30))
    norm = float(np.linalg.norm(dw.astype(np.float64)))
    assume(norm > 1e-3)
    s = np.linalg.svd(dw.astype(np.float64), compute_uv=False)
    rels = []
    for rank in range(1, min(m, n) + 1):
        delta, rel = from_dense(MODULE, dw, rank)
        rels.append(rel)
        # Eckart-Young: the truncated-SVD residual is the tail of the spectrum
        tail = math.sqrt(float((s[rank:] ** 2).sum())) / norm
        assert abs(rel - tail) <= 1e-5
        # the float32 dense differs from the float64 measurement by rounding only
        err = np.linalg.norm(dw.astype(np.float64) - delta.dense().astype(np.float64))
        assert err <= (rel + 1e-5) * norm
    # float32 rounding of the factors can wobble the residual by ~1e-7
    assert all(b <= a + 1e-6 for a, b in itertools.pairwise(rels))
    assert rels[-1] <= 1e-5  # full rank reproduces the input


# --- 4. PEFT round trip ----------------------------------------------------


@st.composite
def peft_edit(draw: st.DrawFn) -> Edit:
    r, n_mod = draw(st.integers(1, 3)), draw(st.integers(1, 3))
    deltas = []
    for i in range(n_mod):
        m, n = draw(dims), draw(dims)
        deltas.append(
            LowRankDelta(
                f"model.layers.{i}.mlp.down_proj", draw(mat(m, r)), draw(mat(r, n))
            )
        )
    return Edit(tuple(deltas), "org/base")


def safetensors_data_start(path: Path) -> int:
    header_len = int.from_bytes(path.read_bytes()[:8], "little")
    return 8 + header_len


@PROP
@given(edit=peft_edit())
def test_peft_roundtrip_bit_identical(edit: Edit) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        save_peft(edit, tmp)
        assert verify_peft(tmp) == []
        loaded = load_peft(tmp)
    want = {d.module: d for d in edit.deltas}
    assert {d.module for d in loaded.deltas} == set(want)
    for d in loaded.deltas:
        assert np.array_equal(d.B, want[d.module].B)
        assert np.array_equal(d.A, want[d.module].A)


@PROP
@given(edit=peft_edit(), data=st.data())
def test_peft_any_flipped_data_byte_is_detected(
    edit: Edit, data: st.DataObject
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        save_peft(edit, tmp)
        path = Path(tmp) / "adapter_model.safetensors"
        raw = bytearray(path.read_bytes())
        start = safetensors_data_start(path)
        pos = data.draw(st.integers(start, len(raw) - 1))
        raw[pos] ^= data.draw(st.integers(1, 255))
        path.write_bytes(bytes(raw))
        assert verify_peft(tmp) != []
        with pytest.raises(ValueError, match=r"."):
            load_peft(tmp)


# --- 5. GGUF adapter -------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(edit=peft_edit())
def test_gguf_adapter_roundtrip(edit: Edit) -> None:
    pytest.importorskip("gguf")
    from astro_weights.export.gguf import (
        gguf_tensor_name,
        read_lora_gguf,
        save_lora_gguf,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "a.gguf"
        save_lora_gguf(edit, path, "qwen35")
        tensors, meta = read_lora_gguf(path)
    assert meta["alpha"] == float(edit.rank)
    assert set(tensors) == {gguf_tensor_name(d.module) for d in edit.deltas}
    for d in edit.deltas:
        b, a = tensors[gguf_tensor_name(d.module)]
        scale = np.float32(meta["alpha"] / a.shape[0])
        assert np.array_equal(b, d.B)
        assert np.array_equal(a, d.A)
        assert np.array_equal(np.asarray(scale * (b @ a), np.float32), d.dense())


# --- 6. eval ---------------------------------------------------------------


@st.composite
def arms(draw: st.DrawFn) -> tuple[list[Record], list[Record]]:
    n_items, n_seeds = draw(st.integers(1, 12)), draw(st.integers(1, 3))
    score = (
        st.sampled_from([0.0, 1.0])
        if draw(st.booleans())
        else st.floats(0, 1, allow_nan=False)
    )
    tokens, caps = st.integers(0, 500), st.booleans()

    def rec(arm: str, i: int, seed: int) -> Record:
        return Record(
            f"item{i}", seed, arm, f"hash{i}", "", draw(score), draw(tokens), draw(caps)
        )

    keys = [(i, s) for i in range(n_items) for s in range(n_seeds)]
    return (
        [rec("base", i, s) for i, s in keys],
        [rec("edit", i, s) for i, s in keys],
    )


def flat(r: PairedResult) -> list[object]:
    """Fields of a result with NaN mapped to a sentinel so equality works."""
    out: list[object] = []
    for v in asdict(r).values():
        for x in v if isinstance(v, tuple) else (v,):
            out.append("nan" if isinstance(x, float) and math.isnan(x) else x)
    return out


NBOOT = 300


@PROP
@given(pair_of_arms=arms(), data=st.data())
def test_compare_ignores_record_order(
    pair_of_arms: tuple[list[Record], list[Record]], data: st.DataObject
) -> None:
    base, edit = pair_of_arms
    shuffled_base = data.draw(st.permutations(base))
    shuffled_edit = data.draw(st.permutations(edit))
    want = compare(base, edit, n_boot=NBOOT)
    assert flat(compare(shuffled_base, shuffled_edit, n_boot=NBOOT)) == flat(want)


@PROP
@given(pair_of_arms=arms())
def test_compare_swap_negates_diff(
    pair_of_arms: tuple[list[Record], list[Record]],
) -> None:
    base, edit = pair_of_arms
    fwd, rev = compare(base, edit, n_boot=NBOOT), compare(edit, base, n_boot=NBOOT)
    assert rev.diff == pytest.approx(-fwd.diff, abs=1e-12)
    assert rev.tokens_diff == pytest.approx(-fwd.tokens_diff, abs=1e-9)
    assert rev.mcnemar_p == fwd.mcnemar_p
    assert (rev.n_base_only, rev.n_edit_only) == (fwd.n_edit_only, fwd.n_base_only)
    assert (rev.mean_base, rev.mean_edit) == (fwd.mean_edit, fwd.mean_base)


@PROP
@given(pair_of_arms=arms())
def test_compare_token_ci_contains_observed_diff(
    pair_of_arms: tuple[list[Record], list[Record]],
) -> None:
    r = compare(*pair_of_arms, n_boot=NBOOT)
    lo, hi = r.tokens_ci95
    eps = 1e-9 * max(1.0, abs(r.tokens_diff))
    assert lo - eps <= r.tokens_diff <= hi + eps


@PROP
@given(pair_of_arms=arms())
def test_compare_identical_arms_have_zero_diff(
    pair_of_arms: tuple[list[Record], list[Record]],
) -> None:
    base, _ = pair_of_arms
    same = [Record(**{**asdict(r), "arm": "edit"}) for r in base]
    r = compare(base, same, n_boot=NBOOT)
    assert r.diff == 0.0
    assert r.tokens_diff == 0.0
    assert r.tokens_ci95 == (0.0, 0.0)
    assert (r.n_base_only, r.n_edit_only) == (0, 0)
    assert r.mcnemar_p in (None, 1.0)


# --- 7. directions ---------------------------------------------------------


@PROP
@given(data=st.data())
def test_direction_unit_norm_and_antisymmetric(data: st.DataObject) -> None:
    n = data.draw(dims)
    a, b = data.draw(vec(n)), data.draw(vec(n))
    assume(not np.array_equal(a, b))
    u, flipped = direction_from_means(a, b), direction_from_means(b, a)
    assert u.dtype == np.float32
    assert abs(float(np.linalg.norm(u.astype(np.float64))) - 1.0) <= 1e-6
    assert np.array_equal(flipped, -u)


# --- 8. runners/hf ---------------------------------------------------------


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(seed=st.integers(0, 2**32 - 1), rank=st.integers(1, 3))
def test_hf_apply_adds_x_dense_and_remove_restores(
    tiny_llama: Any, input_ids: Any, seed: int, rank: int
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from astro_weights.runners.hf import apply, remove

    model = copy.deepcopy(tiny_llama)
    layer = model.get_submodule(MODULE)
    out_f, in_f = layer.weight.shape
    rng = np.random.default_rng(seed)
    delta = LowRankDelta(
        MODULE,
        rng.uniform(-1, 1, (out_f, rank)),
        rng.uniform(-1, 1, (rank, in_f)),
    )
    x = torch.from_numpy(rng.uniform(-1, 1, (5, in_f)).astype(np.float32))
    with torch.no_grad():
        before = layer(x)
        logits_before = model(input_ids).logits

    peft_model = apply(model, Edit((delta,), "tiny"))
    name = next(n for n, _ in peft_model.named_modules() if n.endswith(f"{MODULE}"))
    with torch.no_grad():
        after = peft_model.get_submodule(name)(x)
    want = x @ torch.from_numpy(delta.dense()).T
    assert torch.allclose(after - before, want, atol=1e-4, rtol=1e-4)

    restored = remove(peft_model)
    with torch.no_grad():
        assert torch.equal(restored.get_submodule(MODULE)(x), before)
        assert torch.equal(restored(input_ids).logits, logits_before)
