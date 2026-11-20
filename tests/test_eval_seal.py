import base64
import json
import multiprocessing as mp
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from astro_weights.cli import main
from astro_weights.edits import Edit, LowRankDelta
from astro_weights.eval import Record, read_jsonl, write_jsonl
from astro_weights.eval.seal import (
    FINGERPRINT_TOL,
    LEDGER,
    PROBES,
    Scoring,
    SealedError,
    _distance,
    check_unsealed,
    fingerprint,
    read_seal,
    record_scoring,
    records_sha256,
    scoring_of,
    seal_key,
)
from astro_weights.eval.splits import SplitManifest
from astro_weights.export import load_peft, save_peft

M1, M2 = "model.layers.0.mlp.down_proj", "model.layers.1.self_attn.q_proj"
SHA = "ab" * 32


def _edit(seed: int = 0, rank: int = 3, **kw: Any) -> Edit:
    rng = np.random.default_rng(seed)
    deltas = (
        LowRankDelta(
            M1, rng.standard_normal((48, rank)), rng.standard_normal((rank, 40))
        ),
        LowRankDelta(
            M2, rng.standard_normal((32, rank)), rng.standard_normal((rank, 48))
        ),
    )
    return Edit(deltas, kw.pop("base_model", "tiny"), **kw)


def _map(e: Edit, fn: Any) -> Edit:
    """Apply fn(B, A) -> (B', A') to every delta."""
    return replace(
        e, deltas=tuple(LowRankDelta(d.module, *fn(d.B, d.A)) for d in e.deltas)
    )


def _rotation(r: int, seed: int = 9) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((r, r)))
    return q


def _rotate(e: Edit) -> Edit:
    q = _rotation(e.rank)
    return _map(e, lambda b, a: (b @ q, q.T @ a))


def _rescale(e: Edit, c: float = 2.0) -> Edit:
    return _map(e, lambda b, a: (b * c, a / c))


# ---------------------------------------------------------------- seal_key


def test_key_ignores_names_order_padding_meta_base(tmp_path: Path) -> None:
    e = _edit()
    k = seal_key(e)
    assert len(k) == 64
    save_peft(e, tmp_path / "one")
    shutil.copytree(tmp_path / "one", tmp_path / "renamed")
    assert seal_key(load_peft(tmp_path / "one")) == k
    assert seal_key(load_peft(tmp_path / "renamed")) == k
    assert seal_key(replace(e, meta={"run_id": "other", "name": "x"})) == k
    assert seal_key(replace(e, base_model="other-base")) == k
    assert seal_key(replace(e, base_sha256={M1 + ".weight": "e" * 64})) == k
    assert seal_key(replace(e, deltas=e.deltas[::-1])) == k
    padded = Edit.from_deltas([*e.deltas[:1], e.deltas[1].padded(5)], "tiny")
    assert seal_key(padded) == k
    d = e.deltas[0]
    swapped = LowRankDelta(d.module, d.B[:, ::-1], d.A[::-1, :])
    assert seal_key(replace(e, deltas=(swapped, e.deltas[1]))) == k
    zero = LowRankDelta("model.layers.2.mlp.up_proj", np.zeros((3, 3)), np.ones((3, 3)))
    assert seal_key(replace(e, deltas=(*e.deltas, zero))) == k


def test_key_changes_with_factors() -> None:
    e = _edit()
    k = seal_key(e)
    d = e.deltas[0]
    b = d.B.copy()
    b[0, 0] = np.nextafter(b[0, 0], np.float32(np.inf))
    assert seal_key(replace(e, deltas=(LowRankDelta(M1, b, d.A), e.deltas[1]))) != k
    assert seal_key(_rescale(e)) != k  # same B @ A: the fingerprint catches it
    assert seal_key(_rotate(e)) != k
    renamed = LowRankDelta("model.layers.9.mlp.down_proj", d.B, d.A)
    assert seal_key(replace(e, deltas=(renamed, e.deltas[1]))) != k
    assert seal_key(_edit(seed=1)) != k


def test_key_negative_zero_is_zero() -> None:
    b = np.array([[1.0, 0.0], [0.0, 1.0]])
    a = np.array([[0.0, 2.0], [3.0, 0.0]])
    neg_b = b.copy()
    neg_b[0, 1] = -0.0
    pos = Edit((LowRankDelta(M1, b, a),), "tiny")
    assert seal_key(Edit((LowRankDelta(M1, neg_b, a),), "tiny")) == seal_key(pos)


@dataclass(frozen=True)
class _D:
    module: str
    B: np.ndarray
    A: np.ndarray


@dataclass(frozen=True)
class _E:
    deltas: tuple[_D, ...]


def test_key_refuses_bad_duck_edits() -> None:
    ok = np.ones((2, 2))
    with pytest.raises(ValueError, match="duplicate module"):
        seal_key(_E((_D(M1, ok, ok), _D(M1, ok, ok))))
    with pytest.raises(ValueError, match="finite 2-D"):
        seal_key(_E((_D(M1, ok * np.inf, ok),)))
    with pytest.raises(ValueError, match="finite 2-D"):
        seal_key(_E((_D(M1, np.ones(2), ok),)))
    with pytest.raises(ValueError, match="rank mismatch"):
        fingerprint(_E((_D(M1, np.ones((2, 3)), ok),)))


# ---------------------------------------------------------------- fingerprint


def test_fingerprint_depends_on_dw_only() -> None:
    e = _edit()
    f = fingerprint(e)
    assert sorted(f) == [M1, M2]
    assert all(v.shape == (PROBES, PROBES) for v in f.values())
    assert _distance(f, fingerprint(_rescale(e, 2.0))) < 1e-6
    assert _distance(f, fingerprint(_rescale(e, 3.7))) < 1e-5
    assert _distance(f, fingerprint(_rotate(e))) < 1e-5
    padded = Edit.from_deltas([*e.deltas[:1], e.deltas[1].padded(6)], "tiny")
    assert _distance(f, fingerprint(padded)) == 0.0
    assert _distance(f, fingerprint(replace(e, base_model="other"))) == 0.0


def test_fingerprint_moves_with_the_edit() -> None:
    e = _edit()
    f = fingerprint(e)
    near = _map(e, lambda b, a: (b * 1.0005, a))
    assert _distance(f, fingerprint(near)) <= FINGERPRINT_TOL
    far = _map(e, lambda b, a: (b * 1.01, a))
    assert _distance(f, fingerprint(far)) > FINGERPRINT_TOL
    assert _distance(f, fingerprint(_edit(seed=1))) > 0.5
    one = replace(e, deltas=e.deltas[:1])  # a module dropped counts as zero
    assert _distance(f, fingerprint(one)) > FINGERPRINT_TOL
    assert _distance({}, {}) == 0.0


# ---------------------------------------------------------------- ledger


def _manifest(seed: int = 3, k: int = 4, revision: str = "r1") -> SplitManifest:
    return SplitManifest.build(
        [f"q{i}" for i in range(60)],
        dataset="synthetic",
        revision=revision,
        file_sha256=SHA,
        seed=seed,
        k=k,
    )


def _records(
    m: SplitManifest, tag: str = "out", seeds: range = range(4), arm: str = "edit"
) -> list[Record]:
    return [
        Record(i, s, arm, "h-" + i, f"{tag}-{i}-{s}", 1.0, 5, False)
        for i in sorted(m.test)
        for s in seeds
    ]


def _sc(
    e: Edit | None = None, m: SplitManifest | None = None, tag: str = "out"
) -> Scoring:
    m = m or _manifest()
    return scoring_of(e or _edit(), m, _records(m, tag))


def test_scoring_of() -> None:
    m = _manifest()
    s = _sc(m=m)
    assert s.key == seal_key(_edit())
    assert s.test_ids_sha256 == m.test_ids_sha256
    assert s.manifest_sha256 == m.manifest_sha256
    assert s.records_sha256 == records_sha256(_records(m))
    assert s.prompt_hashes == tuple(sorted("h-" + i for i in m.test))
    assert s.k == 4
    assert records_sha256(_records(m)[::-1]) == s.records_sha256
    assert records_sha256(_records(m, "other")) != s.records_sha256


def test_second_scoring_refused_unseal_recorded(tmp_path: Path) -> None:
    s = _sc()
    check_unsealed(tmp_path / "seal", s)  # no ledger yet: unsealed
    first = record_scoring(tmp_path / "seal", s, note="first")
    assert (first["scoring"], first["unseal"], first["k"]) == (1, False, 4)
    with pytest.raises(SealedError, match="same edit-arm records"):
        check_unsealed(tmp_path / "seal", s)
    with pytest.raises(SealedError, match="--unseal"):
        record_scoring(tmp_path / "seal", s)
    again = record_scoring(tmp_path / "seal", s, unseal=True, note="why")
    assert (again["scoring"], again["unseal"], again["note"]) == (2, True, "why")
    lines = read_seal(tmp_path / "seal")
    assert [(e["scoring"], e["unseal"]) for e in lines] == [(1, False), (2, True)]
    assert lines[0]["prev"] == ""
    assert lines[1]["prev"]
    assert _distance(lines[0]["fingerprint"], s.fingerprint) == 0.0
    assert lines[0]["prompt_hashes"] == list(s.prompt_hashes)


def test_many_reasons_are_cut_short(tmp_path: Path) -> None:
    s = _sc()
    for _ in range(4):
        record_scoring(tmp_path, s, unseal=True)
    with pytest.raises(SealedError, match=r"; \.\.\."):
        check_unsealed(tmp_path, s)


def test_manifest_change_keeping_test_ids_stays_sealed(tmp_path: Path) -> None:
    """Review probe: a retyped revision gave a fresh manifest_sha256 and a rescore."""
    m1, m2 = _manifest(revision="740312ad"), _manifest(revision="740312add88f")
    assert m1.test_ids_sha256 == m2.test_ids_sha256
    record_scoring(tmp_path, _sc(m=m1))
    with pytest.raises(SealedError, match="scored on this test set"):
        record_scoring(tmp_path, _sc(m=m2, tag="new-outputs"))


def test_overlapping_prompts_stay_sealed(tmp_path: Path) -> None:
    m1, m2 = _manifest(seed=3), _manifest(seed=4)
    assert m1.test_ids_sha256 != m2.test_ids_sha256
    assert set(m1.test) & set(m2.test)
    record_scoring(tmp_path, _sc(m=m1))
    with pytest.raises(SealedError, match=r"scored on \d+ of these prompts"):
        record_scoring(tmp_path, _sc(m=m2, tag="new"))


def test_disjoint_test_set_is_allowed(tmp_path: Path) -> None:
    record_scoring(tmp_path, _sc())
    other = SplitManifest.build(
        [f"z{i}" for i in range(60)],
        dataset="synthetic",
        revision="r1",
        file_sha256=SHA,
        seed=3,
    )
    record_scoring(tmp_path, _sc(m=other))
    assert len(read_seal(tmp_path)) == 2


def test_same_records_under_another_adapter_stay_sealed(tmp_path: Path) -> None:
    record_scoring(tmp_path, _sc())
    with pytest.raises(SealedError, match="same edit-arm records"):
        record_scoring(tmp_path, _sc(_edit(seed=7)))


@pytest.mark.parametrize(
    "variant",
    [
        lambda e: _rescale(e, 2.0),
        lambda e: _rescale(e, 0.3),
        _rotate,
        lambda e: replace(e, base_model="other-base"),
        lambda e: _map(e, lambda b, a: (b * 1.0005, a)),
    ],
    ids=["rescale-2", "rescale-0.3", "rotation", "base-name", "within-tol"],
)
def test_same_dw_stays_sealed(tmp_path: Path, variant: Any) -> None:
    record_scoring(tmp_path, _sc())
    with pytest.raises(SealedError, match="scored on this test set"):
        record_scoring(tmp_path, _sc(variant(_edit()), tag="fresh-outputs"))


def test_edit_changed_beyond_tol_is_new(tmp_path: Path) -> None:
    record_scoring(tmp_path, _sc())
    record_scoring(tmp_path, _sc(_map(_edit(), lambda b, a: (b * 1.01, a)), tag="x"))
    record_scoring(tmp_path, _sc(_edit(seed=1), tag="y"))
    assert len(read_seal(tmp_path)) == 3


def test_scoring_validation(tmp_path: Path) -> None:
    s = _sc()
    with pytest.raises(ValueError, match="key must be a sha256"):
        replace(s, key="edit-name")
    with pytest.raises(ValueError, match="test_ids_sha256 must be"):
        replace(s, test_ids_sha256="short")
    with pytest.raises(ValueError, match="k must be an int"):
        replace(s, k=0)
    with pytest.raises(ValueError, match="must be finite 4x4"):
        replace(s, fingerprint={M1: np.ones((2, 2))})
    with pytest.raises(TypeError, match="must be a Scoring"):
        record_scoring(tmp_path, "a" * 64)  # type: ignore[arg-type]


def test_torn_tail_is_cut_earlier_lines_kept(tmp_path: Path) -> None:
    record_scoring(tmp_path, _sc())
    path = tmp_path / LEDGER
    before = path.read_bytes()
    with path.open("ab") as f:
        f.write(b'{"key": "' + b"b" * 20)  # crash mid-write
    assert len(read_seal(tmp_path)) == 1  # reading leaves the file alone
    other = _sc(_edit(seed=5), tag="other")
    check_unsealed(tmp_path, other)
    record_scoring(tmp_path, other)
    assert path.read_bytes().startswith(before)
    assert len(read_seal(tmp_path)) == 2


def test_line_missing_only_its_newline_is_kept(tmp_path: Path) -> None:
    s = _sc()
    record_scoring(tmp_path, s)
    path = tmp_path / LEDGER
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    assert len(read_seal(tmp_path)) == 1
    with pytest.raises(SealedError):
        record_scoring(tmp_path, s)
    record_scoring(tmp_path, _sc(_edit(seed=5), tag="other"))
    assert len(read_seal(tmp_path)) == 2


def test_edited_earlier_line_breaks_the_chain(tmp_path: Path) -> None:
    record_scoring(tmp_path, _sc())
    record_scoring(tmp_path, _sc(_edit(seed=5), tag="other"))
    path = tmp_path / LEDGER
    lines = path.read_bytes().split(b"\n")
    lines[0] = lines[0].replace(b'"note":""', b'"note":"x"')
    path.write_bytes(b"\n".join(lines))
    with pytest.raises(ValueError, match="chain broken"):
        read_seal(tmp_path)
    path.write_bytes(b"\n".join(lines[1:]))  # first line deleted
    with pytest.raises(ValueError, match="chain broken"):
        record_scoring(tmp_path, _sc())


def _line(**change: Any) -> bytes:
    good = {
        "key": "a" * 64,
        "fingerprint": {M1: base64.b64encode(bytes(8 * PROBES * PROBES)).decode()},
        "test_ids_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "records_sha256": "d" * 64,
        "prompt_hashes": ["h"],
        "k": 4,
        "scoring": 1,
        "unseal": False,
        "note": "",
        "time": "t",
        "prev": "",
    }
    return json.dumps(good | change).encode()


def test_well_formed_line_reads(tmp_path: Path) -> None:
    (tmp_path / LEDGER).write_bytes(_line() + b"\n")
    assert read_seal(tmp_path)[0]["k"] == 4


@pytest.mark.parametrize(
    ("line", "match"),
    [
        (b"not json", "invalid JSON"),
        (b"[]", "not a seal line"),
        (b'{"key": "x"}', "not a seal line"),
        (_line(scoring=True), "wrong value type"),
        (_line(k="4"), "wrong value type"),
        (_line(prompt_hashes=[1]), "wrong value type"),
        (_line(fingerprint={M1: 3}), "wrong value type"),
        (_line(fingerprint={M1: "!!"}), "bad fingerprint"),
        (_line(fingerprint={M1: "AAAA"}), "bad fingerprint"),
    ],
)
def test_malformed_line_raises(tmp_path: Path, line: bytes, match: str) -> None:
    (tmp_path / LEDGER).write_bytes(line + b"\n")
    with pytest.raises(ValueError, match=rf"seal.jsonl:1: .*{match}"):
        read_seal(tmp_path)


def _race(directory: str, barrier: Any, out: Any) -> None:
    barrier.wait()
    try:
        record_scoring(directory, _sc(), note="race")
        out.put("won")
    except SealedError:
        out.put("sealed")


def test_racing_processes_exactly_one_wins(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    n = 6
    barrier, out = ctx.Barrier(n), ctx.Queue()
    procs = [
        ctx.Process(target=_race, args=(str(tmp_path), barrier, out)) for _ in range(n)
    ]
    for p in procs:
        p.start()
    results = [out.get(timeout=60) for _ in range(n)]
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    assert sorted(results) == ["sealed"] * (n - 1) + ["won"]
    assert len(read_seal(tmp_path)) == 1


# ---------------------------------------------------------------- CLI


@pytest.fixture
def setup(tmp_path: Path) -> dict[str, Path]:
    m = _manifest()
    m.save(tmp_path / "splits.json")
    write_jsonl(tmp_path / "test-base.jsonl", _records(m, "base", arm="base"))
    write_jsonl(tmp_path / "test-edit.jsonl", _records(m))
    for arm in ("base", "edit"):
        write_jsonl(
            tmp_path / f"dev-{arm}.jsonl",
            [Record(i, 0, arm, "h-" + i, "o", 1.0, 5, False) for i in sorted(m.dev)],
        )
    save_peft(_edit(), tmp_path / "adapter")
    return {
        "dir": tmp_path,
        "manifest": tmp_path / "splits.json",
        "seal": tmp_path / "seal",
        "adapter": tmp_path / "adapter",
    }


def _test_args(
    s: dict[str, Path],
    *extra: str,
    adapter: Path | None = None,
    edit: Path | None = None,
    base: Path | None = None,
) -> list[str]:
    d = s["dir"]
    return [
        "eval",
        str(base or d / "test-base.jsonl"),
        str(edit or d / "test-edit.jsonl"),
        "--split",
        "test",
        "--manifest",
        str(s["manifest"]),
        "--seal-dir",
        str(s["seal"]),
        "--adapter",
        str(adapter or s["adapter"]),
        "--boot",
        "200",
        *extra,
    ]


def _new_outputs(s: dict[str, Path], tag: str) -> Path:
    path = s["dir"] / f"edit-{tag}.jsonl"
    write_jsonl(path, _records(SplitManifest.load(s["manifest"]), tag))
    return path


def test_cli_test_split_scored_once(
    setup: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(_test_args(setup)) == 0
    assert "n=" in capsys.readouterr().out
    lines = read_seal(setup["seal"])
    assert len(lines) == 1
    m = SplitManifest.load(setup["manifest"])
    assert lines[0]["key"] == seal_key(load_peft(setup["adapter"]))
    assert lines[0]["test_ids_sha256"] == m.test_ids_sha256
    assert lines[0]["k"] == 4
    assert "base_model='tiny'" in lines[0]["note"]

    assert main(_test_args(setup, "--json")) == 1
    cap = capsys.readouterr()
    assert cap.out == ""  # the second result is never shown
    assert "already scored" in cap.err

    moved = setup["dir"] / "renamed-adapter"
    shutil.copytree(setup["adapter"], moved)
    assert main(_test_args(setup, adapter=moved)) == 1
    assert "already scored" in capsys.readouterr().err
    assert len(read_seal(setup["seal"])) == 1

    assert main(_test_args(setup, "--unseal")) == 0
    cap = capsys.readouterr()
    assert "n=" in cap.out
    assert "--unseal" in cap.err
    last = read_seal(setup["seal"])[-1]
    assert (last["unseal"], last["scoring"]) == (True, 2)


def test_cli_wrong_adapter_same_records_refused(
    setup: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(_test_args(setup)) == 0
    save_peft(_edit(seed=8), setup["dir"] / "other")
    assert main(_test_args(setup, adapter=setup["dir"] / "other")) == 1
    assert "same edit-arm records" in capsys.readouterr().err


@pytest.mark.parametrize(
    "variant",
    [_rescale, _rotate, lambda e: replace(e, base_model="other-base")],
    ids=["rescale", "rotation", "base-name"],
)
def test_cli_same_dw_new_outputs_refused(
    setup: dict[str, Path], capsys: pytest.CaptureFixture[str], variant: Any
) -> None:
    assert main(_test_args(setup)) == 0
    save_peft(variant(_edit()), setup["dir"] / "variant")
    args = _test_args(
        setup, adapter=setup["dir"] / "variant", edit=_new_outputs(setup, "v")
    )
    assert main(args) == 1
    assert "scored on this test set" in capsys.readouterr().err


def test_cli_changed_factor_is_a_new_edit(
    setup: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(_test_args(setup)) == 0
    e = _edit()
    d = e.deltas[0]
    b = d.B.copy()
    b[0, :] += 0.5
    save_peft(
        replace(e, deltas=(LowRankDelta(M1, b, d.A), e.deltas[1])),
        setup["dir"] / "a2",
    )
    args = _test_args(
        setup, adapter=setup["dir"] / "a2", edit=_new_outputs(setup, "a2")
    )
    assert main(args) == 0
    assert len(read_seal(setup["seal"])) == 2
    capsys.readouterr()


def test_cli_lost_race_shows_nothing(
    setup: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another process records between the early check and the ledger write."""

    def late(*args: Any, **kw: Any) -> dict[str, Any]:
        record_scoring(*args, **kw)
        return record_scoring(*args, **kw)

    monkeypatch.setattr("astro_weights.cli.record_scoring", late)
    assert main(_test_args(setup)) == 1
    cap = capsys.readouterr()
    assert cap.out == ""
    assert "already scored" in cap.err


def test_cli_refuses_records_outside_split(
    setup: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    d = setup["dir"]
    args = _test_args(setup, base=d / "dev-base.jsonl", edit=d / "dev-edit.jsonl")
    assert main(args) == 1
    assert "not in the test split" in capsys.readouterr().err
    assert not (setup["seal"] / LEDGER).exists()

    write_jsonl(d / "part-base.jsonl", read_jsonl(d / "test-base.jsonl")[4:])
    assert main(_test_args(setup, base=d / "part-base.jsonl")) == 1
    assert "have no record" in capsys.readouterr().err


def test_cli_test_split_needs_k_seeds(
    setup: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    d, m = setup["dir"], SplitManifest.load(setup["manifest"])
    write_jsonl(d / "k1.jsonl", _records(m, seeds=range(1)))
    assert main(_test_args(setup, edit=d / "k1.jsonl")) == 1
    assert "k=4" in capsys.readouterr().err
    write_jsonl(d / "s47.jsonl", _records(m, seeds=range(4, 8)))
    assert main(_test_args(setup, edit=d / "s47.jsonl")) == 1
    assert "different seeds" in capsys.readouterr().err
    assert not (setup["seal"] / LEDGER).exists()


def test_cli_floor_arms_checked_against_split(
    setup: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Review probe: --floor EDIT_test BASE_test leaked the test diff on --split dev."""
    d = setup["dir"]
    dev = [
        "eval",
        str(d / "dev-base.jsonl"),
        str(d / "dev-edit.jsonl"),
        "--boot",
        "200",
        "--split",
        "dev",
        "--manifest",
        str(setup["manifest"]),
    ]
    floor = ["--floor", str(d / "test-edit.jsonl"), str(d / "test-base.jsonl")]
    assert main([*dev, *floor]) == 1
    assert "not in the dev split" in capsys.readouterr().err
    write_jsonl(d / "part.jsonl", read_jsonl(d / "test-base.jsonl")[4:])
    part = ["--floor", str(d / "part.jsonl"), str(d / "test-base.jsonl")]
    assert main(_test_args(setup, *part)) == 1
    assert "have no record" in capsys.readouterr().err
    both = str(d / "test-base.jsonl")
    assert main(_test_args(setup, "--floor", both, both)) == 0
    assert "floor" in capsys.readouterr().out.lower()


def test_cli_dev_split_needs_no_seal(
    setup: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    d = setup["dir"]
    base, edit = str(d / "dev-base.jsonl"), str(d / "dev-edit.jsonl")
    common = ["eval", base, edit, "--boot", "200", "--manifest", str(setup["manifest"])]
    assert main([*common, "--split", "dev"]) == 0
    assert main([*common, "--split", "dev"]) == 0  # dev: any number of times
    assert main([*common, "--split", "confirm"]) == 1
    assert "not in the confirm split" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["--seal-dir", "S"], "--seal-dir need --split"),
        (["--unseal", "--adapter", "A"], "--adapter, --unseal need --split"),
        (["--split", "dev"], "--split needs --manifest"),
        (["--split", "test", "--manifest", "M"], "needs --seal-dir and --adapter"),
        (
            ["--split", "dev", "--manifest", "M", "--unseal"],
            "apply only to --split test",
        ),
    ],
)
def test_cli_flag_misuse_exits_2(
    capsys: pytest.CaptureFixture[str], extra: list[str], match: str
) -> None:
    assert main(["eval", "b.jsonl", "e.jsonl", *extra]) == 2
    assert match in capsys.readouterr().err


def test_cli_bad_manifest(
    setup: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    setup["manifest"].write_text("{}", encoding="utf-8")
    assert main(_test_args(setup)) == 2
    assert "missing keys" in capsys.readouterr().err


def test_cli_bad_adapter(
    setup: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(_test_args(setup, adapter=setup["dir"] / "nope")) == 2
    assert "nope" in capsys.readouterr().err


def test_cli_broken_ledger_exits_2(
    setup: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    setup["seal"].mkdir()
    (setup["seal"] / LEDGER).write_bytes(b"junk\n")
    assert main(_test_args(setup)) == 2
    assert "invalid JSON line" in capsys.readouterr().err
    assert main(_test_args(setup, "--unseal")) == 2  # skips the early check
    assert "invalid JSON line" in capsys.readouterr().err
