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
    LEDGER,
    SealedError,
    check_unsealed,
    read_seal,
    record_scoring,
    seal_key,
)
from astro_weights.eval.splits import SplitManifest
from astro_weights.export import load_peft, save_peft

M1, M2 = "model.layers.0.mlp.down_proj", "model.layers.1.self_attn.q_proj"
K1, K2 = "a" * 64, "b" * 64
MAN = "c" * 64


def _edit(seed: int = 0, rank: int = 2, **kw: Any) -> Edit:
    rng = np.random.default_rng(seed)
    deltas = (
        LowRankDelta(
            M1, rng.standard_normal((6, rank)), rng.standard_normal((rank, 5))
        ),
        LowRankDelta(
            M2, rng.standard_normal((4, rank)), rng.standard_normal((rank, 6))
        ),
    )
    return Edit(deltas, kw.pop("base_model", "tiny"), **kw)


# ---------------------------------------------------------------- seal_key


def test_key_ignores_names_order_padding_meta(tmp_path: Path) -> None:
    e = _edit()
    k = seal_key(e)
    assert len(k) == 64
    save_peft(e, tmp_path / "one")
    shutil.copytree(tmp_path / "one", tmp_path / "renamed")
    assert seal_key(load_peft(tmp_path / "one")) == k
    assert seal_key(load_peft(tmp_path / "renamed")) == k
    assert seal_key(replace(e, meta={"run_id": "other", "name": "x"})) == k
    assert seal_key(replace(e, deltas=e.deltas[::-1])) == k
    assert (
        seal_key(Edit.from_deltas([*e.deltas[:1], e.deltas[1].padded(4)], "tiny")) == k
    )
    d = e.deltas[0]
    swapped = LowRankDelta(d.module, d.B[:, ::-1], d.A[::-1, :])
    assert seal_key(replace(e, deltas=(swapped, e.deltas[1]))) == k
    zero = LowRankDelta("model.layers.2.mlp.up_proj", np.zeros((3, 2)), np.ones((2, 3)))
    assert seal_key(replace(e, deltas=(*e.deltas, zero))) == k


def test_key_changes_with_content() -> None:
    e = _edit()
    k = seal_key(e)
    d = e.deltas[0]
    b = d.B.copy()
    b[0, 0] = np.nextafter(b[0, 0], np.float32(np.inf))
    assert seal_key(replace(e, deltas=(LowRankDelta(M1, b, d.A), e.deltas[1]))) != k
    rescaled = LowRankDelta(M1, d.B * 2, d.A / 2)
    assert seal_key(replace(e, deltas=(rescaled, e.deltas[1]))) != k
    renamed = LowRankDelta("model.layers.9.mlp.down_proj", d.B, d.A)
    assert seal_key(replace(e, deltas=(renamed, e.deltas[1]))) != k
    assert seal_key(_edit(seed=1)) != k
    assert seal_key(replace(e, base_model="other")) != k


def test_key_negative_zero_is_zero() -> None:
    b = np.array([[1.0, 0.0], [0.0, 1.0]])
    a = np.array([[0.0, 2.0], [3.0, 0.0]])
    pos = Edit((LowRankDelta(M1, b, a),), "tiny")
    neg = Edit((LowRankDelta(M1, b * -0.0 + b, a),), "tiny")
    neg_b = b.copy()
    neg_b[0, 1] = -0.0
    assert seal_key(Edit((LowRankDelta(M1, neg_b, a),), "tiny")) == seal_key(pos)
    assert seal_key(neg) == seal_key(pos)


def test_key_base_sha_wins_over_base_name() -> None:
    sha = {M1 + ".weight": "d" * 64}
    e = _edit(base_sha256=sha)
    assert seal_key(replace(e, base_model="renamed")) == seal_key(e)
    assert seal_key(replace(e, base_sha256={M1 + ".weight": "e" * 64})) != seal_key(e)


@dataclass(frozen=True)
class _D:
    module: str
    B: np.ndarray
    A: np.ndarray


@dataclass(frozen=True)
class _E:
    deltas: tuple[_D, ...]
    base_model: str = "tiny"
    base_sha256: dict[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_sha256", self.base_sha256 or {})


def test_key_refuses_bad_duck_edits() -> None:
    ok = np.ones((2, 2))
    with pytest.raises(ValueError, match="duplicate module"):
        seal_key(_E((_D(M1, ok, ok), _D(M1, ok, ok))))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite 2-D"):
        seal_key(_E((_D(M1, ok * np.inf, ok),)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite 2-D"):
        seal_key(_E((_D(M1, np.ones(2), ok),)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rank mismatch"):
        seal_key(_E((_D(M1, np.ones((2, 3)), ok),)))  # type: ignore[arg-type]


# ---------------------------------------------------------------- ledger


def test_second_scoring_refused_unseal_recorded(tmp_path: Path) -> None:
    check_unsealed(tmp_path / "seal", K1, MAN)  # no ledger yet: unsealed
    first = record_scoring(tmp_path / "seal", K1, MAN, note="first")
    assert first["scoring"] == 1
    assert first["unseal"] is False
    with pytest.raises(SealedError, match="already scored 1 time"):
        check_unsealed(tmp_path / "seal", K1, MAN)
    with pytest.raises(SealedError, match="--unseal"):
        record_scoring(tmp_path / "seal", K1, MAN)
    record_scoring(tmp_path / "seal", K2, MAN)  # other edit
    record_scoring(tmp_path / "seal", K1, "d" * 64)  # other test manifest
    again = record_scoring(tmp_path / "seal", K1, MAN, unseal=True, note="why")
    assert (again["scoring"], again["unseal"], again["note"]) == (2, True, "why")
    lines = read_seal(tmp_path / "seal")
    assert [(e["key"][:1], e["scoring"], e["unseal"]) for e in lines] == [
        ("a", 1, False),
        ("b", 1, False),
        ("a", 1, False),
        ("a", 2, True),
    ]
    assert lines[0]["prev"] == ""
    assert all(e["prev"] for e in lines[1:])


def test_record_refuses_bad_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="key must be a sha256"):
        record_scoring(tmp_path, "edit-name", MAN)
    with pytest.raises(ValueError, match="manifest_sha256 must be"):
        record_scoring(tmp_path, K1, "short")


def test_torn_tail_is_cut_earlier_lines_kept(tmp_path: Path) -> None:
    record_scoring(tmp_path, K1, MAN)
    path = tmp_path / LEDGER
    before = path.read_bytes()
    with path.open("ab") as f:
        f.write(b'{"key": "' + K2.encode()[:20])  # crash mid-write
    assert len(read_seal(tmp_path)) == 1  # reading leaves the file alone
    check_unsealed(tmp_path, K2, MAN)
    record_scoring(tmp_path, K2, MAN)
    data = path.read_bytes()
    assert data.startswith(before)
    assert [e["key"] for e in read_seal(tmp_path)] == [K1, K2]


def test_line_missing_only_its_newline_is_kept(tmp_path: Path) -> None:
    record_scoring(tmp_path, K1, MAN)
    path = tmp_path / LEDGER
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    assert len(read_seal(tmp_path)) == 1
    with pytest.raises(SealedError):
        record_scoring(tmp_path, K1, MAN)
    record_scoring(tmp_path, K2, MAN)
    assert [e["key"] for e in read_seal(tmp_path)] == [K1, K2]


def test_edited_earlier_line_breaks_the_chain(tmp_path: Path) -> None:
    record_scoring(tmp_path, K1, MAN)
    record_scoring(tmp_path, K2, MAN)
    path = tmp_path / LEDGER
    lines = path.read_bytes().split(b"\n")
    lines[0] = lines[0].replace(b'"note":""', b'"note":"x"')
    path.write_bytes(b"\n".join(lines))
    with pytest.raises(ValueError, match="chain broken"):
        read_seal(tmp_path)
    path.write_bytes(b"\n".join(lines[1:]))  # first line deleted
    with pytest.raises(ValueError, match="chain broken"):
        record_scoring(tmp_path, K1, MAN)


@pytest.mark.parametrize(
    "line",
    [
        b"not json",
        b"[]",
        b'{"key": "x"}',
        json.dumps(
            {
                "key": K1,
                "manifest_sha256": MAN,
                "scoring": True,
                "unseal": False,
                "note": "",
                "time": "t",
                "prev": "",
            }
        ).encode(),
    ],
)
def test_malformed_line_raises(tmp_path: Path, line: bytes) -> None:
    (tmp_path / LEDGER).write_bytes(line + b"\n")
    with pytest.raises(ValueError, match=r"seal.jsonl:1"):
        read_seal(tmp_path)


def _race(directory: str, barrier: Any, out: Any) -> None:
    barrier.wait()
    try:
        record_scoring(directory, K1, MAN, note="race")
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


def _rec(item_id: str, arm: str, score: float) -> Record:
    return Record(item_id, 0, arm, "h-" + item_id, "out", score, 5, False)


@pytest.fixture
def setup(tmp_path: Path) -> dict[str, Path]:
    m = SplitManifest.build(
        [f"q{i}" for i in range(60)],
        dataset="synthetic",
        revision="r1",
        file_sha256="ab" * 32,
        seed=5,
    )
    m.save(tmp_path / "splits.json")
    for split in ("dev", "test"):
        ids = sorted(m.ids(split))
        write_jsonl(tmp_path / f"{split}-base.jsonl", [_rec(i, "base", 0) for i in ids])
        write_jsonl(tmp_path / f"{split}-edit.jsonl", [_rec(i, "edit", 1) for i in ids])
    save_peft(_edit(), tmp_path / "adapter")
    return {
        "dir": tmp_path,
        "manifest": tmp_path / "splits.json",
        "seal": tmp_path / "seal",
        "adapter": tmp_path / "adapter",
    }


def _test_args(
    s: dict[str, Path], *extra: str, adapter: Path | None = None
) -> list[str]:
    d = s["dir"]
    return [
        "eval",
        str(d / "test-base.jsonl"),
        str(d / "test-edit.jsonl"),
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


def test_cli_test_split_scored_once(
    setup: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(_test_args(setup)) == 0
    assert "n=" in capsys.readouterr().out
    lines = read_seal(setup["seal"])
    assert len(lines) == 1
    assert lines[0]["key"] == seal_key(load_peft(setup["adapter"]))
    assert (
        lines[0]["manifest_sha256"]
        == SplitManifest.load(setup["manifest"]).manifest_sha256
    )

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


def test_cli_changed_factor_is_a_new_edit(
    setup: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(_test_args(setup)) == 0
    e = _edit()
    d = e.deltas[0]
    b = d.B.copy()
    b[0, 0] += 0.5
    save_peft(
        replace(e, deltas=(LowRankDelta(M1, b, d.A), e.deltas[1])), setup["dir"] / "a2"
    )
    assert main(_test_args(setup, adapter=setup["dir"] / "a2")) == 0
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
    args = _test_args(setup)
    args[1:3] = [str(d / "dev-base.jsonl"), str(d / "dev-edit.jsonl")]
    assert main(args) == 1
    assert "not in the test split" in capsys.readouterr().err
    assert not (setup["seal"] / LEDGER).exists()

    recs = read_jsonl(d / "test-base.jsonl")[1:]
    write_jsonl(d / "part-base.jsonl", recs)
    args[1] = str(d / "part-base.jsonl")
    args[2] = str(d / "test-edit.jsonl")
    assert main(args) == 1
    assert "have no record" in capsys.readouterr().err


def test_cli_dev_split_needs_no_seal(
    setup: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    d = setup["dir"]
    base, edit = str(d / "dev-base.jsonl"), str(d / "dev-edit.jsonl")
    common = ["eval", base, edit, "--boot", "200", "--manifest", str(setup["manifest"])]
    assert main([*common, "--split", "dev"]) == 0
    assert (
        main([*common, "--split", "dev"]) == 0
    )  # dev may be scored any number of times
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
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str], match: str
) -> None:
    assert main(["eval", "b.jsonl", "e.jsonl", *extra]) == 2
    assert match in capsys.readouterr().err


def test_cli_bad_manifest_adapter_or_ledger(
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
