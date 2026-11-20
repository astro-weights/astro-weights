import json
from collections import Counter
from pathlib import Path

import pytest

from astro_weights.eval import Record
from astro_weights.eval.splits import SPLITS, SplitManifest, assign_splits

SHA = "ab" * 32


def _ids(n: int, prefix: str = "item") -> list[str]:
    return [f"{prefix}-{i}" for i in range(n)]


def _rec(item_id: str) -> Record:
    return Record(item_id, 0, "base", "h", "out", 1.0, 3, False)


def _manifest(n: int = 200, seed: int = 3) -> SplitManifest:
    return SplitManifest.build(
        _ids(n), dataset="synthetic", revision="r1", file_sha256=SHA, seed=seed
    )


# ---------------------------------------------------------------- assign_splits


def test_adding_and_removing_items_never_moves_others() -> None:
    full = assign_splits(_ids(2000), seed=11)
    fewer = assign_splits(_ids(2000)[::3], seed=11)
    more = assign_splits([*_ids(2000), *_ids(500, "new")], seed=11)
    assert all(full[i] == s for i, s in fewer.items())
    assert all(more[i] == s for i, s in full.items())
    assert assign_splits(reversed(_ids(2000)), seed=11) == full


def test_fractions_within_two_points_on_10k() -> None:
    for fractions in [(0.2, 0.3, 0.5), (0.1, 0.1, 0.8)]:
        counts = Counter(
            assign_splits(_ids(10_000), seed=0, fractions=fractions).values()
        )
        for split, f in zip(SPLITS, fractions, strict=True):
            assert abs(counts[split] / 10_000 - f) <= 0.02, (split, counts)


def test_seed_changes_the_assignment() -> None:
    a, b = assign_splits(_ids(1000), 1), assign_splits(_ids(1000), 2)
    moved = sum(a[i] != b[i] for i in a)
    assert moved > 300
    assert assign_splits(_ids(1000), 1) == a


def test_zero_fraction_split_is_empty() -> None:
    out = assign_splits(_ids(500), 0, (0.0, 0.0, 1.0))
    assert set(out.values()) == {"test"}


@pytest.mark.parametrize(
    ("fractions", "match"),
    [
        ((0.5, 0.5), "three numbers"),
        ((0.2, 0.3, 0.6), "sum to 1"),
        ((-0.1, 0.6, 0.5), r"not in \[0, 1\]"),
        ((0.2, float("nan"), 0.8), r"not in \[0, 1\]"),
        ((True, 0.0, 0.0), "not a number"),
        ("abc", "three numbers"),
    ],
)
def test_bad_fractions(fractions: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        assign_splits(["a"], 0, fractions)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [True, 1.0, "1", None])
def test_bad_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="seed must be an int"):
        assign_splits(["a"], seed)  # type: ignore[arg-type]


def test_bad_ids() -> None:
    with pytest.raises(ValueError, match="repeated item ids"):
        assign_splits(["a", "b", "a"], 0)
    with pytest.raises(ValueError, match="non-empty string"):
        assign_splits(["a", 3], 0)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="non-empty string"):
        assign_splits([""], 0)


# ---------------------------------------------------------------- SplitManifest


def test_manifest_round_trip(tmp_path: Path) -> None:
    m = _manifest()
    assert sorted([*m.dev, *m.confirm, *m.test]) == sorted(_ids(200))
    assert all(m.ids(s) for s in SPLITS)
    m.save(tmp_path / "m.json")
    back = SplitManifest.load(tmp_path / "m.json")
    assert back == m
    assert back.manifest_sha256 == m.manifest_sha256
    raw = json.loads((tmp_path / "m.json").read_text("utf-8"))
    assert raw["manifest_sha256"] == m.manifest_sha256
    assert raw["dataset"] == "synthetic"
    assert raw["file_sha256"] == SHA


def test_manifest_sha_depends_on_every_field() -> None:
    m = _manifest()
    others = [
        _manifest(seed=4),
        _manifest(n=201),
        SplitManifest.build(
            _ids(200), dataset="other", revision="r1", file_sha256=SHA, seed=3
        ),
        SplitManifest.build(
            _ids(200), dataset="synthetic", revision="r2", file_sha256=SHA, seed=3
        ),
        SplitManifest.build(
            _ids(200), dataset="synthetic", revision="r1", file_sha256="cd" * 32, seed=3
        ),
    ]
    assert len({m.manifest_sha256, *(o.manifest_sha256 for o in others)}) == 6


def test_manifest_ids_are_sorted_so_order_does_not_matter() -> None:
    a = _manifest()
    b = SplitManifest.build(
        list(reversed(_ids(200))),
        dataset="synthetic",
        revision="r1",
        file_sha256=SHA,
        seed=3,
    )
    assert a.manifest_sha256 == b.manifest_sha256


def test_real_dataset_manifest_format(tmp_path: Path) -> None:
    """The pinned datasets of PDR 0.3 (ids generated; nothing downloaded)."""
    data = [
        (
            "HuggingFaceH4/MATH-500 test",
            "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
            "2a8cc0a4b40586dc982be7c50be33a942790f55d9ef0716d53bf4ba8f17be3ef",
            _ids(500, "math500"),
        ),
        (
            "openai/gsm8k main test",
            "740312add88f781978c0658806c59bc2815b9866",
            "c55b65bc330379188022e1d2c8180e434f828b21bd089814e575d7896d92b8c4",
            _ids(1319, "gsm8k"),
        ),
    ]
    for dataset, revision, sha, ids in data:
        m = SplitManifest.build(
            ids, dataset=dataset, revision=revision, file_sha256=sha, seed=0
        )
        path = tmp_path / f"{len(ids)}.json"
        m.save(path)
        back = SplitManifest.load(path)
        assert back == m
        for split, f in zip(SPLITS, m.fractions, strict=True):
            assert abs(len(back.ids(split)) / len(ids) - f) < 0.05
        raw = json.loads(path.read_text("utf-8"))
        assert sorted(raw) == sorted(
            [
                "dataset",
                "file_sha256",
                "format",
                "fractions",
                "manifest_sha256",
                "revision",
                "seed",
                "splits",
            ]
        )
        assert (raw["dataset"], raw["revision"], raw["file_sha256"]) == (
            dataset,
            revision,
            sha,
        )


def _saved(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    path = tmp_path / "m.json"
    _manifest().save(path)
    return path, json.loads(path.read_text("utf-8"))


def _rewrite(path: Path, raw: dict[str, object]) -> None:
    path.write_text(json.dumps(raw), encoding="utf-8")


def test_load_refuses_unknown_and_missing_keys(tmp_path: Path) -> None:
    path, raw = _saved(tmp_path)
    _rewrite(path, raw | {"extra": 1})
    with pytest.raises(ValueError, match=r"unknown keys \['extra'\]"):
        SplitManifest.load(path)
    del raw["seed"]
    _rewrite(path, raw)
    with pytest.raises(ValueError, match=r"missing keys \['seed'\]"):
        SplitManifest.load(path)


def test_load_refuses_tampered_manifest(tmp_path: Path) -> None:
    path, raw = _saved(tmp_path)
    _rewrite(path, raw | {"revision": "r2"})
    with pytest.raises(ValueError, match="manifest_sha256 does not match"):
        SplitManifest.load(path)


def test_load_refuses_moved_id(tmp_path: Path) -> None:
    """Moving a test id into dev is caught even with a recomputed sha."""
    path, raw = _saved(tmp_path)
    splits = raw["splits"]
    assert isinstance(splits, dict)
    splits["dev"].append(splits["test"].pop())
    _rewrite(path, raw)
    with pytest.raises(ValueError, match="belong to another split"):
        SplitManifest.load(path)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"format": 2}, "format must be 1"),
        ({"format": True}, "format must be 1"),
        ({"splits": {"dev": [], "test": []}}, "exactly the keys"),
        ({"splits": []}, "exactly the keys"),
        ({"file_sha256": "ABC"}, "64 lowercase hex"),
        ({"dataset": ""}, "dataset must be a non-empty string"),
        ({"revision": 5}, "revision must be a non-empty string"),
        ({"splits": {"dev": "a", "confirm": [], "test": []}}, "list of item ids"),
        ({"splits": {"dev": ["x"], "confirm": ["x"], "test": []}}, "repeated"),
    ],
)
def test_load_refuses_bad_values(
    tmp_path: Path, change: dict[str, object], match: str
) -> None:
    path, raw = _saved(tmp_path)
    _rewrite(path, raw | change)
    with pytest.raises(ValueError, match=match):
        SplitManifest.load(path)


def test_load_refuses_non_json(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        SplitManifest.load(path)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        SplitManifest.load(path)


def test_check_records() -> None:
    m = _manifest()
    test_ids = sorted(m.test)
    m.check_records([_rec(i) for i in test_ids[:3]], "test")
    m.check_records([_rec(i) for i in test_ids], "test", complete=True)
    with pytest.raises(ValueError, match=r"1 item id\(s\) are not in the test split"):
        m.check_records([_rec(test_ids[0]), _rec(m.dev[0])], "test")
    with pytest.raises(ValueError, match="have no record"):
        m.check_records([_rec(i) for i in test_ids[1:]], "test", complete=True)
    with pytest.raises(ValueError, match="unknown split"):
        m.ids("train")  # type: ignore[arg-type]
