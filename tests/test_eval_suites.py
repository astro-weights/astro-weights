import builtins
import hashlib
import shutil
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, Self

import pytest

from astro_weights.eval import suites
from astro_weights.eval.suites import PINS, SuiteData, load_suite, suite_grader

DATA = Path(__file__).parent / "data" / "suites"
FILES = {
    "gsm8k": "gsm8k.parquet",
    "ifeval": "ifeval.jsonl",
    "mmlu_pro": "mmlu_pro.parquet",
}
KEYS = {"item_id", "prompt", "gold"}


def _sha(name: str) -> str:
    return hashlib.sha256((DATA / FILES[name]).read_bytes()).hexdigest()


@pytest.fixture
def fixture_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the tiny fixture files instead of the real data."""
    pins = {n: replace(PINS[n], sha256=_sha(n)) for n in PINS}
    monkeypatch.setattr(suites, "PINS", pins)


def test_pins_are_complete() -> None:
    assert set(PINS) == {"gsm8k", "ifeval", "mmlu_pro"}
    for pin in PINS.values():
        assert len(pin.sha256) == 64
        assert len(pin.revision) == 40
    assert PINS["gsm8k"].revision == "740312add88f781978c0658806c59bc2815b9866"
    assert (PINS["gsm8k"].rows, PINS["ifeval"].rows) == (1319, 541)


def test_gsm8k(fixture_pins: None) -> None:
    pytest.importorskip("pyarrow")
    data = load_suite("gsm8k", DATA / FILES["gsm8k"])
    assert data.verified
    assert data.sha256 == _sha("gsm8k")
    assert [i["item_id"] for i in data.items] == ["gsm8k-0", "gsm8k-1"]
    assert [i["gold"] for i in data.items] == ["5", "999"]
    assert data.items[0]["prompt"] == (
        "Ann has 2 apples and buys 3 more. How many?\n\n"
        "Put your final answer in \\boxed{}."
    )
    assert set(data.items[0]) == KEYS


def test_mmlu_pro(fixture_pins: None) -> None:
    pytest.importorskip("pyarrow")
    items = load_suite("mmlu_pro", DATA / FILES["mmlu_pro"]).items
    assert [i["item_id"] for i in items] == [
        "mmlu_pro-70",
        "mmlu_pro-71",
        "mmlu_pro-72",
    ]
    assert [i["gold"] for i in items] == ["B", "D", "A"]
    assert items[1]["prompt"] == (
        "Q two?\n\nA. red\nB. green\nC. blue\nD. cyan\n\n"
        "Answer with the letter of the correct option in \\boxed{}."
    )
    assert items[2]["category"] == "law"
    assert KEYS <= set(items[0])


def test_ifeval(fixture_pins: None) -> None:
    items = load_suite("ifeval", DATA / FILES["ifeval"]).items
    assert [i["item_id"] for i in items] == ["ifeval-1000", "ifeval-1001"]
    assert items[0]["prompt"] == "Write a note without commas."
    assert (
        items[1]["gold"] == "detectable_format:number_bullet_lists,startend:end_checker"
    )
    assert items[1]["kwargs"] == [{"num_bullets": 3}, {"end_phrase": "Done"}]


def test_hash_mismatch_refused_with_both_hashes(tmp_path: Path) -> None:
    f = tmp_path / "ifeval.jsonl"
    shutil.copy(DATA / FILES["ifeval"], f)
    with pytest.raises(ValueError, match="expected") as info:
        load_suite("ifeval", f)
    assert _sha("ifeval") in str(info.value)
    assert PINS["ifeval"].sha256 in str(info.value)


def test_verify_false_records_actual_hash(tmp_path: Path) -> None:
    f = tmp_path / "ifeval.jsonl"
    shutil.copy(DATA / FILES["ifeval"], f)
    data = load_suite("ifeval", f, verify=False)
    assert isinstance(data, SuiteData)
    assert not data.verified
    assert data.sha256 == _sha("ifeval")
    assert data.source == str(f)
    assert len(data.items) == 2


def test_unknown_suite() -> None:
    with pytest.raises(ValueError, match="unknown suite"):
        load_suite("nope")
    with pytest.raises(ValueError, match="unknown suite"):
        suite_grader("nope")


def test_download_uses_the_pin_and_caches(
    fixture_pins: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake(repo: str, revision: str, filename: str, dest: Path) -> None:
        calls.append((repo, revision, filename))
        shutil.copy(DATA / FILES["ifeval"], dest)

    monkeypatch.setattr(suites, "_download", fake)
    data = load_suite("ifeval", cache_dir=tmp_path)
    pin = PINS["ifeval"]
    assert calls == [(pin.repo, pin.revision, pin.filename)]
    assert data.verified
    assert data.source == f"{pin.repo}@{pin.revision}:{pin.filename}"
    assert len(data.items) == 2
    load_suite("ifeval", cache_dir=tmp_path)
    assert len(calls) == 1  # second load comes from the cache
    assert not list(tmp_path.rglob("*.part"))


def test_download_with_wrong_hash_is_refused_and_not_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake(repo: str, revision: str, filename: str, dest: Path) -> None:
        dest.write_bytes(b"not the dataset")

    monkeypatch.setattr(suites, "_download", fake)
    with pytest.raises(ValueError, match="downloaded sha256"):
        load_suite("ifeval", cache_dir=tmp_path)
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_default_cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert suites._cache_root(None) == tmp_path / "astro-weights" / "suites"


def test_download_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[str] = []

    class Resp:
        def __init__(self) -> None:
            self.chunks = [b"abc", b""]

        def read(self, n: int) -> bytes:
            return self.chunks.pop(0)

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *a: object) -> None:
            return None

    def urlopen(url: str, timeout: int) -> Resp:
        seen.append(url)
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    dest = tmp_path / "f"
    suites._download("a/b", "rev", "x/y.parquet", dest)
    assert seen == ["https://huggingface.co/datasets/a/b/resolve/rev/x/y.parquet"]
    assert dest.read_bytes() == b"abc"


def test_gsm8k_row_without_answer_line(
    fixture_pins: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="no '#### '"):
        suites._gsm8k([{"question": "q", "answer": "no marker"}])


def test_parquet_needs_the_extra(
    fixture_pins: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = builtins.__import__

    def fake(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("pyarrow"):
            raise ImportError("blocked", name="pyarrow")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(
        ModuleNotFoundError, match=r"pip install 'astro-weights\[eval\]'"
    ):
        load_suite("gsm8k", DATA / FILES["gsm8k"])


def test_graders() -> None:
    pytest.importorskip("math_verify")
    gsm = suite_grader("gsm8k")
    assert gsm({"gold": "1000"}, r"so \boxed{1,000}") == 1.0
    assert gsm({"gold": "5"}, r"\boxed{4}") == 0.0
    assert gsm({"gold": "5"}, r"\boxed{4}</think>\boxed{5}") == 1.0
    mm = suite_grader("mmlu_pro")
    assert mm({"gold": "B"}, r"\boxed{B}") == 1.0
    assert mm({"gold": "B"}, r"\boxed{C}") == 0.0
    item = {"instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}
    assert suite_grader("ifeval")(item, "fine") == 1.0
    assert suite_grader("ifeval")(item, "a, b") == 0.0
