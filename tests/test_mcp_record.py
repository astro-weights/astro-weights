import json
import math
import platform
import shutil
import subprocess
import threading
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pytest

from astro_weights.mcp import record as record_mod
from astro_weights.mcp.budget import Ledger, Limits
from astro_weights.mcp.record import CATEGORY_MAX, DETAIL_MAX, RunRecord, library_commit


def start(tmp_path: Path, **kw: Any) -> RunRecord:
    args: dict[str, Any] = {
        "seed": 7,
        "config": {"budget": {"forward_passes": 3}, "tags": ["a", "b"]},
        "editor": "scripted",
        "model": None,
    }
    args.update(kw)
    return RunRecord.start(tmp_path / "record.jsonl", **args)


def test_round_trip(tmp_path: Path) -> None:
    rec = start(tmp_path, model="m-1", library_commit="abc123")
    rec.event("forward_pass", handle="h1", n=2)
    rec.refusal(category="cyber", editor="hosted", model="m-1", detail="declined")
    led = Ledger(tmp_path / "ledger.jsonl", Limits(1, 1, 10, 0.5))
    led.charge("forward_passes", ref="h1")
    rec.finish(budget=led.to_json(), outcome="submitted")

    again = RunRecord.open(rec.path)
    lines = again.read()
    assert [x["event"] for x in lines] == ["start", "forward_pass", "refusal", "finish"]
    first = lines[0]
    assert first["seed"] == 7
    assert first["editor"] == "scripted"
    assert first["model"] == "m-1"
    assert first["config"] == {"budget": {"forward_passes": 3}, "tags": ["a", "b"]}
    assert first["library_version"] == version("astro-weights")
    assert first["library_commit"] == "abc123"
    assert first["python"] == platform.python_version()
    assert first["platform"] == platform.platform()
    assert all("time" in x for x in lines)
    assert lines[1]["handle"] == "h1"
    assert lines[1]["n"] == 2
    assert lines[2] == {
        "event": "refusal",
        "category": "cyber",
        "editor": "hosted",
        "model": "m-1",
        "detail": "declined",
        "time": lines[2]["time"],
    }
    assert lines[3]["outcome"] == "submitted"
    assert lines[3]["budget"]["used"]["forward_passes"] == 1
    again.event("resumed")
    assert again.read()[-1]["event"] == "resumed"


def test_start_defaults(tmp_path: Path) -> None:
    first = start(tmp_path).read()[0]
    assert first["model"] is None
    assert first["library_commit"] is None


def test_start_on_empty_file_ok(tmp_path: Path) -> None:
    (tmp_path / "record.jsonl").touch()
    assert start(tmp_path).read()[0]["event"] == "start"


def test_start_on_non_empty_file_refuses(tmp_path: Path) -> None:
    rec = start(tmp_path)
    before = rec.path.read_bytes()
    with pytest.raises(FileExistsError, match=r"RunRecord\.open"):
        start(tmp_path)
    assert rec.path.read_bytes() == before


@pytest.mark.parametrize(
    "kw",
    [
        {"seed": True},
        {"seed": 1.0},
        {"config": [1]},
        {"config": {"x": object()}},
        {"config": {"x": math.nan}},
        {"config": {1: "x"}},
        {"editor": ""},
        {"editor": "  "},
        {"model": 3},
        {"library_commit": b"abc"},
    ],
)
def test_start_rejects_bad_fields(tmp_path: Path, kw: dict[str, Any]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        start(tmp_path, **kw)
    assert not (tmp_path / "record.jsonl").exists()


def test_tuple_written_as_list(tmp_path: Path) -> None:
    rec = start(tmp_path, config={"layers": (1, 2)})
    assert rec.read()[0]["config"] == {"layers": [1, 2]}


@pytest.mark.parametrize(
    "value",
    [object(), math.inf, {1, 2}, b"x", {"k": [1, {"deep": math.nan}]}, {"k": {2: 1}}],
)
def test_event_non_json_rejected_no_partial_line(tmp_path: Path, value: Any) -> None:
    rec = start(tmp_path)
    before = rec.path.read_bytes()
    with pytest.raises(ValueError, match="record"):
        rec.event("x", value=value)
    assert rec.path.read_bytes() == before


def test_finish_non_json_rejected(tmp_path: Path) -> None:
    rec = start(tmp_path)
    before = rec.path.read_bytes()
    with pytest.raises(ValueError, match="record"):
        rec.finish(budget={"used": math.nan}, outcome="x")
    with pytest.raises(ValueError, match="budget"):
        rec.finish(budget=[1], outcome="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="outcome"):
        rec.finish(budget={}, outcome="")
    assert rec.path.read_bytes() == before


@pytest.mark.parametrize("name", ["start", "refusal", "finish", "", 3])
def test_event_reserved_or_bad_name(tmp_path: Path, name: Any) -> None:
    rec = start(tmp_path)
    with pytest.raises(ValueError):  # noqa: PT011
        rec.event(name)


@pytest.mark.parametrize("field", ["event", "time"])
def test_event_reserved_field(tmp_path: Path, field: str) -> None:
    rec = start(tmp_path)
    with pytest.raises(ValueError, match="reserved"):
        rec.event("x", **{field: "y"})


def test_refusal_caps_and_validation(tmp_path: Path) -> None:
    rec = start(tmp_path)
    rec.refusal(category="c" * 500, editor="e", model=None, detail="d" * 5000)
    line = rec.read()[-1]
    assert len(line["category"]) == CATEGORY_MAX
    assert len(line["detail"]) == DETAIL_MAX
    assert line["model"] is None
    before = rec.path.read_bytes()
    for kw in (
        {"category": "", "editor": "e", "model": None},
        {"category": " \n", "editor": "e", "model": None},
        {"category": "c", "editor": "", "model": None},
        {"category": "c", "editor": "e", "model": 1},
        {"category": "c", "editor": "e", "model": None, "detail": None},
    ):
        with pytest.raises(ValueError):  # noqa: PT011
            rec.refusal(**kw)
    assert rec.path.read_bytes() == before


def test_open_missing_or_not_started(tmp_path: Path) -> None:
    path = tmp_path / "record.jsonl"
    with pytest.raises(FileNotFoundError):
        RunRecord.open(path)
    path.touch()
    with pytest.raises(ValueError, match="no start line"):
        RunRecord.open(path)
    path.write_text('{"event":"finish"}\n')
    with pytest.raises(ValueError, match="not a start event"):
        RunRecord.open(path)


def test_open_corrupt_middle_line_raises(tmp_path: Path) -> None:
    rec = start(tmp_path)
    with rec.path.open("ab") as f:
        f.write(b"garbage\n")
    with pytest.raises(ValueError, match=r"record\.jsonl:2"):
        RunRecord.open(rec.path)
    rec.path.write_bytes(rec.path.read_bytes().replace(b"garbage", b'{"no":"event"}'))
    with pytest.raises(ValueError, match='no "event"'):
        RunRecord.open(rec.path)


def test_open_cuts_torn_final_line(tmp_path: Path) -> None:
    rec = start(tmp_path)
    rec.event("a")
    with rec.path.open("ab") as f:
        f.write(b'{"event":"b","x":')
    again = RunRecord.open(rec.path)
    again.event("c")
    assert [x["event"] for x in again.read()] == ["start", "a", "c"]


def test_open_keeps_torn_line_missing_newline(tmp_path: Path) -> None:
    rec = start(tmp_path)
    with rec.path.open("ab") as f:
        f.write(b'{"event":"b"}')
    again = RunRecord.open(rec.path)
    again.event("c")
    assert [x["event"] for x in again.read()] == ["start", "b", "c"]


def test_read_ignores_torn_tail(tmp_path: Path) -> None:
    rec = start(tmp_path)
    with rec.path.open("ab") as f:
        f.write(b'{"event":')
    assert [x["event"] for x in rec.read()] == ["start"]


def test_lines_are_compact_json(tmp_path: Path) -> None:
    rec = start(tmp_path)
    rec.event("unicode", text="é \ud800")
    raw = rec.path.read_bytes().splitlines()
    assert all(json.loads(x) for x in raw)
    assert rec.read()[-1]["text"] == "é \ud800"


def test_open_unterminated_object_failing_check_raises(tmp_path: Path) -> None:
    rec = start(tmp_path)
    with rec.path.open("ab") as f:
        f.write(b'{"no":"event"}')
    before = rec.path.read_bytes()
    with pytest.raises(ValueError, match=r"record\.jsonl:2"):
        RunRecord.open(rec.path)
    assert rec.path.read_bytes() == before


def test_cyclic_config_rejected_no_file(tmp_path: Path) -> None:
    config: dict[str, Any] = {"a": 1}
    config["self"] = config
    with pytest.raises(ValueError, match="contains itself"):
        start(tmp_path, config=config)
    assert not (tmp_path / "record.jsonl").exists()


def test_cyclic_event_rejected_shared_ok(tmp_path: Path) -> None:
    rec = start(tmp_path)
    loop: list[Any] = [1]
    loop.append([loop])
    before = rec.path.read_bytes()
    with pytest.raises(ValueError, match="contains itself"):
        rec.event("x", value=loop)
    assert rec.path.read_bytes() == before
    shared = [1, 2]
    rec.event("x", a=shared, b={"c": shared, "d": shared})  # repeated, not cyclic
    assert rec.read()[-1]["b"] == {"c": [1, 2], "d": [1, 2]}


def test_threads_append_every_line_intact(tmp_path: Path) -> None:
    rec = start(tmp_path)
    threads, per, pad = 16, 25, "x" * 20_000
    barrier = threading.Barrier(threads)

    def worker(t: int) -> None:
        barrier.wait()
        for i in range(per):
            if i % 5 == 0:
                rec.refusal(category="c", editor="e", model=None, detail=f"{t}:{i}")
            else:
                rec.event("e", t=t, i=i, pad=pad)

    ts = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
    for th in ts:
        th.start()
    for th in ts:
        th.join()
    lines = RunRecord.open(rec.path).read()
    assert len(lines) == 1 + threads * per
    events = {(x["t"], x["i"]) for x in lines if x["event"] == "e"}
    refusals = {x["detail"] for x in lines if x["event"] == "refusal"}
    assert len(events) == threads * per * 4 // 5
    assert len(refusals) == threads * per // 5
    assert all(x["pad"] == pad for x in lines if x["event"] == "e")


def test_failed_append_keeps_other_threads_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = start(tmp_path)
    rec.event("a")
    calls = 0
    real = __import__("os").fsync

    def flaky(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("disk hiccup")
        real(fd)

    monkeypatch.setattr("astro_weights.mcp.record.os.fsync", flaky)
    with pytest.raises(OSError, match="hiccup"):
        rec.event("lost")
    rec.event("b")
    monkeypatch.undo()
    assert [x["event"] for x in rec.read()] == ["start", "a", "b"]


# ---------------------------------------------------------------- library_commit


class _Dist:
    def __init__(self, text: str | None) -> None:
        self.text = text

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return self.text


def _installed(monkeypatch: pytest.MonkeyPatch, info: Any) -> None:
    text = info if info is None or isinstance(info, str) else json.dumps(info)
    monkeypatch.setattr(record_mod, "distribution", lambda _: _Dist(text))


def _git_repo(where: Path) -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(where), *args],
            capture_output=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()

    where.mkdir()
    git("init", "-q")
    (where / "f.txt").write_text("a", encoding="utf-8")
    git("add", "f.txt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")
    return git("rev-parse", "HEAD")


def test_library_commit_from_vcs_install(monkeypatch: pytest.MonkeyPatch) -> None:
    vcs = {"vcs": "git", "commit_id": "abc"}
    _installed(monkeypatch, {"url": "https://x/y.git", "vcs_info": vcs})
    assert library_commit() == "abc"


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_library_commit_from_editable_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    head = _git_repo(tmp_path / "repo")
    url = (tmp_path / "repo").as_uri()
    _installed(monkeypatch, {"url": url, "dir_info": {"editable": True}})
    assert library_commit() == head
    (tmp_path / "repo" / "f.txt").write_text("b", encoding="utf-8")
    assert library_commit() == f"{head}-dirty"


@pytest.mark.parametrize(
    "info",
    [
        None,  # a wheel from an index has no direct_url.json
        "not json",
        {"url": "file:///nowhere/at/all", "dir_info": {"editable": True}},
        {"url": "file:///tmp/x.whl", "archive_info": {}},
        {"url": "https://x/y.git", "dir_info": {"editable": True}},
    ],
)
def test_library_commit_none(monkeypatch: pytest.MonkeyPatch, info: Any) -> None:
    _installed(monkeypatch, info)
    assert library_commit() is None


def test_library_commit_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_: str) -> Any:
        raise PackageNotFoundError("astro-weights")

    monkeypatch.setattr(record_mod, "distribution", missing)
    assert library_commit() is None
