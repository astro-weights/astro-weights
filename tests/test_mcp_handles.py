"""Disk-backed job store: lifecycle, restart, hostile handles, atomicity, threads."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from astro_weights.mcp.handles import (
    HANDLE_RE,
    BadTransition,
    HandleStore,
    Job,
    UnknownHandle,
)


@pytest.fixture
def store(tmp_path: Path) -> HandleStore:
    return HandleStore(tmp_path / "run")


def _jobs_dir(store: HandleStore) -> Path:
    return store._dir


def test_creates_jobs_dir(tmp_path: Path) -> None:
    HandleStore(tmp_path / "a" / "b")
    assert (tmp_path / "a" / "b" / "jobs").is_dir()


def test_submit_get_round_trip(store: HandleStore) -> None:
    job = store.submit("forward_pass", {"prompt": "hi", "n": [1, 2]})
    assert HANDLE_RE.fullmatch(job.handle)
    assert job.handle.startswith("forward_pass-")
    assert job.state == "queued"
    assert job.result is None
    assert job.error is None
    assert job.created == job.updated
    assert store.get(job.handle) == job
    assert job.to_json() == {
        "handle": job.handle,
        "kind": "forward_pass",
        "state": "queued",
        "params": {"prompt": "hi", "n": [1, 2]},
        "result": None,
        "error": None,
        "created": job.created,
        "updated": job.updated,
    }
    json.dumps(job.to_json())


def test_handles_are_unique(store: HandleStore) -> None:
    handles = {store.submit("k", {}).handle for _ in range(50)}
    assert len(handles) == 50


def test_restart_sees_old_jobs(tmp_path: Path) -> None:
    first = HandleStore(tmp_path)
    job = first.submit("exec_python", {"code": "1"})
    first.start(job.handle)
    done = first.finish(job.handle, {"out": "ok"})
    second = HandleStore(tmp_path)
    assert second.get(job.handle) == done
    assert second.list() == [done]


def test_full_lifecycle(store: HandleStore) -> None:
    job = store.submit("k", {})
    running = store.start(job.handle)
    assert running.state == "running"
    assert running.updated >= job.updated
    done = store.finish(job.handle, {"x": 1})
    assert done.state == "done"
    assert done.result == {"x": 1}
    assert done.params == job.params
    assert done.created == job.created
    assert store.get(job.handle) == done


def test_fail_from_queued_and_running(store: HandleStore) -> None:
    a = store.submit("k", {})
    failed = store.fail(a.handle, "boom")
    assert failed.state == "failed"
    assert failed.error == "boom"
    assert failed.result is None
    b = store.submit("k", {})
    store.start(b.handle)
    assert store.fail(b.handle, "late").state == "failed"
    assert store.get(b.handle).error == "late"


def test_error_capped(store: HandleStore) -> None:
    job = store.submit("k", {})
    out = store.fail(job.handle, "e" * 5000)
    assert out.error == "e" * 2000
    assert store.get(job.handle).error == "e" * 2000


def _in_state(store: HandleStore, state: str) -> Job:
    job = store.submit("k", {})
    if state == "queued":
        return job
    if state == "failed":
        return store.fail(job.handle, "x")
    store.start(job.handle)
    if state == "running":
        return store.get(job.handle)
    return store.finish(job.handle, {})


@pytest.mark.parametrize("state", ["running", "done", "failed"])
def test_start_illegal(store: HandleStore, state: str) -> None:
    job = _in_state(store, state)
    with pytest.raises(BadTransition):
        store.start(job.handle)
    assert store.get(job.handle).state == state


@pytest.mark.parametrize("state", ["queued", "done", "failed"])
def test_finish_illegal(store: HandleStore, state: str) -> None:
    job = _in_state(store, state)
    with pytest.raises(BadTransition):
        store.finish(job.handle, {"a": 1})
    assert store.get(job.handle) == job


@pytest.mark.parametrize("state", ["done", "failed"])
def test_fail_illegal(store: HandleStore, state: str) -> None:
    job = _in_state(store, state)
    with pytest.raises(BadTransition):
        store.fail(job.handle, "again")
    assert store.get(job.handle) == job


def test_bad_transition_is_value_error() -> None:
    assert issubclass(BadTransition, ValueError)
    assert issubclass(UnknownHandle, KeyError)


MALFORMED = [
    "../x",
    "a/b",
    "",
    "x-zz",
    "K-0123456789abcdef",
    "k-0123456789ABCDEF",
    "a" * 33 + "-0123456789abcdef",
    "k-0123456789abcdef0",
    "k-0123456789abcde",
    "k-0123456789abcdef\n",
    "k-0123456789abcdef\x00",
    "k\x00-0123456789abcdef",
    "..\\k-0123456789abcdef",
    "../k-0123456789abcdef",
    "k-0123456789abcdef/../x",
]


@pytest.mark.parametrize("handle", MALFORMED)
def test_malformed_handles_unknown(store: HandleStore, handle: str) -> None:
    calls: list[Callable[[], object]] = [
        lambda: store.get(handle),
        lambda: store.start(handle),
        lambda: store.finish(handle, {}),
        lambda: store.finish(handle, {"a": object()}),
        lambda: store.fail(handle, "x"),
    ]
    for call in calls:
        with pytest.raises(UnknownHandle):
            call()


def test_malformed_never_touches_disk(
    store: HandleStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("filesystem touched")

    monkeypatch.setattr(Path, "read_text", boom)
    monkeypatch.setattr(Path, "open", boom)
    monkeypatch.setattr(Path, "replace", boom)
    monkeypatch.setattr(Path, "unlink", boom)
    monkeypatch.setattr(Path, "glob", boom)
    for handle in MALFORMED:
        for call in (
            store.get,
            store.start,
            lambda h: store.finish(h, {}),
            lambda h: store.fail(h, "x"),
        ):
            with pytest.raises(UnknownHandle):
                call(handle)
    with pytest.raises(UnknownHandle):
        store.get(123)  # type: ignore[arg-type]


def test_well_formed_but_absent(store: HandleStore) -> None:
    with pytest.raises(UnknownHandle):
        store.get("k-0123456789abcdef")


def test_outside_file_not_readable(tmp_path: Path) -> None:
    (tmp_path / "secret.json").write_text("{}")
    store = HandleStore(tmp_path / "run")
    with pytest.raises(UnknownHandle):
        store.get("../../secret")


@pytest.mark.parametrize(
    "kind", ["", "A", "a-b", "a/b", "a" * 33, "a\x00", "a1", "../x", 5]
)
def test_bad_kind(store: HandleStore, kind: object) -> None:
    with pytest.raises(ValueError, match="kind"):
        store.submit(kind, {})  # type: ignore[arg-type]
    assert list(_jobs_dir(store).iterdir()) == []


def test_kind_max_length_ok(store: HandleStore) -> None:
    assert store.submit("a" * 32, {}).kind == "a" * 32


@pytest.mark.parametrize(
    "params",
    [
        {"x": object()},
        {"x": {1, 2}},
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": b"b"},
    ],
)
def test_non_json_params_rejected(
    store: HandleStore, params: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="params"):
        store.submit("k", params)
    assert list(_jobs_dir(store).iterdir()) == []
    assert store.list() == []


def test_non_json_result_rejected_leaves_job(store: HandleStore) -> None:
    job = store.submit("k", {})
    store.start(job.handle)
    before = store.get(job.handle)
    with pytest.raises(ValueError, match="result"):
        store.finish(job.handle, {"x": object()})
    assert store.get(job.handle) == before
    assert [p.name for p in _jobs_dir(store).iterdir()] == [f"{job.handle}.json"]


def test_params_copied(store: HandleStore) -> None:
    params = {"a": [1]}
    job = store.submit("k", params)
    params["a"].append(2)
    assert store.get(job.handle).params == {"a": [1]}


def test_tmp_file_not_a_job(store: HandleStore) -> None:
    job = store.submit("k", {})
    tmp = _jobs_dir(store) / ".0123456789abcdef.tmp"
    tmp.write_text("{garbage")
    (_jobs_dir(store) / "k-0123456789abcdef.json.tmp").write_text("{garbage")
    assert store.list() == [job]
    with pytest.raises(UnknownHandle):
        store.get("k-0123456789abcdef")


def test_no_tmp_left_after_write_error(
    store: HandleStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise OSError("disk")

    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(OSError, match="disk"):
        store.submit("k", {})
    assert list(_jobs_dir(store).iterdir()) == []


def test_corrupt_file_raises_value_error(store: HandleStore) -> None:
    job = store.submit("k", {})
    path = _jobs_dir(store) / f"{job.handle}.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match=job.handle) as info:
        store.get(job.handle)
    assert not isinstance(info.value, UnknownHandle)
    with pytest.raises(ValueError, match=job.handle):
        store.start(job.handle)
    with pytest.raises(ValueError, match=job.handle):
        store.list()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("state"),
        lambda d: d.update(state="weird"),
        lambda d: d.update(handle="k-ffffffffffffffff"),
        lambda d: d.update(params=[1]),
        lambda d: d.update(created="soon"),
        lambda d: d.update(result=5),
        lambda d: d.update(result=[["a", 1]]),
        lambda d: d.update(params=[["a", 1]]),
        lambda d: d.update(kind=5),
        lambda d: d.update(kind="other"),
        lambda d: d.update(error=5),
        lambda d: d.update(created=True),
        lambda d: d.update(updated=None),
    ],
)
def test_invalid_fields_raise(store: HandleStore, mutate: object) -> None:
    job = store.submit("k", {})
    path = _jobs_dir(store) / f"{job.handle}.json"
    data = json.loads(path.read_text())
    mutate(data)  # type: ignore[operator]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=job.handle):
        store.get(job.handle)


def test_stored_result_and_error_types_read_back(store: HandleStore) -> None:
    done = store.submit("k", {})
    store.start(done.handle)
    store.finish(done.handle, {"n": 1})
    failed = store.submit("k", {})
    store.fail(failed.handle, "bad")
    assert store.get(done.handle).result == {"n": 1}
    assert store.get(failed.handle).error == "bad"


def test_concurrent_finish_one_winner(store: HandleStore) -> None:
    job = store.submit("k", {})
    store.start(job.handle)
    barrier = threading.Barrier(8)
    wins: list[int] = []
    losses: list[BaseException] = []

    def worker(i: int) -> None:
        barrier.wait()
        try:
            store.finish(job.handle, {"winner": i})
            wins.append(i)
        except BaseException as exc:  # noqa: BLE001
            losses.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1
    assert len(losses) == 7
    assert all(isinstance(e, BadTransition) for e in losses)
    assert store.get(job.handle).result == {"winner": wins[0]}


def test_concurrent_submit(store: HandleStore) -> None:
    def worker() -> None:
        for _ in range(10):
            store.submit("k", {})

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store.list()) == 80


def test_list_filter_and_order(
    store: HandleStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks = iter(range(100, 200))
    monkeypatch.setattr(
        "astro_weights.mcp.handles.time.time", lambda: float(next(ticks))
    )
    a = store.submit("aa", {})
    b = store.submit("bb", {})
    c = store.submit("aa", {})
    store.start(a.handle)  # update must not reorder
    assert [j.handle for j in store.list()] == [a.handle, b.handle, c.handle]
    assert [j.handle for j in store.list("aa")] == [a.handle, c.handle]
    assert [j.handle for j in store.list("bb")] == [b.handle]
    assert store.list("zz") == []


def test_list_empty(store: HandleStore) -> None:
    assert store.list() == []


def test_list_ignores_foreign_files(store: HandleStore) -> None:
    job = store.submit("k", {})
    (_jobs_dir(store) / "notes.json").write_text("{}")
    (_jobs_dir(store) / "README").write_text("x")
    assert store.list() == [job]


def test_concurrent_finish_separate_stores_one_winner(tmp_path: Path) -> None:
    job = HandleStore(tmp_path).submit("k", {})
    HandleStore(tmp_path).start(job.handle)
    stores = [HandleStore(tmp_path) for _ in range(8)]
    barrier = threading.Barrier(8)
    wins: list[int] = []
    losses: list[BaseException] = []

    def worker(i: int) -> None:
        barrier.wait()
        try:
            stores[i].finish(job.handle, {"winner": i})
            wins.append(i)
        except BaseException as exc:  # noqa: BLE001
            losses.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1
    assert len(losses) == 7
    assert all(isinstance(e, BadTransition) for e in losses)
    assert HandleStore(tmp_path).get(job.handle).result == {"winner": wins[0]}


def test_same_root_shares_lock(tmp_path: Path) -> None:
    assert HandleStore(tmp_path)._lock is HandleStore(tmp_path / "x" / "..")._lock


@pytest.mark.parametrize(
    "params",
    [{1: "a"}, {1: "a", "1": "b"}, {"x": {2: 3}}, {"x": [{None: 1}]}, {"x": ({3: 1},)}],
)
def test_non_str_keys_rejected(
    store: HandleStore, params: dict[object, object]
) -> None:
    with pytest.raises(ValueError, match="params"):
        store.submit("k", params)  # type: ignore[arg-type]
    assert list(_jobs_dir(store).iterdir()) == []
