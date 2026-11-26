"""The ``confirm`` tool: pass or fail of the stored edits against the current best.

Rule (PDR 0.3 section 4.1, choice c, Ladder style: Blum and Hardt 2015,
https://arxiv.org/abs/1502.04585). A query generates greedy outputs of the edited
model for every confirm item, in file order, and grades them: ``c`` items correct
out of ``n``. The bar ``b`` is the count of the current best: the base model on the
first query, then the last edit that passed. The query passes iff
``c - b > margin * n``. The test is one-sided and strict, so a tie fails even at
margin 0. On a pass the bar becomes ``c``; on a fail it stays. Items, order and
decoding are fixed, so ``(c - b) / n`` is the paired mean difference.

With ``target = "tokens"`` (match the gate's ``control_target = "tokens_diff"``,
PDR 0.3 section 4.6 stage 4) a query passes iff both hold: (a) the edit's total
output tokens over all confirm items are below the token bar (strict, a tie
fails), and (b) ``c - base >= -margin * n``, accuracy non-inferior to the base
model (never to an earlier edit, so passes cannot drift down). The token bar is
the base model's total on the first query, then the total of the last edit that
passed. Base counts are recorded once (``confirm_base``) and restored on a
restart, so the base arm never runs twice.

Cap-hit guard (optional ``max_cap_hit``, a share in [0, 1]; absent = off, as
before). An answer cut at ``max_new_tokens`` is graded wrong and its tokens stop
at the cap, so an edit can look shorter or as accurate only because it is cut.
With the key set, a query fails, whatever the rule above says, when more than
that share of confirm items hit the cap in the edited arm or in the base arm (a
cut base makes the bar itself censored; raise ``max_new_tokens``). The shares are
exact (``Fraction``), a tie with the limit holds. The result records ``hit_cap``
and, on a guard fail, ``cap_over`` (the arms over); with ``target = "accuracy"``
the base counts go to a ``confirm_base`` event too, so a restart keeps them. The
agent still sees only ``{"passed": false}``. Same rule as the gate's
``max_cap_hit`` (``GATE.toml``).

``batch_size > 1`` generates that many items at once with left padding, for both
arms alike; greedy outputs can then differ from ``batch_size = 1`` (see
``generate_records``).

The agent learns ``{"passed": bool}`` and nothing else: no score, no n, no item,
no timing. Any error in a query is also ``{"passed": false}``; only the error's
type name is recorded. The items file is read here, in the server, at start; the
kernel sandbox cannot read it and no tool returns it. Counts and the bar go to the
run record (``confirm_query`` and ``confirm_result`` events), which no tool returns.

Resume: a query counts toward ``max_queries`` once it has a ``confirm_result``. On
start, each ``confirm_query`` without one (the server was killed mid-query, so the
agent never saw a result) gets one ``confirm_interrupted`` event and does not
count; query numbers keep increasing. With a run directory, ``generator``
checkpoints each finished chunk of outputs under ``RUN_DIR/confirm/``, so a
resubmitted query of the same edits generates only the items not yet written; the
base arm's outputs are reused the same way.
"""

import contextlib
import hashlib
import json
import os
import threading
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

from astro_weights.eval import grade_choice, grade_exact, grade_math
from astro_weights.mcp.handles import HandleStore
from astro_weights.mcp.jobs import ConfirmConfig, Executor
from astro_weights.mcp.record import RunRecord

__all__ = [
    "ConfirmLimit",
    "Confirmer",
    "candidate_id",
    "check_grader",
    "generator",
    "passes",
    "passes_tokens",
    "pin",
    "read_items",
]

GRADERS: dict[str, Callable[[str, str], bool]] = {
    "exact": grade_exact,
    "choice": grade_choice,
    "math": grade_math,
}
KIND = "confirm"


class CapHitsMissing(ValueError):
    """``max_cap_hit`` is set but an arm's outputs carry no valid ``hit_cap`` list."""


class ConfirmLimit(Exception):
    """``confirm.max_queries`` queries were already made."""


def passes(correct: int, bar: int, n: int, margin: float) -> bool:
    """True iff ``correct`` beats ``bar`` by more than ``margin * n`` items.

    Exact: the margin is read as its decimal text, so 0.29 * 100 is 29, a tie.
    """
    return correct - bar > Fraction(str(margin)) * n


def passes_tokens(
    correct: int, tokens: int, base_correct: int, bar_tokens: int, n: int, margin: float
) -> bool:
    """True iff ``tokens < bar_tokens`` and ``correct - base_correct >= -margin * n``.

    The margin is exact as in ``passes``: at n 100, margin 0.02, 2 items down holds.
    """
    return tokens < bar_tokens and correct - base_correct >= -Fraction(str(margin)) * n


def candidate_id(applied: dict[str, str]) -> str:
    """sha256 of the applied edits (tensor name to edit file sha256)."""
    text = json.dumps(sorted(applied.items()), separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


Items = tuple[list[str], list[str], str]  # prompts, golds, file sha256


def read_items(path: Path) -> Items:
    """(prompts, golds, file sha256) of a JSONL items file; ValueError if bad.

    Error messages name the line, never its content.
    """
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"confirm.items is not readable: {exc.strerror}") from None
    prompts: list[str] = []
    golds: list[str] = []
    seen: set[str] = set()
    for i, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError:
            raise ValueError(f"confirm.items line {i} is not JSON") from None
        ok = isinstance(item, dict) and all(
            isinstance(item.get(k), str) for k in ("item_id", "prompt", "gold")
        )
        if not ok or item["item_id"] in seen:
            raise ValueError(
                f"confirm.items line {i}: needs unique string item_id, prompt, gold"
            )
        seen.add(item["item_id"])
        prompts.append(item["prompt"])
        golds.append(item["gold"])
    if not prompts:
        raise ValueError("confirm.items has no items")
    return prompts, golds, hashlib.sha256(data).hexdigest()


def pin(cfg: ConfirmConfig, items: Items) -> dict[str, Any]:
    """What the run record pins of the confirm setup: no path, no item.

    ``target`` and ``batch_size`` only when not the default, so a default record
    reads as before.
    """
    out: dict[str, Any] = {
        "items_sha256": items[2],
        "grader": cfg.grader,
        "margin": cfg.margin,
        "max_queries": cfg.max_queries,
    }
    if cfg.target != "accuracy":
        out["target"] = cfg.target
    if cfg.batch_size != 1:
        out["batch_size"] = cfg.batch_size
    if cfg.max_cap_hit is not None:
        out["max_cap_hit"] = cfg.max_cap_hit
    return out


Row = tuple[str, int, bool]  # output text, token count, hit the cap


def _prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def _load_rows(path: Path, shas: list[str]) -> dict[int, Row]:
    """Checkpointed rows of ``path`` whose prompt sha256 still matches, by index.

    A torn last line (a kill mid-write) is cut off the file first.
    """
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return {}
    keep = data[: data.rfind(b"\n") + 1]
    if len(keep) != len(data):
        with path.open("r+b") as f:
            f.truncate(len(keep))
    rows: dict[int, Row] = {}
    for line in keep.decode("utf-8", "replace").splitlines():
        try:
            r = json.loads(line)
            i, row = r["index"], (r["text"], r["tokens"], r["hit_cap"])
        except (ValueError, TypeError, KeyError):
            continue
        ok = isinstance(i, int) and 0 <= i < len(shas) and r.get("sha256") == shas[i]
        types = (str, int, bool)
        if ok and all(isinstance(v, t) for v, t in zip(row, types, strict=True)):
            rows[i] = row
    return rows


def _save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append one chunk of rows; written, flushed and fsynced together."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
    with path.open("ab") as f:
        f.write(data.encode())
        f.flush()
        os.fsync(f.fileno())


def generator(
    backend: Any, max_new_tokens: int, run_dir: Path | None = None
) -> Executor:
    """Executor ``{"arm", "prompts", "batch_size"?}`` on the forward model.

    Returns ``{"texts", "tokens", "hit_cap", "candidate"}``: per prompt the output,
    its token count and whether it hit ``max_new_tokens``. Greedy, chat template
    when the tokenizer has one. ``batch_size`` (default 1) > 1 left-pads that many
    prompts per call, as ``generate_records`` does. ``candidate`` is
    ``candidate_id`` of the edits actually applied, or ``"base"``.

    With ``run_dir``, each finished chunk of rows (index, prompt sha256, text,
    tokens, hit_cap) is appended to
    ``run_dir/confirm/<candidate>-<batch_size>-<max_new_tokens>.jsonl``; a later
    call (after a kill and resume) reuses the rows whose prompt sha256 matches and
    generates only the rest. Chunks sit at fixed positions, so the outputs equal an
    uninterrupted run.
    """

    def run(params: dict[str, Any]) -> dict[str, Any]:
        from astro_weights.runners.generate import _prompt_text, _run_batch, _run_one

        arm = params["arm"]
        size = int(params.get("batch_size", 1))
        prompts = list(params["prompts"])
        with backend._lock:  # one model, shared with forward_pass jobs
            backend._ensure()
            backend._sync(arm)
            model, tok = backend._model, backend._tok
            cand = "base" if arm == "base" else candidate_id(dict(backend._applied))
            eos = model.generation_config.eos_token_id
            eos_ids = set(
                eos if isinstance(eos, list) else [] if eos is None else [eos]
            )
            if size > 1 and tok.pad_token_id is None:
                raise ValueError("confirm.batch_size > 1 needs a pad token")
            shas = [_prompt_sha(p) for p in prompts]
            path = None
            done: dict[int, Row] = {}
            if run_dir is not None:
                path = Path(run_dir) / KIND / f"{cand}-{size}-{max_new_tokens}.jsonl"
                done = _load_rows(path, shas)
            shown = [_prompt_text(tok, p, True) for p in prompts]
            rows: dict[int, Row] = {}
            for start in range(0, len(shown), size):
                idx = range(start, min(start + size, len(shown)))
                if all(i in done for i in idx):
                    rows.update((i, done[i]) for i in idx)
                    continue
                chunk = shown[start : start + size]
                if size == 1:
                    text, chat = chunk[0]
                    got = [_run_one(model, tok, text, chat, max_new_tokens, eos_ids)]
                else:
                    texts = [t for t, _ in chunk]
                    got = _run_batch(
                        model, tok, texts, chunk[0][1], max_new_tokens, eos_ids
                    )
                if path is not None:
                    _save_rows(
                        path,
                        [
                            {"index": i, "sha256": shas[i], "text": r[0]}
                            | {"tokens": r[1], "hit_cap": r[2]}
                            for i, r in zip(idx, got, strict=True)
                        ],
                    )
                rows.update(zip(idx, got, strict=True))
        out = [rows[i] for i in range(len(shown))]
        return {
            "texts": [r[0] for r in out],
            "tokens": [r[1] for r in out],
            "hit_cap": [r[2] for r in out],
            "candidate": cand,
        }

    return run


def check_grader(name: str) -> None:
    """ValueError naming the extra when the grader cannot run here."""
    if name == "math":
        try:
            import math_verify  # noqa: F401
        except ImportError:
            raise ValueError(
                "confirm.grader math needs: pip install 'astro-weights[eval]'"
            ) from None


class _State:
    """What the confirm rule carries from query to query, rebuilt from the record."""

    def __init__(self) -> None:
        self.queries = 0  # counted toward max_queries: not interrupted
        self.last = 0  # highest query number used; numbers are never reused
        self.bar: int | None = None  # accuracy target: correct count of the best
        self.best: str | None = None  # candidate of the last pass
        self.base: tuple[int, int] | None = None  # tokens target: correct, tokens
        self.bar_tokens: int | None = None  # tokens target: total of the best
        self.base_caps: int | None = None  # cap hits of the base arm


def _interrupted(events: list[dict[str, Any]]) -> list[int]:
    """Query numbers with a ``confirm_query`` but no result and no interrupted mark."""
    closed = {
        e.get("query")
        for e in events
        if e["event"] in {"confirm_result", "confirm_interrupted"}
    }
    return [
        e["query"]
        for e in events
        if e["event"] == "confirm_query" and e.get("query") not in closed
    ]


def _restore(events: list[dict[str, Any]]) -> _State:
    """Queries made, bars and best candidate from earlier record events.

    A query counts once it has a result: ``confirm_interrupted`` queries do not.
    """
    st = _State()
    asked = [e for e in events if e["event"] == "confirm_query"]
    cut = sum(e["event"] == "confirm_interrupted" for e in events)
    st.queries = len(asked) - cut
    st.last = max((int(e["query"]) for e in asked), default=0)
    for e in events:
        if e["event"] == "confirm_base":
            if isinstance(e.get("base_hit_cap"), int):
                st.base_caps = e["base_hit_cap"]
            if "base_tokens" in e:  # tokens target
                st.base = (int(e["base_correct"]), int(e["base_tokens"]))
                st.bar_tokens = st.base[1]
            if st.bar is None:  # a crash before the first result: best is still base
                st.bar = int(e["base_correct"])
        if e["event"] != "confirm_result":
            continue
        if isinstance(e.get("best"), int):
            st.bar = e["best"]
        if isinstance(e.get("best_candidate"), str):
            st.best = e["best_candidate"]
            if isinstance(e.get("tokens"), int):
                st.bar_tokens = e["tokens"]
    return st


class Confirmer:
    """Runs confirm queries as jobs in ``store``, one at a time."""

    def __init__(
        self,
        cfg: ConfirmConfig,
        items: Items,
        generate: Executor,
        store: HandleStore,
        record: RunRecord,
    ) -> None:
        self._prompts, self._golds, _ = items
        self._grade = GRADERS[cfg.grader]
        self._cfg = cfg
        self._generate = generate
        self._store = store
        self._record = record
        self._count_lock = threading.Lock()
        self._run_lock = threading.Lock()
        events = record.read()
        for query in _interrupted(events):  # killed mid-query: no result was seen
            record.event("confirm_interrupted", query=query)
            events.append({"event": "confirm_interrupted", "query": query})
        self._st = _restore(events)
        self._queries, self._bar = self._st.queries, self._st.bar
        self._last = self._st.last
        self.best_candidate = self._st.best

    def export_check(self, exported: dict[str, str] | None) -> dict[str, Any]:
        """Record and return whether the exported edits are the last that passed.

        ``exported`` maps tensor name to edit sha256, or None when nothing was
        exported. A test score of other edits is not the edits selection picked.
        """
        with self._run_lock:  # no query mid-way
            cand = None if exported is None else candidate_id(exported)
            out = {
                "exported_candidate": cand,
                "exported_is_best": cand is not None and cand == self.best_candidate,
            }
            self._record.event(
                "confirm_export", best_candidate=self.best_candidate, **out
            )
        return out

    def submit(self) -> dict[str, Any]:
        """Queue one query; ConfirmLimit past ``max_queries`` (nothing queued)."""
        with self._count_lock:
            if self._queries >= self._cfg.max_queries:
                raise ConfirmLimit(
                    f"confirm limit reached: {self._cfg.max_queries} queries"
                )
            self._queries += 1
            self._last += 1
            query = self._last
            self._record.event("confirm_query", query=query)
            job = self._store.submit(KIND, {})
        threading.Thread(
            target=self._run, args=(job.handle, query), daemon=True
        ).start()
        return {"handle": job.handle, "state": job.state}

    def _arm(self, arm: str) -> dict[str, Any]:
        params: dict[str, Any] = {"arm": arm, "prompts": list(self._prompts)}
        if self._cfg.batch_size != 1:
            params["batch_size"] = self._cfg.batch_size
        out = self._generate(params)
        texts = out["texts"]
        if len(texts) != len(self._golds):
            raise ValueError("wrong number of outputs")
        n = sum(
            bool(self._grade(g, t)) for g, t in zip(self._golds, texts, strict=True)
        )
        return {**out, "correct": n, "candidate": str(out["candidate"])}

    @staticmethod
    def _caps(out: dict[str, Any], n: int) -> int:
        """Cap hits of one arm's outputs; ValueError when not one bool per item."""
        caps = out.get("hit_cap")
        if not isinstance(caps, list) or len(caps) != n:
            raise CapHitsMissing("max_cap_hit needs one bool hit_cap per item")
        if not all(isinstance(c, bool) for c in caps):
            raise CapHitsMissing("max_cap_hit needs one bool hit_cap per item")
        return sum(caps)

    def _cap_over(self, edit_caps: int, n: int) -> list[str]:
        """Arms whose cap-hit share is above ``max_cap_hit``; empty when guard off."""
        limit = self._cfg.max_cap_hit
        if limit is None:
            return []
        if self._st.base_caps is None:
            raise CapHitsMissing("base cap hits unknown")
        lim = Fraction(str(limit))
        arms = {"base": self._st.base_caps, "edited": edit_caps}
        return [a for a, c in arms.items() if Fraction(c, n) > lim]

    @staticmethod
    def _tokens(out: dict[str, Any], n: int) -> tuple[int, int]:
        """(total output tokens, cap hits) of one arm's outputs."""
        tokens, caps = out["tokens"], out.get("hit_cap", [])
        ok = isinstance(tokens, list) and len(tokens) == n
        if not ok or not all(isinstance(t, int) and t >= 0 for t in tokens):
            raise ValueError("bad token counts")
        return sum(tokens), sum(bool(c) for c in caps)

    def _run(self, handle: str, query: int) -> None:
        try:
            with self._run_lock:
                self._store.start(handle)
                passed = self._query(query)
                self._store.finish(handle, {"passed": passed})
        except Exception:  # noqa: BLE001 - disk trouble must not strand the job
            with contextlib.suppress(Exception):
                self._store.fail(handle, "confirm failed")

    def _query(self, query: int) -> bool:
        if self._cfg.target == "tokens":
            return self._query_tokens(query)
        n = len(self._golds)
        fields: dict[str, Any] = {"query": query, "n": n}
        guard = self._cfg.max_cap_hit is not None
        passed = False
        try:
            if self._bar is None:
                base = self._arm("base")
                if guard:  # before the bar: a bad base leaves it unset, so it reruns
                    self._st.base_caps = self._caps(base, n)
                    self._record.event(
                        "confirm_base",
                        n=n,
                        base_correct=base["correct"],
                        base_hit_cap=self._st.base_caps,
                    )
                self._bar = base["correct"]
            fields["bar"] = self._bar
            out = self._arm("edited")
            correct, fields["candidate"] = out["correct"], out["candidate"]
            fields["correct"] = correct
            passed = passes(correct, self._bar, n, self._cfg.margin)
            if guard:
                fields["hit_cap"] = self._caps(out, n)
                if over := self._cap_over(fields["hit_cap"], n):
                    fields["cap_over"], passed = over, False
            if passed:
                self._bar = correct
                self.best_candidate = fields["best_candidate"] = fields["candidate"]
        except Exception as exc:  # noqa: BLE001 - the agent sees only passed
            passed = False  # an error after the rule check still fails
            fields["error"] = type(exc).__name__
        self._record.event("confirm_result", passed=passed, best=self._bar, **fields)
        return passed

    def _query_tokens(self, query: int) -> bool:
        n = len(self._golds)
        fields: dict[str, Any] = {"query": query, "n": n, "target": "tokens"}
        passed = False
        st = self._st
        try:
            if st.base is None:
                base = self._arm("base")
                tokens, caps = self._tokens(base, n)
                if self._cfg.max_cap_hit is not None:
                    caps = self._caps(base, n)  # strict, before any state is set
                st.base, st.bar_tokens = (base["correct"], tokens), tokens
                st.base_caps = caps
                st.bar = base["correct"]
                self._record.event(
                    "confirm_base",
                    n=n,
                    base_correct=base["correct"],
                    base_tokens=tokens,
                    base_hit_cap=caps,
                )
            base_correct = st.base[0]
            bar_tokens = st.bar_tokens if st.bar_tokens is not None else st.base[1]
            fields.update(bar=base_correct, base_correct=base_correct)
            fields["bar_tokens"] = bar_tokens
            out = self._arm("edited")
            fields["candidate"], fields["correct"] = out["candidate"], out["correct"]
            fields["tokens"], fields["hit_cap"] = self._tokens(out, n)
            passed = passes_tokens(
                out["correct"],
                fields["tokens"],
                base_correct,
                bar_tokens,
                n,
                self._cfg.margin,
            )
            if self._cfg.max_cap_hit is not None:
                fields["hit_cap"] = self._caps(out, n)
                if over := self._cap_over(fields["hit_cap"], n):
                    fields["cap_over"], passed = over, False
            if passed:
                st.bar_tokens, st.bar = fields["tokens"], out["correct"]
                self.best_candidate = fields["best_candidate"] = out["candidate"]
        except Exception as exc:  # noqa: BLE001 - the agent sees only passed
            passed = False  # an error after the rule check still fails
            fields["error"] = type(exc).__name__
        self._record.event("confirm_result", passed=passed, best=st.bar, **fields)
        return passed
