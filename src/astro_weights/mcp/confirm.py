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
restart, so the base arm never runs twice. Cap hits are recorded, not used.

``batch_size > 1`` generates that many items at once with left padding, for both
arms alike; greedy outputs can then differ from ``batch_size = 1`` (see
``generate_records``).

The agent learns ``{"passed": bool}`` and nothing else: no score, no n, no item,
no timing. Any error in a query is also ``{"passed": false}``; only the error's
type name is recorded. The items file is read here, in the server, at start; the
kernel sandbox cannot read it and no tool returns it. Counts and the bar go to the
run record (``confirm_query`` and ``confirm_result`` events), which no tool returns.
"""

import contextlib
import hashlib
import json
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
    return out


def generator(backend: Any, max_new_tokens: int) -> Executor:
    """Executor ``{"arm", "prompts", "batch_size"?}`` on the forward model.

    Returns ``{"texts", "tokens", "hit_cap", "candidate"}``: per prompt the output,
    its token count and whether it hit ``max_new_tokens``. Greedy, chat template
    when the tokenizer has one. ``batch_size`` (default 1) > 1 left-pads that many
    prompts per call, as ``generate_records`` does. ``candidate`` is
    ``candidate_id`` of the edits actually applied, or ``"base"``.
    """

    def run(params: dict[str, Any]) -> dict[str, Any]:
        from astro_weights.runners.generate import _prompt_text, _run_batch, _run_one

        arm = params["arm"]
        size = int(params.get("batch_size", 1))
        with backend._lock:  # one model, shared with forward_pass jobs
            backend._ensure()
            backend._sync(arm)
            model, tok = backend._model, backend._tok
            eos = model.generation_config.eos_token_id
            eos_ids = set(
                eos if isinstance(eos, list) else [] if eos is None else [eos]
            )
            if size > 1 and tok.pad_token_id is None:
                raise ValueError("confirm.batch_size > 1 needs a pad token")
            shown = [_prompt_text(tok, p, True) for p in params["prompts"]]
            rows: list[tuple[str, int, bool]] = []
            for start in range(0, len(shown), size):
                chunk = shown[start : start + size]
                if size == 1:
                    text, chat = chunk[0]
                    rows.append(
                        _run_one(model, tok, text, chat, max_new_tokens, eos_ids)
                    )
                else:
                    texts = [t for t, _ in chunk]
                    rows += _run_batch(
                        model, tok, texts, chunk[0][1], max_new_tokens, eos_ids
                    )
            applied = dict(backend._applied)
        return {
            "texts": [r[0] for r in rows],
            "tokens": [r[1] for r in rows],
            "hit_cap": [r[2] for r in rows],
            "candidate": "base" if arm == "base" else candidate_id(applied),
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
        self.queries = 0
        self.bar: int | None = None  # accuracy target: correct count of the best
        self.best: str | None = None  # candidate of the last pass
        self.base: tuple[int, int] | None = None  # tokens target: correct, tokens
        self.bar_tokens: int | None = None  # tokens target: total of the best


def _restore(events: list[dict[str, Any]]) -> _State:
    """Queries made, bars and best candidate from earlier record events."""
    st = _State()
    st.queries = sum(e["event"] == "confirm_query" for e in events)
    for e in events:
        if e["event"] == "confirm_base":
            st.base = (int(e["base_correct"]), int(e["base_tokens"]))
            st.bar_tokens = st.base[1]
            if st.bar is None:  # a crash before the first result: best is still base
                st.bar = st.base[0]
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
        self._st = _restore(record.read())
        self._queries, self._bar = self._st.queries, self._st.bar
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
            query = self._queries
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

    def _correct(self, arm: str) -> tuple[int, str]:
        out = self._arm(arm)
        return out["correct"], out["candidate"]

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
        fields: dict[str, Any] = {"query": query, "n": len(self._golds)}
        passed = False
        try:
            if self._bar is None:
                self._bar, _ = self._correct("base")
            fields["bar"] = self._bar
            correct, fields["candidate"] = self._correct("edited")
            fields["correct"] = correct
            passed = passes(correct, self._bar, len(self._golds), self._cfg.margin)
            if passed:
                self._bar = correct
                self.best_candidate = fields["best_candidate"] = fields["candidate"]
        except Exception as exc:  # noqa: BLE001 - the agent sees only passed
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
                st.base, st.bar_tokens = (base["correct"], tokens), tokens
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
            if passed:
                st.bar_tokens, st.bar = fields["tokens"], out["correct"]
                self.best_candidate = fields["best_candidate"] = out["candidate"]
        except Exception as exc:  # noqa: BLE001 - the agent sees only passed
            fields["error"] = type(exc).__name__
        self._record.event("confirm_result", passed=passed, best=st.bar, **fields)
        return passed
