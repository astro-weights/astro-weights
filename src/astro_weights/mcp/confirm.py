"""The ``confirm`` tool: pass or fail of the stored edits against the current best.

Rule (PDR 0.3 section 4.1, choice c, Ladder style: Blum and Hardt 2015,
https://arxiv.org/abs/1502.04585). A query generates greedy outputs of the edited
model for every confirm item, in file order, and grades them: ``c`` items correct
out of ``n``. The bar ``b`` is the count of the current best: the base model on the
first query, then the last edit that passed. The query passes iff
``c - b > margin * n``. The test is one-sided and strict, so a tie fails even at
margin 0. On a pass the bar becomes ``c``; on a fail it stays. Items, order and
decoding are fixed, so ``(c - b) / n`` is the paired mean difference.

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
    "generator",
    "passes",
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
    """True iff ``correct`` beats ``bar`` by more than ``margin * n`` items."""
    return correct - bar > margin * n


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
    """What the run record pins of the confirm setup: no path, no item."""
    return {
        "items_sha256": items[2],
        "grader": cfg.grader,
        "margin": cfg.margin,
        "max_queries": cfg.max_queries,
    }


def generator(backend: Any, max_new_tokens: int) -> Executor:
    """Executor ``{"arm", "prompts"}`` -> ``{"texts", "candidate"}`` on the forward model.

    Greedy, chat template when the tokenizer has one. ``candidate`` is
    ``candidate_id`` of the edits actually applied, or ``"base"``.
    """

    def run(params: dict[str, Any]) -> dict[str, Any]:
        from astro_weights.runners.generate import _prompt_text, _run_one

        arm = params["arm"]
        with backend._lock:  # one model, shared with forward_pass jobs
            backend._ensure()
            backend._sync(arm)
            model, tok = backend._model, backend._tok
            eos = model.generation_config.eos_token_id
            eos_ids = set(
                eos if isinstance(eos, list) else [] if eos is None else [eos]
            )
            texts = []
            for prompt in params["prompts"]:
                text, chat = _prompt_text(tok, prompt, True)
                texts.append(
                    _run_one(model, tok, text, chat, max_new_tokens, eos_ids)[0]
                )
            applied = dict(backend._applied)
        return {
            "texts": texts,
            "candidate": "base" if arm == "base" else candidate_id(applied),
        }

    return run


def _restore(events: list[dict[str, Any]]) -> tuple[int, int | None]:
    """(queries made, bar) from earlier record events."""
    queries = sum(e["event"] == "confirm_query" for e in events)
    bar = None
    for e in events:
        if e["event"] == "confirm_result" and isinstance(e.get("best"), int):
            bar = e["best"]
    return queries, bar


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
        self._queries, self._bar = _restore(record.read())

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

    def _correct(self, arm: str) -> tuple[int, str]:
        out = self._generate({"arm": arm, "prompts": list(self._prompts)})
        texts = out["texts"]
        if len(texts) != len(self._golds):
            raise ValueError("wrong number of outputs")
        n = sum(
            bool(self._grade(g, t)) for g, t in zip(self._golds, texts, strict=True)
        )
        return n, str(out["candidate"])

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
        except Exception as exc:  # noqa: BLE001 - the agent sees only passed
            fields["error"] = type(exc).__name__
        self._record.event("confirm_result", passed=passed, best=self._bar, **fields)
        return passed
