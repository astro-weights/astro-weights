"""``forward_pass``: request checks in the tool call, the model run in a job.

``check_request`` runs in the tool call, before anything is submitted. A request it
refuses (``BadRequest``) costs nothing and leaves no job. Token counts come from the
model's tokenizer.

``ForwardBackend.run`` runs in the job thread, one job at a time. It loads the model
on first use (offline), makes its weights match the arm (stored edits copied in for
``edited``, originals restored for ``base``; only changed tensors are touched), runs
the request and writes the arrays to ``run_dir/forward/<out>/<key>.npy``, a folder
the kernel cannot write. ``read_result`` reads them back for the kernel's
``forward_result`` request, only for a finished forward job.

torch and transformers are imported only inside functions (the ``hf`` extra).
"""

import importlib.util
import json
import math
import re
import shutil
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from astro_weights.mcp.edit_store import EditStore
from astro_weights.mcp.handles import HANDLE_RE, HandleStore, Job, UnknownHandle

__all__ = [
    "BadRequest",
    "ForwardBackend",
    "check_request",
    "discard_results",
    "executor",
    "hf_available",
    "read_result",
]

MAX_TEXTS = 1024
MAX_TEXT_TOKENS = 32768
MAX_TOTAL_TOKENS = 524288
MAX_TF_ELEMENTS = 4_000_000  # n_texts * last_tokens: sizes the logprob arrays
MAX_ATTN_TEXTS = 2
MAX_ATTN_TOKENS = 2048
MAX_PROMPTS = 512
MAX_SEED = 2**63
TEXT_PREVIEW = 4096  # chars per text in a generate summary
PREVIEW_TOTAL = 32768  # chars over all texts of a generate summary
MAX_ATTN_ELEMENTS = 2**28  # texts * layers * heads * tokens**2: 1 GiB of float32
LAST_TOKENS = 36
RESULTS = "forward"
HF_HINT = "it needs torch and transformers: pip install 'astro-weights[hf]'"
KEY_RE = re.compile(r"[a-z0-9_]{1,64}")
OUT_RE = re.compile(r"[0-9a-f]{32}")
_WANTS = ("attn", "hidden", "logprobs", "mlp")
_TF_KEYS = frozenset({"kind", "texts", "return", "layers", "last_tokens", "arm"})
_GEN_KEYS = frozenset(
    {"kind", "prompts", "max_new_tokens", "arm", "seed", "temperature"}
)
_DIRTY = "?"  # applied-map value for a tensor in an unknown state


class BadRequest(ValueError):
    """The request is malformed or over a limit; refused for free."""


class Model(Protocol):
    """What ``check_request`` needs from the model: token counts and layers."""

    def count_tokens(self, texts: Sequence[str], *, chat: bool) -> list[int]:
        """Tokens per text; ``chat`` applies the chat template as generation does."""
        ...

    def layers(self) -> tuple[int, list[int], int]:
        """(decoder layers, indices of full attention layers, attention heads)."""
        ...


def hf_available() -> bool:
    """True when torch, transformers and peft (the ``hf`` extra) can be imported."""
    return all(
        importlib.util.find_spec(m) is not None
        for m in ("torch", "transformers", "peft")
    )


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _strings(req: Mapping[str, Any], key: str, cap: int) -> list[str]:
    value: Any = req.get(key)
    ok = isinstance(value, list) and value and all(isinstance(t, str) for t in value)
    if not ok:
        raise BadRequest(f'"{key}" must be a non-empty list of strings')
    if len(value) > cap:
        raise BadRequest(f'"{key}" holds {len(value)} items; at most {cap}')
    if any(not t.strip() for t in value):
        raise BadRequest(f'"{key}" must not hold empty or whitespace-only strings')
    return list(value)


def _int(
    req: Mapping[str, Any], key: str, default: int | None, lo: int, hi: int
) -> int:
    value = req.get(key, default)
    if not _is_int(value) or not lo <= value <= hi:
        raise BadRequest(f'"{key}" must be an integer from {lo} to {hi}')
    return int(value)


def _check_keys(req: Mapping[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(str(k)[:80] for k in set(req) - allowed)
    if unknown:
        raise BadRequest(f"unknown keys {unknown}; allowed: {sorted(allowed)}")


def _check_arm(arm: Any, arms: tuple[str, ...], store: EditStore) -> str:
    if arm not in arms:
        raise BadRequest(f'"arm" must be one of {list(arms)}')
    if arm != "base":
        if not store.edited():
            raise BadRequest('there are no stored edits: save one or use "arm": "base"')
        try:
            store.validate()
        except ValueError as e:
            raise BadRequest(f"stored edits are invalid: {e}") from None
    return str(arm)


def _check_tokens(counts: Sequence[int]) -> None:
    if min(counts) == 0:
        raise BadRequest("every text must have at least one token")
    if max(counts) > MAX_TEXT_TOKENS or sum(counts) > MAX_TOTAL_TOKENS:
        raise BadRequest(
            f"token limits: {MAX_TEXT_TOKENS} per text and {MAX_TOTAL_TOKENS} in "
            f"total; got max {max(counts)}, total {sum(counts)}"
        )


def _teacher_forced(
    req: Mapping[str, Any], model: Model, store: EditStore
) -> dict[str, Any]:
    _check_keys(req, _TF_KEYS)
    texts = _strings(req, "texts", MAX_TEXTS)
    ret: Any = req.get("return")
    ok = isinstance(ret, list) and ret and all(x in _WANTS for x in ret)
    if not ok:
        raise BadRequest(f'"return" must be a non-empty list drawn from {list(_WANTS)}')
    want = sorted(set(ret))
    last = _int(req, "last_tokens", LAST_TOKENS, 1, MAX_TEXT_TOKENS)
    if len(texts) * last > MAX_TF_ELEMENTS:
        raise BadRequest(
            f"len(texts) * last_tokens must be at most {MAX_TF_ELEMENTS}; "
            f"got {len(texts) * last}"
        )
    layers = req.get("layers", [])
    n, full, heads = model.layers()
    if not isinstance(layers, list) or not all(
        _is_int(i) and 0 <= i < n for i in layers
    ):
        raise BadRequest(f'"layers" must be a list of layer indices from 0 to {n - 1}')
    if len(set(layers)) != len(layers):
        raise BadRequest('"layers" must not repeat a layer')
    if set(want) & {"hidden", "mlp", "attn"} and not layers:
        raise BadRequest('"layers" is required for hidden, mlp and attn')
    if "attn" in want:
        no = [i for i in layers if i not in full]
        if no:
            raise BadRequest(
                f'"attn": layers {no} have no attention weights; full attention '
                f"layers: {full}"
            )
        if len(texts) > MAX_ATTN_TEXTS:
            raise BadRequest(f'"attn" allows at most {MAX_ATTN_TEXTS} texts')
    arm = _check_arm(req.get("arm"), ("base", "edited"), store)
    counts = model.count_tokens(texts, chat=False)
    _check_tokens(counts)
    if "attn" in want:
        if max(counts) > MAX_ATTN_TOKENS:
            raise BadRequest(f'"attn" allows texts of at most {MAX_ATTN_TOKENS} tokens')
        size = len(texts) * len(layers) * heads * max(counts) ** 2
        if size > MAX_ATTN_ELEMENTS:
            raise BadRequest(
                f'"attn": texts * layers * heads * max_tokens**2 must be at most '
                f"{MAX_ATTN_ELEMENTS}; got {size} ({heads} heads)"
            )
    return {
        "kind": "teacher_forced",
        "texts": texts,
        "return": want,
        "layers": list(layers),
        "last_tokens": last,
        "arm": arm,
    }


def _generate(
    req: Mapping[str, Any], model: Model, store: EditStore, max_new_tokens: int
) -> dict[str, Any]:
    _check_keys(req, _GEN_KEYS)
    prompts = _strings(req, "prompts", MAX_PROMPTS)
    new = _int(req, "max_new_tokens", None, 1, max_new_tokens)
    seed = _int(req, "seed", 0, 0, MAX_SEED - 1)
    temp = req.get("temperature", 0.0)
    number = isinstance(temp, int | float) and not isinstance(temp, bool)
    if not number or not math.isfinite(temp) or temp < 0:
        raise BadRequest('"temperature" must be a finite number >= 0 (0 is greedy)')
    arm = _check_arm(req.get("arm"), ("base", "edited", "both"), store)
    _check_tokens(model.count_tokens(prompts, chat=True))
    return {
        "kind": "generate",
        "prompts": prompts,
        "max_new_tokens": new,
        "arm": arm,
        "seed": seed,
        "temperature": float(temp),
    }


def check_request(
    request: Any, model: Model, store: EditStore, *, max_new_tokens: int
) -> dict[str, Any]:
    """The request with defaults filled in; BadRequest if it may not run.

    ``max_new_tokens`` is the run's cap on generated tokens per prompt.
    """
    if not isinstance(request, Mapping):
        raise BadRequest("request must be a JSON object")
    kind = request.get("kind")
    if kind == "teacher_forced":
        return _teacher_forced(request, model, store)
    if kind == "generate":
        return _generate(request, model, store, max_new_tokens)
    raise BadRequest('"kind" must be "teacher_forced" or "generate"')


# results on disk


def _write_atomic(path: Path, write: Callable[[Any], None]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as f:
            write(f)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _save_array(path: Path, arr: np.ndarray) -> None:
    _write_atomic(
        path, lambda f: np.save(f, np.ascontiguousarray(arr), allow_pickle=False)
    )


def _save_texts(path: Path, texts: list[str]) -> None:
    _write_atomic(path, lambda f: f.write(json.dumps(texts).encode()))


def discard_results(run_dir: Path, job: Job) -> None:
    """Delete the result folder of a forward job that is not done: nobody can read it."""
    out = job.params.get("out")
    if job.kind != "forward" or job.state == "done":
        return
    if isinstance(out, str) and OUT_RE.fullmatch(out) is not None:
        shutil.rmtree(Path(run_dir) / RESULTS / out, ignore_errors=True)


def read_result(
    run_dir: Path, jobs: HandleStore, handle: Any, key: Any
) -> np.ndarray | list[str]:
    """Result ``key`` of finished forward job ``handle``; ValueError if there is none.

    The path is built from the job file, never from the request: the handle and key
    are checked first, the job must be a done ``forward`` job listing ``key``.
    """
    if not isinstance(handle, str) or HANDLE_RE.fullmatch(handle) is None:
        raise ValueError("bad handle")
    if not isinstance(key, str) or KEY_RE.fullmatch(key) is None:
        raise ValueError("bad key: expected [a-z0-9_]{1,64}")
    try:
        job = jobs.get(handle)
    except UnknownHandle:
        raise ValueError(f"unknown handle {handle}") from None
    if job.kind != "forward":
        raise ValueError(f"{handle} is not a forward_pass job")
    if job.state != "done":
        raise ValueError(f"{handle} is {job.state}, not done")
    out = job.params.get("out")
    keys = (job.result or {}).get("keys")
    if not isinstance(out, str) or OUT_RE.fullmatch(out) is None:
        raise ValueError(f"{handle} has no stored results")
    if not isinstance(keys, list) or key not in keys:
        raise ValueError(f"{handle} has no result {key!r}; results: {keys}")
    root = (Path(run_dir) / RESULTS).resolve()
    path = root / out / (f"{key}.json" if key.startswith("texts") else f"{key}.npy")
    if path.resolve().parent != root / out:
        raise ValueError(f"{handle}: result {key!r} is not a plain file")
    if key.startswith("texts"):
        texts: list[str] = json.loads(path.read_text(encoding="utf-8"))
        return texts
    arr: np.ndarray = np.load(path, allow_pickle=False)
    return arr


# the model


def _import_hf() -> tuple[Any, Any]:
    """(torch, transformers); ModuleNotFoundError naming the extra if missing."""
    try:
        import torch
        import transformers
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"forward_pass {HF_HINT}", name=e.name) from e
    return torch, transformers


def _auto_device(torch: Any) -> str:
    """cuda if present, else Apple mps, else cpu."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def device_problem(device: str) -> str | None:
    """Why an explicit ``[forward] device`` cannot run on this machine, else None.

    Checked before the editor starts, so a CPU-only torch fails the run before any
    editor turn is paid for. ``auto`` always has a device.
    """
    if device == "auto":
        return None
    torch, _ = _import_hf()
    try:
        dev = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        return f"[forward] device {device!r} is not a torch device: {exc}"
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            return (
                f"[forward] device {device!r}: this torch has no CUDA "
                f'(torch {torch.__version__}); install a CUDA build or set device = "auto"'
            )
        if dev.index is not None and dev.index >= torch.cuda.device_count():
            return f"[forward] device {device!r}: only {torch.cuda.device_count()} CUDA device(s)"
    if dev.type == "mps" and not torch.backends.mps.is_available():
        return f"[forward] device {device!r}: Apple MPS is not available"
    return None


class ForwardBackend:
    """The HF model behind ``forward_pass``; one per server, jobs run one at a time."""

    def __init__(
        self,
        model_dir: Path,
        store: EditStore,
        *,
        device: str = "auto",
        dtype: str = "auto",
    ) -> None:
        """Nothing is loaded here; the tokenizer loads on the first check, the model
        on the first job."""
        self.model_dir = Path(model_dir)
        self.store = store
        self._device = device
        self._dtype = dtype
        self._lock = threading.Lock()  # jobs, the model and the applied map
        self._tok_lock = threading.Lock()  # the tool call's own tokenizer
        self._counter: Any = None
        self._layers: tuple[int, list[int], int] | None = None
        self._model: Any = None
        self._tok: Any = None
        self._params: dict[str, Any] = {}
        self._applied: dict[str, str] = {}  # name -> sha256 of the edit in the param

    # tool call side

    def count_tokens(self, texts: Sequence[str], *, chat: bool) -> list[int]:
        """Tokens per text, as the job will tokenize them; counts stop past the limit."""
        from astro_weights.runners.generate import _prompt_text

        _, transformers = _import_hf()
        with self._tok_lock:
            if self._counter is None:
                self._counter = transformers.AutoTokenizer.from_pretrained(
                    self.model_dir, local_files_only=True
                )
            tok = self._counter
            special = False
            if chat:
                pairs = [_prompt_text(tok, t, True) for t in texts]
                texts = [t for t, _ in pairs]
                special = not pairs[0][1]
            enc = tok(
                list(texts),
                add_special_tokens=special,
                truncation=True,
                max_length=MAX_TEXT_TOKENS + 1,
            )
            return [len(ids) for ids in enc["input_ids"]]

    def layers(self) -> tuple[int, list[int], int]:
        """(layers, full attention layers, heads) of a weightless copy of the model."""
        from astro_weights.runners.hooks import full_attention_layers, layer_count

        torch, transformers = _import_hf()
        with self._tok_lock:
            if self._layers is None:
                config = transformers.AutoConfig.from_pretrained(
                    self.model_dir, local_files_only=True
                )
                with torch.device("meta"):
                    shell = transformers.AutoModelForCausalLM.from_config(config)
                heads = int(config.get_text_config().num_attention_heads)
                self._layers = (
                    layer_count(shell),
                    full_attention_layers(shell),
                    heads,
                )
            return self._layers

    # job side

    def run(self, request: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
        """Run a checked request; results go to ``out_dir``, removed on any error."""
        with self._lock:
            out_dir.mkdir(parents=True)
            try:
                self._ensure()
                if request["kind"] == "teacher_forced":
                    return self._teacher_forced(request, out_dir)
                return self._generate(request, out_dir)
            except BaseException:
                shutil.rmtree(out_dir, ignore_errors=True)
                raise

    def _ensure(self) -> None:
        if self._model is not None:
            return
        torch, transformers = _import_hf()
        device = self._device
        if device == "auto":
            device = _auto_device(torch)
        dtype = self._dtype if self._dtype == "auto" else getattr(torch, self._dtype)
        tok = transformers.AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=True
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_dir, local_files_only=True, dtype=dtype, device_map=device
        ).eval()
        self._params = dict(model.named_parameters())
        self._tok, self._model = tok, model

    def _param(self, name: str) -> Any:
        """The parameter for a checkpoint name, in either key layout."""
        for cand in (
            name,
            name.replace("model.language_model.", "model.", 1),
            name.replace("model.", "model.language_model.", 1),
        ):
            if cand in self._params:
                return self._params[cand]
        raise KeyError(f"the model has no parameter for {name}")

    def _copy(self, name: str, w: np.ndarray) -> None:
        torch, _ = _import_hf()
        p = self._param(name)
        p.copy_(torch.tensor(w).to(device=p.device, dtype=p.dtype))

    def _sync(self, arm: str) -> None:
        """Make the weights match ``arm``; on any error restore every original."""
        torch, _ = _import_hf()
        store = self.store
        with store._lock, torch.no_grad():  # no save or drop while edits are read
            recs = store.validate() if arm == "edited" else {}
            try:
                if arm == "edited" and not recs:  # dropped since the check
                    raise ValueError("there are no stored edits for the edited arm")
                for name in sorted(set(self._applied) | set(recs)):
                    sha = recs[name]["sha256"] if name in recs else None
                    if self._applied.get(name) == sha:
                        continue
                    if sha:
                        w = store._read_edit(name, recs[name])
                    else:
                        w = store.original(name)
                    self._applied[name] = _DIRTY
                    self._copy(name, w)
                    if sha:
                        self._applied[name] = sha
                    else:
                        del self._applied[name]
            except BaseException:
                self._restore()
                raise

    def _restore(self) -> None:
        """Copy back the original of every tensor that may differ from it."""
        for name in sorted(self._applied):
            self._copy(name, self.store.original(name))
            del self._applied[name]

    def _teacher_forced(self, req: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
        from astro_weights.runners.hooks import capture

        self._sync(req["arm"])
        ids = [
            list(self._tok(t, add_special_tokens=False)["input_ids"])
            for t in req["texts"]
        ]
        arrays = capture(
            self._model,
            ids,
            want=req["return"],
            layers=req["layers"],
            last_tokens=req["last_tokens"],
        )
        shapes: dict[str, list[int]] = {}
        for key, arr in arrays.items():
            name = key.lower()
            _save_array(out_dir / f"{name}.npy", arr)
            shapes[name] = list(arr.shape)
        return {
            "kind": "teacher_forced",
            "arm": req["arm"],
            "n": len(ids),
            "total_tokens": sum(len(i) for i in ids),
            "last_tokens": req["last_tokens"],
            "edits": dict(self._applied),
            "arrays": shapes,
            "keys": sorted(shapes),
        }

    def _generate(self, req: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
        from astro_weights.runners.generate import _prompt_text, _run_one

        torch, _ = _import_hf()
        model, tok = self._model, self._tok
        temp = req["temperature"]
        sampling = {"temperature": temp} if temp > 0 else None
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
        prompts = [_prompt_text(tok, p, True) for p in req["prompts"]]
        total = sum(
            len(tok(t, add_special_tokens=not chat)["input_ids"]) for t, chat in prompts
        )
        both = req["arm"] == "both"
        arms = ["base", "edited"] if both else [req["arm"]]
        summary: dict[str, Any] = {}
        keys: list[str] = []
        left = PREVIEW_TOTAL
        for arm in arms:
            self._sync(arm)
            torch.manual_seed(req["seed"])
            rows = [
                _run_one(
                    model, tok, text, chat, req["max_new_tokens"], eos_ids, sampling
                )
                for text, chat in prompts
            ]
            key = f"texts_{arm}" if both else "texts"
            _save_texts(out_dir / f"{key}.json", [r[0] for r in rows])
            keys.append(key)
            shown = []
            for r in rows:
                shown.append(r[0][: min(TEXT_PREVIEW, left)])
                left -= len(shown[-1])
            summary[arm] = {
                "edits": dict(self._applied),
                "texts": shown,
                "truncated": [
                    len(t) < len(r[0]) for t, r in zip(shown, rows, strict=True)
                ],
                "n_tokens": [r[1] for r in rows],
                "hit_cap": [r[2] for r in rows],
            }
        return {
            "kind": "generate",
            "arm": req["arm"],
            "n": len(prompts),
            "total_tokens": total,
            "max_new_tokens": req["max_new_tokens"],
            "seed": req["seed"],
            "temperature": temp,
            "arms": summary,
            "keys": keys,
        }


def executor(
    backend: ForwardBackend, run_dir: Path
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The ``forward`` job executor: params ``{"request", "out"}`` from the server."""
    root = Path(run_dir) / RESULTS

    def run(params: dict[str, Any]) -> dict[str, Any]:
        out = params["out"]
        if not isinstance(out, str) or OUT_RE.fullmatch(out) is None:
            raise ValueError("bad result folder name")
        return backend.run(params["request"], root / out)

    return run
