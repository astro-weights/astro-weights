"""``astro-weights run CONFIG``: one TOML file starts the server and an editor.

The config names the run directory, the budget, the editor and its options::

    [run]
    dir = "runs/first"        # relative paths resolve against the config's folder
    seed = 0
    model_dir = "models/tiny" # optional; without it exec_python and forward_pass are off

    [limits]
    forward_passes = 20
    code_runs = 40
    tokens = 2_000_000
    dollars = 20.0            # default 20.0 for the hosted editor, 0 for the others

    [editor]
    kind = "local"            # "scripted", "local" or "hosted", then that editor's options
    base_url = "http://127.0.0.1:8000"
    model = "my-model"

    [kernel]                  # optional: timeout_s, mem_gib, unsafe, allow_nested_userns
    [forward]                 # optional: device, dtype, max_new_tokens
    [export]                  # optional: rank, max_rel_residual, base_gguf, gguf_out_type

``start`` writes ``RUN_DIR/run.json`` and a copy of the config (``config.toml``),
then runs the editor's ``main`` on the run directory; the editor starts the server.
Nothing in a run directory is ever deleted. A fresh start refuses a folder that is
not empty, unless all it holds is what an aborted start of the same run wrote. A
directory that holds a run (a record, a ledger, jobs or edits) is refused unless
``resume`` is set, and a resume must keep the seed, model directory, editor, editor
model and limits, so the budget already used still counts; it rewrites run.json
and adds ``config.resume-<n>.toml``, keeping the first config copy. ``status`` reads the run from disk and writes
nothing. Standard library only; the editor needs its extras.
"""

import importlib
import importlib.util
import json
import math
import os
import re
import secrets
import shutil
import tempfile
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeGuard

from astro_weights.mcp import forward, jobs
from astro_weights.mcp.budget import Ledger, Limits
from astro_weights.mcp.editors import prices
from astro_weights.mcp.record import RunRecord

__all__ = ["RunError", "RunSpec", "format_status", "load_config", "start", "status"]

HOSTED_DOLLARS = 20.0
LAST_EVENTS = 10
CONFIG_COPY = "config.toml"
HOSTED_KEY_ENV = "ANTHROPIC_API_KEY"
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# all an aborted start (checks passed, files written, editor never ran) can leave
_ABORTED = frozenset({"run.json", CONFIG_COPY})
# any of these in a run directory means a server ran there: never overwrite it
_STARTED = ("record.jsonl", "ledger.jsonl", "jobs", "edits")
_SAME_ON_RESUME = ("seed", "model_dir", "editor", "model", "limits")
_REQUIRED = object()

# key -> (type, default); _REQUIRED marks a key that must be given, None an unset option
_Fields = dict[str, tuple[str, object]]
_RUN: _Fields = {
    "dir": ("path", _REQUIRED),
    "seed": ("int", _REQUIRED),
    "model_dir": ("path", None),
}
_LIMITS: _Fields = {
    "forward_passes": ("count", _REQUIRED),
    "code_runs": ("count", _REQUIRED),
    "tokens": ("count", _REQUIRED),
    "dollars": ("money", None),
}
_KERNEL: _Fields = {
    "timeout_s": ("number", None),
    "mem_gib": ("int", None),
    "unsafe": ("bool", False),
    "allow_nested_userns": ("bool", False),
}
_FORWARD: _Fields = {
    "device": ("str", None),
    "dtype": ("str", None),
    "max_new_tokens": ("int", None),
}
_EXPORT: _Fields = {
    "rank": ("positive", None),
    "max_rel_residual": ("money", None),
    "base_gguf": ("path", None),
    "gguf_out_type": ("str", None),
}
# editor options map one to one onto the editor's command-line flags
_EDITORS: dict[str, _Fields] = {
    "scripted": {},
    "local": {
        "base_url": ("str", _REQUIRED),
        "model": ("str", _REQUIRED),
        "api_key_env": ("env", None),
        "max_turns": ("positive", None),
        "temperature": ("money", None),
    },
    "hosted": {
        "model": ("str", _REQUIRED),
        "price": ("str", None),
        "max_output_tokens": ("positive", None),
        "max_turns": ("positive", None),
        "fallback_base_url": ("str", None),
        "fallback_model": ("str", None),
        "no_fallback": ("bool", False),
    },
}
_TABLES = ("run", "limits", "editor", "kernel", "forward", "export")
_TYPE_TEXT = {
    "path": "a non-empty string",
    "str": "a non-empty string",
    "env": "the name of an environment variable",
    "int": "an int",
    "count": "an int >= 0",
    "positive": "an int >= 1",
    "number": "a number",
    "money": "a finite number >= 0",
    "bool": "true or false",
}


class RunError(ValueError):
    """The config is bad, or the run cannot start, resume or be read; nothing was written."""


def _whole(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _finite(value: float) -> bool:
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def _ok(kind: str, value: object) -> bool:
    if kind in ("path", "str"):
        return isinstance(value, str) and value.strip() != ""
    if kind == "env":
        return isinstance(value, str) and _ENV_NAME.fullmatch(value) is not None
    if kind == "int":
        return _whole(value)
    if kind in ("count", "positive"):
        return _whole(value) and value >= (1 if kind == "positive" else 0)
    if kind == "number":
        return _number(value)
    if kind == "money":
        return _number(value) and _finite(value) and value >= 0
    return isinstance(value, bool)


def _table(where: str, raw: object, fields: _Fields, name: str) -> dict[str, Any]:
    """The checked keys of one TOML table, with defaults; RunError naming the bad key."""
    if not isinstance(raw, dict):
        raise RunError(f"{where}: [{name}] must be a table")
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise RunError(
            f"{where}: unknown key [{name}] {unknown[0]}; expected one of {sorted(fields)}"
        )
    out: dict[str, Any] = {}
    for key, (kind, default) in fields.items():
        if key not in raw:
            if default is _REQUIRED:
                raise RunError(f"{where}: [{name}] {key} is missing")
            out[key] = default
            continue
        value = raw[key]
        if not _ok(kind, value):
            # a pasted secret in place of a variable name is never echoed
            got = "" if kind == "env" else f", got {value!r}"
            raise RunError(f"{where}: [{name}] {key} must be {_TYPE_TEXT[kind]}{got}")
        out[key] = value
    return out


def _path(base: Path, text: str) -> Path:
    path = Path(text)
    return (path if path.is_absolute() else base / path).resolve()


@dataclass(frozen=True)
class RunSpec:
    """A checked run config. Paths are absolute."""

    source: Path
    run_dir: Path
    seed: int
    model_dir: Path | None
    limits: Limits
    editor: str
    options: Mapping[str, Any]
    kernel: Mapping[str, Any]
    forward: Mapping[str, Any]
    export: Mapping[str, Any] = field(default_factory=dict)
    unsafe_kernel: bool = False
    text: str = field(default="", repr=False)  # the config as read, copied to the run

    @property
    def model(self) -> str | None:
        """The editor's model name; None for the scripted editor."""
        model = self.options.get("model")
        return model if isinstance(model, str) else None

    def run_json(self) -> dict[str, Any]:
        """The contents of ``RUN_DIR/run.json`` (see ``jobs.load_run_config``)."""
        out: dict[str, Any] = {
            "seed": self.seed,
            "editor": self.editor,
            "model": self.model,
            "limits": asdict(self.limits),
            "kernel": dict(self.kernel),
            "forward": dict(self.forward),
            "export": dict(self.export),
        }
        if self.model_dir is not None:
            out["model_dir"] = str(self.model_dir)
        return out

    def editor_argv(self) -> list[str]:
        """The arguments for the editor's ``main``."""
        argv = [str(self.run_dir)]
        for key, value in self.options.items():
            flag = "--" + key.replace("_", "-")
            if value is True:
                argv.append(flag)
            elif value is not None and value is not False:
                argv += [flag, str(value)]
        if self.editor == "hosted":  # the config's dollar limit is the cap
            argv += ["--max-dollars", repr(self.limits.dollars)]
        if self.unsafe_kernel:
            argv.append("--unsafe-kernel")
        return argv


def _parse(where: str, raw: dict[str, Any], base: Path, text: str) -> RunSpec:
    unknown = sorted(set(raw) - set(_TABLES))
    if unknown:
        raise RunError(
            f"{where}: unknown table [{unknown[0]}]; expected {list(_TABLES)}"
        )
    for name in ("run", "limits", "editor"):
        if name not in raw:
            raise RunError(f"{where}: [{name}] table is missing")
    run = _table(where, raw["run"], _RUN, "run")
    editor_raw = raw["editor"]
    if not isinstance(editor_raw, dict):
        raise RunError(f"{where}: [editor] must be a table")
    kind = editor_raw.get("kind")
    if not isinstance(kind, str) or kind not in _EDITORS:
        raise RunError(
            f"{where}: [editor] kind must be one of {sorted(_EDITORS)}, got {kind!r}"
        )
    options = {k: v for k, v in editor_raw.items() if k != "kind"}
    options = _table(where, options, _EDITORS[kind], "editor")
    limits = _table(where, raw["limits"], _LIMITS, "limits")
    if limits["dollars"] is None:
        limits["dollars"] = HOSTED_DOLLARS if kind == "hosted" else 0.0
    limits["dollars"] = float(limits["dollars"])
    if kind == "hosted":
        _check_hosted(where, options, limits["dollars"])
    kernel = _table(where, raw.get("kernel", {}), _KERNEL, "kernel")
    unsafe = bool(kernel.pop("unsafe"))
    kernel = {k: v for k, v in kernel.items() if v is not None}
    fwd = {
        k: v
        for k, v in _table(where, raw.get("forward", {}), _FORWARD, "forward").items()
        if v is not None
    }
    export = {
        k: v
        for k, v in _table(where, raw.get("export", {}), _EXPORT, "export").items()
        if v is not None
    }
    if "base_gguf" in export:
        export["base_gguf"] = str(_path(base, export["base_gguf"]))
    try:  # the server's own checks, so run.json is valid when written
        timeout_s, mem_gib, nested = jobs._kernel_config(Path(where), kernel)
        device, dtype, max_new = jobs._forward_config(Path(where), fwd)
        rank, residual, _gguf, out_type = jobs._export_config(Path(where), export)
    except ValueError as exc:
        raise RunError(str(exc)) from None
    export.update(rank=rank, max_rel_residual=residual, gguf_out_type=out_type)
    return RunSpec(
        source=Path(where),
        run_dir=_path(base, run["dir"]),
        seed=run["seed"],
        model_dir=None if run["model_dir"] is None else _path(base, run["model_dir"]),
        limits=Limits(**limits),
        editor=kind,
        options=options,
        kernel={"timeout_s": timeout_s, "mem_gib": mem_gib}
        | ({"allow_nested_userns": True} if nested else {}),
        forward={"device": device, "dtype": dtype, "max_new_tokens": max_new},
        export=export,
        unsafe_kernel=unsafe,
        text=text,
    )


def _check_hosted(where: str, options: dict[str, Any], dollars: float) -> None:
    if dollars <= 0:
        raise RunError(f"{where}: [limits] dollars must be > 0 for the hosted editor")
    try:
        text = options["price"]
        own = None if text is None else prices.parse_price(text)
        prices.price(options["model"], given=own)
    except ValueError as exc:
        raise RunError(f"{where}: [editor] {exc}") from None
    given = [options["fallback_base_url"], options["fallback_model"]]
    if options["no_fallback"] and any(given):
        raise RunError(
            f"{where}: [editor] no_fallback takes no fallback_base_url or fallback_model"
        )
    if not options["no_fallback"] and not all(given):
        raise RunError(
            f"{where}: [editor] give fallback_base_url and fallback_model, "
            "or no_fallback = true"
        )


def load_config(path: Path) -> RunSpec:
    """Read and check a TOML run config; RunError with a clear message if bad."""
    path = Path(path).resolve()
    try:
        # bytes, not read_text: newline translation would make the copy differ from the file (CRLF)
        text = path.read_bytes().decode("utf-8")
        raw = tomllib.loads(text)
    except FileNotFoundError:
        raise RunError(f"{path} not found") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise RunError(f"{path} is not readable: {exc}") from None
    except tomllib.TOMLDecodeError as exc:
        raise RunError(f"{path} is not valid TOML: {exc}") from None
    return _parse(str(path), raw, path.parent, text)


def _started(run_dir: Path) -> bool:
    return any((run_dir / name).exists() for name in _STARTED)


def _saved(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        raise RunError(f"cannot read the saved {path}: {exc}") from None
    if not isinstance(saved, dict):
        raise RunError(f"the saved {path} is not a JSON object")
    return saved


def _events(run_dir: Path) -> list[dict[str, Any]]:
    """The record's complete lines. Read only: a torn last line is skipped, not cut."""
    path = run_dir / "record.jsonl"
    if not path.is_file():
        return []
    try:
        return RunRecord(path).read()
    except (OSError, ValueError) as exc:
        raise RunError(f"cannot read {path}: {exc}") from None


def _finish(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((e for e in reversed(events) if e.get("event") == "finish"), None)


def _differences(saved: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    return [
        f"  {key}: run has {json.dumps(saved.get(key))}, config has {json.dumps(new.get(key))}"
        for key in _SAME_ON_RESUME
        if saved.get(key) != new.get(key)
    ]


def _check_resume(spec: RunSpec) -> None:
    run_dir = spec.run_dir
    if not _started(run_dir):
        raise RunError(f"{run_dir} holds no run to resume; start it without --resume")
    diff = _differences(_saved(run_dir), spec.run_json())
    if diff:
        raise RunError(
            f"the config does not match the run in {run_dir}; nothing was changed:\n"
            + "\n".join(diff)
        )
    if _finish(_events(run_dir)) is not None:
        raise RunError(f"the run in {run_dir} already finished")


def _check_start(spec: RunSpec) -> None:
    """A fresh start needs a new or empty run dir, or one an aborted start of this run left."""
    run_dir = spec.run_dir
    if run_dir.exists() and not run_dir.is_dir():
        raise RunError(f"{run_dir} exists and is not a directory")
    if _started(run_dir):
        raise RunError(
            f"{run_dir} already holds a run; pass --resume to go on with it, "
            "or set a new [run] dir"
        )
    if not run_dir.is_dir():
        return
    try:
        names = {p.name for p in run_dir.iterdir()}
    except OSError as exc:
        raise RunError(f"cannot read {run_dir}: {exc}") from None
    if not names:
        return
    aborted = names <= _ABORTED and "run.json" in names
    if aborted:
        try:
            aborted = not _differences(_saved(run_dir), spec.run_json())
        except RunError:
            aborted = False
    if not aborted:
        raise RunError(
            f"{run_dir} is not empty and holds no run started from this config; "
            "set a new [run] dir (files there are never overwritten)"
        )


def _check_keys(spec: RunSpec) -> None:
    """The editor's API key variable must be set; checked before anything is written."""
    names = [spec.options.get("api_key_env")]
    if spec.editor == "hosted":
        names.append(HOSTED_KEY_ENV)
    for name in names:
        if isinstance(name, str) and not os.environ.get(name):
            raise RunError(f"${name} is not set")


def _config_copy(run_dir: Path, resume: bool) -> Path:
    """Where the config copy goes; a resume never replaces an earlier copy."""
    first = run_dir / CONFIG_COPY
    if not resume or not first.exists():
        return first
    n = 1
    while (run_dir / f"config.resume-{n}.toml").exists():
        n += 1
    return run_dir / f"config.resume-{n}.toml"


def _editor_main(kind: str) -> Callable[[list[str]], int]:
    """The editor's ``main``; RunError with the extras hint when one is missing."""
    try:
        module = importlib.import_module(f"astro_weights.mcp.editors.{kind}")
    except ModuleNotFoundError as exc:
        raise RunError(str(exc)) from None
    main: Callable[[list[str]], int] = module.main
    return main


def _write(path: Path, data: bytes) -> None:
    """Atomic write: a temporary file in the same folder, then a rename over the target."""
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _gguf_installed() -> bool:
    return importlib.util.find_spec("gguf") is not None


def start(spec: RunSpec, *, resume: bool = False) -> int:
    """Check, write run.json and the config copy, run the editor; its exit code.

    RunError, before anything is written, when the run directory already holds a
    run (and ``resume`` is not set), when a resume does not match the saved run or
    the run already finished, when the model directory or ``export.base_gguf`` is
    missing, when the editor's API key variable is not set, or when an extra the
    run needs is not installed. A fresh start needs a new or empty run directory
    (or one only an aborted start of this same run wrote to); a resume keeps
    ``config.toml`` and writes its own copy as ``config.resume-<n>.toml``.
    """
    if resume:
        _check_resume(spec)
    else:
        _check_start(spec)
    main = _editor_main(spec.editor)
    _check_keys(spec)
    if spec.model_dir is not None:
        if not forward.hf_available():
            raise RunError(f"forward_pass {forward.HF_HINT}")
        if not spec.model_dir.is_dir():
            raise RunError(f"model_dir {spec.model_dir} is not a directory")
    gguf = spec.export.get("base_gguf")
    if gguf is not None:  # found now, not at finish after the budget is spent
        if not _gguf_installed():
            raise RunError("export.base_gguf needs: pip install 'astro-weights[gguf]'")
        if not Path(gguf).is_file():
            raise RunError(f"export.base_gguf {gguf} is not a file")
    run_json = json.dumps(spec.run_json(), indent=1) + "\n"
    try:
        spec.run_dir.mkdir(parents=True, exist_ok=True)
        _write(spec.run_dir / "run.json", run_json.encode())
        _write(_config_copy(spec.run_dir, resume), spec.text.encode())
    except OSError as exc:
        raise RunError(f"cannot write the run directory: {exc}") from None
    try:
        return main(spec.editor_argv())
    except SystemExit as exc:  # argparse in the editor; the config was checked
        return exc.code if isinstance(exc.code, int) else 2


def _budget(run_dir: Path, limits: Limits) -> dict[str, Any]:
    """The ledger's budget, replayed from a copy so the run's file is never touched."""
    path = run_dir / "ledger.jsonl"
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "ledger.jsonl"
        if path.is_file():
            shutil.copyfile(path, copy)
        try:
            return Ledger(copy, limits).to_json()
        except ValueError as exc:
            raise RunError(f"cannot read {path}: {exc}") from None


def _edits(run_dir: Path) -> list[str]:
    try:
        manifest = json.loads((run_dir / "edits" / "manifest.json").read_text("utf-8"))
    except (OSError, ValueError, RecursionError):
        return []
    return sorted(manifest) if isinstance(manifest, dict) else []


def status(spec: RunSpec, last: int = LAST_EVENTS) -> dict[str, Any]:
    """The run as it is on disk: state, budget, edits, the last events. Writes nothing.

    ``state`` is ``not_started``, ``unfinished`` (a server ran; it may still be
    running) or ``finished``.
    """
    run_dir = spec.run_dir
    out: dict[str, Any] = {"run_dir": str(run_dir), "state": "not_started"}
    if not (run_dir / "run.json").exists() and not _started(run_dir):
        return out
    try:
        cfg = jobs.load_run_config(run_dir)
    except ValueError as exc:
        raise RunError(str(exc)) from None
    events = _events(run_dir)
    finish = _finish(events)
    if finish is not None:
        out["state"] = "finished"
    elif _started(run_dir):
        out["state"] = "unfinished"
    out.update(
        editor=cfg.editor,
        model=cfg.model,
        seed=cfg.seed,
        budget=_budget(run_dir, cfg.limits),
        finish=None
        if finish is None
        else {"outcome": finish.get("outcome"), "time": finish.get("time")},
        edits=_edits(run_dir),
        events=events[-last:] if last > 0 else [],
    )
    return out


def _brief(event: Mapping[str, Any]) -> str:
    rest = {k: v for k, v in event.items() if k not in ("event", "time")}
    text = json.dumps(rest, default=str)
    text = text if len(text) <= 100 else text[:97] + "..."
    return f"  {event.get('time', '?')}  {event.get('event')}  {text}"


def format_status(info: Mapping[str, Any]) -> str:
    """``status`` as text for a terminal."""
    lines = [f"run: {info['run_dir']}", f"state: {info['state']}"]
    if info["state"] == "not_started":
        return "\n".join(lines)
    lines.append(
        f"editor: {info['editor']}  model: {info['model']}  seed: {info['seed']}"
    )
    finish = info["finish"]
    if finish is not None:
        lines.append(f"finish: {finish['outcome']} at {finish['time']}")
    budget = info["budget"]
    lines.append("budget (used / limit):")
    for kind, limit in budget["limits"].items():
        used = budget["used"][kind]
        streak = budget["streaks"][kind]
        tail = f"  failures in a row: {streak}" if streak else ""
        lines.append(f"  {kind}: {used} / {limit}{tail}")
    lines.append(f"edits: {', '.join(info['edits']) or 'none'}")
    lines.append(f"last {len(info['events'])} events:")
    lines += [_brief(e) for e in info["events"]]
    return "\n".join(lines)
