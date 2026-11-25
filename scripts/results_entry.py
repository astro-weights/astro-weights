"""Turn one gate run into a results-page entry.

Usage:
    python scripts/results_entry.py --name NAME --base BASE [--verdict verdict.json] --gate gate.json [--append results.json]

``gate.json`` is the stdout of ``astro-weights gate GATE.toml --json``. ``verdict.json``
is the file the gate writes beside ``GATE.toml``; a screened-out run writes none, so its
entry carries ``gate`` only. Both files are checked against the library's own schema,
so a malformed or mismatched pair fails here, not on the site. With ``--append``, the
entry is added to ``runs`` in that file (a placeholder file starts empty, a same-name
entry is replaced); without it, the entry is printed.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from astro_weights.eval.gate import OUTCOMES, SCREENED_OUT
from astro_weights.eval.verdict import Verdict


def entry(name: str, base: str, verdict: Path | None, gate: Path) -> dict[str, Any]:
    """The run entry; ValueError when the files are bad or disagree."""
    g = json.loads(gate.read_text("utf-8"))
    if not isinstance(g, dict) or g.get("outcome") not in OUTCOMES:
        raise ValueError(f"{gate}: not 'astro-weights gate --json' output")
    out: dict[str, Any] = {"name": name, "base": base}
    if g["outcome"] == SCREENED_OUT:
        if verdict is not None:
            raise ValueError("screened out runs write no verdict.json; drop --verdict")
    else:
        if verdict is None:
            raise ValueError(f"outcome {g['outcome']!r} needs --verdict verdict.json")
        v = Verdict.from_json(verdict.read_text("utf-8"))
        if v.verdict != g["outcome"]:
            raise ValueError(
                f"verdict.json says {v.verdict!r}, gate.json {g['outcome']!r}"
            )
        out["verdict"] = json.loads(v.to_json())
    out["gate"] = g
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", required=True, help="base model name")
    ap.add_argument("--verdict", type=Path)
    ap.add_argument("--gate", type=Path, required=True)
    ap.add_argument("--append", type=Path, help="results.json to add the entry to")
    a = ap.parse_args(argv)
    try:
        e = entry(a.name, a.base, a.verdict, a.gate)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if a.append is None:
        print(json.dumps(e, indent=1))
        return 0
    data = json.loads(a.append.read_text("utf-8")) if a.append.exists() else {}
    if data.pop("placeholder", False):
        data["runs"] = []  # the site's example runs go once real data arrives
    data["runs"] = [r for r in data.get("runs", []) if r.get("name") != a.name] + [e]
    a.append.write_text(json.dumps(data, indent=1) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
