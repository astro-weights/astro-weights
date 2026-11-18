"""Compare the metadata (chat template, tokenizer, config keys) of two GGUF files.

Reads only the GGUF header, over HTTP range requests or from a local path, so the
multi-GB weights are never downloaded. Standard library only.

    python gguf_meta_diff.py A.gguf-or-URL B.gguf-or-URL [--dump-template DIR]

Prints one line per metadata key: "same" or "DIFF", with a sha256 of each value.
Exit code 0 if every tokenizer.* key (chat template included) is byte-identical.
"""

import hashlib
import struct
import sys
import urllib.request
from pathlib import Path

CHUNK = 8 << 20
MAX_HEADER = 256 << 20  # a header that claims more is refused, not downloaded


class Src:
    def __init__(self, where):
        self.where, self.buf, self.pos = where, b"", 0

    def _fill(self, need):
        if need > MAX_HEADER:
            raise ValueError(f"{self.where}: header larger than {MAX_HEADER} bytes")
        while len(self.buf) < need:
            start = len(self.buf)
            if self.where.startswith(("http://", "https://")):
                req = urllib.request.Request(
                    self.where, headers={"Range": f"bytes={start}-{start + CHUNK - 1}"}
                )
                with urllib.request.urlopen(req) as r:
                    if r.status != 206:  # Range ignored: would stream the whole file
                        raise ValueError(f"{self.where}: server ignored Range request")
                    data = r.read()
            else:
                with Path(self.where).open("rb") as f:
                    f.seek(start)
                    data = f.read(CHUNK)
            if not data:
                raise EOFError(self.where)
            self.buf += data

    def read(self, n):
        self._fill(self.pos + n)
        b = self.buf[self.pos : self.pos + n]
        self.pos += n
        return b

    def u(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize(fmt)))[0]


SCALAR = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}


def value(s, t):
    """Return the raw bytes of one value (used for hashing) and, for strings, the text."""
    if t in SCALAR:
        return s.read(struct.calcsize(SCALAR[t])), None
    if t == 8:
        n = s.u("Q")
        raw = s.read(n)
        return raw, raw.decode("utf-8", "replace")
    if t == 9:
        et, n = s.u("I"), s.u("Q")
        parts = [struct.pack("<IQ", et, n)]
        for _ in range(n):
            if et == 8:
                m = s.u("Q")
                parts.append(struct.pack("<Q", m) + s.read(m))
            else:
                parts.append(value(s, et)[0])
        return b"".join(parts), None
    raise ValueError(f"unknown GGUF type {t}")


def meta(where):
    s = Src(where)
    assert s.read(4) == b"GGUF", f"{where}: not a GGUF file"
    version, _tensors, nkv = s.u("I"), s.u("Q"), s.u("Q")
    out = {}
    for _ in range(nkv):
        key = s.read(s.u("Q")).decode()
        raw, text = value(s, s.u("I"))
        out[key] = (hashlib.sha256(raw).hexdigest(), text)
    return version, out


def main():
    a, b = sys.argv[1], sys.argv[2]
    dump = (
        Path(sys.argv[sys.argv.index("--dump-template") + 1])
        if "--dump-template" in sys.argv
        else None
    )
    va, ma = meta(a)
    vb, mb = meta(b)
    print(f"GGUF version {va} / {vb}; keys {len(ma)} / {len(mb)}")
    bad = 0
    for k in sorted(set(ma) | set(mb)):
        ha, hb = ma.get(k, ("-", None))[0], mb.get(k, ("-", None))[0]
        same = ha == hb
        if not same and k.startswith("tokenizer."):
            bad += 1
        print(f"{'same' if same else 'DIFF'}  {k}  {ha[:16]}  {hb[:16]}")
    if dump:
        dump.mkdir(parents=True, exist_ok=True)
        for tag, m in (("a", ma), ("b", mb)):
            t = m.get("tokenizer.chat_template", (None, None))[1]
            if t is not None:
                (dump / f"chat_template_{tag}.jinja").write_text(t, encoding="utf-8")
    print(
        "tokenizer keys (chat template included):",
        "IDENTICAL" if bad == 0 else f"{bad} DIFFER",
    )
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
