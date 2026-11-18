"""Compare the metadata (chat template, tokenizer, config keys) of two GGUF files.

Reads only the GGUF header, over HTTP range requests or from a local path, so the
multi-GB weights are never downloaded. Standard library only.

    python gguf_meta_diff.py A.gguf-or-URL B.gguf-or-URL [--dump-template DIR]
    python gguf_meta_diff.py A.gguf B.gguf --tensors

Prints one line per metadata key: "same" or "DIFF", with a sha256 of each value.
Exit code 0 if every tokenizer.* key (chat template included) is byte-identical.

--tensors (local files only) instead hashes each tensor's data, from its offset to the
next tensor's offset (alignment padding included), and lists the tensors that differ
in bytes, type or shape. Exit code 0 if none differ.
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


def tensors(path):
    """sha256 of each tensor's bytes in a local GGUF (range = this offset to next offset)."""
    s = Src(path)
    assert s.read(4) == b"GGUF", f"{path}: not a GGUF file"
    s.u("I")
    nt, nkv = s.u("Q"), s.u("Q")
    align = 32
    for _ in range(nkv):
        key = s.read(s.u("Q")).decode()
        t = s.u("I")
        raw, _ = value(s, t)
        if key == "general.alignment":
            align = struct.unpack("<I", raw)[0]
    info = []
    for _ in range(nt):
        name = s.read(s.u("Q")).decode()
        nd = s.u("I")
        shape = [s.u("Q") for _ in range(nd)]
        ttype, off = s.u("I"), s.u("Q")
        info.append((off, name, ttype, shape))
    start = (s.pos + align - 1) // align * align
    size = Path(path).stat().st_size
    info.sort()
    out = {}
    with Path(path).open("rb") as f:
        for i, (off, name, ttype, shape) in enumerate(info):
            end = info[i + 1][0] if i + 1 < len(info) else size - start
            if start + end > size:
                raise ValueError(f"{path}: tensor {name} runs past end of file")
            f.seek(start + off)
            h, left = hashlib.sha256(), end - off
            while left:
                chunk = f.read(min(left, CHUNK))
                if not chunk:
                    raise EOFError(path)
                h.update(chunk)
                left -= len(chunk)
            out[name] = (h.hexdigest(), ttype, shape)
    return out


def main():
    a, b = sys.argv[1], sys.argv[2]
    if "--tensors" in sys.argv:  # local files only: hashes every tensor's data
        ta, tb = tensors(a), tensors(b)
        same = [k for k in ta if k in tb and ta[k] == tb[k]]
        diff = sorted(set(ta) ^ set(tb) | {k for k in ta if k in tb and ta[k] != tb[k]})
        print(
            f"tensors {len(ta)} / {len(tb)}; identical {len(same)}; differ {len(diff)}"
        )
        for k in diff:
            print(
                f"DIFF  {k}  {ta.get(k, ('-',))[0][:16]} type {ta.get(k, ('', '-'))[1]}  {tb.get(k, ('-',))[0][:16]} type {tb.get(k, ('', '-'))[1]}"
            )
        return 0 if not diff else 1
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
