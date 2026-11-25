"""generate_records: header beside out, resume refuses a file made under another config."""

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest
from fake_endpoint import ok, serve

from astro_weights.eval import Record, grade_exact, header_path, read_header, read_jsonl

pytest.importorskip("torch")
pytest.importorskip("peft")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from astro_weights.runners.endpoint import generate_records_http
from astro_weights.runners.generate import generate_records, run_header

N_ITEMS = 8
ITEMS = [
    {"item_id": f"i{k}", "prompt": " ".join(["w1", "w2", f"w{k + 3}"]), "gold": "w5"}
    for k in range(N_ITEMS)
]


def _tok() -> Any:
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2} | {f"w{i}": i + 3 for i in range(125)}
    t = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=t,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="</s>",
    )


@pytest.fixture
def tok() -> Any:
    return _tok()


def _run(
    model: Any, tok: Any, *, cap: int = 6, batch_size: int = 1, **kw: Any
) -> list[Record]:
    kw.setdefault(
        "header", run_header(model, tok, cap, 0, False, batch_size=batch_size)
    )
    return generate_records(
        model, tok, ITEMS, "base", grade_exact, cap, chat=False,
        batch_size=batch_size, **kw,
    )  # fmt: skip


def test_header_written_beside_out(tiny_llama: Any, tok: Any, tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    header = run_header(tiny_llama, tok, 6, 0, False)
    _run(tiny_llama, tok, out=out, header=header)
    assert read_header(header_path(out)) == header


def test_no_header_no_file(tiny_llama: Any, tok: Any, tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    _run(tiny_llama, tok, out=out, header=None)
    assert not header_path(out).exists()


def test_resume_keeps_earlier_header(tiny_llama: Any, tok: Any, tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    first = run_header(tiny_llama, tok, 6, 0, False, run_id="first")
    _run(tiny_llama, tok, out=out, header=first)
    again = run_header(tiny_llama, tok, 6, 0, False, run_id="second")
    _run(tiny_llama, tok, out=out, resume=True, header=again)
    assert read_header(header_path(out))["run_id"] == "first"  # same config: kept


def test_resume_without_header_file_writes_it(
    tiny_llama: Any, tok: Any, tmp_path: Path
) -> None:
    out = tmp_path / "r.jsonl"
    _run(tiny_llama, tok, out=out, header=None)
    header = run_header(tiny_llama, tok, 6, 0, False)
    _run(tiny_llama, tok, out=out, resume=True, header=header)
    assert read_header(header_path(out)) == header


@pytest.mark.parametrize(
    "change",
    [{"max_new_tokens": 7}, {"batch_size": 2}, {"sampling": {"temperature": 0.7}}],
)
def test_resume_other_config_refused(
    tiny_llama: Any, tok: Any, tmp_path: Path, change: dict[str, Any]
) -> None:
    out = tmp_path / "r.jsonl"
    old = run_header(tiny_llama, tok, 6, 0, False)
    whole = _run(tiny_llama, tok, out=out, header=old)
    # a crash's partial last line: a refused resume must not repair it either
    with out.open("ab") as f:
        f.write(b'{"item_id": "i0", "ar')
    out_bytes, head_bytes = out.read_bytes(), header_path(out).read_bytes()
    new = run_header(
        tiny_llama, tok, change.get("max_new_tokens", 6), 0, False,
        batch_size=change.get("batch_size", 1), sampling=change.get("sampling"),
    )  # fmt: skip
    with pytest.raises(ValueError, match="another config"):
        _run(tiny_llama, tok, out=out, resume=True, header=new)
    assert out.read_bytes() == out_bytes
    assert header_path(out).read_bytes() == head_bytes
    assert len(whole) == N_ITEMS


def test_resume_other_system_refused(tiny_llama: Any, tok: Any, tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    _run(tiny_llama, tok, out=out)
    before = out.read_bytes()
    other = run_header(tiny_llama, tok, 6, 0, False, system="be brief")
    with pytest.raises(ValueError, match="another config"):
        _run(tiny_llama, tok, out=out, resume=True, header=other)
    assert out.read_bytes() == before


def test_refused_resume_generates_nothing(
    tiny_llama: Any, tok: Any, tmp_path: Path
) -> None:
    out = tmp_path / "r.jsonl"
    _run(tiny_llama, tok, out=out)
    other = run_header(tiny_llama, tok, 7, 0, False)
    calls: list[str] = []

    def scorer(gold: str, output: str) -> bool:
        calls.append(output)
        return True

    with pytest.raises(ValueError, match="another config"):
        generate_records(
            tiny_llama, tok, ITEMS, "base", scorer, 7, chat=False,
            out=out, resume=True, header=other,
        )  # fmt: skip
    assert calls == []


_CHILD = textwrap.dedent(
    """
    import sys
    import time

    import tokenizers
    import torch
    import transformers

    from astro_weights.eval import grade_exact
    from astro_weights.runners.generate import generate_records, run_header

    out, batch_size = sys.argv[1], int(sys.argv[2])
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64,
        )
    ).eval()
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2} | {f"w{i}": i + 3 for i in range(125)}
    t = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=t, unk_token="<unk>", bos_token="<s>",
        eos_token="</s>", pad_token="</s>",
    )
    items = [
        {"item_id": f"i{k}", "prompt": " ".join(["w1", "w2", f"w{k + 3}"]), "gold": "w5"}
        for k in range(%d)
    ]

    def slow(gold, output):  # keeps the child killable between records
        time.sleep(0.4)
        return grade_exact(gold, output)

    generate_records(
        model, tok, items, "base", slow, 6, chat=False, out=out,
        batch_size=batch_size,
        header=run_header(model, tok, 6, 0, False, batch_size=batch_size),
    )
    """.replace("%d", str(N_ITEMS))
)


def _lines(path: Path) -> int:
    return path.read_bytes().count(b"\n") if path.exists() else 0


@pytest.mark.skipif(sys.platform == "win32", reason="needs SIGKILL")
@pytest.mark.parametrize("batch_size", [1, 2])
def test_sigkill_then_resume(
    tiny_llama: Any, tok: Any, tmp_path: Path, batch_size: int
) -> None:
    whole = _run(tiny_llama, tok, out=tmp_path / "whole.jsonl", batch_size=batch_size)
    out = tmp_path / "r.jsonl"
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", _CHILD, str(out), str(batch_size)]
    )
    try:
        # a busy CI box loads torch slowly; the child-alive check ends a crash early
        deadline = time.monotonic() + 300
        while _lines(out) < 2:
            assert child.poll() is None, "child ended before it could be killed"
            assert time.monotonic() < deadline, "child wrote no records in 300 s"
            time.sleep(0.02)
        os.kill(child.pid, signal.SIGKILL)
    finally:
        child.kill()
        child.wait()
    kept = read_jsonl_prefix(out)
    assert 2 <= len(kept) < N_ITEMS
    assert header_path(out).exists()

    resumed = _run(tiny_llama, tok, out=out, resume=True, batch_size=batch_size)
    assert resumed == whole
    assert [r.item_id for r in read_jsonl(out)] == [str(i["item_id"]) for i in ITEMS]
    data = out.read_bytes()
    assert data.startswith(b"".join(line + b"\n" for line in kept))  # kept bytes intact
    assert data == (tmp_path / "whole.jsonl").read_bytes()


def read_jsonl_prefix(path: Path) -> list[bytes]:
    """Whole lines in `path`; a SIGKILL can leave a partial last one."""
    return path.read_bytes().split(b"\n")[:-1]


def _count_fsync(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    real = os.fsync

    def counting(fd: int) -> None:
        calls.append(fd)
        real(fd)

    monkeypatch.setattr(os, "fsync", counting)
    return calls


@pytest.mark.parametrize("batch_size", [1, 2])
def test_generate_fsyncs_each_record(
    tiny_llama: Any,
    tok: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
) -> None:
    calls = _count_fsync(monkeypatch)
    _run(tiny_llama, tok, out=tmp_path / "r.jsonl", batch_size=batch_size)
    assert len(calls) >= N_ITEMS


def test_http_fsyncs_each_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _count_fsync(monkeypatch)
    with serve(lambda _b: ok("2")) as (url, _state):
        generate_records_http(
            url, "m", ITEMS, "base", grade_exact, 4, out=tmp_path / "h.jsonl"
        )
    assert len(calls) >= N_ITEMS
