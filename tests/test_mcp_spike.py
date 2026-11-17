import importlib
import sys

import pytest

pytest.importorskip("mcp")

import anyio
from mcp import Client, StdioServerParameters

from astro_weights.mcp.server import build_server

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_in_process_round_trip() -> None:
    async with Client(build_server()) as client:
        tools = await client.list_tools()
        assert [t.name for t in tools.tools] == ["status"]
        result = await client.call_tool("status", {"handle": "h1"})
    assert not result.is_error
    assert result.structured_content == {"handle": "h1", "state": "unknown"}


async def test_stdio_round_trip() -> None:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "astro_weights.mcp"]
    )
    with anyio.fail_after(30):
        async with Client(params) as client:
            assert client.protocol_version == "2026-07-28"
            result = await client.call_tool("status", {"handle": "h2"})
    assert result.structured_content == {"handle": "h2", "state": "unknown"}


def test_missing_extra_names_the_install(monkeypatch: pytest.MonkeyPatch) -> None:
    module = "astro_weights.mcp.server"
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "mcp.server", None)
    monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", None)
    monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.delitem(sys.modules, "astro_weights.mcp", raising=False)
    with pytest.raises(ModuleNotFoundError) as info:
        importlib.import_module(module)
    assert info.value.name is not None
    assert info.value.name.startswith("mcp")
    assert str(info.value).startswith(module)
    assert str(info.value).endswith("pip install 'astro-weights[mcp]'")
