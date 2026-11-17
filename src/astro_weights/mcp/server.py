"""Stdio MCP server exposing astro-weights tools."""

from typing import Any

try:
    from mcp.server.mcpserver import MCPServer
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"{__name__} needs the official MCP SDK: pip install 'astro-weights[mcp]'",
        name=e.name,
    ) from e


def build_server() -> MCPServer[Any]:
    """Build the server with its tools registered."""
    server: MCPServer[Any] = MCPServer("astro-weights")

    @server.tool()
    def status(handle: str) -> dict[str, str]:
        """Return the state of a job handle (stub: always unknown)."""
        return {"handle": handle, "state": "unknown"}

    return server


def main() -> None:
    """Run the server on stdio."""
    build_server().run("stdio")
