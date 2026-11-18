# Drive the agent tools from the Claude Agent SDK

!!! note "New in 0.2"

The built-in route to Claude is the hosted editor: [How-to: run the agent on Claude with your own key](agent-hosted.md). It calls the Messages API, checks the dollar budget before every call and reports every turn's cost to the ledger. With the Agent SDK as client, the server never sees the model calls, so it cannot stop one that would pass the dollar limit.

If you still want the Agent SDK as your client, register the server as a stdio MCP server. Write `RUN_DIR/run.json` first, as in [Drive the agent tools from Claude Code](agent-claude-code.md#1-make-a-run-directory).

Config sketch, not tested:

```py
import asyncio

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query


async def main():
    options = ClaudeAgentOptions(
        mcp_servers={
            "astro-weights": {
                "command": "astro-weights",
                "args": ["mcp", "/path/to/run"],
            }
        },
        allowed_tools=[
            "mcp__astro-weights__exec_python",
            "mcp__astro-weights__forward_pass",
            "mcp__astro-weights__status",
            "mcp__astro-weights__finish",
        ],
    )
    async for message in query(
        prompt="Use the astro-weights tools to edit the model. Call finish when done.",
        options=options,
    ):
        if isinstance(message, ResultMessage):
            print(message.result)


asyncio.run(main())
```

`mcp_servers`, `allowed_tools` and the `mcp__<server>__<tool>` names come from the [Agent SDK MCP docs](https://code.claude.com/docs/en/agent-sdk/mcp). Listing four tools leaves `report_usage`, `check_spend` and `report_refusal` unapproved.

## Limits

- The SDK's tokens and dollars are not reported to the server, so `tokens` and `dollars` limits do not cap it. Your own provider's terms apply.
- With no OS sandbox, `exec_python` returns an error; see [Explanation: agent budgets and sandbox](../explanation/agent-budgets-and-sandbox.md).
