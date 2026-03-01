"""Helpers for connecting LangChain agents to local FastMCP servers."""

from __future__ import annotations

import os
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

_HERE = os.path.dirname(os.path.abspath(__file__))  # .../src/openai_langchain_agents/
_SRC_DIR = os.path.dirname(_HERE)  # .../src/
_CLAUDE_AGENTS_DIR = os.path.join(_SRC_DIR, "claude_agents")

MCP_SERVER_NAME = "mcpComputerVision"
MCP_SERVER_SCRIPT = os.path.join(_CLAUDE_AGENTS_DIR, "mcp_server_computer_vision.py")


def build_mcp_client() -> MultiServerMCPClient:
    """Return a MultiServerMCPClient configured for mcpComputerVision over stdio."""
    return MultiServerMCPClient(
        {
            MCP_SERVER_NAME: {
                "command": sys.executable,
                "args": [MCP_SERVER_SCRIPT],
                "transport": "stdio",
            }
        }
    )


async def load_tools() -> list:
    """Load all tools exposed by the configured FastMCP server."""
    client = build_mcp_client()
    return await client.get_tools()
