# ============================================================
# LANGGRAPH MULTI-AGENT PIPELINE — overlander26 (OpenAI)
# FILE   : src/openai_langchain_agents/main_agents_claude.py
#
# AGENTS (3 mandatory):
#   1. Orchestrator_Agent     → LCEL chain, parses user message, routes
#   2. get_youTubeVid_agent   → ReAct, calls get_youTubeVid_tool (MCP)
#   3. procs_youTubeVid_agent → ReAct, calls procs_youTubeVid_tool (MCP)
#
# OPTIONAL AGENTS (not wired by default):
#   send_email_agent          → calls send_email_tool (MCP)
#   procs_IPCAM_Vid_agent     → calls procs_IPCAM_Vid_tool (MCP)
#
# FLOW:
#   START
#     │
#     ▼
#   Orchestrator_Agent  ← HUMAN trigger (Streamlit button)
#     │  LCEL: ChatPromptTemplate | llm.with_structured_output(OrchestratorOutput)
#     │  extracts task_type = "youtube", youtube_url
#     ▼
#   get_youTubeVid_agent  ↔  ToolNode(get_youTubeVid_tool)
#     │  on tool completion →
#     ▼
#   procs_youTubeVid_agent  ↔  ToolNode(procs_youTubeVid_tool)
#     │
#     ▼
#   END
#
# MODEL: gpt-5.2 (OpenAI flagship, 2026-03-01)
# API KEY: OPENAI_API_KEY from DATA_DIR/secrets/.env only
#
# LCEL USAGE:
#   orchestrator_chain  = ChatPromptTemplate | llm.with_structured_output(OrchestratorOutput)
#   agent nodes         = llm.bind_tools(tools).ainvoke([SystemMessage] + state["messages"])
#
# MCP SERVER (reused, unchanged):
#   src/claude_agents/mcp_server_computer_vision.py  (FastMCP, stdio transport)
#
# INSTALL:
#   pip install langgraph langchain-openai langchain-mcp-adapters fastmcp openai
#
# RUN:
#   cd src && python openai_langchain_agents/main_agents_claude.py
# ============================================================

from __future__ import annotations

import asyncio
import os
import sys
from typing import Annotated, Literal, TypedDict

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))   # .../src/openai_langchain_agents/
_SRC_DIR     = os.path.dirname(_HERE)                        # .../src/
_PROJECT_DIR = os.path.dirname(_SRC_DIR)                     # .../overlander26/
_MCP_SERVER  = os.path.join(_SRC_DIR, "claude_agents", "mcp_server_computer_vision.py")
sys.path.insert(0, _SRC_DIR)

from util_logger import setup_logger
logger = setup_logger(module_name=str(__name__))

from dotenv import load_dotenv

# ── Load secrets from DATA_DIR/secrets/.env ───────────────────────────────────
_ENV_FILE = os.path.join(_PROJECT_DIR, "DATA_DIR", "secrets", ".env")
load_dotenv(dotenv_path=_ENV_FILE, override=True)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

# ── Model ─────────────────────────────────────────────────────────────────────
#   API key is read from OPENAI_API_KEY env var (loaded above from .env).
#   Never pass the key as a literal value.
OPENAI_MODEL = "gpt-5.2"   # OpenAI flagship (2026-03-01)

_LLM = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0,          # deterministic tool calls
    max_tokens=1024,
)


# ============================================================
# ORCHESTRATOR OUTPUT SCHEMA  (LCEL structured output)
# ============================================================

class OrchestratorOutput(BaseModel):
    """Structured fields extracted by the Orchestrator_Agent LCEL chain."""
    task_type:   str   # "youtube" | "ipcam" | "unknown"
    youtube_url: str = ""
    ipcam_url:   str = ""


# ============================================================
# STATE
# ============================================================

class OverlanderStateOpenAI(TypedDict):
    # ── task routing ──────────────────────────────────────────
    task_type:   str   # "youtube" | "ipcam"

    # ── youtube path inputs ───────────────────────────────────
    youtube_url: str

    # ── ipcam path inputs ─────────────────────────────────────
    ipcam_url:   str
    max_frames:  int

    # ── shared results (populated by process agents) ──────────
    video_file_path:  str
    frames_sampled:   int
    poses_detected:   int
    detection_detail: str

    # ── email settings ────────────────────────────────────────
    email_to: str

    # ── LangGraph message bus (appended across all turns) ─────
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# ORCHESTRATOR CHAIN (LCEL)
# ============================================================

_ORCHESTRATOR_SYSTEM = """\
You are the Orchestrator Agent for the Overlander computer-vision pipeline.

Parse the user's message and extract:
- task_type: "youtube" if the message contains a YouTube URL or asks to process a
  YouTube video; "ipcam" if it contains an RTSP or HTTP stream URL; otherwise "unknown".
- youtube_url: the full YouTube URL if present, empty string otherwise.
- ipcam_url: the RTSP/HTTP stream URL if present, empty string otherwise.

Return ONLY the structured fields — no extra explanation.
"""

_orchestrator_chain = (
    ChatPromptTemplate.from_messages([
        ("system", _ORCHESTRATOR_SYSTEM),
        ("human", "{user_message}"),
    ])
    | _LLM.with_structured_output(OrchestratorOutput)
)


# ============================================================
# AGENT SYSTEM PROMPTS
# ============================================================

_YT_DOWNLOAD_PROMPT = """\
You are the YouTube Download Agent in the Overlander computer-vision pipeline.

Your ONE job: call get_youTubeVid_tool with the YouTube URL provided in the
conversation, then confirm the downloaded file path.

- Always call get_youTubeVid_tool exactly once.
- Do not attempt any other task.
"""

_YT_PROCESS_PROMPT = """\
You are the YouTube Process Agent in the Overlander computer-vision pipeline.

Your ONE job: call procs_youTubeVid_tool with the local video file path that
was just downloaded, then summarise how many frames were sampled and how many
poses were detected.

- Use the file_path from the previous agent's result.
- Call procs_youTubeVid_tool exactly once.
"""


# ============================================================
# HELPER — build one ReAct agent node
# ============================================================

def _make_agent_node(system_prompt: str, tools: list):
    """
    Returns an async LangGraph node function that runs one ReAct step.

    LCEL-style composition:
        [SystemMessage(system_prompt)] + state["messages"]  →  llm.bind_tools(tools).ainvoke(...)

    The node:
      1. Prepends a system message that scopes the agent's role.
      2. Calls the LLM (bound to `tools`) with the full conversation history.
      3. Appends the LLM response (with tool_calls if any) to state["messages"].
    """
    llm_with_tools = _LLM.bind_tools(tools)

    async def _node(state: OverlanderStateOpenAI) -> dict:
        conversation = [SystemMessage(content=system_prompt)] + state["messages"]
        response = await llm_with_tools.ainvoke(conversation)
        return {"messages": [response]}

    return _node


# ============================================================
# ORCHESTRATOR NODE
# ============================================================

async def _orchestrator_node(state: OverlanderStateOpenAI) -> dict:
    """
    Agent_1: Orchestrator_Agent

    Uses the LCEL chain (_orchestrator_chain) to parse the human's raw message
    and extract task_type + URL.  Does NOT call any MCP tool — pure routing.
    Updates task_type (and optionally youtube_url / ipcam_url) in state.
    """
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        state.get("youtube_url", ""),   # fallback: already set in initial_state
    )
    logger.debug("--- Orchestrator_Agent parsing message len=%d", len(last_human))

    result: OrchestratorOutput = await _orchestrator_chain.ainvoke(
        {"user_message": last_human}
    )
    logger.debug(
        "--- Orchestrator_Agent task_type=%s youtube_url=%s",
        result.task_type, result.youtube_url,
    )

    updates: dict = {"task_type": result.task_type}
    if result.youtube_url:
        updates["youtube_url"] = result.youtube_url
    if result.ipcam_url:
        updates["ipcam_url"] = result.ipcam_url
    return updates


# ============================================================
# ROUTING FUNCTION
# ============================================================

def _route_after_orchestrator(
    state: OverlanderStateOpenAI,
) -> Literal["yt_download_agent"]:
    """
    Routes after Orchestrator_Agent based on state.task_type.
    Only the YouTube path is wired (mandatory 3-agent pipeline).
    """
    if state.get("task_type", "").lower() == "youtube":
        return "yt_download_agent"
    # Default: YouTube path (handles "unknown" gracefully)
    return "yt_download_agent"


# ============================================================
# FULL 3-AGENT YOUTUBE PIPELINE
# ============================================================

async def _run_youtube_pipeline_openai(
    youtube_url: str,
    email_to: str = "",
) -> OverlanderStateOpenAI:
    """
    Full pipeline:
        Orchestrator_Agent → get_youTubeVid_agent → procs_youTubeVid_agent → END

    MCP tools loaded from the existing FastMCP server via stdio transport.
    """
    mcp_client = MultiServerMCPClient(
        {
            "mcpComputerVision": {
                "command":   sys.executable,
                "args":      [_MCP_SERVER],
                "transport": "stdio",
            }
        }
    )
    all_tools: list = await mcp_client.get_tools()

    yt_dl_tools   = [t for t in all_tools if t.name == "get_youTubeVid_tool"]
    yt_proc_tools = [t for t in all_tools if t.name == "procs_youTubeVid_tool"]

    yt_download_agent = _make_agent_node(_YT_DOWNLOAD_PROMPT,  yt_dl_tools)
    yt_process_agent  = _make_agent_node(_YT_PROCESS_PROMPT,   yt_proc_tools)
    yt_dl_tool_node   = ToolNode(yt_dl_tools)
    yt_proc_tool_node = ToolNode(yt_proc_tools)

    # ── Assemble StateGraph ───────────────────────────────────────────────────
    graph = StateGraph(OverlanderStateOpenAI)

    graph.add_node("orchestrator_agent", _orchestrator_node)
    graph.add_node("yt_download_agent",  yt_download_agent)
    graph.add_node("yt_download_tools",  yt_dl_tool_node)
    graph.add_node("yt_process_agent",   yt_process_agent)
    graph.add_node("yt_process_tools",   yt_proc_tool_node)

    # START → Orchestrator_Agent
    graph.add_edge(START, "orchestrator_agent")

    # Orchestrator_Agent → (conditional) → yt_download_agent
    graph.add_conditional_edges("orchestrator_agent", _route_after_orchestrator)

    # get_youTubeVid_agent ReAct loop; on finish → procs_youTubeVid_agent
    graph.add_conditional_edges(
        "yt_download_agent", tools_condition,
        {"tools": "yt_download_tools", END: "yt_process_agent"},
    )
    graph.add_edge("yt_download_tools", "yt_download_agent")

    # procs_youTubeVid_agent ReAct loop; on finish → END
    graph.add_conditional_edges(
        "yt_process_agent", tools_condition,
        {"tools": "yt_process_tools", END: END},
    )
    graph.add_edge("yt_process_tools", "yt_process_agent")

    compiled_graph = graph.compile()

    init_state: OverlanderStateOpenAI = {
        "task_type":        "youtube",
        "youtube_url":      youtube_url,
        "ipcam_url":        "",
        "max_frames":       0,
        "video_file_path":  "",
        "frames_sampled":   0,
        "poses_detected":   0,
        "detection_detail": "",
        "email_to":         email_to,
        "messages": [
            HumanMessage(
                content=(
                    f"Process the YouTube video at: {youtube_url}\n"
                    f"Download it with get_youTubeVid_tool, then run pose "
                    f"detection on the downloaded file with procs_youTubeVid_tool."
                )
            )
        ],
    }
    return await compiled_graph.ainvoke(init_state)


# ============================================================
# DOWNLOAD-ONLY PIPELINE  (used by Streamlit youtube_pipeline page)
# ============================================================

async def _run_yt_download_only_openai(
    youtube_url: str,
    output_path: str,
) -> OverlanderStateOpenAI:
    """
    Minimal 2-agent pipeline:
        Orchestrator_Agent → get_youTubeVid_agent → END

    Used by the Streamlit UI for a quick download-only test.
    """
    mcp_client = MultiServerMCPClient(
        {
            "mcpComputerVision": {
                "command":   sys.executable,
                "args":      [_MCP_SERVER],
                "transport": "stdio",
            }
        }
    )
    all_tools   = await mcp_client.get_tools()
    yt_dl_tools = [t for t in all_tools if t.name == "get_youTubeVid_tool"]

    yt_download_agent = _make_agent_node(_YT_DOWNLOAD_PROMPT, yt_dl_tools)
    yt_dl_tool_node   = ToolNode(yt_dl_tools)

    graph = StateGraph(OverlanderStateOpenAI)
    graph.add_node("orchestrator_agent", _orchestrator_node)
    graph.add_node("yt_download_agent",  yt_download_agent)
    graph.add_node("yt_download_tools",  yt_dl_tool_node)

    graph.add_edge(START, "orchestrator_agent")
    graph.add_conditional_edges("orchestrator_agent", _route_after_orchestrator)
    graph.add_conditional_edges(
        "yt_download_agent",
        tools_condition,
        {"tools": "yt_download_tools", END: END},
    )
    graph.add_edge("yt_download_tools", "yt_download_agent")

    compiled = graph.compile()

    init_state: OverlanderStateOpenAI = {
        "task_type":        "youtube",
        "youtube_url":      youtube_url,
        "ipcam_url":        "",
        "max_frames":       0,
        "video_file_path":  "",
        "frames_sampled":   0,
        "poses_detected":   0,
        "detection_detail": "",
        "email_to":         "",
        "messages": [
            HumanMessage(
                content=(
                    f"Download the YouTube video at: {youtube_url}\n"
                    f"Save it to this directory: {output_path}\n"
                    f"Call get_youTubeVid_tool with youtube_url={youtube_url!r} "
                    f"and output_path={output_path!r}."
                )
            )
        ],
    }
    return await compiled.ainvoke(init_state)


# ============================================================
# PUBLIC SYNCHRONOUS WRAPPERS  (Streamlit-safe)
# ============================================================

def run_youtube_pipeline_openai(
    youtube_url: str,
    email_to: str = "",
) -> OverlanderStateOpenAI:
    """
    Full 3-agent pipeline (Orchestrator + Download + Process).
    Synchronous wrapper — safe to call from Streamlit.

    Args:
        youtube_url: Full YouTube URL.
        email_to:    Optional; not wired to email agent by default.

    Returns:
        Final OverlanderStateOpenAI. Check state["messages"] for agent results.
    """
    return asyncio.run(_run_youtube_pipeline_openai(youtube_url, email_to))


def run_yt_download_only_openai(
    youtube_url: str,
    output_path: str,
) -> OverlanderStateOpenAI:
    """
    Download-only pipeline (Orchestrator + Download agent only).
    Synchronous wrapper — safe to call from Streamlit.

    Args:
        youtube_url: Full YouTube URL.
        output_path: Directory to save the downloaded MP4.

    Returns:
        Final OverlanderStateOpenAI.
    """
    return asyncio.run(_run_yt_download_only_openai(youtube_url, output_path))


# ============================================================
# ENTRY POINT — smoke test
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=f"Overlander OpenAI LangGraph pipeline ({OPENAI_MODEL})"
    )
    parser.add_argument(
        "--youtube-url",
        default="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        help="YouTube URL to download and process",
    )
    parser.add_argument(
        "--email-to",
        default=os.getenv("OVERLANDER_ALERT_EMAIL", ""),
        help="Optional recipient email for detection summary",
    )
    args = parser.parse_args()

    logger.debug(
        "--- OpenAI LangGraph pipeline model=%s url=%s",
        OPENAI_MODEL, args.youtube_url,
    )
    result = run_youtube_pipeline_openai(args.youtube_url, args.email_to)

    logger.debug("--- Pipeline complete")
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            logger.debug("--- Agent summary %s", msg.content)
            break
