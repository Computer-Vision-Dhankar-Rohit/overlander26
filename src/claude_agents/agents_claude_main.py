# ============================================================
# LANGRAPH MULTI-AGENT PIPELINE — overlander26
# FILE   : src/claude_agents/agents_claude_main.py
#
# AGENTS (4):
#   1. YouTubeDownloadAgent  → calls get_youTubeVid_tool
#   2. YouTubeProcessAgent   → calls procs_youTubeVid_tool
#   3. IPCamProcessAgent     → calls procs_IPCAM_Vid_tool
#   4. EmailAlertAgent       → calls send_email_tool
#
# FLOW:
#   START
#     │
#     ├─[task_type="youtube"]─► YouTubeDownloadAgent
#     │                               │
#     │                         YouTubeProcessAgent
#     │                               │
#     └─[task_type="ipcam"]──► IPCamProcessAgent
#                                     │
#                               EmailAlertAgent
#                                     │
#                                    END
#
# INSTALL:
#   pip install langgraph langchain-anthropic langchain-mcp-adapters fastmcp
#
# ENV VARS:
#   ANTHROPIC_API_KEY           — Claude API key
#   OVERLANDER_EMAIL_SENDER     — Gmail sender address
#   OVERLANDER_EMAIL_PASSWORD   — Gmail app password
#
# RUN:
#   cd src && python claude_agents/agents_claude_main.py
# ============================================================

from __future__ import annotations

import asyncio
import os
import sys
from typing import Annotated, Literal, TypedDict

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))   # .../src/claude_agents/
_SRC_DIR     = os.path.dirname(_HERE)                        # .../src/
_PROJECT_DIR = os.path.dirname(_SRC_DIR)                     # .../overlander26/
_MCP_SERVER  = os.path.join(_HERE, "mcp_server_computer_vision.py")
sys.path.insert(0, _SRC_DIR)

from util_logger import setup_logger
logger = setup_logger(module_name=str(__name__))

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ── Load secrets from DATA_DIR/secrets/.env ───────────────────────────────────
_ENV_FILE = os.path.join(_PROJECT_DIR, "DATA_DIR", "secrets", ".env")
load_dotenv(dotenv_path=_ENV_FILE, override=True)

# ── LLM (Claude) ──────────────────────────────────────────────────────────────
#   Uses ANTHROPIC_API_KEY from environment.
#   Swap to ChatOpenAI / ChatOllama if needed.
_LLM = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0,          # deterministic tool calls
    max_tokens=1024,
)


# ============================================================
# STATE
# ============================================================

class OverlanderState(TypedDict):
    # ── task routing ──────────────────────────────────────────
    task_type:   str   # "youtube" | "ipcam"

    # ── youtube path inputs ───────────────────────────────────
    youtube_url: str

    # ── ipcam path inputs ─────────────────────────────────────
    ipcam_url:   str
    max_frames:  int   # how many raw frames to consume from stream

    # ── shared results (populated by process agents) ──────────
    video_file_path:  str   # set by YouTubeDownloadAgent
    frames_sampled:   int   # set by process agents
    poses_detected:   int   # set by process agents
    detection_detail: str   # human-readable summary

    # ── email settings ────────────────────────────────────────
    email_to: str

    # ── LangGraph message bus (appended across all turns) ─────
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# HELPER — build one ReAct agent node
# ============================================================

def _make_agent_node(system_prompt: str, tools: list):
    """
    Returns an async LangGraph node function that runs one ReAct step.

    The node:
      1. Prepends a system message that scopes the agent's role.
      2. Calls the LLM (bound to `tools`) with the full conversation history.
      3. Appends the LLM response (with tool_calls if any) to state["messages"].
    """
    llm_with_tools = _LLM.bind_tools(tools)

    async def _node(state: OverlanderState) -> dict:
        conversation = [SystemMessage(content=system_prompt)] + state["messages"]
        response = await llm_with_tools.ainvoke(conversation)
        return {"messages": [response]}

    return _node


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

_IPCAM_PROCESS_PROMPT = """\
You are the IP Camera Process Agent in the Overlander computer-vision pipeline.

Your ONE job: call procs_IPCAM_Vid_tool with the RTSP / HTTP stream URL
provided in the conversation, then summarise how many frames were processed
and how many poses were detected.

- Call procs_IPCAM_Vid_tool exactly once.
"""

_EMAIL_ALERT_PROMPT = """\
You are the Email Alert Agent in the Overlander computer-vision pipeline.

Your ONE job: compose and send an email alert summarising the detection results
from the previous agents in the conversation.

- Use send_email_tool to send the alert.
- Write a clear, concise subject line and body.
- The body should include: number of frames sampled, number of poses detected,
  timestamp, and the source (YouTube URL or IP camera URL).
- Call send_email_tool exactly once.
"""


# ============================================================
# ROUTING FUNCTION
# ============================================================

def _route_task(state: OverlanderState) -> Literal["yt_download_agent", "ipcam_process_agent"]:
    """
    Branch from START based on task_type.

      task_type="youtube"  →  YouTubeDownloadAgent  →  YouTubeProcessAgent
      task_type="ipcam"    →  IPCamProcessAgent
      (both paths converge at EmailAlertAgent)
    """
    if state.get("task_type", "").lower() == "youtube":
        return "yt_download_agent"
    return "ipcam_process_agent"


# ============================================================
# BUILD AND RUN PIPELINE
# ============================================================

async def run_pipeline(initial_state: OverlanderState) -> OverlanderState:
    """
    Connects to the FastMCP server (mcpComputerVision) via stdio,
    loads the 4 MCP tools, wires up the LangGraph StateGraph, and
    runs the full agent pipeline.

    Args:
        initial_state: Populated OverlanderState dict from the caller.

    Returns:
        Final OverlanderState after all agents have completed.
    """
    # ── Connect to MCP server ─────────────────────────────────────────────────
    # langchain-mcp-adapters >= 0.1.0: no longer an async context manager.
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

    # ── Filter tools per agent ────────────────────────────────────────────────
    yt_dl_tools   = [t for t in all_tools if t.name == "get_youTubeVid_tool"]
    yt_proc_tools = [t for t in all_tools if t.name == "procs_youTubeVid_tool"]
    ipcam_tools   = [t for t in all_tools if t.name == "procs_IPCAM_Vid_tool"]
    email_tools   = [t for t in all_tools if t.name == "send_email_tool"]

    # ── Build agent nodes ─────────────────────────────────────────────────────
    yt_download_agent   = _make_agent_node(_YT_DOWNLOAD_PROMPT,   yt_dl_tools)
    yt_process_agent    = _make_agent_node(_YT_PROCESS_PROMPT,    yt_proc_tools)
    ipcam_process_agent = _make_agent_node(_IPCAM_PROCESS_PROMPT, ipcam_tools)
    email_alert_agent   = _make_agent_node(_EMAIL_ALERT_PROMPT,   email_tools)

    # ── Build ToolNodes ───────────────────────────────────────────────────────
    yt_dl_tool_node   = ToolNode(yt_dl_tools)
    yt_proc_tool_node = ToolNode(yt_proc_tools)
    ipcam_tool_node   = ToolNode(ipcam_tools)
    email_tool_node   = ToolNode(email_tools)

    # ── Assemble StateGraph ───────────────────────────────────────────────────
    graph = StateGraph(OverlanderState)

    graph.add_node("yt_download_agent",   yt_download_agent)
    graph.add_node("yt_download_tools",   yt_dl_tool_node)
    graph.add_node("yt_process_agent",    yt_process_agent)
    graph.add_node("yt_process_tools",    yt_proc_tool_node)
    graph.add_node("ipcam_process_agent", ipcam_process_agent)
    graph.add_node("ipcam_process_tools", ipcam_tool_node)
    graph.add_node("email_alert_agent",   email_alert_agent)
    graph.add_node("email_alert_tools",   email_tool_node)

    # START → route by task_type
    graph.add_conditional_edges(START, _route_task)

    # YouTube path
    graph.add_conditional_edges(
        "yt_download_agent", tools_condition,
        {"tools": "yt_download_tools", END: "yt_process_agent"},
    )
    graph.add_edge("yt_download_tools", "yt_download_agent")

    graph.add_conditional_edges(
        "yt_process_agent", tools_condition,
        {"tools": "yt_process_tools", END: "email_alert_agent"},
    )
    graph.add_edge("yt_process_tools", "yt_process_agent")

    # IP Cam path
    graph.add_conditional_edges(
        "ipcam_process_agent", tools_condition,
        {"tools": "ipcam_process_tools", END: "email_alert_agent"},
    )
    graph.add_edge("ipcam_process_tools", "ipcam_process_agent")

    # Email agent (both paths converge here)
    graph.add_conditional_edges(
        "email_alert_agent", tools_condition,
        {"tools": "email_alert_tools", END: END},
    )
    graph.add_edge("email_alert_tools", "email_alert_agent")

    compiled_graph = graph.compile()
    return await compiled_graph.ainvoke(initial_state)


# ============================================================
# YOUTUBE-DOWNLOAD-ONLY PIPELINE  (used by Streamlit UI)
# ============================================================

async def _run_yt_download_only(
    youtube_url: str,
    output_path: str,
) -> OverlanderState:
    """
    Minimal LangGraph graph: just the YouTubeDownloadAgent.

    START → yt_download_agent ↔ yt_download_tools → END

    Used by the Streamlit UI so it can invoke the agent without
    running the full pose-detection + email pipeline.
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

    graph = StateGraph(OverlanderState)
    graph.add_node("yt_download_agent", yt_download_agent)
    graph.add_node("yt_download_tools", yt_dl_tool_node)

    graph.add_edge(START, "yt_download_agent")
    graph.add_conditional_edges(
        "yt_download_agent",
        tools_condition,
        {"tools": "yt_download_tools", END: END},
    )
    graph.add_edge("yt_download_tools", "yt_download_agent")

    compiled = graph.compile()

    init_state: OverlanderState = {
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


def run_yt_download_only(youtube_url: str, output_path: str) -> OverlanderState:
    """
    Synchronous wrapper — safe to call from Streamlit.

    Args:
        youtube_url: Full YouTube URL.
        output_path: Directory to save the downloaded MP4.

    Returns:
        Final OverlanderState. Check state["messages"][-1] for the
        agent's summary, or parse tool message content for file_path.
    """
    return asyncio.run(_run_yt_download_only(youtube_url, output_path))


# ============================================================
# CONVENIENCE WRAPPERS
# ============================================================

def run_youtube_pipeline(
    youtube_url: str,
    email_to: str,
) -> OverlanderState:
    """
    Synchronous entry point for the YouTube pipeline.

    Downloads the video, runs MediaPipe pose detection on sampled frames,
    and emails a detection summary.

    Example:
        from claude_agents.agents_claude_main import run_youtube_pipeline
        result = run_youtube_pipeline(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            email_to="analyst@example.com",
        )
    """
    initial_state: OverlanderState = {
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
                    f"Process the YouTube video at {youtube_url}. "
                    f"Download it, run pose detection on sampled frames, "
                    f"then send a detection summary email to {email_to}."
                )
            )
        ],
    }
    return asyncio.run(run_pipeline(initial_state))


def run_ipcam_pipeline(
    ipcam_url: str,
    email_to: str,
    max_frames: int = 150,
) -> OverlanderState:
    """
    Synchronous entry point for the IP camera pipeline.

    Connects to the RTSP/HTTP stream, runs MediaPipe pose detection on
    sampled frames, and emails a detection summary.

    Example:
        from claude_agents.agents_claude_main import run_ipcam_pipeline
        result = run_ipcam_pipeline(
            ipcam_url="rtsp://192.168.1.64:554/stream",
            email_to="analyst@example.com",
        )
    """
    initial_state: OverlanderState = {
        "task_type":        "ipcam",
        "youtube_url":      "",
        "ipcam_url":        ipcam_url,
        "max_frames":       max_frames,
        "video_file_path":  "",
        "frames_sampled":   0,
        "poses_detected":   0,
        "detection_detail": "",
        "email_to":         email_to,
        "messages": [
            HumanMessage(
                content=(
                    f"Connect to the IP camera at {ipcam_url}. "
                    f"Read up to {max_frames} frames, run pose detection, "
                    f"then send a detection summary email to {email_to}."
                )
            )
        ],
    }
    return asyncio.run(run_pipeline(initial_state))


# ============================================================
# ENTRY POINT — quick smoke test
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Overlander LangGraph multi-agent pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["youtube", "ipcam"],
        default="ipcam",
        help="Pipeline mode: 'youtube' to process a YouTube video, "
             "'ipcam' for a live IP camera stream.",
    )
    parser.add_argument(
        "--youtube-url",
        default="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        help="YouTube URL (used when --mode=youtube)",
    )
    parser.add_argument(
        "--ipcam-url",
        default="rtsp://192.168.1.64:554/stream",
        help="RTSP/HTTP stream URL (used when --mode=ipcam)",
    )
    parser.add_argument(
        "--email-to",
        default=os.getenv("OVERLANDER_ALERT_EMAIL", "alert@example.com"),
        help="Recipient email for the detection summary.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="Max raw frames to read from IP camera stream.",
    )

    args = parser.parse_args()

    logger.debug("--- Overlander LangGraph Agent Pipeline mode %s email_to %s", args.mode, args.email_to)

    if args.mode == "youtube":
        logger.debug("--- youtube_url %s", args.youtube_url)
        result = run_youtube_pipeline(args.youtube_url, args.email_to)
    else:
        logger.debug("--- ipcam_url %s max_frames %s", args.ipcam_url, args.max_frames)
        result = run_ipcam_pipeline(args.ipcam_url, args.email_to, args.max_frames)

    logger.debug("--- Pipeline complete")
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            logger.debug("--- Agent summary %s", msg.content)
            break
