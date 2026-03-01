# ============================================================
# LANGGRAPH MULTI-AGENT PIPELINE (OpenAI) — overlander26
# FILE   : src/openai_langchain_agents/main_agents_codex.py
#
# MANDATORY AGENTS:
#   1. Orchestrator_Agent
#   2. get_youTubeVid_agent
#   3. procs_youTubeVid_agent
#
# OPTIONAL AGENTS:
#   4. send_email_agent
#   5. procs_IPCAM_Vid_agent
# ============================================================

from __future__ import annotations

import asyncio
import os
import sys

try:
    from typing import Annotated, Literal, TypedDict
except ImportError:  # Python 3.7 fallback
    from typing_extensions import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

_HERE = os.path.dirname(os.path.abspath(__file__))  # .../src/openai_langchain_agents/
_SRC_DIR = os.path.dirname(_HERE)  # .../src/
_PROJECT_DIR = os.path.dirname(_SRC_DIR)  # .../overlander26/
sys.path.insert(0, _SRC_DIR)

from openai_langchain_agents.mcp_servers import MCP_SERVER_NAME, build_mcp_client
from util_logger import setup_logger

logger = setup_logger(module_name=str(__name__))

# Load secrets exactly from project secret env file.
_ENV_FILE = os.path.join(_PROJECT_DIR, "DATA_DIR", "secrets", ".env")
load_dotenv(dotenv_path=_ENV_FILE, override=True)

_DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-5.2")
_DEFAULT_MAX_FRAMES = int(os.getenv("OVERLANDER_DEFAULT_MAX_FRAMES", "150"))


class OverlanderStateCodex(TypedDict):
    task_type: str
    youtube_url: str
    ipcam_url: str
    max_frames: int
    video_file_path: str
    frames_sampled: int
    poses_detected: int
    detection_detail: str
    email_to: str
    messages: Annotated[list[BaseMessage], add_messages]


class OrchestratorDecision(BaseModel):
    task_type: Literal["youtube", "ipcam"] = Field(
        default="youtube",
        description="Route target. Prefer youtube unless request clearly asks for IP camera.",
    )
    youtube_url: str = Field(default="", description="YouTube URL if present.")
    ipcam_url: str = Field(default="", description="RTSP/HTTP IP camera URL if present.")
    email_to: str = Field(default="", description="Recipient email address if user asks for alerts.")
    max_frames: int = Field(
        default=_DEFAULT_MAX_FRAMES,
        ge=1,
        le=10000,
        description="Frame budget for IP camera processing.",
    )


def _build_llm(model_name: str | None = None) -> ChatOpenAI:
    selected_model = model_name or _DEFAULT_OPENAI_MODEL
    return ChatOpenAI(
        model=selected_model,
        temperature=0,
        max_retries=2,
    )


_LLM = _build_llm()

_ORCHESTRATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are Orchestrator_Agent for a computer-vision pipeline.\n"
                "Return structured routing for either 'youtube' or 'ipcam'.\n"
                "Rules:\n"
                "- Default to 'youtube' if unclear.\n"
                "- Extract youtube_url if present.\n"
                "- Extract ipcam_url if present.\n"
                "- Keep email_to only if user requested email alerts.\n"
                "- Keep max_frames positive."
            ),
        ),
        (
            "human",
            (
                "User request:\n{user_request}\n\n"
                "Current defaults:\n"
                "- task_type={default_task_type}\n"
                "- youtube_url={default_youtube_url}\n"
                "- ipcam_url={default_ipcam_url}\n"
                "- email_to={default_email_to}\n"
                "- max_frames={default_max_frames}\n"
            ),
        ),
    ]
)

# LCEL structured-routing chain for Orchestrator_Agent.
_ORCHESTRATOR_CHAIN = _ORCHESTRATOR_PROMPT | _LLM.with_structured_output(OrchestratorDecision)

_FINAL_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Summarise the pipeline outcome in 2-4 concise bullet points.",
        ),
        ("human", "{transcript}"),
    ]
)

# LCEL text parser chain for optional concise summaries.
_FINAL_SUMMARY_CHAIN = _FINAL_SUMMARY_PROMPT | _LLM | StrOutputParser()

_YT_DOWNLOAD_PROMPT = """\
You are get_youTubeVid_agent.

Your one job is to call get_youTubeVid_tool exactly once using the available
YouTube URL and output path in conversation context.

After tool execution, return a short confirmation message.
"""

_YT_PROCESS_PROMPT = """\
You are procs_youTubeVid_agent.

Your one job is to call procs_youTubeVid_tool exactly once using the downloaded
video file path from previous tool output.

After tool execution, return a short summary with sampled_frames and poses_detected.
"""

_IPCAM_PROCESS_PROMPT = """\
You are procs_IPCAM_Vid_agent.

Your one job is to call procs_IPCAM_Vid_tool exactly once with the stream URL.
Use max_frames if provided by the user.
"""

_SEND_EMAIL_PROMPT = """\
You are send_email_agent.

Your one job is to call send_email_tool exactly once to send a concise alert
about the latest detection result.
"""


def _make_agent_node(agent_name: str, system_prompt: str, tools: list):
    """Build one LCEL ReAct-style LangGraph node scoped to a specific tool set."""
    llm_with_tools = _LLM.bind_tools(tools)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"{system_prompt}\nAgentName: {agent_name}",
            ),
            MessagesPlaceholder("messages"),
        ]
    )
    chain = prompt | llm_with_tools

    async def _node(state: OverlanderStateCodex) -> dict:
        response = await chain.ainvoke({"messages": state["messages"]})
        return {"messages": [response]}

    return _node


def _latest_human_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


async def _orchestrator_agent(state: OverlanderStateCodex) -> dict:
    user_request = _latest_human_text(state.get("messages", []))
    if not user_request.strip():
        user_request = (
            "No explicit user message found. Use defaults and route safely."
        )

    decision = await _ORCHESTRATOR_CHAIN.ainvoke(
        {
            "user_request": user_request,
            "default_task_type": state.get("task_type", "youtube") or "youtube",
            "default_youtube_url": state.get("youtube_url", "") or "",
            "default_ipcam_url": state.get("ipcam_url", "") or "",
            "default_email_to": state.get("email_to", "") or "",
            "default_max_frames": int(state.get("max_frames", _DEFAULT_MAX_FRAMES)),
        }
    )

    task_type = decision.task_type.lower().strip()
    if task_type not in {"youtube", "ipcam"}:
        task_type = "youtube"

    youtube_url = (decision.youtube_url or state.get("youtube_url", "")).strip()
    ipcam_url = (decision.ipcam_url or state.get("ipcam_url", "")).strip()
    email_to = (decision.email_to or state.get("email_to", "")).strip()
    max_frames = int(decision.max_frames) if int(decision.max_frames) > 0 else _DEFAULT_MAX_FRAMES

    summary = AIMessage(
        content=(
            "Orchestrator_Agent routing decision: "
            f"task_type={task_type}, "
            f"youtube_url={'set' if youtube_url else 'empty'}, "
            f"ipcam_url={'set' if ipcam_url else 'empty'}, "
            f"email_to={'set' if email_to else 'empty'}, "
            f"max_frames={max_frames}"
        )
    )

    return {
        "task_type": task_type,
        "youtube_url": youtube_url,
        "ipcam_url": ipcam_url,
        "email_to": email_to,
        "max_frames": max_frames,
        "messages": [summary],
    }


def _post_task_router(_: OverlanderStateCodex) -> dict:
    """No-op node used before optional send_email routing."""
    return {}


async def _build_graph() -> tuple:
    mcp_client: MultiServerMCPClient = build_mcp_client()
    all_tools: list = await mcp_client.get_tools()

    def _pick(name: str) -> list:
        return [tool for tool in all_tools if tool.name == name]

    yt_dl_tools = _pick("get_youTubeVid_tool")
    yt_proc_tools = _pick("procs_youTubeVid_tool")
    ipcam_tools = _pick("procs_IPCAM_Vid_tool")
    email_tools = _pick("send_email_tool")

    if not yt_dl_tools:
        raise RuntimeError("Missing MCP tool: get_youTubeVid_tool")
    if not yt_proc_tools:
        raise RuntimeError("Missing MCP tool: procs_youTubeVid_tool")

    has_ipcam = bool(ipcam_tools)
    has_email = bool(email_tools)

    yt_download_agent = _make_agent_node(
        "get_youTubeVid_agent",
        _YT_DOWNLOAD_PROMPT,
        yt_dl_tools,
    )
    yt_process_agent = _make_agent_node(
        "procs_youTubeVid_agent",
        _YT_PROCESS_PROMPT,
        yt_proc_tools,
    )

    if has_ipcam:
        ipcam_process_agent = _make_agent_node(
            "procs_IPCAM_Vid_agent",
            _IPCAM_PROCESS_PROMPT,
            ipcam_tools,
        )
    if has_email:
        send_email_agent = _make_agent_node(
            "send_email_agent",
            _SEND_EMAIL_PROMPT,
            email_tools,
        )

    graph = StateGraph(OverlanderStateCodex)

    graph.add_node("orchestrator_agent", _orchestrator_agent)
    graph.add_node("yt_download_agent", yt_download_agent)
    graph.add_node("yt_download_tools", ToolNode(yt_dl_tools))
    graph.add_node("yt_process_agent", yt_process_agent)
    graph.add_node("yt_process_tools", ToolNode(yt_proc_tools))
    graph.add_node("post_task_router", _post_task_router)

    if has_ipcam:
        graph.add_node("ipcam_process_agent", ipcam_process_agent)
        graph.add_node("ipcam_process_tools", ToolNode(ipcam_tools))

    if has_email:
        graph.add_node("send_email_agent", send_email_agent)
        graph.add_node("send_email_tools", ToolNode(email_tools))

    graph.add_edge(START, "orchestrator_agent")

    if has_ipcam:
        def _route_from_orchestrator(state: OverlanderStateCodex) -> Literal["yt_download_agent", "ipcam_process_agent"]:
            if state.get("task_type", "").lower() == "ipcam":
                return "ipcam_process_agent"
            return "yt_download_agent"

        graph.add_conditional_edges(
            "orchestrator_agent",
            _route_from_orchestrator,
        )
    else:
        graph.add_edge("orchestrator_agent", "yt_download_agent")

    graph.add_conditional_edges(
        "yt_download_agent",
        tools_condition,
        {"tools": "yt_download_tools", END: "yt_process_agent"},
    )
    graph.add_edge("yt_download_tools", "yt_download_agent")

    graph.add_conditional_edges(
        "yt_process_agent",
        tools_condition,
        {"tools": "yt_process_tools", END: "post_task_router"},
    )
    graph.add_edge("yt_process_tools", "yt_process_agent")

    if has_ipcam:
        graph.add_conditional_edges(
            "ipcam_process_agent",
            tools_condition,
            {"tools": "ipcam_process_tools", END: "post_task_router"},
        )
        graph.add_edge("ipcam_process_tools", "ipcam_process_agent")

    if has_email:
        def _route_email_or_end(state: OverlanderStateCodex) -> Literal["send_email_agent", "end"]:
            if state.get("email_to", "").strip():
                return "send_email_agent"
            return "end"

        graph.add_conditional_edges(
            "post_task_router",
            _route_email_or_end,
            {"send_email_agent": "send_email_agent", "end": END},
        )
        graph.add_conditional_edges(
            "send_email_agent",
            tools_condition,
            {"tools": "send_email_tools", END: END},
        )
        graph.add_edge("send_email_tools", "send_email_agent")
    else:
        graph.add_edge("post_task_router", END)

    logger.debug(
        "--- OpenAI graph ready model=%s has_ipcam=%s has_email=%s mcp_server=%s",
        _DEFAULT_OPENAI_MODEL,
        has_ipcam,
        has_email,
        MCP_SERVER_NAME,
    )
    return graph.compile(), has_email


async def run_pipeline_codex(initial_state: OverlanderStateCodex) -> OverlanderStateCodex:
    compiled_graph, has_email_tool = await _build_graph()
    result: OverlanderStateCodex = await compiled_graph.ainvoke(initial_state)

    if has_email_tool:
        transcript = "\n".join(
            str(msg.content) for msg in result.get("messages", []) if hasattr(msg, "content")
        )
        try:
            summary = await _FINAL_SUMMARY_CHAIN.ainvoke({"transcript": transcript})
            result.setdefault("messages", []).append(AIMessage(content=f"Pipeline summary:\n{summary}"))
        except Exception:
            # Optional summary should never fail the pipeline.
            pass

    return result


async def _run_yt_download_only_codex(youtube_url: str, output_path: str) -> OverlanderStateCodex:
    mcp_client: MultiServerMCPClient = build_mcp_client()
    all_tools: list = await mcp_client.get_tools()
    yt_dl_tools = [tool for tool in all_tools if tool.name == "get_youTubeVid_tool"]

    if not yt_dl_tools:
        raise RuntimeError("Missing MCP tool: get_youTubeVid_tool")

    yt_download_agent = _make_agent_node(
        "get_youTubeVid_agent",
        _YT_DOWNLOAD_PROMPT,
        yt_dl_tools,
    )
    yt_dl_tool_node = ToolNode(yt_dl_tools)

    graph = StateGraph(OverlanderStateCodex)
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

    init_state: OverlanderStateCodex = {
        "task_type": "youtube",
        "youtube_url": youtube_url,
        "ipcam_url": "",
        "max_frames": _DEFAULT_MAX_FRAMES,
        "video_file_path": "",
        "frames_sampled": 0,
        "poses_detected": 0,
        "detection_detail": "",
        "email_to": "",
        "messages": [
            HumanMessage(
                content=(
                    f"Download the YouTube video at: {youtube_url}\n"
                    f"Save it to: {output_path}\n"
                    f"Call get_youTubeVid_tool with youtube_url={youtube_url!r} "
                    f"and output_path={output_path!r}."
                )
            )
        ],
    }
    return await compiled.ainvoke(init_state)


def run_yt_download_only_codex(youtube_url: str, output_path: str) -> OverlanderStateCodex:
    return asyncio.run(_run_yt_download_only_codex(youtube_url, output_path))


def run_youtube_pipeline_codex(youtube_url: str, email_to: str = "") -> OverlanderStateCodex:
    initial_state: OverlanderStateCodex = {
        "task_type": "youtube",
        "youtube_url": youtube_url,
        "ipcam_url": "",
        "max_frames": _DEFAULT_MAX_FRAMES,
        "video_file_path": "",
        "frames_sampled": 0,
        "poses_detected": 0,
        "detection_detail": "",
        "email_to": email_to.strip(),
        "messages": [
            HumanMessage(
                content=(
                    "Orchestrator_Agent: Process this YouTube video with the mandatory flow.\n"
                    f"YouTube URL: {youtube_url}\n"
                    "Steps: get_youTubeVid_agent -> procs_youTubeVid_agent.\n"
                    f"Email recipient: {email_to.strip() or 'none'}."
                )
            )
        ],
    }
    return asyncio.run(run_pipeline_codex(initial_state))


def run_ipcam_pipeline_codex(
    ipcam_url: str,
    email_to: str = "",
    max_frames: int = _DEFAULT_MAX_FRAMES,
) -> OverlanderStateCodex:
    initial_state: OverlanderStateCodex = {
        "task_type": "ipcam",
        "youtube_url": "",
        "ipcam_url": ipcam_url,
        "max_frames": max(1, int(max_frames)),
        "video_file_path": "",
        "frames_sampled": 0,
        "poses_detected": 0,
        "detection_detail": "",
        "email_to": email_to.strip(),
        "messages": [
            HumanMessage(
                content=(
                    "Orchestrator_Agent: Process this IP camera stream.\n"
                    f"Stream URL: {ipcam_url}\n"
                    f"max_frames: {max_frames}\n"
                    f"Email recipient: {email_to.strip() or 'none'}."
                )
            )
        ],
    }
    return asyncio.run(run_pipeline_codex(initial_state))


async def _run_send_email_only_codex(
    receiver_email: str,
    subject: str,
    body: str,
) -> OverlanderStateCodex:
    mcp_client: MultiServerMCPClient = build_mcp_client()
    all_tools: list = await mcp_client.get_tools()
    email_tools = [tool for tool in all_tools if tool.name == "send_email_tool"]
    if not email_tools:
        raise RuntimeError("Missing MCP tool: send_email_tool")

    send_email_agent = _make_agent_node(
        "send_email_agent",
        _SEND_EMAIL_PROMPT,
        email_tools,
    )
    email_tool_node = ToolNode(email_tools)

    graph = StateGraph(OverlanderStateCodex)
    graph.add_node("send_email_agent", send_email_agent)
    graph.add_node("send_email_tools", email_tool_node)
    graph.add_edge(START, "send_email_agent")
    graph.add_conditional_edges(
        "send_email_agent",
        tools_condition,
        {"tools": "send_email_tools", END: END},
    )
    graph.add_edge("send_email_tools", "send_email_agent")
    compiled = graph.compile()

    init_state: OverlanderStateCodex = {
        "task_type": "youtube",
        "youtube_url": "",
        "ipcam_url": "",
        "max_frames": _DEFAULT_MAX_FRAMES,
        "video_file_path": "",
        "frames_sampled": 0,
        "poses_detected": 0,
        "detection_detail": "",
        "email_to": receiver_email.strip(),
        "messages": [
            HumanMessage(
                content=(
                    f"Send an alert email.\n"
                    f"receiver_email={receiver_email!r}\n"
                    f"subject={subject!r}\n"
                    f"body={body!r}\n"
                    "Call send_email_tool exactly once."
                )
            )
        ],
    }
    return await compiled.ainvoke(init_state)


def run_send_email_only_codex(
    receiver_email: str,
    subject: str,
    body: str,
) -> OverlanderStateCodex:
    return asyncio.run(_run_send_email_only_codex(receiver_email, subject, body))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenAI LangGraph pipeline (codex)")
    parser.add_argument(
        "--mode",
        choices=["youtube", "ipcam", "download-only"],
        default="youtube",
    )
    parser.add_argument(
        "--youtube-url",
        default="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    )
    parser.add_argument(
        "--ipcam-url",
        default="rtsp://192.168.1.64:554/stream",
    )
    parser.add_argument(
        "--email-to",
        default="",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=_DEFAULT_MAX_FRAMES,
    )
    parser.add_argument(
        "--output-path",
        default=os.path.join(_PROJECT_DIR, "DATA_DIR", "youTubeVids"),
    )
    args = parser.parse_args()

    logger.debug("--- OpenAI pipeline mode=%s model=%s", args.mode, _DEFAULT_OPENAI_MODEL)

    if args.mode == "download-only":
        final_state = run_yt_download_only_codex(args.youtube_url, args.output_path)
    elif args.mode == "ipcam":
        final_state = run_ipcam_pipeline_codex(
            ipcam_url=args.ipcam_url,
            email_to=args.email_to,
            max_frames=args.max_frames,
        )
    else:
        final_state = run_youtube_pipeline_codex(
            youtube_url=args.youtube_url,
            email_to=args.email_to,
        )

    logger.debug("--- Pipeline completed. messages=%s", len(final_state.get("messages", [])))
