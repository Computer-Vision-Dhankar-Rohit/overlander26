# ============================================================
# PAGE 1 — YouTube Video Download
#
# Flow:
#   User enters URL
#     → get_youTubeVid_agent (LangGraph)
#       → get_youTubeVid_tool (MCP)
#         → file saved to DATA_DIR/youTubeVids/
#           → result shown in UI
# ============================================================

import json
import os
import sys
from typing import Any, Dict, Optional

from dotenv import load_dotenv
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
_PAGE_DIR    = os.path.dirname(os.path.abspath(__file__))   # .../ui_streamlit/pages/
_UI_DIR      = os.path.dirname(_PAGE_DIR)                    # .../ui_streamlit/
_PROJECT_DIR = os.path.dirname(_UI_DIR)                      # .../overlander26/
_SRC_DIR     = os.path.join(_PROJECT_DIR, "src")

# ── Load secrets before any agent import ─────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(_PROJECT_DIR, "DATA_DIR", "secrets", ".env"), override=True)

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from openai_langchain_agents.main_agents import run_yt_download_only

# ── Download target directory ─────────────────────────────────────────────────
_YT_DOWNLOAD_DIR = os.path.join(_PROJECT_DIR, "DATA_DIR", "youTubeVids")
os.makedirs(_YT_DOWNLOAD_DIR, exist_ok=True)


def _parse_tool_result(messages: list, tool_name: str) -> Optional[Dict[str, Any]]:
    for msg in reversed(messages):
        if getattr(msg, "type", "") != "tool":
            continue
        if getattr(msg, "name", "") != tool_name:
            continue
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            try:
                return json.loads(content)
            except Exception:
                return {"raw": content}
        return {"raw": content}
    return None


def _latest_agent_summary(messages: list) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", "") != "ai":
            continue
        if getattr(msg, "tool_calls", []):
            continue
        return str(getattr(msg, "content", ""))
    return ""


# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Download Video", page_icon="🎬", layout="centered")

st.title("YouTube Video Download")
st.caption("Agent: **get_youTubeVid_agent**  |  Tool: `get_youTubeVid_tool`")
st.markdown("---")

# ── Input ─────────────────────────────────────────────────────────────────────
youtube_url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

st.markdown(f"**Save to:** `{_YT_DOWNLOAD_DIR}`")

download_btn = st.button("Download via get_youTubeVid_agent", type="primary")

# ── Run agent ─────────────────────────────────────────────────────────────────
if download_btn:
    if not youtube_url.strip():
        st.warning("Please enter a YouTube URL first.")
        st.stop()

    with st.status("Invoking get_youTubeVid_agent...", expanded=True) as status:
        st.write("Connecting to MCP server `mcpComputerVision`...")
        st.write("Agent calling `get_youTubeVid_tool`...")

        try:
            final_state = run_yt_download_only(
                youtube_url=youtube_url.strip(),
                output_path=_YT_DOWNLOAD_DIR,
            )
            status.update(label="Agent completed", state="complete")
        except Exception as exc:
            status.update(label="Agent error", state="error")
            st.error(f"Error: {exc}")
            st.stop()

    # ── Parse result from agent messages ──────────────────────────────────────
    messages = final_state.get("messages", [])
    tool_result = _parse_tool_result(messages, "get_youTubeVid_tool")
    agent_summary = _latest_agent_summary(messages)

    # ── Display result ───────────────────────────────────────────────────────
    if tool_result:
        tool_status = tool_result.get("status", "unknown")
        file_path = tool_result.get("file_path", "")

        if tool_status == "downloaded" and file_path:
            file_size_mb = (
                os.path.getsize(file_path) / (1024 * 1024)
                if os.path.isfile(file_path)
                else 0
            )
            st.success(f"Downloaded successfully  ({file_size_mb:.1f} MB)")
            st.code(file_path, language=None)
        else:
            detail = tool_result.get("detail", "Unknown error")
            st.error(f"Download failed: {detail}")
    else:
        st.warning("No tool output found for `get_youTubeVid_tool`.")

    if agent_summary:
        with st.expander("Agent summary", expanded=False):
            st.write(agent_summary)

st.markdown("---")

# ── List files already in the download directory ──────────────────────────────
with st.expander(f"Files in DATA_DIR/youTubeVids/", expanded=True):
    video_exts = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
    try:
        files = [
            f for f in os.listdir(_YT_DOWNLOAD_DIR)
            if os.path.splitext(f)[1].lower() in video_exts
        ]
    except Exception:
        files = []

    if files:
        for f in sorted(files):
            full = os.path.join(_YT_DOWNLOAD_DIR, f)
            size_mb = os.path.getsize(full) / (1024 * 1024)
            st.markdown(f"- `{f}` &nbsp; ({size_mb:.1f} MB)")
    else:
        st.caption("No videos downloaded yet.")
