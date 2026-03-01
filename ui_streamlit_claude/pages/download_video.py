# ============================================================
# PAGE 1 — YouTube Video Download
#
# Flow:
#   User enters URL
#     → YouTubeDownloadAgent (LangGraph)
#       → get_youTubeVid_tool (MCP)
#         → file saved to DATA_DIR/youTubeVids/
#           → result shown in UI
# ============================================================

import os
import sys

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

from claude_agents.agents_claude_main import run_yt_download_only

# ── Download target directory ─────────────────────────────────────────────────
_YT_DOWNLOAD_DIR = os.path.join(_PROJECT_DIR, "DATA_DIR", "youTubeVids")
os.makedirs(_YT_DOWNLOAD_DIR, exist_ok=True)

# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Download Video", page_icon="🎬", layout="centered")

st.title("YouTube Video Download")
st.caption("Agent: **YouTubeDownloadAgent**  |  Tool: `get_youTubeVid_tool`")
st.markdown("---")

# ── Input ─────────────────────────────────────────────────────────────────────
youtube_url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

st.markdown(f"**Save to:** `{_YT_DOWNLOAD_DIR}`")

download_btn = st.button("Download via YouTubeDownloadAgent", type="primary")

# ── Run agent ─────────────────────────────────────────────────────────────────
if download_btn:
    if not youtube_url.strip():
        st.warning("Please enter a YouTube URL first.")
        st.stop()

    with st.status("Invoking YouTubeDownloadAgent...", expanded=True) as status:
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

    # ── Parse result from agent messages ─────────────────────────────────────
    messages = final_state.get("messages", [])

    # The tool message carries the raw JSON result from get_youTubeVid_tool
    tool_result = None
    agent_summary = ""
    for msg in messages:
        # ToolMessage has type="tool"
        if hasattr(msg, "type") and msg.type == "tool":
            tool_result = msg.content
        # Last AIMessage with no tool_calls is the agent's final summary
        if hasattr(msg, "type") and msg.type == "ai" and not getattr(msg, "tool_calls", []):
            agent_summary = msg.content

    # ── Display result ────────────────────────────────────────────────────────
    if tool_result:
        import json
        try:
            result_dict = json.loads(tool_result)
        except Exception:
            result_dict = {"raw": tool_result}

        tool_status = result_dict.get("status", "unknown")
        file_path   = result_dict.get("file_path", "")

        if tool_status == "downloaded" and file_path:
            file_size_mb = (
                os.path.getsize(file_path) / (1024 * 1024)
                if os.path.isfile(file_path)
                else 0
            )
            st.success(f"Downloaded successfully  ({file_size_mb:.1f} MB)")
            st.code(file_path, language=None)
        else:
            detail = result_dict.get("detail", "Unknown error")
            st.error(f"Download failed: {detail}")

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
