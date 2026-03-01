# ============================================================
# PAGE 2 — YouTube Full Pipeline (OpenAI Agents)
#
# Flow (human-in-the-loop):
#   User enters YouTube URL
#     → st.button "Run Full Pipeline (OpenAI Agents)"
#       → Orchestrator_Agent parses URL
#         → get_youTubeVid_agent downloads video
#           → procs_youTubeVid_agent detects poses
#             → result table shown in UI
#
# Agents:  OpenAI gpt-5.2 via LangGraph StateGraph
# Tools:   get_youTubeVid_tool + procs_youTubeVid_tool (FastMCP server)
# ============================================================

import json
import os
import sys

from dotenv import load_dotenv
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
_PAGE_DIR    = os.path.dirname(os.path.abspath(__file__))   # .../ui_streamlit_claude/pages/
_UI_DIR      = os.path.dirname(_PAGE_DIR)                    # .../ui_streamlit_claude/
_PROJECT_DIR = os.path.dirname(_UI_DIR)                      # .../overlander26/
_SRC_DIR     = os.path.join(_PROJECT_DIR, "src")

# ── Load secrets before any agent import ─────────────────────────────────────
load_dotenv(
    dotenv_path=os.path.join(_PROJECT_DIR, "DATA_DIR", "secrets", ".env"),
    override=True,
)

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from openai_langchain_agents.main_agents_claude import run_youtube_pipeline_openai

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Full Pipeline (OpenAI)",
    page_icon="🤖",
    layout="centered",
)

st.title("YouTube Full Pipeline")
st.caption(
    "Agents: **Orchestrator** → **YouTubeDownload** → **PoseDetection**  "
    f"| Model: `gpt-5.2` | Tools: `get_youTubeVid_tool`, `procs_youTubeVid_tool`"
)
st.markdown("---")

# ── Input ─────────────────────────────────────────────────────────────────────
youtube_url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

run_btn = st.button("Run Full Pipeline (OpenAI Agents)", type="primary")

# ── Run pipeline on human action ──────────────────────────────────────────────
if run_btn:
    if not youtube_url.strip():
        st.warning("Please enter a YouTube URL first.")
        st.stop()

    with st.status("Running OpenAI agent pipeline...", expanded=True) as status:
        st.write("Agent 1: **Orchestrator_Agent** — parsing your request...")
        st.write("Agent 2: **get_youTubeVid_agent** — downloading video via MCP...")
        st.write("Agent 3: **procs_youTubeVid_agent** — running pose detection via MCP...")

        try:
            final_state = run_youtube_pipeline_openai(youtube_url=youtube_url.strip())
            status.update(label="Pipeline completed", state="complete")
        except Exception as exc:
            status.update(label="Pipeline error", state="error")
            st.error(f"Error: {exc}")
            st.stop()

    # ── Parse results from agent messages ─────────────────────────────────────
    messages = final_state.get("messages", [])

    dl_result   = None   # result from get_youTubeVid_tool
    proc_result = None   # result from procs_youTubeVid_tool
    agent_summary = ""

    for msg in messages:
        if hasattr(msg, "type") and msg.type == "tool":
            # Parse JSON content from each ToolMessage
            try:
                data = json.loads(msg.content)
            except Exception:
                data = {"raw": msg.content}

            if "file_path" in data:
                dl_result = data
            elif "sampled_frames" in data or "total_frames" in data:
                proc_result = data

        # Last AIMessage without tool_calls is the agent's final summary
        if (
            hasattr(msg, "type")
            and msg.type == "ai"
            and not getattr(msg, "tool_calls", [])
        ):
            agent_summary = msg.content

    # ── Download result ────────────────────────────────────────────────────────
    st.subheader("Download Result")
    if dl_result:
        dl_status = dl_result.get("status", "unknown")
        file_path = dl_result.get("file_path", "")

        if dl_status == "downloaded" and file_path:
            file_size_mb = (
                os.path.getsize(file_path) / (1024 * 1024)
                if os.path.isfile(file_path)
                else 0
            )
            st.success(f"Downloaded  ({file_size_mb:.1f} MB)")
            st.code(file_path, language=None)
        else:
            st.error(f"Download failed: {dl_result.get('detail', 'Unknown error')}")
    else:
        st.info("No download result captured.")

    # ── Pose detection result ──────────────────────────────────────────────────
    st.subheader("Pose Detection Result")
    if proc_result:
        proc_status = proc_result.get("status", "unknown")
        if proc_status == "processed":
            st.success(proc_result.get("detail", "Pose detection completed."))
            st.table(
                {
                    "Metric":  ["Total Frames", "Sampled Frames", "Poses Detected", "Output Dir"],
                    "Value": [
                        str(proc_result.get("total_frames",   "—")),
                        str(proc_result.get("sampled_frames", "—")),
                        str(proc_result.get("poses_detected", "—")),
                        proc_result.get("output_dir", "—"),
                    ],
                }
            )
        else:
            st.error(f"Processing failed: {proc_result.get('detail', 'Unknown error')}")
    else:
        st.info("No pose detection result captured.")

    # ── Agent summary ──────────────────────────────────────────────────────────
    if agent_summary:
        with st.expander("Agent final summary", expanded=False):
            st.write(agent_summary)

st.markdown("---")
st.caption("Human-in-the-loop: every pipeline run is triggered by the button above.")
