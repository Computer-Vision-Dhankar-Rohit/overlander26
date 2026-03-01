# ============================================================
# PAGE — Trigger OpenAI Agents (minimal)
#
# Invokes agents from:
#   src/openai_langchain_agents/main_agents_claude.py
#
# Two triggers:
#   1. run_youtube_pipeline_openai   → full 3-agent pipeline
#   2. run_yt_download_only_openai   → download-only (2-agent)
# ============================================================

import json
import os
import sys

from dotenv import load_dotenv
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
_PAGE_DIR    = os.path.dirname(os.path.abspath(__file__))
_UI_DIR      = os.path.dirname(_PAGE_DIR)
_PROJECT_DIR = os.path.dirname(_UI_DIR)
_SRC_DIR     = os.path.join(_PROJECT_DIR, "src")

load_dotenv(os.path.join(_PROJECT_DIR, "DATA_DIR", "secrets", ".env"), override=True)

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from util_logger import setup_logger
logger = setup_logger(module_name=str(__name__))

from openai_langchain_agents.main_agents_claude import (
    run_youtube_pipeline_openai,
    run_yt_download_only_openai,
    OPENAI_MODEL,
)

# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Computer Vision - LangChain OpenAI Agents", page_icon="🤖", layout="centered")

# ── Header with Framework Icons ───────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

with col1:
    st.title("🤖 Trigger OpenAI Agents")

with col5:
    st.markdown(
        """
        <div style="display: flex; gap: 10px; justify-content: flex-end; margin-top: 10px;">
            <span title="OpenAI GPT" style="font-size: 24px; cursor: help;">🔮</span>
            <span title="LangGraph" style="font-size: 24px; cursor: help;">🕸️</span>
            <span title="LangChain" style="font-size: 24px; cursor: help;">⛓️</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.caption(f"Model: `{OPENAI_MODEL}`  |  MCP Server: `mcpComputerVision`  |  Transport: `stdio`")
st.markdown("---")

# ── Input ─────────────────────────────────────────────────────────────────────
youtube_url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

# ── Pipeline selector ─────────────────────────────────────────────────────────
pipeline = st.radio(
    "Select pipeline",
    options=[
        "Full Pipeline  (Orchestrator → Download → Pose Detection)",
        "Download Only  (Orchestrator → Download)",
    ],
    index=0,
)

run_btn = st.button("Trigger Agents", type="primary", use_container_width=True)

# ── Agent execution ───────────────────────────────────────────────────────────
if run_btn:
    if not youtube_url.strip():
        st.warning("Enter a YouTube URL first.")
        st.stop()

    url = youtube_url.strip()
    full_pipeline = pipeline.startswith("Full")

    logger.debug("--- trigger_agents_OpenAI url=%s full_pipeline=%s", url, full_pipeline)

    with st.status("Invoking OpenAI agents...", expanded=True) as status:
        st.write("**Agent 1** — Orchestrator_Agent (LCEL): parsing request...")
        if full_pipeline:
            st.write("**Agent 2** — get_youTubeVid_agent: downloading via MCP...")
            st.write("**Agent 3** — procs_youTubeVid_agent: running pose detection via MCP...")
        else:
            st.write("**Agent 2** — get_youTubeVid_agent: downloading via MCP...")

        try:
            if full_pipeline:
                state = run_youtube_pipeline_openai(youtube_url=url)
                logger.debug("--AAA-full_pipeline-STATE--=%s",state)

            else:
                out_dir = os.path.join(_PROJECT_DIR, "DATA_DIR", "youTubeVids")
                os.makedirs(out_dir, exist_ok=True)
                state = run_yt_download_only_openai(youtube_url=url, output_path=out_dir)
                logger.debug("-bbb--STATE-run_yt_download_only_openai-=%s",state)

            status.update(label="Agents completed", state="complete")
            logger.debug("--- trigger_agents_OpenAI completed ok")

        except Exception as exc:
            status.update(label="Agent error", state="error")
            logger.error("--- trigger_agents_OpenAI error %s", exc)
            st.error(f"Error: {exc}")
            st.stop()

    # ── Parse messages ────────────────────────────────────────────────────────
    messages = state.get("messages", [])

    def _parse_tool_content(content):
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    try:    return json.loads(item["text"])
                    except: pass
        elif isinstance(content, str):
            try:    return json.loads(content)
            except: pass
        return {}

    tool_results = [
        _parse_tool_content(m.content)
        for m in messages
        if getattr(m, "type", "") == "tool"
    ]
    final_ai = next(
        (m.content for m in reversed(messages)
         if getattr(m, "type", "") == "ai" and not getattr(m, "tool_calls", [])),
        None,
    )

    # ── AI MESSAGES ───────────────────────────────────────────────────────────────
    st.markdown("### Messages in STATE of the - Tool Calling OpenAI-AGENTS...")
    
    if messages:
        with st.expander("📋 View All AI Messages (JSON Format)", expanded=False):
            # Format all messages as a list of dictionaries for JSON display
            messages_json = []
            for i, m in enumerate(messages):
                # Extract tool_calls (handle both dict and object formats)
                tool_calls_list = []
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for tc in m.tool_calls:
                        if isinstance(tc, dict):
                            # If it's a dict, extract fields safely
                            tool_calls_list.append({
                                "id": tc.get("id", "unknown"),
                                "name": tc.get("function", {}).get("name", "unknown") if isinstance(tc.get("function"), dict) else "unknown",
                                "arguments": tc.get("function", {}).get("arguments", "") if isinstance(tc.get("function"), dict) else ""
                            })
                        else:
                            # If it's an object, access attributes
                            try:
                                tool_calls_list.append({
                                    "id": tc.id,
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                })
                            except AttributeError:
                                tool_calls_list.append({"raw": str(tc)})
                
                msg_dict = {
                    "index": i,
                    "type": getattr(m, "type", "unknown"),
                    "content": getattr(m, "content", ""),
                    "tool_calls": tool_calls_list,
                    "raw": str(m)
                }
                messages_json.append(msg_dict)
            
            # Display as JSON
            st.json(messages_json)
    else:
        st.info("ℹ️ No messages available yet. Run the agent to generate messages.")

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("### Results from Tool Calling OpenAI-AGENTS...")

    for i, result in enumerate(tool_results, 1):
        tool_status = result.get("status", "unknown")
        icon = "✅" if tool_status not in ("error", "unknown") else "❌"
        with st.expander(f"{icon} Tool result {i} — `{tool_status}`", expanded=True):
            st.json(result)

    if final_ai:
        st.markdown("### Agent Summary")
        st.write(final_ai)

    # ── Pose-Detected Images ──────────────────────────────────────────────────
    # Images written by media_pipe.py → pose_media_pipe_google_0() line 446:
    #   cv2.imwrite(pose_write_path, annotated_image)
    # Saved to: DATA_DIR/pose_detected/pose_id_not_ipcam/
    # Only images where flag_pose_landmarks_detected == "YES_LANDMARKS" are here.

    st.markdown("---")
    st.markdown("### Pose-Detected Frames — Faces Identified by MediaPipe")

    _pose_img_dir = os.path.join(_PROJECT_DIR, "DATA_DIR", "pose_detected", "pose_id_not_ipcam")
    logger.debug("--- trigger_agents_OpenAI pose_img_dir %s", _pose_img_dir)

    if os.path.isdir(_pose_img_dir):
        _pose_imgs = sorted(
            [f for f in os.listdir(_pose_img_dir) if f.lower().endswith(".png")],
        )
        if _pose_imgs:
            st.caption(f"{len(_pose_imgs)} pose-annotated frame(s) found in `pose_id_not_ipcam/`")
            cols = st.columns(3)
            for idx, fname in enumerate(_pose_imgs):
                fpath = os.path.join(_pose_img_dir, fname)
                with cols[idx % 3]:
                    st.image(fpath, caption=fname, use_container_width=True)
                logger.debug("--- trigger_agents_OpenAI displaying pose img %s", fname)
        else:
            st.info("No pose-annotated frames found yet. Run the Full Pipeline to generate them.")
    else:
        st.warning(f"Output directory not found: `{_pose_img_dir}`")

st.markdown("---")
st.caption("Agent flow: Orchestrator → get_youTubeVid_agent → procs_youTubeVid_agent → END")
