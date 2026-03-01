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
st.set_page_config(page_title="Trigger OpenAI Agents", page_icon="⚡", layout="centered")

st.title("Trigger OpenAI Agents")
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
            else:
                out_dir = os.path.join(_PROJECT_DIR, "DATA_DIR", "youTubeVids")
                os.makedirs(out_dir, exist_ok=True)
                state = run_yt_download_only_openai(youtube_url=url, output_path=out_dir)

            status.update(label="Agents completed", state="complete")
            logger.debug("--- trigger_agents_OpenAI completed ok")

        except Exception as exc:
            status.update(label="Agent error", state="error")
            logger.debug("--- trigger_agents_OpenAI error %s", exc)
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

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("### Results")

    for i, result in enumerate(tool_results, 1):
        tool_status = result.get("status", "unknown")
        icon = "✅" if tool_status not in ("error", "unknown") else "❌"
        with st.expander(f"{icon} Tool result {i} — `{tool_status}`", expanded=True):
            st.json(result)

    if final_ai:
        st.markdown("### Agent Summary")
        st.write(final_ai)

st.markdown("---")
st.caption("Agent flow: Orchestrator → get_youTubeVid_agent → procs_youTubeVid_agent → END")
