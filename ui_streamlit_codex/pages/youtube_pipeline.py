# ============================================================
# PAGE 2 — YouTube Full Pipeline (Mandatory 3-Agent Flow)
#
# Human trigger (button) ->
#   Orchestrator_Agent ->
#   get_youTubeVid_agent ->
#   procs_youTubeVid_agent
# ============================================================

import json
import os
import sys
from typing import Any, Dict, Optional

from dotenv import load_dotenv
import streamlit as st

_PAGE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../ui_streamlit_codex/pages/
_UI_DIR = os.path.dirname(_PAGE_DIR)  # .../ui_streamlit_codex/
_PROJECT_DIR = os.path.dirname(_UI_DIR)  # .../overlander26/
_SRC_DIR = os.path.join(_PROJECT_DIR, "src")

load_dotenv(
    dotenv_path=os.path.join(_PROJECT_DIR, "DATA_DIR", "secrets", ".env"),
    override=True,
)

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from openai_langchain_agents.main_agents import run_youtube_pipeline

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


st.set_page_config(page_title="YouTube Pipeline", page_icon="🧠", layout="centered")
st.title("YouTube Full Pipeline (OpenAI LangGraph)")
st.caption(
    "Flow: **Orchestrator_Agent -> get_youTubeVid_agent -> procs_youTubeVid_agent**"
)
st.markdown("---")

youtube_url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

email_to = st.text_input(
    "Alert Email (optional)",
    placeholder="analyst@example.com",
)

run_btn = st.button("Run Mandatory 3-Agent Pipeline", type="primary")

if run_btn:
    if not youtube_url.strip():
        st.warning("Please enter a YouTube URL first.")
        st.stop()

    with st.status("Running LangGraph pipeline...", expanded=True) as status:
        st.write("1) Orchestrator_Agent routing request...")
        st.write("2) get_youTubeVid_agent calling get_youTubeVid_tool...")
        st.write("3) procs_youTubeVid_agent calling procs_youTubeVid_tool...")
        if email_to.strip():
            st.write("Optional) send_email_agent may send alert...")

        try:
            final_state = run_youtube_pipeline(
                youtube_url=youtube_url.strip(),
                email_to=email_to.strip(),
            )
            status.update(label="Pipeline completed", state="complete")
        except Exception as exc:
            status.update(label="Pipeline failed", state="error")
            st.error(f"Error: {exc}")
            st.stop()

    messages = final_state.get("messages", [])
    download_result = _parse_tool_result(messages, "get_youTubeVid_tool")
    process_result = _parse_tool_result(messages, "procs_youTubeVid_tool")
    email_result = _parse_tool_result(messages, "send_email_tool")
    summary = _latest_agent_summary(messages)

    st.subheader("Run Result")

    if download_result:
        tool_status = download_result.get("status", "unknown")
        file_path = download_result.get("file_path", "")
        if tool_status == "downloaded" and file_path:
            size_mb = (
                os.path.getsize(file_path) / (1024 * 1024)
                if os.path.isfile(file_path)
                else 0.0
            )
            st.success(f"Download complete ({size_mb:.1f} MB)")
            st.code(file_path, language=None)
        else:
            st.error(f"Download failed: {download_result.get('detail', 'Unknown error')}")
    else:
        st.warning("No result found for `get_youTubeVid_tool`.")

    if process_result:
        if process_result.get("status") == "processed":
            c1, c2, c3 = st.columns(3)
            c1.metric("Sampled Frames", int(process_result.get("sampled_frames", 0)))
            c2.metric("Poses Detected", int(process_result.get("poses_detected", 0)))
            c3.metric("Total Frames", int(process_result.get("total_frames", 0)))
            st.caption(process_result.get("detail", ""))
        else:
            st.error(f"Processing failed: {process_result.get('detail', 'Unknown error')}")
    else:
        st.warning("No result found for `procs_youTubeVid_tool`.")

    if email_to.strip():
        if email_result:
            if email_result.get("status") == "sent":
                st.success(f"Email sent: {email_result.get('detail', '')}")
            else:
                st.error(f"Email failed: {email_result.get('detail', 'Unknown error')}")
        else:
            st.info("No email tool output found. Email step may have been skipped.")

    if summary:
        with st.expander("Agent Summary", expanded=False):
            st.write(summary)

    with st.expander("Raw Tool Payloads", expanded=False):
        st.json(
            {
                "get_youTubeVid_tool": download_result,
                "procs_youTubeVid_tool": process_result,
                "send_email_tool": email_result,
            }
        )
