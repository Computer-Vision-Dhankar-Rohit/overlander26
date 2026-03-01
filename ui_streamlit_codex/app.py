# ============================================================
# Overlander — Streamlit UI entry point
#
# RUN:
#   cd /home/dhankar/temp/26_02/git_over/overlander26
#   streamlit run ui_streamlit_codex/app.py
# ============================================================

import streamlit as st

st.set_page_config(
    page_title="Overlander CV Pipeline",
    page_icon="🎯",
    layout="centered",
)

st.title("Overlander — Computer Vision Pipeline")
st.markdown("---")

st.markdown(
    """
    Use the **sidebar** to trigger LangGraph runs (human-in-the-loop).

    | Page | Flow | Primary Tools |
    |---|---|---|
    | Download Video | `get_youTubeVid_agent` | `get_youTubeVid_tool` |
    | YouTube Pipeline | `Orchestrator_Agent -> get_youTubeVid_agent -> procs_youTubeVid_agent` | `get_youTubeVid_tool`, `procs_youTubeVid_tool` |
    """
)

st.info("Select a page from the sidebar to start a run.")
