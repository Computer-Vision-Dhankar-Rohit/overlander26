# ============================================================
# Overlander — Streamlit UI entry point
#
# RUN:
#   cd /home/dhankar/temp/26_02/git_over/overlander26
#   streamlit run ui_streamlit/app.py
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
    Use the **sidebar** to navigate between tools.

    | Page | Agents | Tools |
    |---|---|---|
    | Download Video | YouTubeDownloadAgent (Claude) | `get_youTubeVid_tool` |
    | Full Pipeline (OpenAI) | Orchestrator → YouTubeDownload → PoseDetection | `get_youTubeVid_tool`, `procs_youTubeVid_tool` |
    """
)

st.info("Select a page from the **sidebar** to get started.")
