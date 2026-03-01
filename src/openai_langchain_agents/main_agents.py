"""Compatibility entrypoint for OpenAI LangChain/LangGraph agents."""

from __future__ import annotations

from openai_langchain_agents.main_agents_codex import (
    OverlanderStateCodex,
    run_ipcam_pipeline_codex,
    run_send_email_only_codex,
    run_youtube_pipeline_codex,
    run_yt_download_only_codex,
)

# Backward-compatible names for app imports.
run_yt_download_only = run_yt_download_only_codex
run_youtube_pipeline = run_youtube_pipeline_codex
run_ipcam_pipeline = run_ipcam_pipeline_codex
run_send_email_only = run_send_email_only_codex

__all__ = [
    "OverlanderStateCodex",
    "run_yt_download_only_codex",
    "run_youtube_pipeline_codex",
    "run_ipcam_pipeline_codex",
    "run_send_email_only_codex",
    "run_yt_download_only",
    "run_youtube_pipeline",
    "run_ipcam_pipeline",
    "run_send_email_only",
]
