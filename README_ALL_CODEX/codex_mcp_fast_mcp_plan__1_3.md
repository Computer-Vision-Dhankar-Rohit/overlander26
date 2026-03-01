# TODO-6 — FastMCP + LangChain/LangGraph (OpenAI): Detailed Execution Plan

**Branch**: `feature/dev_codex_1`  
**Date**: 2026-03-01  
**Status**: PRE-IMPLEMENTATION PLAN

---

## Fixed Architecture (Will Not Change)

This plan keeps your broad system architecture unchanged:

1. Reuse existing FastMCP server `mcpComputerVision`
2. Build OpenAI-based LangChain + LangGraph agents
3. Mandatory 3-agent flow:
   - `Orchestrator_Agent`
   - `get_youTubeVid_agent`
   - `procs_youTubeVid_agent`
4. Human-in-the-loop trigger from Streamlit (`ui_streamlit_codex`)

Optional agents remain available but not required by default:
- `send_email_agent`
- `procs_IPCAM_Vid_agent`

---

## PLAN_POINT_1 — FastMCP Server Reuse

### Existing Reference (No architecture change)
- `/home/dhankar/temp/26_02/git_over/overlander26/src/claude_agents/mcp_server_computer_vision.py`
- Server init already present: `mcp = FastMCP("mcpComputerVision")`
- Existing tools already exposed:
  - `send_email_tool`
  - `get_youTubeVid_tool`
  - `procs_youTubeVid_tool`
  - `procs_IPCAM_Vid_tool`

### Plan
- Reuse this MCP server as-is as the tool backend for LangGraph.
- Connect via `langchain-mcp-adapters` over `stdio` transport.

### Non-breaking Optimizations (Allowed)
- Normalize output directory casing (`DATA_DIR` vs `data_dir`) in follow-up cleanup.
- Keep secrets strictly in `.env` and inherited process env (no CLI arg secret passing).

---

## PLAN_POINT_2 — OpenAI LangChain + LCEL Agent Layer

### Target Files
- `/home/dhankar/temp/26_02/git_over/overlander26/src/openai_langchain_agents/main_agents_codex.py` (primary implementation)
- `/home/dhankar/temp/26_02/git_over/overlander26/src/openai_langchain_agents/main_agents.py` (compatibility entry/wrapper)
- `/home/dhankar/temp/26_02/git_over/overlander26/src/openai_langchain_agents/mcp_servers.py` (MCP server config constants/helper)

### Model Requirement (Latest GPT-5 family)
- Base default model: `gpt-5.2` (latest GPT-5 flagship family model as of 2026-03-01).
- Configurable via env:
  - `OPENAI_MODEL_NAME` (default `gpt-5.2`)
  - Optional override to `gpt-5.2-codex` for coding-heavy agent runs.

### API Key and Security
- Load only from project `.env`:
  - `/home/dhankar/temp/26_02/git_over/overlander26/DATA_DIR/secrets/.env`
- Use only `OPENAI_API_KEY` from this file.
- Never print key in logs, never write key to code, never commit key.

### LCEL Plan

Use LCEL in three places:

1. Orchestrator structured routing chain:
   - `ChatPromptTemplate | llm.with_structured_output(...)`
2. Agent execution chain per node:
   - system prompt + state messages + `llm.bind_tools(tools)`
3. Optional text-summary parser chain:
   - `prompt | llm | StrOutputParser()`

### Planned Public Functions
- `run_youtube_pipeline_codex(youtube_url: str, email_to: str = "")`
- `run_yt_download_only_codex(youtube_url: str, output_path: str)`
- Optional wrappers:
  - `run_ipcam_pipeline_codex(...)`
  - `run_send_email_only_codex(...)`

---

## PLAN_POINT_3 — LangGraph Flow (Mandatory + Optional)

### Mandatory Flow (Enabled by default)

```text
START
  ->
Orchestrator_Agent
  ->
get_youTubeVid_agent <-> ToolNode(get_youTubeVid_tool)
  ->
procs_youTubeVid_agent <-> ToolNode(procs_youTubeVid_tool)
  ->
END
```

### Optional Nodes (Implemented, feature-flagged or separate wrappers)
- `send_email_agent <-> ToolNode(send_email_tool)`
- `procs_IPCAM_Vid_agent <-> ToolNode(procs_IPCAM_Vid_tool)`

### Planned State Schema

```python
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
```

### Routing Strategy
- Orchestrator parses human request and sets `task_type`.
- Primary default route for TODO-6:
  - `task_type="youtube"` -> mandatory 3-agent path.
- Optional route availability:
  - `task_type="ipcam"` for future optional path.

---

## PLAN_POINT_4 — Streamlit Human-in-the-Loop (Codex UI)

### UI Root
- `/home/dhankar/temp/26_02/git_over/overlander26/ui_streamlit_codex/app.py`

### Human Trigger Rule
- Every run starts from user action in Streamlit (button click).
- No background autonomous execution without user trigger.

### Planned UI Changes
- Update `app.py` home instructions to show OpenAI LangGraph flow.
- Update existing page:
  - `/home/dhankar/temp/26_02/git_over/overlander26/ui_streamlit_codex/pages/download_video.py`
  - Move import from Claude pipeline to Codex OpenAI pipeline wrapper.
- Add full-pipeline page (new):
  - `/home/dhankar/temp/26_02/git_over/overlander26/ui_streamlit_codex/pages/youtube_pipeline.py`
  - Calls full 3-agent mandatory flow:
    - Orchestrator -> Download -> Process.

### UI Runtime Progress
- Use `st.status(...)` for step visibility:
  - Orchestrator routing
  - YouTube tool call
  - Video processing tool call
  - Final summary output

---

## Dependencies Plan

Update `/home/dhankar/temp/26_02/git_over/overlander26/requirements.txt` with:

- `langchain-openai>=0.3.0`
- `openai>=1.0.0`

Existing required packages already present:
- `langgraph`
- `langchain-core`
- `langchain-mcp-adapters`
- `fastmcp`
- `streamlit`

---

## Implementation Sequence (Execution Order)

1. Create `main_agents_codex.py` with OpenAI + LCEL + LangGraph core flow.
2. Add lightweight wrappers/exports in `main_agents.py`.
3. Add MCP server path/config helper in `mcp_servers.py`.
4. Update `ui_streamlit_codex/app.py`.
5. Update `ui_streamlit_codex/pages/download_video.py`.
6. Create `ui_streamlit_codex/pages/youtube_pipeline.py`.
7. Update `requirements.txt`.
8. Run local checks and smoke tests.
9. Write post-implementation report:
   - `/home/dhankar/temp/26_02/git_over/overlander26/README_ALL_CODEX/CODEX_mcp_fast_mcp_plan__implemented__1_3.md`

---

## Validation Plan

### Static checks
- Import checks for new modules.
- Type/schema sanity for LangGraph state and wrappers.

### Runtime checks
- MCP server discovery via `MultiServerMCPClient.get_tools()`.
- Confirm tool names found:
  - `get_youTubeVid_tool`
  - `procs_youTubeVid_tool`
- Streamlit smoke run:
  - `streamlit run ui_streamlit_codex/app.py`

### Functional checks
- Mandatory path run from UI:
  - Provide YouTube URL
  - Confirm download tool output includes `file_path`
  - Confirm processing tool output includes sampled frames / poses detected

---

## Risks and Mitigations

1. Path casing mismatch (`DATA_DIR` vs `data_dir`)
- Mitigation: standardize references during implementation and verify file writes.

2. Existing codex UI page currently imports Claude wrapper
- Mitigation: switch to OpenAI codex wrappers and retest end-to-end.

3. MCP tool call parsing variation by message type
- Mitigation: normalize message parsing helpers for `ToolMessage` and final `AIMessage`.

4. Missing optional reference doc (`Skills__1.md`) at provided path
- Mitigation: proceed without it; architecture and core references are already sufficient.

---

## Commit Policy

- All changes for this TODO will be committed only on:
  - `feature/dev_codex_1`
- No secrets and no `.env` content in commits.

