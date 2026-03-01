# TODO-6 — FastMCP + LangGraph OpenAI Agents: Detailed Plan

**Branch**: `feature/dev_claude_1`
**Date**: 2026-03-01
**Status**: PLANNED (implementation follows)

---

## Overview

Build an OpenAI-powered LangGraph multi-agent pipeline that:
1. Is triggered by a human via Streamlit UI
2. Uses GPT-5.2 (OpenAI's current flagship, API id `gpt-5.2`) for all agents
3. Uses LangChain Expression Language (LCEL) for composable chain construction
4. Connects to the existing FastMCP server (`mcpComputerVision`) for tool execution
5. Exposes a Streamlit UI (`ui_streamlit_claude/`) as the human-in-the-loop entry point

---

## PLAN_POINT_1 — Reuse Existing FastMCP Server

**Reference file** (unchanged):
```
src/claude_agents/mcp_server_computer_vision.py
```

The FastMCP server `mcpComputerVision` already hosts 4 tools:

| Tool                  | Purpose                                |
|-----------------------|----------------------------------------|
| `send_email_tool`     | Send Gmail alert                       |
| `get_youTubeVid_tool` | Download YouTube video via yt-dlp      |
| `procs_youTubeVid_tool` | Run MediaPipe pose detection on video |
| `procs_IPCAM_Vid_tool`  | Connect to RTSP/HTTP stream + detect  |

The MCP server is launched as a subprocess by `langchain-mcp-adapters` via **stdio transport**.
No changes required to `mcp_server_computer_vision.py`.

---

## PLAN_POINT_2 — OpenAI Agents with LCEL

**New file**:
```
src/openai_langchain_agents/main_agents_claude.py
```

### Model
```python
OPENAI_MODEL = "gpt-5.2"   # OpenAI current flagship (2026-03-01)
```
API key loaded exclusively from `DATA_DIR/secrets/.env` → `OPENAI_API_KEY`.
Key is already git-ignored (`DATA_DIR/secrets/` in `.gitignore`).

### LCEL Usage

LangChain Expression Language (`|` pipe operator) is used for:

1. **Orchestrator chain** — parses the user's raw message and extracts structured routing info:
   ```python
   orchestrator_chain = (
       ChatPromptTemplate.from_messages([...])
       | llm.with_structured_output(OrchestratorOutput)
   )
   ```

2. **ReAct agent nodes** — each agent node composes prompt + LLM-bound-tools as a chain:
   ```python
   llm_with_tools = llm.bind_tools(tools)
   agent_chain = [SystemMessage(system_prompt)] + state["messages"]
   response = await llm_with_tools.ainvoke(agent_chain)
   ```

3. **String output parser** — for any agent producing free-text:
   ```python
   summary_chain = prompt | llm | StrOutputParser()
   ```

### LangGraph State
```python
class OverlanderStateOpenAI(TypedDict):
    task_type:        str   # "youtube" | "ipcam"
    youtube_url:      str
    ipcam_url:        str
    max_frames:       int
    video_file_path:  str
    frames_sampled:   int
    poses_detected:   int
    detection_detail: str
    email_to:         str
    messages: Annotated[list[BaseMessage], add_messages]
```

---

## PLAN_POINT_3 — LangGraph Agents Flow

### Mandatory Agents (3)

```
START
  │
  ▼
Agent_1: Orchestrator_Agent   ← triggered by HUMAN via Streamlit
  │  (LCEL chain: parse user message → extract task_type + youtube_url)
  │  routes task_type = "youtube"
  ▼
Agent_2: get_youTubeVid_agent  ← ReAct, calls get_youTubeVid_tool (MCP)
  │  ↔ ToolNode(get_youTubeVid_tool)
  │  on tool completion →
  ▼
Agent_3: procs_youTubeVid_agent  ← ReAct, calls procs_youTubeVid_tool (MCP)
  │  ↔ ToolNode(procs_youTubeVid_tool)
  │  on completion →
  ▼
END
```

### Optional Agents (not wired by default)

```
Agent_3a: send_email_agent     — calls send_email_tool (MCP)
Agent_4:  procs_IPCAM_Vid_agent — calls procs_IPCAM_Vid_tool (MCP)
```

### LangGraph Conditional Edges

```python
# After Orchestrator (no tools — direct routing based on state.task_type)
graph.add_conditional_edges(START, _route_from_orchestrator)
# _route_from_orchestrator: "youtube" → "yt_download_agent"

# Each ReAct agent uses tools_condition for tool ↔ agent loop:
graph.add_conditional_edges(
    "yt_download_agent", tools_condition,
    {"tools": "yt_download_tools", END: "yt_process_agent"}
)
graph.add_conditional_edges(
    "yt_process_agent", tools_condition,
    {"tools": "yt_process_tools", END: END}
)
```

### Exposed Public Functions

```python
# Full 3-agent pipeline (Orchestrator + Download + Process)
run_youtube_pipeline_openai(youtube_url: str, email_to: str = "") -> OverlanderStateOpenAI

# Download-only minimal pipeline (for Streamlit Page 1)
run_yt_download_only_openai(youtube_url: str, output_path: str) -> OverlanderStateOpenAI
```

---

## PLAN_POINT_4 — Streamlit UI

**Directory**: `ui_streamlit_claude/`
**Entry point**: `ui_streamlit_claude/app.py`

### Run Command
```bash
cd /home/dhankar/temp/26_02/git_over/overlander26
streamlit run ui_streamlit_claude/app.py
```

### Pages

```
ui_streamlit_claude/
├── app.py                        ← Home page (navigation table)
└── pages/
    ├── download_video.py         ← Existing: Claude agents, download only
    └── youtube_pipeline.py       ← NEW: OpenAI agents, full pipeline
```

### `youtube_pipeline.py` — Full Pipeline Page

Flow triggered by HUMAN:
```
User enters YouTube URL in text_input
  → st.button "Run Full Pipeline (OpenAI Agents)"
    → Orchestrator_Agent parses URL
      → get_youTubeVid_agent downloads video
        → procs_youTubeVid_agent detects poses
          → st.success + result table shown
```

### Human-in-the-Loop Principle

- Every pipeline execution starts with a **human action** (button click in Streamlit)
- The Orchestrator_Agent receives the human's raw message and orchestrates the rest
- Agents execute autonomously but the human can see real-time progress via `st.status()`

---

## File Map

| File | Action | Description |
|------|--------|-------------|
| `src/claude_agents/mcp_server_computer_vision.py` | REUSE | FastMCP server (unchanged) |
| `src/openai_langchain_agents/main_agents_claude.py` | CREATE | OpenAI LangGraph agents |
| `src/openai_langchain_agents/main_agents.py` | EMPTY | Placeholder (not used) |
| `src/openai_langchain_agents/mcp_servers.py` | EMPTY | Placeholder (not used) |
| `ui_streamlit_claude/app.py` | UPDATE | Home page |
| `ui_streamlit_claude/pages/youtube_pipeline.py` | CREATE | Full pipeline page |
| `ui_streamlit_claude/pages/download_video.py` | REUSE | Existing Claude agent page |

---

## Dependencies to Add to `requirements.txt`

```
langchain-openai>=0.3.0
openai>=1.0.0
```

---

## Environment Variables

| Variable | Source | Usage |
|----------|--------|-------|
| `OPENAI_API_KEY` | `DATA_DIR/secrets/.env` | OpenAI API authentication |
| `ANTHROPIC_API_KEY` | `DATA_DIR/secrets/.env` | Existing Claude agents (unchanged) |
| `OVERLANDER_EMAIL_SENDER` | `DATA_DIR/secrets/.env` | Optional email alerts |
| `OVERLANDER_EMAIL_PASSWORD` | `DATA_DIR/secrets/.env` | Optional email alerts |

---

## Security

- `DATA_DIR/secrets/.env` is git-ignored — API key is never committed
- Key is loaded with `load_dotenv(override=True)` at module import time
- MCP server subprocess inherits the environment — no key is passed as argument

---

*Post-implementation document*: `README_ALL_CLAUDE_CODE/CLAUDE_mcp_fast_mcp_plan__implemented__1_3.md`
