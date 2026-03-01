# TODO-6 — FastMCP + LangGraph OpenAI Agents: Implementation Record

**Branch**: `feature/dev_claude_1`
**Commit**: `52da2bf`
**Date**: 2026-03-01
**Status**: IMPLEMENTED

---

## Files Created / Modified

| File | Action | Description |
|------|--------|-------------|
| `src/claude_agents/mcp_server_computer_vision.py` | REUSED | FastMCP server (unchanged) |
| `src/openai_langchain_agents/main_agents_claude.py` | CREATED | OpenAI LangGraph 3-agent pipeline |
| `src/openai_langchain_agents/main_agents.py` | EMPTY | Placeholder (not used) |
| `src/openai_langchain_agents/mcp_servers.py` | EMPTY | Placeholder (not used) |
| `ui_streamlit_claude/app.py` | UPDATED | Navigation table updated |
| `ui_streamlit_claude/pages/youtube_pipeline.py` | CREATED | Full pipeline Streamlit page |
| `ui_streamlit_claude/pages/download_video.py` | REUSED | Existing Claude agent page |
| `README_ALL_CLAUDE_CODE/CLAUDE_mcp_fast_mcp_plan__1_3.md` | CREATED | Plan document |

---

## Implementation Notes

### PLAN_POINT_1 — FastMCP Server (REUSED)

`src/claude_agents/mcp_server_computer_vision.py` was reused unchanged.
The 4 tools remain identical: `send_email_tool`, `get_youTubeVid_tool`,
`procs_youTubeVid_tool`, `procs_IPCAM_Vid_tool`.

MCP server launched as subprocess via `langchain-mcp-adapters` / `MultiServerMCPClient`:
```python
_MCP_SERVER = os.path.join(_SRC_DIR, "claude_agents", "mcp_server_computer_vision.py")
mcp_client  = MultiServerMCPClient({
    "mcpComputerVision": {
        "command": sys.executable,
        "args":    [_MCP_SERVER],
        "transport": "stdio",
    }
})
```

### PLAN_POINT_2 — OpenAI Agents with LCEL

**File**: `src/openai_langchain_agents/main_agents_claude.py`

**Model**:
```python
OPENAI_MODEL = "gpt-5.2"   # OpenAI flagship (2026-03-01)
_LLM = ChatOpenAI(model=OPENAI_MODEL, temperature=0, max_tokens=1024)
```

**LCEL — Orchestrator chain** (structured output):
```python
_orchestrator_chain = (
    ChatPromptTemplate.from_messages([
        ("system", _ORCHESTRATOR_SYSTEM),
        ("human", "{user_message}"),
    ])
    | _LLM.with_structured_output(OrchestratorOutput)
)
```

**LCEL — ReAct agent nodes** (bind_tools + ainvoke):
```python
llm_with_tools = _LLM.bind_tools(tools)
conversation   = [SystemMessage(content=system_prompt)] + state["messages"]
response       = await llm_with_tools.ainvoke(conversation)
```

**State**:
```python
class OverlanderStateOpenAI(TypedDict):
    task_type:        str
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

### PLAN_POINT_3 — LangGraph Agent Flow

```
START
  │
  ▼
orchestrator_agent  ← _orchestrator_node (LCEL chain, no tools)
  │  add_conditional_edges → _route_after_orchestrator
  ▼
yt_download_agent   ← _make_agent_node(_YT_DOWNLOAD_PROMPT, yt_dl_tools)
  ↔  yt_download_tools (ToolNode)
  │  tools_condition: "tools" → yt_download_tools, END → yt_process_agent
  ▼
yt_process_agent    ← _make_agent_node(_YT_PROCESS_PROMPT, yt_proc_tools)
  ↔  yt_process_tools (ToolNode)
  │  tools_condition: "tools" → yt_process_tools, END → END
  ▼
END
```

**Public functions**:
```python
run_youtube_pipeline_openai(youtube_url: str, email_to: str = "") -> OverlanderStateOpenAI
run_yt_download_only_openai(youtube_url: str, output_path: str)   -> OverlanderStateOpenAI
```

### PLAN_POINT_4 — Streamlit UI

**Run command**:
```bash
cd /home/dhankar/temp/26_02/git_over/overlander26
streamlit run ui_streamlit_claude/app.py
```

**Pages**:
```
ui_streamlit_claude/
├── app.py                        ← Home page (updated navigation table)
└── pages/
    ├── download_video.py         ← EXISTING: Claude agents, download only
    └── youtube_pipeline.py       ← NEW: OpenAI agents, full pipeline
```

`youtube_pipeline.py` flow:
1. Human enters YouTube URL in `st.text_input`
2. Human clicks **Run Full Pipeline (OpenAI Agents)** button
3. `st.status()` shows real-time progress across 3 agent steps
4. Results displayed in separate sections: Download Result + Pose Detection Result
5. Agent final summary shown in collapsible expander

---

## Security

- `OPENAI_API_KEY` loaded from `DATA_DIR/secrets/.env` via `load_dotenv(override=True)`
- `DATA_DIR/secrets/` is git-ignored — key never committed
- MCP server subprocess inherits the environment — no key passed as argument

---

## Dependencies Added

Added to `reqmts.log` (if not already present):
```
langchain-openai>=0.3.0
openai>=1.0.0
```
