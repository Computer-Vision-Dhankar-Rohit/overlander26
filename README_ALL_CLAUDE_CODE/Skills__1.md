Perfect ✅ — this fits **exactly** into the MCP Tool-Hosted Architecture pattern you’ve already been building for your **some_tool_sample Query MCP** and **LangGraph Tool Invocation** pipelines.

Now we will create an **END-to-END single-file Python MCP Server** using
👉 `PrefectHQ / fastmcp`
Server Name 👉 `mcpComputerVision`
Hosting 4 Internal Tools:

| Tool Name               | Purpose                      |
| ----------------------- | ---------------------------- |
| `send_email_tool`       | Send Email Alerts            |
| `get_youTubeVid_tool`   | Download YouTube Video (MP4) |
| `procs_youTubeVid_tool` | Process Downloaded Video     |
| `procs_IPCAM_Vid_tool`  | Process IP Camera Stream     |

---

# ✅ FINAL ARCHITECTURE

```
LANGGRAPH AGENT
        │
        ▼
 MCP CLIENT (FASTMCP)
        │
        ▼
==================================================
        FAST MCP SERVER
        NAME = mcpComputerVision
==================================================
        │
        ├── send_email_tool()
        ├── get_youTubeVid_tool()
        ├── procs_youTubeVid_tool()
        └── procs_IPCAM_Vid_tool()
```

Exactly the **same Tool Invocation model** you used earlier for:

> AGENT → STATE → MCP TOOL → RETURN TO STATE

---

# 📦 INSTALL REQUIREMENTS

```bash
pip install fastmcp yt-dlp opencv-python smtplib email-validator
```

---

# 🚀 COMPLETE MCP SERVER

## 📄 FILE NAME → `mcp_server_computer_vision.py`

```python
# ============================================================
# MCP SERVER NAME : mcpComputerVision
# HOSTING 4 TOOLS
# ============================================================

from fastmcp import FastMCP
from pydantic import BaseModel
import yt_dlp
import cv2
import smtplib
from email.message import EmailMessage
import os

# ============================================================
# INITIALIZE MCP SERVER
# ============================================================

mcp = FastMCP("mcpComputerVision")

# ============================================================
# TOOL 1 : SEND EMAIL
# ============================================================

class EmailInput(BaseModel):
    receiver_email: str
    subject: str
    body: str

@mcp.tool(name="send_email_tool")
def send_email_tool(input: EmailInput):

    sender_email = "YOUR_EMAIL@gmail.com"
    sender_password = "YOUR_APP_PASSWORD"

    msg = EmailMessage()
    msg.set_content(input.body)

    msg["Subject"] = input.subject
    msg["From"] = sender_email
    msg["To"] = input.receiver_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)

    return {"status": "Email Sent Successfully"}

# ============================================================
# TOOL 2 : DOWNLOAD YOUTUBE VIDEO
# ============================================================

class YouTubeInput(BaseModel):
    youtube_url: str
    output_path: str = "./downloads"

@mcp.tool(name="get_youTubeVid_tool")
def get_youTubeVid_tool(input: YouTubeInput):

    os.makedirs(input.output_path, exist_ok=True)

    ydl_opts = {
        'format': 'mp4',
        'outtmpl': f"{input.output_path}/%(title)s.%(ext)s"
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([input.youtube_url])

    return {"status": "Video Downloaded"}

# ============================================================
# TOOL 3 : PROCESS DOWNLOADED YOUTUBE VIDEO
# ============================================================

class ProcessVideoInput(BaseModel):
    video_path: str

@mcp.tool(name="procs_youTubeVid_tool")
def procs_youTubeVid_tool(input: ProcessVideoInput):

    cap = cv2.VideoCapture(input.video_path)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

    cap.release()

    return {"frame_count": frame_count}

# ============================================================
# TOOL 4 : PROCESS IP CAMERA STREAM
# ============================================================

class IPCamInput(BaseModel):
    ipcam_url: str

@mcp.tool(name="procs_IPCAM_Vid_tool")
def procs_IPCAM_Vid_tool(input: IPCamInput):

    cap = cv2.VideoCapture(input.ipcam_url)

    frame_count = 0
    while frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

    cap.release()

    return {"processed_frames": frame_count}

# ============================================================
# RUN MCP SERVER
# ============================================================

if __name__ == "__main__":
    mcp.run()
```

---

# ▶️ RUN MCP SERVER

```bash
python mcp_server_computer_vision.py
```

---

# 🧠 LANGGRAPH AGENT CAN NOW CALL:

```
mcpComputerVision.send_email_tool
mcpComputerVision.get_youTubeVid_tool
mcpComputerVision.procs_youTubeVid_tool
mcpComputerVision.procs_IPCAM_Vid_tool
```

Exactly same way you earlier called:

```
some_tool_sample_query_tool
```

inside your:

```
STATEGRAPH
AGENT → TOOL → STATE UPDATE
```

---

# 🔁 MCP TOOL CALL JSON EXAMPLE

(For your LangGraph ToolNode)

```json
{
  "tool": "procs_IPCAM_Vid_tool",
  "arguments": {
    "ipcam_url": "rtsp://192.168.1.64:554/stream"
  }
}
```

---

# 🔐 IMPORTANT (Production Note)

Move:

```
EMAIL PASSWORD
SMTP CONFIG
DOWNLOAD PATH
MODEL PATH
```

into:

```
.env
Kubernetes Secret
Vault
some_tool_sample Secret Manager
```
