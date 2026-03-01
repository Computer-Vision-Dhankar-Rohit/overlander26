# ============================================================
# MCP SERVER  : mcpComputerVision
# FRAMEWORK   : FastMCP (PrefectHQ/fastmcp)
# TOOLS HOSTED: 4
#   1. send_email_tool
#   2. get_youTubeVid_tool
#   3. procs_youTubeVid_tool
#   4. procs_IPCAM_Vid_tool
#
# RUN:
#   cd src && python claude_agents/mcp_server_computer_vision.py
#
# INSTALL:
#   pip install fastmcp yt-dlp opencv-python
# ============================================================

import os
import sys
import smtplib
from email.message import EmailMessage
from datetime import datetime

import cv2
import yt_dlp
from dotenv import load_dotenv
from fastmcp import FastMCP

# ── Load secrets ──────────────────────────────────────────────────────────────
_HERE_MCP    = os.path.dirname(os.path.abspath(__file__))
_SRC_MCP     = os.path.dirname(_HERE_MCP)
_PROJECT_MCP = os.path.dirname(_SRC_MCP)
_ENV_FILE    = os.path.join(_PROJECT_MCP, "DATA_DIR", "secrets", ".env")
load_dotenv(dotenv_path=_ENV_FILE, override=True)

# ── path so we can import MediaPipeGoog from sibling analysis/ dir ────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util_logger import setup_logger
logger = setup_logger(module_name=str(__name__))

# ── Config from environment variables (never hard-code credentials) ───────────
EMAIL_SENDER       = os.getenv("OVERLANDER_EMAIL_SENDER",   "")
EMAIL_APP_PASSWORD = os.getenv("OVERLANDER_EMAIL_PASSWORD", "")
SMTP_HOST          = os.getenv("OVERLANDER_SMTP_HOST",      "smtp.gmail.com")
SMTP_PORT          = int(os.getenv("OVERLANDER_SMTP_PORT",  "465"))

# Default output dirs (relative to project root, i.e. one level above src/)
_HERE        = os.path.dirname(os.path.abspath(__file__))   # .../src/claude_agents/
_SRC_DIR     = os.path.dirname(_HERE)                        # .../src/
_PROJECT_DIR = os.path.dirname(_SRC_DIR)                     # .../overlander26/

DEFAULT_YT_DOWNLOAD_PATH = os.getenv(
    "OVERLANDER_YT_DOWNLOAD_PATH",
    os.path.join(_PROJECT_DIR, "data_dir", "youtube_downloads")
)
DEFAULT_POSE_MODEL_PATH = os.path.join(
    _PROJECT_DIR, "data_dir", "pose_models", "pose_landmarker.task"
)

# ── Initialise FastMCP server ─────────────────────────────────────────────────
mcp = FastMCP("mcpComputerVision")


# ============================================================
# TOOL 1 — send_email_tool
# ============================================================

@mcp.tool()
def send_email_tool(
    receiver_email: str,
    subject: str,
    body: str,
) -> dict:
    """
    Send an email alert via Gmail SMTP.

    Uses OVERLANDER_EMAIL_SENDER and OVERLANDER_EMAIL_PASSWORD env vars.

    Args:
        receiver_email: Destination email address.
        subject:        Email subject line.
        body:           Plain-text email body (detection summary, timestamps, etc.)

    Returns:
        dict with key 'status' ("sent" | "error") and 'detail'.
    """
    if not EMAIL_SENDER or not EMAIL_APP_PASSWORD:
        return {
            "status": "error",
            "detail": (
                "Email credentials not configured. "
                "Set OVERLANDER_EMAIL_SENDER and OVERLANDER_EMAIL_PASSWORD env vars."
            ),
        }

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = receiver_email
        msg.set_content(body)

        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_APP_PASSWORD)
            smtp.send_message(msg)

        return {
            "status": "sent",
            "detail": f"Email delivered to {receiver_email} at {datetime.now().isoformat()}",
        }

    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


# ============================================================
# TOOL 2 — get_youTubeVid_tool
# ============================================================

@mcp.tool()
def get_youTubeVid_tool(
    youtube_url: str,
    output_path: str = DEFAULT_YT_DOWNLOAD_PATH,
) -> dict:
    """
    Download a YouTube video as an MP4 file using yt-dlp.

    Args:
        youtube_url: Full YouTube URL (e.g. https://www.youtube.com/watch?v=xxx).
        output_path: Directory to save the downloaded video.
                     Defaults to data_dir/youtube_downloads/.

    Returns:
        dict with keys:
            status      — "downloaded" | "error"
            file_path   — absolute path to the saved MP4 (on success)
            detail      — error message (on failure)
    """
    os.makedirs(output_path, exist_ok=True)

    # Capture the actual filename yt-dlp chooses
    downloaded_files: list[str] = []

    def _progress_hook(d: dict) -> None:
        if d["status"] == "finished":
            downloaded_files.append(d["filename"])

    # Redirect ALL yt-dlp output to stderr so stdout stays clean for
    # MCP stdio JSON-RPC messages (yt-dlp \r progress lines corrupt the stream).
    class _StderrLogger:
        def debug(self, msg: str) -> None:
            pass   # suppress debug/progress lines entirely
        def info(self, msg: str) -> None:
            pass
        def warning(self, msg: str) -> None:
            import sys
            sys.stderr.write(f"yt-dlp WARNING: {msg}\n")
        def error(self, msg: str) -> None:
            import sys
            sys.stderr.write(f"yt-dlp ERROR: {msg}\n")

    ydl_opts = {
        "format":   "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl":  os.path.join(output_path, "%(title)s.%(ext)s"),
        "progress_hooks": [_progress_hook],
        "quiet":    True,
        "no_warnings": True,
        "noprogress": True,
        "logger":   _StderrLogger(),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        file_path = downloaded_files[0] if downloaded_files else output_path
        return {
            "status":    "downloaded",
            "file_path": os.path.abspath(file_path),
            "detail":    f"Saved to {file_path}",
        }

    except Exception as exc:
        return {"status": "error", "file_path": "", "detail": str(exc)}


# ============================================================
# TOOL 3 — procs_youTubeVid_tool
# ============================================================

@mcp.tool()
def procs_youTubeVid_tool(
    video_path: str,
    sample_every_n_seconds: int = 5,
) -> dict:
    """
    Process a downloaded YouTube video through MediaPipe pose detection.

    Samples one frame every `sample_every_n_seconds` seconds and runs the
    MediaPipeGoog pose detection pipeline on each sampled frame.

    Args:
        video_path:             Absolute path to a local MP4/AVI/MOV video file.
        sample_every_n_seconds: How often (in seconds) to sample a frame.
                                Default is 5 (same as the live pipeline).

    Returns:
        dict with keys:
            status          — "processed" | "error"
            total_frames    — total frame count in the video
            sampled_frames  — how many frames were sent for pose detection
            poses_detected  — frames where at least one pose was found
            output_dir      — directory where annotated frames were saved
            detail          — error message (on failure)
    """
    if not os.path.isfile(video_path):
        return {
            "status": "error",
            "detail": f"Video file not found: {video_path}",
        }

    try:
        # Import here so the MCP server can start even if mediapipe is absent
        from analysis.media_pipe import MediaPipeGoog

        # Ensure detector is loaded once
        MediaPipeGoog.init_pose_detector(DEFAULT_POSE_MODEL_PATH)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "error", "detail": f"Cannot open video: {video_path}"}

        fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps * sample_every_n_seconds))

        # Output dir for raw sampled frames (same layout as the main pipeline)
        out_raw_dir = os.path.join(
            _PROJECT_DIR, "data_dir", "pose_detected", "detected_pose"
        )
        os.makedirs(out_raw_dir, exist_ok=True)

        sampled = 0
        poses_detected = 0
        current_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame % frame_interval == 0:
                # Resize to match the main pipeline dimensions
                resized = cv2.resize(frame, (500, 650))
                frame_path = os.path.join(out_raw_dir, f"yt_frame_{sampled:04d}.jpg")
                cv2.imwrite(frame_path, resized)

                # Run pose detection (same function used by main pipeline)
                result_path = MediaPipeGoog.pose_media_pipe_google_0(frame_path)
                sampled += 1
                if isinstance(result_path, str) and result_path != "EMPTY_STR":
                    # _0() saves to pose_id_not_ipcam when landmarks found
                    if "pose_id_not_ipcam" in result_path:
                        poses_detected += 1

            current_frame += 1

        cap.release()

        return {
            "status":        "processed",
            "total_frames":  total_frames,
            "sampled_frames": sampled,
            "poses_detected": poses_detected,
            "output_dir":    out_raw_dir,
            "detail":        f"Processed {sampled} frames, {poses_detected} poses found",
        }

    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


# ============================================================
# TOOL 4 — procs_IPCAM_Vid_tool
# ============================================================

@mcp.tool()
def procs_IPCAM_Vid_tool(
    ipcam_url: str,
    max_frames: int = 150,
    sample_every_n_seconds: int = 5,
) -> dict:
    """
    Connect to an IP camera / RTSP stream and run MediaPipe pose detection.

    Reads up to `max_frames` frames from the live stream, sampling one frame
    every `sample_every_n_seconds` seconds for pose detection.

    Args:
        ipcam_url:              RTSP / HTTP stream URL
                                (e.g. "rtsp://192.168.1.64:554/stream"
                                 or   "http://192.168.1.100:8080/video").
        max_frames:             Hard cap on how many raw frames to read.
                                Prevents infinite loops on live streams.
        sample_every_n_seconds: Pose detection is run on one frame per this
                                many seconds of stream time.

    Returns:
        dict with keys:
            status          — "processed" | "error"
            frames_read     — how many frames were consumed from the stream
            sampled_frames  — how many frames were sent for pose detection
            poses_detected  — frames where at least one pose was found
            output_dir      — directory where annotated frames were saved
            detail          — human-readable summary or error message
    """
    try:
        from analysis.media_pipe import MediaPipeGoog

        MediaPipeGoog.init_pose_detector(DEFAULT_POSE_MODEL_PATH)

        cap = cv2.VideoCapture(ipcam_url)
        if not cap.isOpened():
            return {
                "status": "error",
                "detail": f"Cannot connect to stream: {ipcam_url}",
            }

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0  # Assume 25 fps if the stream doesn't report it

        frame_interval = max(1, int(fps * sample_every_n_seconds))

        out_raw_dir = os.path.join(
            _PROJECT_DIR, "data_dir", "pose_detected", "detected_pose"
        )
        os.makedirs(out_raw_dir, exist_ok=True)

        frames_read    = 0
        sampled        = 0
        poses_detected = 0

        while frames_read < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frames_read % frame_interval == 0:
                resized    = cv2.resize(frame, (500, 650))
                frame_path = os.path.join(out_raw_dir, f"ipcam_frame_{sampled:04d}.jpg")
                cv2.imwrite(frame_path, resized)

                result_path = MediaPipeGoog.pose_media_pipe_google_0(frame_path)
                sampled += 1
                if isinstance(result_path, str) and "pose_id_not_ipcam" in result_path:
                    poses_detected += 1

            frames_read += 1

        cap.release()

        return {
            "status":        "processed",
            "frames_read":   frames_read,
            "sampled_frames": sampled,
            "poses_detected": poses_detected,
            "output_dir":    out_raw_dir,
            "detail":        (
                f"Stream {ipcam_url}: read {frames_read} frames, "
                f"sampled {sampled}, {poses_detected} poses detected"
            ),
        }

    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Runs the MCP server over stdio — LangChain agents connect via subprocess
    logger.debug("--- mcpComputerVision Starting FastMCP server")
    logger.debug("--- mcpComputerVision YT download dir %s", DEFAULT_YT_DOWNLOAD_PATH)
    logger.debug("--- mcpComputerVision Pose model %s", DEFAULT_POSE_MODEL_PATH)
    mcp.run()
