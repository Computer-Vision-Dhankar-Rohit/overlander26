# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Computer vision pipeline for human pose detection, facial landmark detection, and object detection using MediaPipe Tasks API 0.10+, DETR, and YOLO models. Key use case: virtual fence boundary detection for intrusion monitoring.

## Setup & Run

```bash
# Install dependencies (note: file is reqmts.log, not requirements.txt)
pip install -r reqmts.log

# Download required MediaPipe model
mkdir -p data_dir/pose_models
wget -O data_dir/pose_models/pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

# Run (entry point is src/main.py, not root main.py)
cd src && python main.py
```

The active pipeline is controlled by the single method call at the bottom of `src/main.py` (currently `IPWebCam().face_detect_and_landmarks_combined()`). Change that line to switch pipelines.

## Architecture

### Entry Point & Orchestration (`src/main.py`)
`IPWebCam` class routes to the various detection pipelines. Each public method corresponds to a distinct pipeline (pose, face, object detection). To run a different pipeline, update the bottom invocation line.

### Pose Detection (`src/analysis/media_pipe.py`)
`MediaPipeGoog` class using MediaPipe Tasks API 0.10+. Uses a **singleton pattern** for the detector — `init_pose_detector()` is called once and reused across all frames for 3–5x performance. The primary method is `pose_media_pipe_google_2()`, which accepts a video path, RTSP URL, integer webcam index, or defaults to scanning `../data_dir/pose_detected/init_video/`.

**Virtual fence logic** (lines ~139–165): landmark #5 (Left Eye) crossing a horizontal line at 70% frame height triggers a red bounding box and "-FACE-" label.

### Object & Face Detection (`src/analysis/detr_hugging_face.py`)
Multiple classes handling distinct tasks:
- `GetFramesFromVids` — extracts frames at fixed indices from video files
- `FaceDetection` — YOLOv8 face detector (loaded from Hugging Face)
- `ObjDetHFRtDetr` — RT-DETR v2 via Hugging Face pipeline (`PekingU/rtdetr_v2_r50vd`)
- `FacialLandmarksDetection` — crops detected faces and runs landmark detection

Several DETR models are **pre-initialized at module import time** (globals at top of file), so importing `detr_hugging_face.py` triggers model downloads on first run.

### Logging (`src/util_logger.py`)
Call `setup_logger(__name__)` at the top of any module. Writes rotating daily logs to `../logs_dir/` with format `ipwebcam_log_MM_DD_YYYY_HH00h_.log`.

## Data Directory Layout

All I/O paths are relative to `src/` (one level up = project root):

```
../data_dir/
├── pose_models/          # MediaPipe .task model files
├── pose_detected/
│   ├── init_video/       # Input videos for pose pipeline
│   ├── detected_pose/    # Output frames (no landmarks)
│   └── pose_id_not_ipcam/ # Output frames (with landmarks)
├── out_vid_frames_dir/   # Extracted video frames (DETR pipeline)
└── out_dir/              # Face detection + combined outputs
../logs_dir/              # Rotating log files
```

## Key Patterns

- **No test suite, no linter config, no build system** — direct Python execution only.
- Model initialization is expensive; the singleton/global pattern intentionally avoids re-loading on each call.
- `pose_media_pipe_google_2()` handles all input types (file, RTSP, USB webcam int, IP camera URL) via type/prefix checks.
- Frame extraction in DETR pipeline uses a fixed index list `[4,11,17,25,30,37,45,55,66,77,88,100,110]`.
