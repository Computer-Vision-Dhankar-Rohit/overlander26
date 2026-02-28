# OPENCV GRID VIEW — System Architecture & Method Reasoning
## File: `src/analysis/media_pipe.py`

---

## Overview

`media_pipe.py` implements the **`MediaPipeGoog`** class — the core pose detection engine for this project. It uses the **MediaPipe Tasks API 0.10+** (not the older Solutions API), which means:
- No protobuf conversion required
- Direct `.x`, `.y`, `.z` attribute access on landmarks
- Detector is created via `PoseLandmarker.create_from_options()`

The central design principle is a **singleton detector pattern**: the expensive model is loaded once at class level, then reused across all frames. This gives a 3–5x performance improvement over re-loading per frame.

The class name derives from "Google" (MediaPipe is a Google framework) and "IP webcam" (the original use case was an Android IP webcam app).

---

## Class-Level Singleton: `detector = None`

```python
class MediaPipeGoog():
    detector = None
```

This is a **class variable**, not an instance variable. It is shared across all instances of `MediaPipeGoog`. The first time `init_pose_detector()` is called, it creates the detector and stores it here. Subsequent calls (from any frame, any loop iteration) skip initialization because `detector` is no longer `None`.

**Why this matters:** Loading a `.task` model file is slow (hundreds of milliseconds). In a video loop processing thousands of frames, re-loading on every frame would make the pipeline unusable.

---

## Method 1: `init_pose_detector(cls, model_path)`

**Lines:** ~33–66

### What it does
Initializes the MediaPipe `PoseLandmarker` detector exactly **once** and stores it as a class-level attribute.

### How it works

```
model_path (default: '../data_dir/pose_models/pose_landmarker.task')
    ↓
BaseOptions(model_asset_path=model_path)     ← points to .task binary file
    ↓
PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.IMAGE,          ← static frame mode, not video/live
    output_segmentation_masks=False          ← disabled for speed
)
    ↓
PoseLandmarker.create_from_options(options)  ← heavy model load happens here
    ↓
cls.detector = <PoseLandmarker object>       ← stored at class level
```

### Key design decisions

| Decision | Reason |
|---|---|
| `running_mode=IMAGE` | Processes each frame independently; no temporal tracking between frames. Safe for both files and live streams. |
| `output_segmentation_masks=False` | Segmentation masks are expensive to compute; not needed for fence-crossing detection. |
| `if cls.detector is None` guard | Prevents double-initialization. Once set, all subsequent calls are no-ops. |

### Guard behavior
```
First call  → cls.detector is None → loads model → sets cls.detector
Second call → cls.detector is NOT None → logs "already initialized" → returns
```

---

## Method 2: `get_horizontal_line_position(self, height, width)`

**Lines:** ~68–75

### What it does
Calculates the **Y-coordinate** of the virtual fence line — the horizontal boundary that determines if a person has "intruded."

### How it works
```python
line_y = int(height * 0.70)
```

- The frame height is multiplied by `0.70`
- Returns a pixel Y-coordinate at **70% of the frame height** (measured from top)
- So for a 650px tall frame: `line_y = 455`

### Why 70%?
The fence line is positioned **low in the frame**, near the bottom third. This makes sense for a ground-level intrusion scenario — a person standing normally will have their head/eyes (landmark #5) **above** this line (smaller Y value). When they crouch, move into a restricted zone, or the camera is oriented such that the boundary matters at this height, the fence triggers.

**Note:** The code comment says `#TODO# 0.30 -- from TOP 30%` — the original intent was 30% from the top, but it was changed to 70%. The line is the same — `0.30` from top = `0.70` from top in terms of frame-bottom orientation.

---

## Method 3: `draw_horizontal_line(self, pose_annotated_image)`

**Lines:** ~77–101

### What it does
Draws the visible **green horizontal fence line** onto the annotated frame.

### How it works
```
pose_annotated_image (numpy array, BGR)
    ↓
height, width = image.shape[:2]
    ↓
line_y = int(height * 0.70)              ← same 70% calculation
    ↓
cv2.line(
    pose_annotated_image,
    (0, line_y),                         ← left edge of frame
    (width - 1, line_y),                 ← right edge of frame
    (0, 255, 0),                         ← GREEN in BGR
    5                                    ← 5px thick
)
    ↓
return pose_annotated_image              ← image modified in-place AND returned
```

### Visual result
A solid green horizontal line stretching the full width of the frame, sitting at 70% of the frame's height. This is the **virtual fence boundary** — it's the visual reference for the intrusion detection zone.

### Note on `cv2.line` coordinate system
OpenCV's origin (0,0) is at the **top-left** corner. Y increases downward. So a line at `line_y = int(height * 0.70)` is 70% of the way DOWN from the top.

---

## Method 4: `pose_draw_landmarks_on_image(self, rgb_image, detection_result)`

**Lines:** ~103–193

### What it does
This is the **core annotation engine**. It takes:
- `rgb_image`: the raw numpy frame
- `detection_result`: the MediaPipe `PoseLandmarkerResult` object

And returns:
- `annotated_image`: the frame with drawings overlaid
- `flag_pose_landmarks_detected`: `"YES_LANDMARKS"` or `"NO_LANDMARKS"`

### Step-by-step flow

```
rgb_image + detection_result
    ↓
annotated_image = np.copy(rgb_image)          ← deep copy, don't modify original
    ↓
pose_landmarks_list = detection_result.pose_landmarks
    ↓
[GUARD] if len(pose_landmarks_list) < 1:
    → flag = "NO_LANDMARKS"
    → return early (no drawing)
    ↓
line_y_coord = get_horizontal_line_position(height, width)   ← fence Y coord
    ↓
for each detected pose:
    pose_landmarks = pose_landmarks_list[idx]
        ↓
    for each landmark in pose:
        if landmark_idx == 5:                 ← ONLY landmark #5 (Left Eye)
            x = int(landmark.x * width)       ← normalize [0,1] → pixel
            y = int(landmark.y * height)      ← normalize [0,1] → pixel
                ↓
            if line_y_coord > y:              ← eye is ABOVE the fence line
                [ALERT MODE]
                → draw RED rectangle around the eye position
                → draw "-FACE-" label in RED
                ↓
    annotated_image = draw_horizontal_line(annotated_image)  ← always draw fence
    return annotated_image, "YES_LANDMARKS"
```

### The Virtual Fence Logic (lines ~140–163)

```python
if landmark_idx == 5:                      # Left Eye landmark
    x = int(landmark.x * width)
    y_coord_height = int(landmark.y * height)

    if line_y_coord > y_coord_height:      # Eye is ABOVE the fence line
        # RED alert box around the eye
        cv2.rectangle(annotated_image, top_left, bottom_right, (0,0,255), 5)
        # RED "-FACE-" label
        cv2.putText(annotated_image, "-FACE-", (x, y_coord_height), ...)
```

**Why landmark #5 (Left Eye)?**
- Landmark #5 in MediaPipe's 33-point body model is the **left eye outer corner**
- The eye is a stable, reliably-detected landmark near the head
- Using the head/eye position as a proxy for "person's location in frame" is practical
- If the eye is above the fence line (smaller Y = higher in frame), an alert triggers

**Coordinate comparison logic:**
```
line_y_coord = 70% of frame height (e.g., 455 for 650px frame)
y_coord_height = eye's Y position in pixels

if line_y_coord > y_coord_height:
    → eye Y < fence Y → eye is HIGHER in frame than fence → ABOVE fence → ALERT
```

**Alert drawing:**
- Red rectangle: centered on the eye position, 35×55 pixels (portrait orientation box)
- "-FACE-" text: printed at the eye coordinate in RED

### Font/text settings
```python
font = cv2.FONT_HERSHEY_PLAIN    # thinnest available font
font_scale = 10                   # large scale for the landmark numbers (unused in active code)
font_scale_alert = 2              # smaller scale for the "-FACE-" alert text
thickness = 3                     # line thickness
```

### Commented-out code (lines ~171–188)
There is substantial commented-out code for:
- Drawing when landmark crosses the OTHER direction (`line_y_coord < y_coord_height`)
- Printing landmark index numbers on the image
- Drawing green circles at landmark positions

These represent earlier iterations of the annotation that were superseded.

---

## Method 5: `pose_media_pipe_google_2(self, video_source=None)`

**Lines:** ~195–476

### What it does
The **main pipeline method** and entry point for the OpenCV grid view. It:
1. Accepts any video source (file, RTSP, USB webcam, IP camera)
2. Reads frames in a loop
3. Every 5 seconds of video, runs pose detection
4. Displays a side-by-side grid: `[original frame | pose-annotated frame]`

### Input handling — source type detection

```
video_source
    ├── None        → default to gym_1_h264.mp4 in data_dir
    ├── int         → USB_WEBCAM (device index)
    └── str
        ├── "rtsp://"   → RTSP_STREAM
        ├── "http://"   → IP_CAMERA
        ├── "https://"  → IP_CAMERA
        ├── os.path.isfile() == True → LOCAL_FILE
        └── else        → ERROR + file search diagnostics
```

### Error handling for missing files
If a local file path is given but doesn't exist, the method performs **smart diagnostics**:
1. Searches the exact filename in alternate `data_dir` subdirectories
2. Recursively scans all of `data_dir` for any `.mp4/.avi/.mov/.mkv` files
3. Prints actionable suggestions (e.g., `mv` command to move files to the right location)

### Path resolution
```python
script_dir = os.path.dirname(os.path.abspath(__file__))   # .../src/analysis/
project_root = os.path.dirname(script_dir)                 # .../src/
git_up_root = os.path.dirname(project_root)                # .../overlander26/
```

So `git_up_root` = the project root (`overlander26/`), and all `data_dir` paths are absolute from there.

### Frame capture loop

```
cv2.VideoCapture(video_source)
    ↓
fps, frame_count = get video properties
    ↓
is_live_stream = (frame_count <= 0 OR source is RTSP/IP/USB)
    ↓
frame_interval = int(fps * 5)           ← skip N frames = 5 seconds of video
    ↓
LOOP:
    ret, frame = capture.read()
        ↓
    [if not ret AND live stream] → reconnect and retry
    [if not ret AND file] → break (end of file)
        ↓
    resized_frame_0 = cv2.resize(frame, (500, 650))   ← always resize raw frame
        ↓
    if current_frame % frame_interval == 0:            ← every 5 seconds
        resized_frame1 = cv2.resize(frame, (500, 650))
        cv2.imwrite(frame_pose_save_path, resized_frame1)  ← save to disk
            ↓
        pose_write_path = pose_media_pipe_google_0(frame_pose_save_path)
            ↓
        [if pose_write_path is valid]
        image_pose_saved_last = cv2.imread(pose_write_path)
        resized_pose_frame = cv2.resize(image_pose_saved_last, (500, 650))
            ↓
        top_row = np.hstack((resized_frame_0, resized_pose_frame))
        cv2.imshow('OVERLANDER__GRID_VIEW', top_row)    ← THE GRID VIEW
            ↓
        if cv2.waitKey(1) == 'q': break
        ↓
    current_frame += 1
```

### The OpenCV Grid View — `np.hstack`

This is the key visual output:

```
np.hstack((resized_frame_0, resized_pose_frame))

┌─────────────────────┬─────────────────────┐
│                     │                     │
│   RAW FRAME         │   POSE-ANNOTATED    │
│   (500 × 650)       │   (500 × 650)       │
│   No annotations    │   Green fence line  │
│                     │   Red box if alert  │
│                     │   "-FACE-" label    │
└─────────────────────┴─────────────────────┘
        Total: 1000 × 650 px
        Window: 'OVERLANDER__GRID_VIEW'
```

`np.hstack` horizontally concatenates two numpy arrays of the **same height**. Both frames are resized to `(500, 650)` before stacking, guaranteeing shape compatibility.

### Indirect pipeline: write-then-read pattern
Note the indirect flow: the raw frame is **written to disk** first, then `pose_media_pipe_google_0()` reads it from disk, annotates it, writes the annotated version, and returns the path. The main loop then reads the annotated image back from disk.

```
frame (numpy) → imwrite → [disk: detected_pose/frame_NNNN.jpg]
                               ↓
                    pose_media_pipe_google_0(path)
                               ↓
                    annotated image written to [disk: pose_id_not_ipcam/...]
                               ↓
                    pose_write_path (string) returned
                               ↓
                    cv2.imread(pose_write_path) → back to numpy
```

This disk round-trip is inefficient (vs. passing numpy arrays directly), but it decouples frame capture from pose inference and provides saved frames for inspection.

### Live stream reconnect logic
```python
if not ret and is_live_stream:
    time.sleep(2)
    capture.release()
    capture = cv2.VideoCapture(video_source)   ← full reconnect
    if not capture.isOpened():
        break  ← give up if still can't connect
    continue
```

### Quit key
```python
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```
`cv2.waitKey(1)` waits 1ms between frames (non-blocking for video). The `& 0xFF` mask handles cross-platform key code differences.

---

## Method 6: `pose_media_pipe_google_0(self, image_saved_ipcam)`

**Lines:** ~529–588

### What it does
**Single-frame pose detection.** Takes a path to a saved image, runs the full MediaPipe detection pipeline on it, annotates the image, and saves the annotated result to disk. Returns the path to the annotated image.

### How it works

```
image_saved_ipcam (str: path to image on disk)
    ↓
[guard] if "detected_pose" in path:
    image_name_pose_detect = filename extracted after "detected_pose/"
    ↓
[singleton guard] if self.detector is None:
    self.init_pose_detector()
    ↓
image = mp.Image.create_from_file(image_saved_ipcam)   ← MediaPipe Image object
    ↓
detection_result = self.detector.detect(image)          ← run inference
    ↓
annotated_image, flag = pose_draw_landmarks_on_image(
    image.numpy_view(), detection_result
)
    ↓
if flag == "YES_LANDMARKS":
    → save to: ../data_dir/pose_detected/pose_id_not_ipcam/<name>.png
    → return that path
else:
    → save to: ../data_dir/pose_detected/pose_not_ipcam/<name>.png
    → return that path
```

### Output directory routing based on detection flag

| Flag | Directory | Meaning |
|---|---|---|
| `YES_LANDMARKS` | `pose_id_not_ipcam/` | Pose was detected — annotated frame with landmarks |
| `NO_LANDMARKS` | `pose_not_ipcam/` | No pose detected — unannotated frame still saved |

This bifurcation lets you sort output frames by detection success just by looking at which directory they landed in.

### Filename construction
```python
name_to_write = image_name_pose_detect + "_frame_pose_" + second_now + "__" + str(frame_counter) + "__.png"
```
- `image_name_pose_detect`: the original input filename
- `second_now`: timestamp at seconds precision (e.g., `45`)
- `frame_counter`: always `0` (resets per call — not a persistent counter)

**Note:** `frame_counter` is always `0` in this method because it's a local variable reset to `0` on every call. The timestamp provides uniqueness.

---

## Method 7: `pose_media_pipe_google_1(self, image_saved_ipcam)`

**Lines:** ~592–626

### What it does
An **older, simpler variant** of `pose_media_pipe_google_0`. Performs the same single-frame detection but with fewer features.

### Differences from `_0`

| Feature | `_0` (current) | `_1` (older) |
|---|---|---|
| Output directory | Two dirs (`pose_id_not_ipcam` / `pose_not_ipcam`) | One dir (`pose_rect_only`) |
| Detection flag | Checks `flag` → routes to different dirs | No flag check |
| Filename extraction | Looks for `"detected_pose"` in path | Looks for `"frame_for_pose"` |
| Return value | Always returns a path | Always returns a path |

### Path extraction heuristic
```python
if "frame_for_pose" in str(image_saved_ipcam):
    image_name_pose_detect = str(str(image_saved_ipcam).rsplit("frame_for_pose/",1)[1])
```

This was written for an earlier directory structure where frames were saved to `frame_for_pose/`. The current pipeline uses `detected_pose/` (handled by `_0`).

**Note:** If the path doesn't contain `"frame_for_pose"`, `image_name_pose_detect` is never set, and the `name_to_write` line will raise a `NameError`. This method is effectively a legacy artifact.

---

## Data Flow Summary

```
                     ┌──────────────────────────────┐
                     │   pose_media_pipe_google_2()  │
                     │   (Main Pipeline / Grid View) │
                     └──────────────┬───────────────┘
                                    │
                      Every 5 seconds of video
                                    │
                     ┌──────────────▼───────────────┐
                     │   Save raw frame to disk      │
                     │   data_dir/detected_pose/     │
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │  pose_media_pipe_google_0()   │
                     │  (Single Frame Inference)     │
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │      init_pose_detector()     │
                     │      (Singleton guard)        │
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │  mp.Image.create_from_file()  │
                     │  detector.detect(image)       │
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │  pose_draw_landmarks_on_image │
                     │  (Annotation Engine)          │
                     │  ┌─────────────────────────┐ │
                     │  │ get_horizontal_line_pos  │ │
                     │  │ Virtual fence @ 70%      │ │
                     │  └─────────────────────────┘ │
                     │  ┌─────────────────────────┐ │
                     │  │ Landmark #5 (Left Eye)   │ │
                     │  │ if eye above fence:      │ │
                     │  │   RED box + "-FACE-"     │ │
                     │  └─────────────────────────┘ │
                     │  ┌─────────────────────────┐ │
                     │  │ draw_horizontal_line()   │ │
                     │  │ Green fence drawn        │ │
                     │  └─────────────────────────┘ │
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │  Save annotated frame to disk │
                     │  YES → pose_id_not_ipcam/    │
                     │  NO  → pose_not_ipcam/       │
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │  np.hstack(raw, annotated)    │
                     │  cv2.imshow('OVERLANDER__     │
                     │            GRID_VIEW')        │
                     │  [1000 × 650 side-by-side]    │
                     └──────────────────────────────┘
```

---

## Key Constants & Magic Numbers

| Value | Location | Meaning |
|---|---|---|
| `0.70` | `get_horizontal_line_position`, `draw_horizontal_line` | Fence line at 70% frame height |
| `5` | landmark filter | Only check landmark index #5 (MediaPipe Left Eye outer) |
| `(500, 650)` | `pose_media_pipe_google_2` | All frames resized to this before hstack |
| `5` (seconds) | `capture_interval` | How often to sample frames for pose detection |
| `(0,0,255)` | alert rectangle | RED in BGR — intrusion alert color |
| `(0,255,0)` | fence line | GREEN in BGR — virtual fence visualization |
| `55×35` | alert box size | Bounding box drawn around the eye landmark |
| `"OVERLANDER__GRID_VIEW"` | window title | The OpenCV display window name |

---

## API Migration Notes (Tasks API 0.10+)

The file contains important comments about the migration away from the older MediaPipe Solutions API:

```python
# MediaPipe 0.10+ Tasks API - No protobuf or solutions needed
# Removed: from mediapipe.framework.formats import landmark_pb2
# Removed: from mediapipe import solutions
```

**Old API (removed):**
```python
# landmark_pb2.NormalizedLandmarkList() — protobuf conversion required
# mp_pose.Pose() — Solutions API
# mp_drawing.draw_landmarks() — Solutions drawing utilities
```

**New API (current):**
```python
mp.Image.create_from_file()          # MediaPipe native image
vision.PoseLandmarker               # Tasks API detector class
detection_result.pose_landmarks     # Direct list access, no protobuf
landmark.x, landmark.y, landmark.z  # Direct float attributes
```

The commented-out code block at the bottom (~lines 630–716) preserves the entire old Solutions API approach for reference, including both static image and webcam modes.
