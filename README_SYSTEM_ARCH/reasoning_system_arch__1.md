# reasoning_system_arch__1.md

## TASK-1 — Identify Central Object: HUMAN / DOG / CAT / OTHER ANIMAL

**Goal:** Given an image, determine whether the most prominent / central object is a:
- HUMAN (person)
- DOG
- CAT
- OTHER ANIMAL (bird, horse, cow, bear, elephant, etc.)

---

## SECTION 1 — What the Existing Code Does (and Its Limits)

The current `media_pipe.py` uses **MediaPipe Tasks API 0.10+ — Pose Landmarker**.

- It detects **human body pose landmarks** (33 keypoints per person).
- If `pose_landmarks` list is empty → returns `NO_LANDMARKS`.
- It tells us "a human was found" or "no human was found".
- It does **NOT** tell us what the object IS when it's not a human.
- A dog, cat, or other animal will simply return `NO_LANDMARKS` — no distinction between species.

**Gap:** The current system is binary: human vs. nothing.
**Need:** A multi-class classifier/detector: HUMAN / DOG / CAT / OTHER ANIMAL.

---

## SECTION 2 — Option A: MediaPipe Object Detector (Preferred for MediaPipe Path)

### What it is
MediaPipe Tasks API 0.10+ includes an **ObjectDetector** task (separate from PoseLandmarker).
It uses **EfficientDet-Lite** models trained on COCO 80 classes.

### Relevant COCO Classes
| Class ID | Label    |
|----------|----------|
| 0        | person   |
| 15       | cat      |
| 16       | dog      |
| 17       | horse    |
| 18       | sheep    |
| 19       | cow      |
| 20       | elephant |
| 21       | bear     |
| 22       | zebra    |
| 23       | giraffe  |
| 14       | bird     |

All needed categories (HUMAN, DOG, CAT, OTHER ANIMAL) map directly to COCO classes.

### Model Files (download once)
```
efficientdet_lite0.tflite   (~5MB,  fastest, least accurate)
efficientdet_lite2.tflite   (~9MB,  balanced)
efficientdet_lite4.tflite   (~19MB, most accurate, slower)
```
Download URL pattern:
```
https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
```

### API Usage Pattern (Tasks API 0.10+)
```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    score_threshold=0.5,
    max_results=5
)
detector = vision.ObjectDetector.create_from_options(options)

image = mp.Image.create_from_file("input.jpg")
detection_result = detector.detect(image)
# detection_result.detections → list of Detection objects
# Each Detection: .bounding_box, .categories[0].category_name, .categories[0].score
```

### Pros
- Fully consistent with existing codebase pattern (singleton, Tasks API 0.10+)
- Lightweight model (~5MB)
- Fast inference: 30+ FPS on CPU
- No new pip dependencies — `mediapipe` already installed
- Detects all required categories in one pass

### Cons
- Requires downloading a new `.tflite` model file
- EfficientDet-Lite accuracy is lower than transformer-based models
- Fixed to COCO 80 classes (no open vocabulary)

### Verdict: **RECOMMENDED if staying within MediaPipe**

---

## SECTION 3 — Option B: RT-DETR v2 (HuggingFace — Already in Codebase)

### What it is
Already used in `detr_hugging_face.py` (`ObjDetHFRtDetr` class).
Checkpoint: `PekingU/rtdetr_v2_r50vd`

### How it maps to our task
RT-DETR outputs COCO class labels. Same class mapping as above.
`person`, `cat`, `dog`, + all other animal classes are covered.

### Central Object Logic
Post-process the detections: pick the box whose center is closest to image center, or the box with the largest area. Check its label.

### Pros
- **Zero new code dependencies** — already imported in project
- State-of-the-art accuracy (outperforms DETR, close to YOLO)
- HuggingFace hosted, Hugging Face `pipeline()` abstraction available

### Cons
- Heavy model (~100MB+ weights)
- Requires GPU for real-time performance; slow on CPU
- Model is **pre-initialized at import time** (global in `detr_hugging_face.py`) — adds startup cost

### Verdict: **Good secondary option — reuses existing code, high accuracy but heavier**

---

## SECTION 4 — Option C: DETR ResNet-101 (HuggingFace — Already in Codebase)

Already initialized globally in `detr_hugging_face.py`:
```python
detr_image_processor_detr_resnet101 = ...  # facebook/detr-resnet-101
model_detr_resnet101 = ...
```

Same COCO-class coverage. Slightly lower accuracy than RT-DETR v2.
Same pros/cons as Option B, slightly more weight.

### Verdict: **Available but superseded by RT-DETR v2 already in codebase**

---

## SECTION 5 — Option D: YOLOv8 via HuggingFace / Ultralytics

### What it is
`ultralytics` is already listed in `reqmts.log`. YOLOv8 nano/small models are ~6MB.

### Usage
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")   # auto-downloads
results = model("input.jpg")
for box in results[0].boxes:
    print(box.cls, box.conf, box.xyxy)
```

### Pros
- Extremely fast (real-time on CPU with nano model)
- COCO 80 classes — same coverage
- `ultralytics` dependency already in `reqmts.log`
- HuggingFace hosted: `Ultralytics/assets`

### Cons
- Different API pattern from the rest of the codebase
- FaceDetection class already uses YOLOv8 for faces — architecture is already hybrid

### Verdict: **Strong option if speed is prioritized over architecture consistency**

---

## SECTION 6 — Option E: Grounding DINO (HuggingFace — Open Vocabulary)

### What it is
`IDEA-Research/grounding-dino-tiny` (HuggingFace)
Open-vocabulary detection — query with **text prompts** like `"human . dog . cat . animal"`.

### Why it's interesting
- No fixed class list — detects whatever text you describe
- Future-proof: add new categories without retraining
- Example:
```python
from transformers import pipeline
pipe = pipeline("zero-shot-object-detection", model="IDEA-Research/grounding-dino-tiny")
results = pipe("input.jpg", candidate_labels=["human", "dog", "cat", "animal"])
```

### Pros
- Maximum flexibility — add any category via text
- HuggingFace hosted
- State-of-the-art open-vocabulary detection

### Cons
- Heavier model than EfficientDet
- Requires `transformers >= 4.38` (GroundingDINO support added late)
- Slower than YOLO/EfficientDet on CPU

### Verdict: **Best for future extensibility, overkill for fixed 4-class task**

---

## SECTION 7 — "Central Object" Detection Strategy

Regardless of which model is chosen, the logic to identify the **central** object is:

### Strategy 1: Largest Bounding Box Area (Recommended)
The object that occupies the most image area is likely the "main subject".
```
area = (box.width * box.height)
central_object = max(detections, key=lambda d: area(d.bounding_box))
```

### Strategy 2: Closest Center to Image Center
```
image_center = (image_width / 2, image_height / 2)
dist = euclidean_distance(box_center, image_center)
central_object = min(detections, key=lambda d: dist(d.bounding_box))
```

### Strategy 3: Combination Score (Most Robust)
```
score = (area_weight * normalized_area) + (center_weight * (1 - normalized_distance))
central_object = max(detections, key=lambda d: score(d))
```

### Label Mapping Post-Detection
```python
HUMAN_LABELS   = {"person"}
DOG_LABELS     = {"dog"}
CAT_LABELS     = {"cat"}
ANIMAL_LABELS  = {"horse", "sheep", "cow", "elephant", "bear", "zebra",
                  "giraffe", "bird", "cat", "dog"}  # superset

def classify_detection(label):
    if label in HUMAN_LABELS:   return "HUMAN"
    if label in DOG_LABELS:     return "DOG"
    if label in CAT_LABELS:     return "CAT"
    if label in ANIMAL_LABELS:  return "ANIMAL"
    return "OTHER"
```

---

## SECTION 8 — Recommendation Summary

| Option | Model | Speed (CPU) | Accuracy | New Deps | Consistency |
|--------|-------|-------------|----------|----------|-------------|
| **A — MediaPipe ObjectDetector** | EfficientDet-Lite0 | ★★★★★ | ★★★ | None (only .tflite file) | Highest |
| B — RT-DETR v2 | rtdetr_v2_r50vd | ★★ | ★★★★★ | None (already used) | High |
| C — DETR ResNet-101 | detr-resnet-101 | ★★ | ★★★★ | None (already used) | High |
| D — YOLOv8 | yolov8n.pt | ★★★★★ | ★★★★ | None (ultralytics in reqmts.log) | Medium |
| E — Grounding DINO | grounding-dino-tiny | ★★★ | ★★★★★ | New: transformers >= 4.38 | Low |

### **Primary Recommendation: Option A — MediaPipe ObjectDetector**
- Stays entirely within the MediaPipe Tasks API ecosystem already used
- Singleton pattern reused (same as existing `init_pose_detector`)
- Fastest CPU inference — matches the 25-30 FPS goal
- Only requires downloading one `.tflite` file (~5MB)
- No code architecture changes to `detr_hugging_face.py`

### **Fallback Recommendation: Option B — RT-DETR v2**
- If higher accuracy is required and GPU is available
- Reuses `ObjDetHFRtDetr` class already written in `detr_hugging_face.py`
- Minimal new code needed

---

## SECTION 9 — Proposed New File Location

New code should go in:
```
src/analysis/object_classifier.py
```

New class: `CentralObjectClassifier`
Methods:
- `init_detector()` — singleton initialization (MediaPipe ObjectDetector)
- `classify_central_object(image_path)` → returns `"HUMAN"` / `"DOG"` / `"CAT"` / `"ANIMAL"` / `"UNKNOWN"`
- `classify_central_object_from_frame(cv2_frame)` → same but from numpy array

Integration point: called from `IPWebCam` in `src/main.py`, same pattern as `pose_media_pipe_google()`.

---

## SECTION 10 — Next Steps (Pending Approval)

1. Download `efficientdet_lite0.tflite` to `data_dir/pose_models/`
2. Create `src/analysis/object_classifier.py` with `CentralObjectClassifier` class
3. Add entry method `classify_central_object_in_video()` to `IPWebCam` in `src/main.py`
4. Update `CUSTOM_UTILS_SKILLS.md` with the new pipeline pattern

**No code changes will be made until this reasoning is reviewed and approved.**
