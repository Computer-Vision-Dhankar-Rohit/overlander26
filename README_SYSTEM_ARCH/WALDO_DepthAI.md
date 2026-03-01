# DepthAI YOLO Object Detection - Comprehensive Code Analysis

# ORIGINAL AUTHOR 
- https://huggingface.co/StephanST/WALDO30
# ORIGINAL LICENSE 
- https://huggingface.co/StephanST/WALDO30


## Document Information

- **Code Origin:** Adapted from Luxonis DepthAI Documentation (tiny_yolo sample)
- https://huggingface.co/StephanST/WALDO30

- **Purpose:** Real-time object detection using YOLO on OAK-D cameras with custom model configuration

---

## Executive Summary

This Python script implements a **real-time object detection pipeline** using:
- **Hardware:** OAK-D (OpenCV AI Kit with Depth) camera from Luxonis
- **Framework:** DepthAI SDK for on-device neural network inference
- **Model:** Custom YOLO (You Only Look Once) model in OpenVINO blob format
- **Configuration:** JSON-based model metadata and parameter loading

**Key Innovation:** Dynamic configuration loading from JSON files instead of hardcoded parameters, enabling easy model swapping without code changes.

---

## 1. Import Dependencies & Libraries

### Standard Library Imports
```python
from pathlib import Path
import sys
import cv2
import numpy as np
import time
import argparse
import json
```

**Purpose & Usage:**
- `pathlib.Path`: Modern path handling (cross-platform, object-oriented)
- `sys`: System-specific parameters (not actively used but imported)
- `cv2`: OpenCV for image processing and display
- `numpy (np)`: Numerical operations for bounding box normalization
- `time`: Performance monitoring (FPS calculation)
- `argparse`: Command-line argument parsing for flexible execution
- `json`: Parse model configuration files

### Specialized Libraries
```python
import depthai as dai
import blobconverter
```

**Purpose & Usage:**
- `depthai (dai)`: Luxonis DepthAI SDK for OAK camera communication and pipeline creation
- `blobconverter`: Automatic model conversion/download from DepthAI model zoo

---

## 2. Command-Line Argument Parsing

### Argument Parser Configuration

```python
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", 
                    help="Provide model name or model path for inference",
                    default='./model/best_openvino_2022.1_6shave.blob', 
                    type=str)
parser.add_argument("-c", "--config", 
                    help="Provide config path for inference",
                    default='./model/best.json', 
                    type=str)
args = parser.parse_args()
```

### Detailed Parameter Analysis

#### Parameter 1: `--model` / `-m`
- **Type:** String (file path or model name)
- **Default Value:** `./model/best_openvino_2022.1_6shave.blob`
- **Purpose:** Specify neural network model in OpenVINO IR blob format
- **Format Details:**
  - `.blob` extension indicates OpenVINO compiled model
  - `6shave` indicates 6 SHAVE cores allocated for inference
  - SHAVE (Streaming Hybrid Architecture Vector Engine) = specialized neural network accelerator in Myriad X VPU
- **Usage Examples:**
  ```bash
  # Local model file
  python script.py -m ./my_custom_model.blob
  
  # Model zoo name (auto-downloaded)
  python script.py -m yolov5n_coco_416x416
  ```

#### Parameter 2: `--config` / `-c`
- **Type:** String (JSON file path)
- **Default Value:** `./model/best.json`
- **Purpose:** Load model metadata and inference parameters
- **Configuration Contents:**
  - Input dimensions (width x height)
  - Class names/labels
  - Anchor boxes for YOLO detection
  - Confidence/IoU thresholds
  - Output coordinate system specifications

### Error Handling & Validation
```python
configPath = Path(args.config)
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))
```

**Validation Logic:**
1. Convert string path to `Path` object
2. Check file existence using `.exists()` method
3. Raise descriptive `ValueError` if config missing
4. **Critical:** Script exits immediately if config not found (fail-fast pattern)

---

## 3. Configuration File Parsing & Metadata Extraction

### JSON Configuration Loading

```python
with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})
```

**Processing Steps:**
1. Open JSON file using context manager (`with` statement)
2. Parse entire JSON into Python dictionary
3. Extract `nn_config` section (neural network specific settings)
4. Use `.get()` with empty dict `{}` as default to prevent KeyError

### Input Dimension Extraction

```python
if "input_size" in nnConfig:
    W, H = tuple(map(int, nnConfig.get("input_size").split('x')))
```

**Detailed Processing Flow:**
1. **Check:** Verify `input_size` key exists in config
2. **Extract:** Get string value (e.g., "416x416" or "640x640")
3. **Split:** Separate width and height using 'x' delimiter
4. **Convert:** Map string values to integers
5. **Unpack:** Assign to W (width) and H (height) variables

**Example Transformation:**
```
Input: "416x416" (string)
Split: ["416", "416"] (list of strings)
Map:   [416, 416] (list of integers)
Unpack: W=416, H=416 (separate variables)
```

### YOLO-Specific Metadata Extraction

```python
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})
```

#### Metadata Parameters Explained

| Parameter | Data Type | Purpose | Example Value |
|-----------|-----------|---------|---------------|
| `classes` | Integer | Number of object classes model can detect | 80 (COCO dataset) |
| `coordinates` | Integer | Bounding box coordinate format (usually 4) | 4 (x, y, w, h) |
| `anchors` | List[Float] | Pre-defined anchor boxes for detection | [10,13, 16,30, 33,23, ...] |
| `anchorMasks` | Dict | Maps detection layers to anchor indices | {"side80": [0,1,2], "side40": [3,4,5]} |
| `iouThreshold` | Float | IoU threshold for NMS (Non-Max Suppression) | 0.45 |
| `confidenceThreshold` | Float | Minimum confidence for detection validity | 0.25 |

**YOLO Anchor Boxes Concept:**
- Pre-computed aspect ratios/sizes based on training dataset
- Help model predict bounding boxes more accurately
- Different scales for different detection layers (multi-scale detection)
- Example: Small anchors for small objects, large anchors for large objects

### Label Mapping Extraction

```python
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})
```

**Purpose:** Map class indices (0, 1, 2...) to human-readable labels
**Example Mapping:**
```json
{
  "0": "person",
  "1": "bicycle",
  "2": "car",
  "79": "toothbrush"
}
```

---

## 4. Model Loading & Validation

### Model Path Resolution

```python
nnPath = args.model
if not Path(nnPath).exists():
    print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
    nnPath = str(blobconverter.from_zoo(args.model, shaves=6, zoo_type="depthai", use_cache=True))
```

**Intelligent Model Loading Logic:**

1. **Primary Path:** Try to load model from specified file path
2. **Fallback Mechanism:** If file not found, attempt model zoo download
3. **Auto-Download:** `blobconverter.from_zoo()` handles:
   - Model download from cloud repository
   - On-the-fly OpenVINO conversion
   - Local caching (subsequent runs use cached version)
4. **Configuration:**
   - `shaves=6`: Allocate 6 SHAVE cores (balance speed/power)
   - `zoo_type="depthai"`: Use DepthAI-specific model zoo
   - `use_cache=True`: Enable local caching for faster subsequent loads

**Performance Impact:**
- First run: ~30-60 seconds (download + conversion)
- Subsequent runs: <1 second (cached model loaded from disk)

---

## 5. DepthAI Pipeline Construction

### Pipeline Architecture Overview

```python
syncNN = True  # Synchronize neural network outputs with camera frames
pipeline = dai.Pipeline()
```

**Pipeline Concept:**
- Directed graph of processing nodes
- Data flows through nodes via links
- Executed entirely on OAK camera's VPU (not host CPU)
- Enables low-latency, high-throughput processing

### Node Creation

```python
# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)
```

#### Node Type Descriptions

**1. ColorCamera Node (`camRgb`)**
- **Purpose:** Capture RGB video from camera sensor
- **Hardware:** Interfaces with OAK camera's color sensor (IMX378 or similar)
- **Capabilities:** 
  - Resolution up to 4K (12MP)
  - Frame rate control
  - Auto-focus, auto-exposure
  - ISP (Image Signal Processing)

**2. YoloDetectionNetwork Node (`detectionNetwork`)**
- **Purpose:** Run YOLO inference on VPU (Vision Processing Unit)
- **Hardware Acceleration:** Executes on Myriad X 16 SHAVE cores
- **Processing:** 
  - Input: Preprocessed image (resized, normalized)
  - Output: Bounding boxes, class labels, confidence scores
  - Post-processing: NMS (Non-Maximum Suppression) on-device

**3. XLinkOut Nodes (`xoutRgb`, `nnOut`)**
- **Purpose:** Stream data from camera to host computer
- **Protocol:** XLink (Luxonis proprietary high-speed USB protocol)
- **Bandwidth:** Up to 5 Gbps (USB 3.0)
- **Use Cases:**
  - `xoutRgb`: Send camera frames for display
  - `nnOut`: Send detection results to host

### Stream Name Assignment

```python
xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")
```

**Purpose:** Named channels for accessing data on host side
**Usage:** `device.getOutputQueue(name="rgb")` retrieves this stream

---

## 6. Camera Configuration

### Preview Size Configuration

```python
camRgb.setPreviewSize(W, H)
```

**Purpose:** Set neural network input dimensions
**Details:**
- Matches model's expected input size (e.g., 416x416)
- Camera automatically resizes/crops to these dimensions
- Lower resolution = faster inference, less accuracy
- Higher resolution = slower inference, better accuracy

**Common YOLO Sizes:**
- 320x320: Ultra-fast, mobile devices
- 416x416: Balanced speed/accuracy
- 640x640: High accuracy, slower

### Sensor Resolution

```python
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
```

**Available Options:**
- `THE_1080_P`: 1920x1080 (Full HD) - **Selected**
- `THE_4_K`: 3840x2160 (4K UHD)
- `THE_12_MP`: 4032x3040 (12 Megapixel)
- `THE_13_MP`: 4208x3120 (13 Megapixel)

**Why 1080P?**
- Good balance between field-of-view and processing speed
- Lower USB bandwidth requirements
- Sufficient quality for most detection tasks

### Image Format Settings

```python
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
```

**Interleaved Setting:**
- `False`: Planar format (RRRRR...GGGGG...BBBBB)
- `True`: Interleaved format (RGBRGBRGB...)
- **Choice Rationale:** Planar format preferred for neural networks

**Color Order:**
- `BGR`: OpenCV standard (Blue-Green-Red)
- `RGB`: Alternative format
- **Reason for BGR:** Direct compatibility with OpenCV functions

### Frame Rate Control

```python
camRgb.setFps(15)
```

**FPS (Frames Per Second) Configuration:**
- **Set to 15 FPS:** Conservative setting for stable inference
- **Trade-off:**
  - Lower FPS: More processing time per frame, higher accuracy
  - Higher FPS: Smoother video, potential dropped frames if inference slow
- **Typical Range:** 5-30 FPS for object detection

**Optional Manual Focus (Commented Out):**
```python
#camRgb.initialControl.setManualFocus(70)
```
- Range: 0-255 (0=infinite focus, 255=closest focus)
- Use case: Fixed-distance object detection

---

## 7. YOLO Detection Network Configuration

### Detection Parameters

```python
detectionNetwork.setConfidenceThreshold(confidenceThreshold)
detectionNetwork.setNumClasses(classes)
detectionNetwork.setCoordinateSize(coordinates)
detectionNetwork.setAnchors(anchors)
detectionNetwork.setAnchorMasks(anchorMasks)
detectionNetwork.setIouThreshold(iouThreshold)
```

#### Parameter-by-Parameter Breakdown

**1. Confidence Threshold**
```python
detectionNetwork.setConfidenceThreshold(confidenceThreshold)
```
- **Purpose:** Filter out low-confidence detections
- **Range:** 0.0 to 1.0
- **Typical Value:** 0.25-0.5
- **Effect:** Higher = fewer false positives, more false negatives

**2. Number of Classes**
```python
detectionNetwork.setNumClasses(classes)
```
- **Purpose:** Tell network how many object types to detect
- **Examples:**
  - 80 classes: COCO dataset (person, car, dog, etc.)
  - 1 class: Custom single-object detector (e.g., face only)
  - 20 classes: VOC dataset

**3. Coordinate Size**
```python
detectionNetwork.setCoordinateSize(coordinates)
```
- **Purpose:** Bounding box parameter count
- **Standard Value:** 4 (x, y, width, height)
- **Alternative:** 5 (x, y, w, h, rotation angle)

**4. Anchors**
```python
detectionNetwork.setAnchors(anchors)
```
- **Purpose:** Pre-computed bounding box templates
- **Format:** Flat list [w1,h1, w2,h2, w3,h3, ...]
- **Example:** [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
- **Count:** Typically 9 anchors (3 per detection layer)

**5. Anchor Masks**
```python
detectionNetwork.setAnchorMasks(anchorMasks)
```
- **Purpose:** Map detection layers to anchor indices
- **Multi-Scale Detection:**
  - Large grid (e.g., 13x13): Detects large objects using large anchors
  - Medium grid (e.g., 26x26): Detects medium objects
  - Small grid (e.g., 52x52): Detects small objects using small anchors
- **Example Mapping:**
  ```python
  {
    "side80": [0, 1, 2],    # Smallest objects
    "side40": [3, 4, 5],    # Medium objects
    "side20": [6, 7, 8]     # Largest objects
  }
  ```

**6. IoU Threshold**
```python
detectionNetwork.setIouThreshold(iouThreshold)
```
- **Purpose:** Non-Maximum Suppression (NMS) threshold
- **Range:** 0.0 to 1.0
- **Typical Value:** 0.4-0.5
- **Function:** Suppress overlapping duplicate detections
- **Algorithm:**
  1. Sort detections by confidence
  2. Keep highest confidence detection
  3. Remove all detections with IoU > threshold with kept detection
  4. Repeat for next highest confidence

**IoU (Intersection over Union) Formula:**
```
IoU = Area of Overlap / Area of Union
```

### Model Blob Path

```python
detectionNetwork.setBlobPath(nnPath)
```
- **Purpose:** Point network node to model file
- **Format:** OpenVINO IR blob (.blob extension)
- **Loading:** Model copied to VPU memory on pipeline start

### Inference Threading

```python
detectionNetwork.setNumInferenceThreads(2)
```
- **Purpose:** Parallel inference execution
- **Range:** 1-2 threads (hardware limitation)
- **Effect:** 2 threads can process overlapping frames for higher throughput

### Input Queue Behavior

```python
detectionNetwork.input.setBlocking(False)
```
- **Blocking=False:** Drop frames if inference queue full
- **Blocking=True:** Wait for queue space (may cause lag)
- **Rationale:** Non-blocking prevents pipeline stalls

---

## 8. Pipeline Linking (Data Flow)

### Link Configuration

```python
camRgb.preview.link(detectionNetwork.input)
detectionNetwork.passthrough.link(xoutRgb.input)
detectionNetwork.out.link(nnOut.input)
```

### Visual Data Flow Diagram

```
┌─────────────┐
│ ColorCamera │
│   (camRgb)  │
└──────┬──────┘
       │ preview (resized frames)
       ▼
┌──────────────────┐
│YoloDetectionNet  │
│(detectionNetwork)│
└─────┬─────┬──────┘
      │     │
      │     └─ passthrough (original frames)
      │            │
      │            ▼
      │     ┌──────────────┐
      │     │  XLinkOut    │──► Host: qRgb (display)
      │     │  (xoutRgb)   │
      │     └──────────────┘
      │
      └─ out (detection results)
             │
             ▼
      ┌──────────────┐
      │  XLinkOut    │──► Host: qDet (bounding boxes)
      │  (nnOut)     │
      └──────────────┘
```

### Link Explanations

**Link 1: Camera → Detection Network**
```python
camRgb.preview.link(detectionNetwork.input)
```
- **Source:** Camera preview output (resized to W×H)
- **Destination:** Neural network input
- **Data Type:** Image frame (BGR format)
- **Processing:** On-device (no USB transfer)

**Link 2: Detection Network → RGB Output**
```python
detectionNetwork.passthrough.link(xoutRgb.input)
```
- **Purpose:** Send original frames to host for display
- **Passthrough:** Network node forwards input frame unchanged
- **Use Case:** Overlay bounding boxes on original frame

**Link 3: Detection Network → NN Output**
```python
detectionNetwork.out.link(nnOut.input)
```
- **Purpose:** Send detection results to host
- **Data Type:** List of Detection objects (bbox, class, confidence)
- **Format:** Normalized coordinates (0.0-1.0)

---

## 9. Device Connection & Pipeline Execution

### Device Context Manager

```python
with dai.Device(pipeline) as device:
```

**Context Manager Benefits:**
1. **Auto-cleanup:** Device closed automatically on exit
2. **Exception safety:** Resources released even if error occurs
3. **USB connection:** Opens USB communication channel
4. **Pipeline upload:** Transfers pipeline graph to camera VPU
5. **Pipeline start:** Begins execution on device

**Device Discovery:**
- Automatically finds connected OAK camera
- Raises error if no device found
- Supports multiple devices (requires device ID specification)

### Output Queue Creation

```python
qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
```

#### Queue Parameters

**name** (str)
- Matches `setStreamName()` from XLinkOut nodes
- "rgb": Camera frames queue
- "nn": Neural network results queue

**maxSize** (int)
- **Value:** 4
- **Purpose:** Buffer size for incoming data
- **Behavior:** If queue full and blocking=False, oldest data dropped
- **Trade-off:** 
  - Larger: More memory, less data loss
  - Smaller: Less latency, potential drops

**blocking** (bool)
- **Value:** False
- **Effect:** `.get()` returns immediately (None if empty)
- **Alternative:** True = wait until data available

### State Variables

```python
frame = None
detections = []
startTime = time.monotonic()
counter = 0
color2 = (255, 255, 255)  # White color for text
```

**Purpose of Each Variable:**
- `frame`: Latest camera frame (numpy array)
- `detections`: List of Detection objects from NN
- `startTime`: Reference time for FPS calculation
- `counter`: Count of processed NN outputs (for FPS)
- `color2`: RGB color tuple for UI text

---

## 10. Utility Functions

### Function 1: frameNorm (Coordinate Normalization)

```python
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
```

#### Detailed Step-by-Step Breakdown

**Input Parameters:**
- `frame` (numpy.ndarray): Image frame with shape (height, width, channels)
- `bbox` (tuple): Normalized bounding box coordinates (xmin, ymin, xmax, ymax) in range [0.0, 1.0]

**Processing Steps:**

**Step 1: Create Base Normalization Array**
```python
normVals = np.full(len(bbox), frame.shape[0])
# Result: [height, height, height, height]
```
- `len(bbox)`: 4 (for xmin, ymin, xmax, ymax)
- `frame.shape[0]`: Frame height
- Creates array filled with height value

**Step 2: Set Width Values for X Coordinates**
```python
normVals[::2] = frame.shape[1]
# Result: [width, height, width, height]
```
- `[::2]`: Slice with step 2 (indices 0, 2)
- `frame.shape[1]`: Frame width
- X coordinates (indices 0, 2) multiplied by width
- Y coordinates (indices 1, 3) multiplied by height

**Step 3: Convert to Pixel Coordinates**
```python
return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
```
- `np.array(bbox)`: Convert tuple to numpy array
- `np.clip(..., 0, 1)`: Clamp values to [0, 1] range (safety check)
- `* normVals`: Element-wise multiplication
- `.astype(int)`: Convert floats to integers

**Example Transformation:**
```
Input:  bbox = (0.25, 0.30, 0.75, 0.80)
Frame:  1920×1080 (width × height)
normVals = [1920, 1080, 1920, 1080]

Calculation:
[0.25, 0.30, 0.75, 0.80] * [1920, 1080, 1920, 1080]
= [480, 324, 1440, 864]

Output: [480, 324, 1440, 864] (pixel coordinates)
```

---

### Function 2: displayFrame (Visualization)

```python
def displayFrame(name, frame, detections):
    color = (255, 0, 255)  # Magenta for bounding boxes
    for detection in detections:
        bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        
        # Draw confidence text
        cv2.putText(frame, f"{int(detection.confidence * 100)}%", 
                    (bbox[0] + 10, bbox[1] + 20), 
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        
        # Draw bounding box rectangle
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)
    
    # Show the frame
    cv2.imshow(name, frame)
    cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
```

#### Function Component Analysis

**Loop Through Detections:**
```python
for detection in detections:
```
- `detection` object attributes:
  - `.xmin, .ymin, .xmax, .ymax`: Normalized bbox (0-1)
  - `.confidence`: Detection confidence (0-1)
  - `.label`: Class index (0 to num_classes-1)

**Convert Coordinates:**
```python
bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
```
- Transforms normalized [0,1] → pixel coordinates

**Draw Confidence Text:**
```python
cv2.putText(frame, f"{int(detection.confidence * 100)}%", 
            (bbox[0] + 10, bbox[1] + 20),  # Position: top-left corner offset
            cv2.FONT_HERSHEY_TRIPLEX,       # Font family
            0.5,                             # Font scale
            255)                             # Color (white)
```

**Text Positioning:**
- `bbox[0] + 10`: 10 pixels right of left edge
- `bbox[1] + 20`: 20 pixels below top edge
- Ensures text stays inside bounding box

**Draw Bounding Box:**
```python
cv2.rectangle(frame, 
              (bbox[0], bbox[1]),    # Top-left corner
              (bbox[2], bbox[3]),    # Bottom-right corner
              color,                  # Magenta (255, 0, 255)
              1)                      # Line thickness (1 pixel)
```

**Display Window:**
```python
cv2.imshow(name, frame)
cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
```
- `imshow`: Create/update OpenCV window
- `setWindowProperty`: Set window to always-on-top
- Ensures detection window stays visible

---

## 11. Main Processing Loop

### Loop Structure

```python
while True:
    inRgb = qRgb.get()
    inDet = qDet.get()
    
    # Frame processing...
    # Detection processing...
    # Display...
    
    if cv2.waitKey(1) == ord('q'):
        break
```

### Detailed Loop Analysis

#### Queue Data Retrieval

**Get Camera Frame:**
```python
inRgb = qRgb.get()
```
- **Blocking:** False (returns immediately)
- **Return Value:** 
  - `ImgFrame` object if data available
  - `None` if queue empty
- **Rate:** ~15 FPS (set by camera configuration)

**Get Detection Results:**
```python
inDet = qDet.get()
```
- **Return Value:** 
  - `ImgDetections` object if inference complete
  - `None` if no new detections
- **Rate:** Depends on inference time (typically 30-60ms per frame)

#### Frame Processing Block

```python
if inRgb is not None:
    frame = inRgb.getCvFrame()
    cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                (2, frame.shape[0] - 4), 
                cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)
```

**Frame Extraction:**
- `getCvFrame()`: Convert to numpy array (OpenCV format)
- Format: BGR, uint8, shape (H, W, 3)

**FPS Calculation:**
```python
counter / (time.monotonic() - startTime)
```
- `counter`: Number of NN inferences completed
- `time.monotonic()`: Current time (monotonic clock)
- `startTime`: Loop start time
- **Formula:** FPS = Total Inferences / Elapsed Time

**FPS Text Overlay:**
- **Position:** Bottom-left corner `(2, frame.shape[0] - 4)`
- **Format:** "NN fps: 14.32" (2 decimal places)
- **Purpose:** Real-time performance monitoring

#### Detection Processing Block

```python
if inDet is not None:
    detections = inDet.detections
    counter += 1
```

**Detection Update:**
- `inDet.detections`: List of Detection objects
- Each Detection contains: bbox, confidence, label
- **Counter Increment:** Tracks total inferences for FPS

#### Display & Visualization

```python
if frame is not None:
    displayFrame("rgb", frame, detections)
```

**Display Conditions:**
- Only display when frame available
- Uses latest detections (may be from previous frame)
- Handles asynchronous frame/detection timing

#### Exit Condition

```python
if cv2.waitKey(1) == ord('q'):
    break
```

**Keyboard Polling:**
- `cv2.waitKey(1)`: Wait 1ms for key press
- Returns ASCII code of pressed key
- `ord('q')`: ASCII code for 'q' key (113)
- **Effect:** Press 'Q' to exit loop

---

## 12. Performance Characteristics

### Inference Speed Analysis

**Typical Performance Metrics:**
- **Model:** YOLOv5n (nano) 416×416
- **Hardware:** OAK-D (Myriad X VPU)
- **Inference Time:** 30-40ms per frame
- **Throughput:** ~25-30 FPS
- **Latency:** 50-60ms (camera + inference + display)

**Performance Factors:**
1. **Model Size:**
   - Nano: ~30ms
   - Small: ~50ms
   - Medium: ~100ms
   - Large: ~200ms

2. **Input Resolution:**
   - 320×320: Fastest
   - 416×416: Balanced
   - 640×640: Slowest but most accurate

3. **SHAVE Allocation:**
   - 4 SHAVEs: Slower
   - 6 SHAVEs: Optimal (used in this code)
   - 8 SHAVEs: Minimal improvement

### PROBABLE - Resource Utilization

**VPU Memory:**
- Model blob: ~10-20MB
- Input buffer: ~1MB
- Output buffer: ~100KB
- Total: ~12-22MB

**USB Bandwidth:**
- RGB stream (1080p @ 15fps): ~45 MB/s
- Detection data: ~10 KB/s
- Total: ~45 MB/s (< 1% of USB 3.0 bandwidth)

**Host CPU Usage:**
- OpenCV display: ~2-5%
- Queue management: <1%
- Total: <10% of one core

---

## 13. Error Handling & Edge Cases

### Potential Issues & Solutions

**Issue 1: Model File Not Found**
```python
if not Path(nnPath).exists():
    print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
```
- **Solution:** Auto-download from model zoo
- **Fallback:** Use cached version if available

**Issue 2: Config File Missing**
```python
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))
```
- **Solution:** Fail fast with descriptive error
- **User Action:** Check config file path

**Issue 3: No Camera Connected**
- **Error:** `RuntimeError: Failed to find device`
- **Solution:** Check USB connection, try different port

**Issue 4: Queue Overflow**
- **Symptom:** Dropped frames
- **Cause:** Host processing slower than camera
- **Solution:** Non-blocking queues drop old data

**Issue 5: Low FPS**
- **Causes:**
  - Model too large
  - USB 2.0 instead of 3.0
  - Too few SHAVE cores
- **Solutions:**
  - Use smaller model
  - Check USB connection
  - Increase SHAVE allocation

---


---

## 17. Glossary of Technical Terms

| Term | Definition |
|------|------------|
| **Blob** | Compiled neural network model for Myriad X VPU |
| **SHAVE** | Streaming Hybrid Architecture Vector Engine (NN accelerator core) |
| **VPU** | Vision Processing Unit (Myriad X chip in OAK cameras) |
| **NMS** | Non-Maximum Suppression (removes duplicate detections) |
| **IoU** | Intersection over Union (overlap metric for bounding boxes) |
| **Anchor Boxes** | Pre-defined bounding box templates for YOLO |
| **XLink** | Luxonis USB protocol for camera-host communication |
| **Pipeline** | Directed graph of processing nodes on OAK camera |
| **DepthAI** | Luxonis SDK for OAK camera programming |
| **OpenVINO** | Intel's toolkit for optimizing neural network inference |

---

## 18. References & Further Reading

### Official Documentation
1. **DepthAI Documentation:** https://docs.luxonis.com
2. **YOLO Papers:** 
   - YOLOv3: https://arxiv.org/abs/1804.02767
   - YOLOv5: https://github.com/ultralytics/yolov5
3. **OpenVINO Toolkit:** https://docs.openvino.ai

### Related Technologies
- **OAK Camera Hardware:** https://store.opencv.ai
- **Myriad X VPU:** https://www.intel.com/content/www/us/en/products/details/processors/movidius-vpu.html
- **OpenCV:** https://docs.opencv.org

### Tutorials
- DepthAI Bootcamp: https://docs.luxonis.com/projects/api/en/latest/tutorials/


---

## 19. Conclusion

This code represents a **production-ready** object detection system with:

✅ **Flexibility:** JSON-based configuration for easy model swapping
✅ **Performance:** On-device inference at 25-30 FPS
✅ **Reliability:** Automatic model download and error handling
✅ **Efficiency:** Low host CPU usage, VPU-accelerated processing
✅ **Usability:** Real-time visualization with FPS monitoring

**Ideal Use Cases:**
- Smart surveillance systems
- Retail analytics (people counting)
- Autonomous vehicles (object detection)
- Industrial automation (defect detection)
- Robotics (environmental perception)

**Learning Value:**
This code demonstrates professional practices in:
- Hardware-accelerated AI deployment
- Configuration-driven architecture
- Real-time computer vision pipelines
- Resource management and optimization

---


