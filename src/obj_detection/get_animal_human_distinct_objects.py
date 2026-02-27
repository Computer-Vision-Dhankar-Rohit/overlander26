# TODO__get_animal_human_distinct_objects
#
# Task: Identify if the central/dominant object in a video frame is:
#   HUMAN / DOG / CAT / ANIMAL / OTHER
#
# Model: MediaPipe ObjectDetector — EfficientDet-Lite0 (COCO 80 classes)
# API:   MediaPipe Tasks API 0.10+  (same pattern as existing media_pipe.py)
# Input: MP4 from DATA_DIR/youtube_videos/
# Output: Annotated frames in DATA_DIR/obj_detection_output/

import sys
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.append(str(Path(__file__).parent.parent))
from util_logger import setup_logger
logger = setup_logger(module_name=str(__name__))

# ---------------------------------------------------------------------------
# Paths  (all relative via pathlib — auto-created on import)
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).parent.parent.parent          # overlander26/
DATA_DIR      = PROJECT_ROOT / "DATA_DIR"
VIDEOS_DIR    = DATA_DIR / "youtube_videos"
OUTPUT_DIR    = DATA_DIR / "obj_detection_output"
MODELS_DIR    = DATA_DIR / "models"

for _d in [VIDEOS_DIR, OUTPUT_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

MODEL_PATH    = MODELS_DIR / "efficientdet_lite0.tflite"
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
)

# ---------------------------------------------------------------------------
# Label classification maps
# ---------------------------------------------------------------------------
HUMAN_LABELS  = {"person"}
DOG_LABELS    = {"dog"}
CAT_LABELS    = {"cat"}
ANIMAL_LABELS = {
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "bird",
}

# Bounding box colour per category (BGR)
CATEGORY_COLORS = {
    "HUMAN":   (0,   255,   0),    # green
    "DOG":     (0,   165, 255),    # orange
    "CAT":     (255,   0, 255),    # magenta
    "ANIMAL":  (255, 255,   0),    # cyan
    "OTHER":   (200, 200, 200),    # grey
    "UNKNOWN": (128, 128, 128),    # dark grey
}


def _classify_label(label: str) -> str:
    label = label.strip().lower()
    if label in HUMAN_LABELS:   return "HUMAN"
    if label in DOG_LABELS:     return "DOG"
    if label in CAT_LABELS:     return "CAT"
    if label in ANIMAL_LABELS:  return "ANIMAL"
    return "OTHER"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class CentralObjectClassifier:
    """
    MediaPipe ObjectDetector — EfficientDet-Lite0
    Identifies the dominant (largest bounding box) object in a frame.

    Singleton detector pattern — initialized once, reused across all frames.
    """

    detector = None   # Class-level singleton

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    @classmethod
    def init_detector(cls, model_path: Path = MODEL_PATH) -> bool:
        """
        Initialize MediaPipe ObjectDetector ONCE at class level.

        Args:
            model_path: Path to efficientdet_lite0.tflite

        Returns:
            True if ready, False if model file missing.
        """
        model_path = Path(model_path)

        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            print(f"❌ Model not found: {model_path}")
            print(f"   Download with:")
            print(f"   wget -O {model_path} {MODEL_DOWNLOAD_URL}")
            return False

        if cls.detector is None:
            base_options = mp_python.BaseOptions(
                model_asset_path=str(model_path)
            )
            options = mp_vision.ObjectDetectorOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                score_threshold=0.40,
                max_results=10,
            )
            cls.detector = mp_vision.ObjectDetector.create_from_options(options)
            logger.info("MediaPipe ObjectDetector initialized — EfficientDet-Lite0")
            print("✅ MediaPipe ObjectDetector initialized (EfficientDet-Lite0)")
        else:
            logger.debug("ObjectDetector already initialized — reusing singleton")

        return True

    # ------------------------------------------------------------------
    # Per-frame classification
    # ------------------------------------------------------------------
    @classmethod
    def classify_frame(cls, cv2_frame: np.ndarray) -> dict:
        """
        Classify the dominant object in a single OpenCV BGR frame.

        Args:
            cv2_frame: numpy array (H, W, 3) in BGR

        Returns:
            dict:
              category       — "HUMAN" / "DOG" / "CAT" / "ANIMAL" / "OTHER" / "UNKNOWN"
              label          — raw COCO label string
              score          — confidence float
              bounding_box   — mediapipe BoundingBox (or None)
              all_detections — list of {label, score, category} for every detected object
        """
        if cls.detector is None:
            logger.error("Detector not initialized. Call init_detector() first.")
            return {"category": "UNKNOWN", "label": "", "score": 0.0,
                    "bounding_box": None, "all_detections": []}

        rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = cls.detector.detect(mp_image)

        all_detections = [
            {
                "label":    d.categories[0].category_name,
                "score":    round(d.categories[0].score, 3),
                "category": _classify_label(d.categories[0].category_name),
            }
            for d in detection_result.detections
        ]

        if not detection_result.detections:
            return {"category": "UNKNOWN", "label": "", "score": 0.0,
                    "bounding_box": None, "all_detections": all_detections}

        # Central object = largest bounding box area
        best = max(
            detection_result.detections,
            key=lambda d: d.bounding_box.width * d.bounding_box.height,
        )
        label    = best.categories[0].category_name
        score    = best.categories[0].score
        category = _classify_label(label)

        return {
            "category":      category,
            "label":         label,
            "score":         round(score, 3),
            "bounding_box":  best.bounding_box,
            "all_detections": all_detections,
        }

    # ------------------------------------------------------------------
    # Video processing
    # ------------------------------------------------------------------
    @classmethod
    def process_video(cls, video_path: Path = None, frame_interval_sec: int = 5):
        """
        Process a video file, classifying the dominant object every N seconds.
        Saves annotated frames to DATA_DIR/obj_detection_output/

        Args:
            video_path:          Path to MP4. If None, auto-picks first file in VIDEOS_DIR.
            frame_interval_sec:  Capture / classify every N seconds (default 5).
        """
        # Resolve video
        if video_path is None:
            mp4_files = sorted(VIDEOS_DIR.glob("*.mp4"))
            if not mp4_files:
                print(f"❌ No MP4 files in {VIDEOS_DIR}")
                print(f"   Run: YouTubeDownloader.download('<url>') first.")
                return
            video_path = mp4_files[0]
            print(f"📹 Auto-selected: {video_path.name}")

        video_path = Path(video_path)
        if not video_path.exists():
            print(f"❌ Video not found: {video_path}")
            return

        # Init detector
        if not cls.init_detector():
            return

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Cannot open: {video_path}")
            return

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step   = max(1, int(fps * frame_interval_sec))
        duration_s   = total_frames / fps if fps > 0 else 0

        print(f"\n📁 Video     : {video_path.name}")
        print(f"   FPS       : {fps:.1f}")
        print(f"   Frames    : {total_frames}  ({duration_s:.1f}s)")
        print(f"   Interval  : every {frame_interval_sec}s  (step={frame_step})")
        print(f"   Output    : {OUTPUT_DIR}\n")
        logger.info(f"process_video start — {video_path.name} fps={fps} frames={total_frames}")

        frame_idx   = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                result   = cls.classify_frame(frame)
                category = result["category"]
                label    = result["label"]
                score    = result["score"]

                print(f"  [{frame_idx:06d}] → {category:7s} | {label:<12s} score={score:.2f}")
                logger.info(
                    f"frame={frame_idx} category={category} label={label} score={score}"
                )

                annotated = cls._annotate_frame(frame, result)
                out_name  = (
                    f"{video_path.stem}"
                    f"__frame_{frame_idx:06d}"
                    f"__{category}.jpg"
                )
                cv2.imwrite(str(OUTPUT_DIR / out_name), annotated)
                saved_count += 1

            frame_idx += 1

        cap.release()
        print(f"\n✅ Done — {saved_count} frames saved to: {OUTPUT_DIR}")
        logger.info(f"process_video complete — saved={saved_count}")

    # ------------------------------------------------------------------
    # Annotation helper
    # ------------------------------------------------------------------
    @staticmethod
    def _annotate_frame(frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw bounding box + category label on a copy of the frame."""
        annotated = frame.copy()
        box = result.get("bounding_box")
        if box is None:
            return annotated

        x  = int(box.origin_x)
        y  = int(box.origin_y)
        x2 = x + int(box.width)
        y2 = y + int(box.height)

        color      = CATEGORY_COLORS.get(result["category"], (200, 200, 200))
        label_text = f"{result['category']} | {result['label']} {result['score']:.2f}"

        cv2.rectangle(annotated, (x, y), (x2, y2), color, 3)
        cv2.putText(
            annotated, label_text,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
        )

        # Mini legend — all detected objects in top-left corner
        for i, det in enumerate(result.get("all_detections", [])):
            det_color = CATEGORY_COLORS.get(det["category"], (200, 200, 200))
            legend    = f"{det['category']:7s} {det['label']} {det['score']:.2f}"
            cv2.putText(
                annotated, legend,
                (10, 25 + i * 22),
                cv2.FONT_HERSHEY_PLAIN, 1.2, det_color, 1,
            )

        return annotated


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1 — download a video (comment out if already downloaded)
    # from youtube_downloader import YouTubeDownloader
    # YouTubeDownloader.download("https://www.youtube.com/shorts/qlPfknL2P9A")

    # Step 2 — run classification on downloaded video
    CentralObjectClassifier.process_video()
