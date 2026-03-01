from datetime import datetime
import os, sys
sys.path.append('..')
from util_logger import setup_logger
logger = setup_logger(module_name=str(__name__))

from util_video_converter import util_convert_mp4, VideoCodecConverter

import mediapipe as mp
from mediapipe.tasks import python as media_pipe_python_api
from mediapipe.tasks.python import vision as media_pipe_vision_api

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 0.10+ Tasks API - No protobuf or solutions needed
# Removed: from mediapipe.framework.formats import landmark_pb2
# Removed: from mediapipe import solutions

class MediaPipeGoog():
  """
  MediaPipe Pose Detection using Tasks API (0.10+)
  Features:
  - One-time detector initialization for performance
  - Direct landmark access (no protobuf conversion)
  - Compatible with RTSP/IP camera streams
  """

  # Class-level detector (initialized once)
  detector = None

  @classmethod
  def init_pose_detector(cls, model_path=None):
    """
    Initialize MediaPipe Pose Detector ONCE at class level.
    CRITICAL: Call this before processing any frames to avoid memory leaks and performance issues.

    Input Parameters:
    - model_path (str): Path to pose_landmarker.task model file

    Processing:
    - Creates detector with Tasks API
    - Stores at class level for reuse
    - Configures for IMAGE mode (static frames)

    Output:
    - None (sets cls.detector)
    """
    if model_path is None:
        _here = os.path.dirname(os.path.abspath(__file__))   # .../src/analysis/
        _git_root = os.path.dirname(os.path.dirname(_here))   # .../overlander26/
        model_path = os.path.join(_git_root, "DATA_DIR", "pose_models", "pose_landmarker.task")

    logger.debug("--- init_pose_detector model_path type %s", type(model_path))
    logger.debug("--- init_pose_detector model_path %s", model_path)

    if cls.detector is None:
      from mediapipe.tasks import python
      from mediapipe.tasks.python import vision

      base_options = python.BaseOptions(model_asset_path=model_path)
      options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_masks=False  # Disable for performance
      )
      cls.detector = vision.PoseLandmarker.create_from_options(options)
      logger.info("--MediaPipe Pose Detector initialized successfully--")
      logger.debug("--- MediaPipe Pose Detector initialized Tasks API 0.10+")
    else:
      logger.debug("--Detector already initialized, reusing existing--")

  @classmethod
  def get_horizontal_line_position(self, height, width):
    """
    """
    # Calculate the Y-coordinate for the horizontal line (70% of the height)
    line_y = int(height * 0.70)
    return line_y

  @classmethod
  def draw_horizontal_line(self, pose_annotated_image):
    """
    """
    # Get image dimensions
    height, width, _ = pose_annotated_image.shape

    # Calculate the Y-coordinate for the horizontal line (70% of the height)
    line_y = int(height * 0.70)
    # Draw the horizontal line
    line_color = (0, 255, 0)  # Green color in BGR format
    line_thickness = 5        # Thickness of the line
    cv2.line(pose_annotated_image, (0, line_y), (width - 1, line_y), line_color, line_thickness)
    logger.debug("--pose_annotated_image--AAbb---line_y--> %s", line_y)
    return pose_annotated_image

  @classmethod
  def pose_draw_landmarks_on_image(self,
                                   rgb_image,
                                   detection_result):
    """
    def draw_landmarks_on_image(rgb_image, detection_result):
    """
    flag_pose_landamrks_detected = "YES_LANDMARKS"
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    pose_landmarks_list = detection_result.pose_landmarks
    logger.debug("--- LEN-LIST pose_landmarks_list %s", len(pose_landmarks_list))
    if len(pose_landmarks_list) < 1:
       flag_pose_landamrks_detected = "NO_LANDMARKS"
       return annotated_image, flag_pose_landamrks_detected

    line_y_coord = self.get_horizontal_line_position(height, width)
    text_color   = (0, 255, 0)
    text_color_1 = (0, 0, 255)
    font             = cv2.FONT_HERSHEY_PLAIN  # Thinnest font available
    font_scale       = 10
    font_scale_alert = 2
    thickness        = 3

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]  # Draw the pose landmarks.

        # Tasks API returns direct landmarks - NO protobuf conversion needed
        # Each landmark already has .x, .y, .z attributes

        # Draw landmark numbers
        for landmark_idx, landmark in enumerate(pose_landmarks):
            if landmark_idx == 5:  # Left Eye outer — only landmark checked
                x = int(landmark.x * width)
                y_coord_height = int(landmark.y * height)

                if line_y_coord > y_coord_height:
                    line_color_2  = (0, 0, 255)  # RED in BGR
                    line_thickness = 5

                    point_x, point_y = x, y_coord_height
                    square_height = 55
                    square_width  = 35
                    top_left_x     = point_x - square_width  // 2
                    top_left_y     = point_y - square_height // 2
                    bottom_right_x = point_x + square_width  // 2
                    bottom_right_y = point_y + square_height // 2
                    cv2.rectangle(annotated_image,
                                  (top_left_x, top_left_y),
                                  (bottom_right_x, bottom_right_y),
                                  line_color_2, line_thickness)
                    cv2.putText(annotated_image, "-FACE-",
                                (x, y_coord_height),
                                font, font_scale_alert, text_color_1, thickness)

        annotated_image = self.draw_horizontal_line(annotated_image)
        return annotated_image, flag_pose_landamrks_detected

  @classmethod
  def pose_media_pipe_google_2(self, video_source=None):
    """
    Desc:
      - Works with BOTH local video files AND live camera feeds
      - Supports: MP4, AVI, MOV, MKV (local files)
      - Supports: RTSP streams, USB webcams, IP cameras (live feeds)

    Args:
      video_source (str/int, optional):
        - Local file: "../../DATA_DIR/pose_detected/init_video/gym_1.mp4"
        - RTSP stream: "rtsp://192.168.1.100:8080/video"
        - USB webcam: 0 (device index)
        - IP webcam: "http://192.168.1.100:8080/video"
        - If None, uses default local file

    Usage Examples:
      # Local MP4 file (default)
      MediaPipeGoog().pose_media_pipe_google_2()

      # RTSP stream from IP camera
      MediaPipeGoog().pose_media_pipe_google_2("rtsp://192.168.1.100:8080/video")

      # USB webcam (device 0)
      MediaPipeGoog().pose_media_pipe_google_2(0)
    """
    import time
    import os

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    git_up_root  = os.path.dirname(project_root)

    dir_pose_init_video = os.path.join(git_up_root, "DATA_DIR", "pose_detected", "init_video")
    os.makedirs(dir_pose_init_video, exist_ok=True)
    dir_pose_detected_pose = os.path.join(git_up_root, "DATA_DIR", "pose_detected", "detected_pose")
    os.makedirs(dir_pose_detected_pose, exist_ok=True)

    logger.debug("--- git_up_root %s", git_up_root)
    logger.debug("--- dir_pose_init_video %s", dir_pose_init_video)
    logger.info(f"   Input video dir: {dir_pose_init_video}")
    logger.info(f"   Output frames dir: {dir_pose_detected_pose}")

    # Determine video source type
    if video_source is None:
        video_source = os.path.join(dir_pose_init_video, "gym_1_h264.mp4")
        logger.info(f"📹 Using default video source: {video_source}")
        logger.debug("--- default video source %s", video_source)

    # Detect source type for logging
    if isinstance(video_source, int):
        source_type = "USB_WEBCAM"
        logger.info(f"📹 Video source: USB Webcam (device {video_source})")
    elif isinstance(video_source, str):
        if video_source.startswith("rtsp://"):
            source_type = "RTSP_STREAM"
            logger.info(f"📹 Video source: RTSP Stream ({video_source})")
        elif video_source.startswith("http://") or video_source.startswith("https://"):
            source_type = "IP_CAMERA"
            logger.info(f"📹 Video source: IP Camera ({video_source})")
        elif os.path.isfile(video_source):
            source_type = "LOCAL_FILE"
            file_ext = os.path.splitext(video_source)[1].upper()
            logger.info(f"📹 Video source: Local File ({file_ext}) - {video_source}")
        else:
            logger.error(f"❌ Video source not found: {video_source}")
            logger.debug("--- video source does not exist %s", video_source)

            video_filename = os.path.basename(video_source)
            alt_locations = [
                dir_pose_detected_pose,
                os.path.join(git_up_root, "DATA_DIR", "pose_detected"),
                os.path.join(git_up_root, "DATA_DIR"),
            ]

            found_alternatives = []
            for alt_dir in alt_locations:
                alt_path = os.path.join(alt_dir, video_filename)
                if os.path.isfile(alt_path):
                    found_alternatives.append(alt_path)

            all_videos = []
            try:
                for root, dirs, files in os.walk(os.path.join(git_up_root, "DATA_DIR")):
                    for file in files:
                        if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                            all_videos.append(os.path.join(root, file))
            except Exception as e:
                logger.warning(f"Error scanning for videos: {e}")

            if found_alternatives:
                logger.debug("--- found video in wrong location %s", found_alternatives[0])
            elif all_videos:
                logger.debug("--- found %s video files in data_dir", len(all_videos))
                for vid in all_videos[:10]:
                    logger.debug("--- available video %s", vid)
            else:
                logger.debug("--- no video files found in data_dir %s",
                             os.path.join(git_up_root, "DATA_DIR"))
            return
    else:
        logger.error(f"❌ Invalid video source type: {type(video_source)}")
        return

    # Check if video needs H.264 conversion (for OpenCV compatibility)
    if source_type == "LOCAL_FILE":
        logger.info("🔍 Checking video codec compatibility...")
        safe_video_path = util_convert_mp4(video_source)
        
        if safe_video_path is None:
            logger.error(f"❌ Failed to prepare video for playback: {video_source}")
            logger.info("💡 Ensure ffmpeg is installed: sudo apt install ffmpeg")
            return
        
        if safe_video_path != video_source:
            logger.info(f"📺 Using converted video: {safe_video_path}")
            video_source = safe_video_path
        else:
            logger.debug("✅ Video codec is compatible, no conversion needed")

    # Initialize video capture
    capture_vid_init = cv2.VideoCapture(video_source)
    logger.debug("--- capture_vid_init type %s", type(capture_vid_init))

    # Check if the video was opened successfully
    if not capture_vid_init.isOpened():
        logger.error(f"❌ Failed to open video source: {video_source}")
        logger.debug("--- unable to open video source %s", video_source)
        if source_type == "LOCAL_FILE":
            abs_path = os.path.abspath(video_source)
            logger.debug("--- absolute path %s", abs_path)
            logger.debug("--- file exists %s", os.path.exists(video_source))
            try:
                video_files = [f for f in os.listdir(dir_pose_init_video)
                               if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                for vf in video_files:
                    logger.debug("--- available video file %s", vf)
            except Exception as e:
                logger.debug("--- error listing directory %s", e)
        elif source_type in ("RTSP_STREAM", "IP_CAMERA"):
            logger.debug("--- verify camera/stream is running and URL is correct")
        elif source_type == "USB_WEBCAM":
            logger.debug("--- verify camera is connected and check device index")
        return

    # Get video properties
    fps         = capture_vid_init.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture_vid_init.get(cv2.CAP_PROP_FRAME_COUNT))

    # Handle live streams (frame_count is 0 or -1 for live feeds)
    is_live_stream = (frame_count <= 0 or source_type in ["RTSP_STREAM", "IP_CAMERA", "USB_WEBCAM"])

    if is_live_stream:
        logger.info(f"🔴 LIVE STREAM MODE - Running indefinitely (Press 'q' to quit)")
        logger.debug("--- FPS %s", fps if fps > 0 else "Unknown")
        if fps <= 0:
            fps = 30.0
    else:
        duration_of_video = frame_count / fps if fps > 0 else 0
        logger.info(f"📁 FILE MODE - Processing video file")
        logger.debug("--- FPS %s", fps)
        logger.debug("--- Total Frames %s", frame_count)
        logger.debug("--- Video Duration %s", duration_of_video)

    logger.debug("--- Video FPS %s", fps)

    # Frame capture interval (every N seconds)
    capture_interval = 5  # in seconds
    frame_interval   = int(fps * capture_interval)

    # Initialize variables
    current_frame        = 0
    captured_frame_count = 0

    logger.info(f"🎬 Starting frame processing (capture every {capture_interval}s)")

    while True:  # Loop through the video frames
        ret, frame = capture_vid_init.read()  # Read the next frame

        if not ret:  # Break the loop if no more frames are available
            if is_live_stream:
                logger.warning("⚠️ Lost connection to live stream, retrying...")
                time.sleep(2)
                capture_vid_init.release()
                capture_vid_init = cv2.VideoCapture(video_source)
                if not capture_vid_init.isOpened():
                    logger.error("❌ Failed to reconnect to stream")
                    break
                continue
            else:
                logger.info("✅ Reached end of video file")
                break

        resized_frame_0 = cv2.resize(frame, (500, 650))

        if current_frame % frame_interval == 0:  # Capture a frame every N seconds
            resized_frame1 = cv2.resize(frame, (500, 650))

            # Save the captured frame
            frame_pose_save_path = os.path.join(dir_pose_detected_pose,
                                                 f"frame_{captured_frame_count:04d}.jpg")
            cv2.imwrite(frame_pose_save_path, resized_frame1)

            logger.info(f"💾 Frame {captured_frame_count} saved to {frame_pose_save_path}")
            logger.debug("--- frame saved %s", frame_pose_save_path)
            captured_frame_count += 1

            # Run pose detection
            pose_write_path = MediaPipeGoog().pose_media_pipe_google_0(frame_pose_save_path)
            logger.debug("--- pose_write_path %s", pose_write_path)

            if isinstance(pose_write_path, str):
                if pose_write_path == "EMPTY_STR":
                    continue

                image_pose_saved_last = cv2.imread(pose_write_path)
                if image_pose_saved_last is not None:
                    resized_pose_frame = cv2.resize(image_pose_saved_last, (500, 650))
                    top_row = np.hstack((resized_frame_0, resized_pose_frame))
                    cv2.imshow('OVERLANDER__GRID_VIEW', top_row)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("🛑 User requested quit (pressed 'q')")
                        break

        current_frame += 1

        if is_live_stream and current_frame % 300 == 0:
            logger.info(f"🔴 Live stream running... Frames processed: {current_frame}, "
                        f"Poses detected: {captured_frame_count}")

    capture_vid_init.release()
    cv2.destroyAllWindows()

    logger.info(f"✅ Frame capture completed. Total frames: {current_frame}, "
                f"Poses detected: {captured_frame_count}")
    logger.debug("--- total frames processed %s", current_frame)
    logger.debug("--- poses detected %s", captured_frame_count)


  @classmethod
  def pose_media_pipe_google_0(self, image_saved_ipcam):
    """
    This is INIT Frame -- frame_pose_save_path == image_saved_ipcam
    """
    logger.warning("-HIT-pose_media_pipe_google_0--->>")
    logger.debug("--- pose_media_pipe_google_0 hit")

    _here_0     = os.path.dirname(os.path.abspath(__file__))   # .../src/analysis/
    _git_root_0 = os.path.dirname(os.path.dirname(_here_0))    # .../overlander26/
    dir_pose_not_ipcam        = os.path.join(_git_root_0, "DATA_DIR", "pose_detected", "pose_not_ipcam") + "/"
    dir_got_pose_id_not_ipcam = os.path.join(_git_root_0, "DATA_DIR", "pose_detected", "pose_id_not_ipcam") + "/"
    os.makedirs(dir_pose_not_ipcam, exist_ok=True)
    os.makedirs(dir_got_pose_id_not_ipcam, exist_ok=True)

    pose_write_path = "EMPTY_STR"
    frame_counter   = 0
    dt_time_now     = datetime.now()
    time_minute_now = dt_time_now.strftime("_%m_%d_%Y_%H_%M_%S")
    logger.debug("--- time_minute_now %s", time_minute_now)
    second_now = str(time_minute_now).rsplit("_", 1)[1]
    logger.debug("--- second_now %s", second_now)

    logger.warning("-pose_media_pipe_google_0--image_saved_ipcam->> %s", image_saved_ipcam)
    if "detected_pose" in str(image_saved_ipcam):
        image_name_pose_detect = str(str(image_saved_ipcam).rsplit("detected_pose/", 1)[1])
        logger.debug("--- image_name_pose_detect %s", image_name_pose_detect)

    # Initialize detector if not already done
    if self.detector is None:
        self.init_pose_detector()

    # Use class-level detector (NO recreation per frame)
    image = mp.Image.create_from_file(image_saved_ipcam)
    logger.debug("--- image type %s", type(image))

    detection_result = self.detector.detect(image)
    logger.debug("--- detection_result %s", detection_result)
    logger.warning("---YES_LANDMARKS-detection_result-- %s", detection_result)

    annotated_image, flag_pose_landamrks_detected = self.pose_draw_landmarks_on_image(
        image.numpy_view(), detection_result
    )

    if flag_pose_landamrks_detected == "YES_LANDMARKS":
        name_to_write  = (image_name_pose_detect + "_frame_pose_"
                          + str(second_now) + "__" + str(frame_counter) + "__.png")
        pose_write_path = dir_got_pose_id_not_ipcam + name_to_write
        cv2.imwrite(pose_write_path, annotated_image)
        frame_counter += 1
        logger.warning("---YES_LANDMARKS-pose_write_path-POSE FRAMES WRITTEN--- %s", pose_write_path)
        logger.debug("--- YES_LANDMARKS pose_write_path %s", pose_write_path)
        return pose_write_path
    else:
        name_to_write  = (image_name_pose_detect + "_frame_pose_"
                          + str(second_now) + "__" + str(frame_counter) + "__.png")
        pose_write_path = dir_pose_not_ipcam + name_to_write
        cv2.imwrite(pose_write_path, annotated_image)
        frame_counter += 1
        logger.warning("--pose_write_path-POSE FRAMES WRITTEN--- %s", pose_write_path)
        logger.debug("--- pose_write_path %s", pose_write_path)
        return pose_write_path


  @classmethod
  def pose_media_pipe_google_1(self, image_saved_ipcam):
    """
    This is INIT Frame -- frame_pose_save_path == image_saved_ipcam
    """
    dir_pose_rect_only = "../../DATA_DIR/pose_detected/pose_rect_only/"
    pose_write_path = "EMPTY_STR"
    frame_counter   = 0
    dt_time_now     = datetime.now()
    time_minute_now = dt_time_now.strftime("_%m_%d_%Y_%H_%M_%S")
    logger.debug("--- time_minute_now %s", time_minute_now)
    second_now = str(time_minute_now).rsplit("_", 1)[1]
    logger.debug("--- second_now %s", second_now)

    logger.warning("-POSE---TYPE---image_saved_ipcam->> %s", image_saved_ipcam)
    if "frame_for_pose" in str(image_saved_ipcam):
        image_name_pose_detect = str(str(image_saved_ipcam).rsplit("frame_for_pose/", 1)[1])

    # Initialize detector if not already done
    if self.detector is None:
        self.init_pose_detector()

    # Use class-level detector (NO recreation per frame)
    image            = mp.Image.create_from_file(image_saved_ipcam)
    detection_result = self.detector.detect(image)
    annotated_image  = self.pose_draw_landmarks_on_image(image.numpy_view(), detection_result)
    name_to_write    = (image_name_pose_detect + "_frame_pose_"
                        + str(second_now) + "__" + str(frame_counter) + "__.png")
    pose_write_path  = dir_pose_rect_only + name_to_write
    cv2.imwrite(pose_write_path, annotated_image)
    frame_counter += 1
    logger.warning("--pose_write_path-POSE FRAMES WRITTEN--- %s", pose_write_path)
    logger.debug("--- pose_write_path %s", pose_write_path)
    return pose_write_path


# # For static images (legacy reference — Solutions API):
# IMAGE_FILES = []
# BG_COLOR = (192, 192, 192) # gray
# with mp_pose.Pose(
#     static_image_mode=True,
#     model_complexity=2,
#     enable_segmentation=True,
#     min_detection_confidence=0.5) as pose:
#   for idx, file in enumerate(IMAGE_FILES):
#     image = cv2.imread(file)
#     image_height, image_width, _ = image.shape
#     results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     if not results.pose_landmarks:
#       continue
#     annotated_image = image.copy()
#     condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
#     bg_image = np.zeros(image.shape, dtype=np.uint8)
#     bg_image[:] = BG_COLOR
#     annotated_image = np.where(condition, annotated_image, bg_image)
#     mp_drawing.draw_landmarks(
#         annotated_image,
#         results.pose_landmarks,
#         mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#     cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
#     mp_drawing.plot_landmarks(
#         results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# # For webcam input (legacy reference — Solutions API):
# cap = cv2.VideoCapture(0)
# with mp_pose.Pose(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as pose:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       continue
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = pose.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     mp_drawing.draw_landmarks(
#         image,
#         results.pose_landmarks,
#         mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#     cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()
