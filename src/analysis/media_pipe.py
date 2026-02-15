from datetime import datetime
import os , sys 
sys.path.append('..')
from util_logger import setup_logger
logger = setup_logger(module_name=str(__name__))

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
  def init_pose_detector(cls, model_path='../data_dir/pose_models/pose_landmarker.task'):
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
    print("-HIT--InitializeMediaPipePoseDetector-model_path-->> %s",type(model_path))
    print("-detection_result--AAA----model_path-->> %s",model_path)

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
      print("✅ MediaPipe Pose Detector initialized (Tasks API 0.10+)")
    else:
      logger.debug("--Detector already initialized, reusing existing--")

  @classmethod
  def get_horizontal_line_position(self,height, width):
    """ 
    """
    # Calculate the Y-coordinate for the horizontal line (30% of the height)
    line_y = int(height * 0.70) #TODO# 0.30 -- from TOP 30% --horizontal line (30% of the height)
    # Draw the horizontal line
    return line_y

  @classmethod
  def draw_horizontal_line(self,pose_annotated_image):
    """ 
    """
    # Get image dimensions
    height, width, _ = pose_annotated_image.shape
    
    # Calculate the Y-coordinate for the horizontal line (30% of the height)
    line_y = int(height * 0.70) #TODO# 0.30 -- from TOP 30% --horizontal line (30% of the height)
    # Draw the horizontal line
    line_color = (0, 255, 0)  # Green color in BGR format
    line_thickness = 5  # Thickness of the line
    cv2.line(pose_annotated_image, (0, line_y), (width - 1, line_y), line_color, line_thickness)
    # Print the pixel coordinates of the line
    print(f"Line coordinates:")
    print(f"Start: (0, {line_y})")
    print(f"End: ({width - 1}, {line_y})")
    """ 
    Line coordinates:
        Start: (0, 324)
        End: (1919, 324)

    """
    logger.debug("--pose_annotated_image--AAbb---line_y--> %s" ,line_y)
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
    print("--LEN-LIST---pose_landmarks_list-",len(pose_landmarks_list))
    print("   "*200)
    if len(pose_landmarks_list) <1:
       flag_pose_landamrks_detected = "NO_LANDMARKS"
       return annotated_image , flag_pose_landamrks_detected

    line_y_coord = self.get_horizontal_line_position(height, width)
    text_color = (0, 255, 0)
    text_color_1 = (0,0,255)
    font = cv2.FONT_HERSHEY_PLAIN  # Thinnest font available
    font_scale = 10 # 0.5 == Smaller font scale for thinner appearance
    font_scale_alert = 2 # 0.5 == Smaller font scale for thinner appearance
    thickness = 3 # 1 -- Thinnest possible thickness
    
    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx] # Draw the pose landmarks.
        
        # Tasks API returns direct landmarks - NO protobuf conversion needed
        # Each landmark already has .x, .y, .z attributes
        # No need for landmark_pb2.NormalizedLandmarkList() anymore
        
        # Draw landmark numbers
        #ls_hand_only_landmarks = [27,29,31] 
        for landmark_idx, landmark in enumerate(pose_landmarks):
            if landmark_idx == 5 :#or landmark_idx == 6 :#or landmark_idx == 31: 
                ##TODO # if landmark_idx in ls_hand_only_landmarks:
                x = int(landmark.x * width) # Get the landmark position in pixel coordinates
                y_coord_height = int(landmark.y * height)
                
                if line_y_coord > y_coord_height:
                    line_color_2 = (0, 0, 255)  # RED - color in BGR format
                    line_thickness = 5  # Thickness of the line

                    # Define the point coordinate (center of the square)
                    point_x, point_y = x, y_coord_height  # Replace with your point coordinates
                    # Define the height and width of the square
                    square_height = 55
                    square_width = 35
                    # Calculate the top-left and bottom-right corners of the square
                    top_left_x = point_x - square_width // 2
                    top_left_y = point_y - square_height // 2
                    bottom_right_x = point_x + square_width // 2
                    bottom_right_y = point_y + square_height // 2
                    cv2.rectangle(annotated_image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), line_color_2, line_thickness)

                    #Draw the horizontal line
                    #cv2.line(annotated_image, (0, y_coord_height), (width - 1, 50), line_color_2, line_thickness)
                    cv2.putText(annotated_image, "-FACE-", (x, y_coord_height), font, font_scale_alert, text_color_1, thickness)

                    # #cv2.line(image, (0, y_coord_height), (width - 1, y_coord_height), line_color, line_thickness)

                    # logger.debug("-AAbb----LINE--CROSSED---landmark_idx--> %s" ,landmark_idx)
                    # logger.debug("-AAbb------LINE--CROSSED---landmark---> %s" ,landmark)
                    # logger.debug("-AAbb------LINE--CROSSED---landmark.y----> %s" ,int(landmark.y))

                # if line_y_coord < y_coord_height:
                #     crossed_landmark_idx = str(landmark_idx) + "--CROSS-AA"
                #     cv2.putText(annotated_image, str(crossed_landmark_idx), (x, y_coord_height), font, font_scale_alert, text_color_1, thickness)
                #     print("   -- "*10)
                #     print("--LINE--CROSSED----------------------------------------")
                #     print("   -- "*10)
                #     logger.debug("--LINE--CROSSED---landmark_idx--> %s" ,landmark_idx)
                #     logger.debug("--LINE--CROSSED---line_y_coord--> %s" ,line_y_coord)
                #     logger.debug("--LINE--CROSSED---landmark.y--> %s" ,int(landmark.y))
                #     logger.debug("--LINE--CROSSED---y_coord_height--> %s" ,y_coord_height)
                
                # else:
                #    continue

                # Draw the landmark number
                # logger.debug("--a--landmark_idx--> %s" ,landmark_idx)
                # cv2.putText(annotated_image, str(landmark_idx), (x, y_coord_height), font, font_scale, text_color, thickness)
                # # #cv2.putText(annotated_image, str(landmark_idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                # cv2.circle(annotated_image, (x, y_coord_height), 7, text_color_1, -1)  # Green circle with radius 5

        annotated_image = self.draw_horizontal_line(annotated_image) #
        # TODO -- # OK Dont Draw Horizonta Line -- Keep it INVISIBLE 
        return annotated_image , flag_pose_landamrks_detected

  @classmethod
  def pose_media_pipe_google_2(self, video_source=None):
    """
    Desc:
      - Works with BOTH local video files AND live camera feeds
      - Supports: MP4, AVI, MOV, MKV (local files)
      - Supports: RTSP streams, USB webcams, IP cameras (live feeds)
    
    Args:
      video_source (str/int, optional): 
        - Local file: "../data_dir/pose_detected/init_video/gym_1.mp4"
        - RTSP stream: "rtsp://192.168.1.100:8080/video"
        - USB webcam: 0 (device index)
        - IP webcam: "http://192.168.1.100:8080/video"
        - If None, uses default local file
    
    Usage Examples:
      # Local MP4 file (default)
      MediaPipeGoog().pose_media_pipe_google_2()
      
      # Local MP4 file (explicit)
      MediaPipeGoog().pose_media_pipe_google_2("../data_dir/pose_detected/init_video/Columbia_.mp4")
      
      # RTSP stream from IP camera
      MediaPipeGoog().pose_media_pipe_google_2("rtsp://192.168.1.100:8080/video")
      
      # USB webcam (device 0)
      MediaPipeGoog().pose_media_pipe_google_2(0)
      
      # IP webcam HTTP stream
      MediaPipeGoog().pose_media_pipe_google_2("http://192.168.1.100:8080/video")
    """
    import time
    import os
    
    # Get the absolute path to the script's directory
    # media_pipe.py is in: /home/dhankar/temp/26_02/git_up/ipWebCam/analysis/
    # We need to go UP to: /home/dhankar/temp/26_02/git_up/
    script_dir = os.path.dirname(os.path.abspath(__file__))  # .../ipWebCam/analysis/
    project_root = os.path.dirname(script_dir)  # .../ipWebCam/
    git_up_root = os.path.dirname(project_root)  # .../git_up/
    
    # Create ABSOLUTE output directories
    dir_pose_init_video = os.path.join(git_up_root, "data_dir", "pose_detected", "init_video")
    os.makedirs(dir_pose_init_video, exist_ok=True)
    dir_pose_detected_pose = os.path.join(git_up_root, "data_dir", "pose_detected", "detected_pose")
    os.makedirs(dir_pose_detected_pose, exist_ok=True)
    
    logger.info(f"📂 Working directories:")
    print(f"   Input video dir---git_up_root: {git_up_root}")
    print(f"   Input video dir: {dir_pose_init_video}")

    logger.info(f"   Input video dir: {dir_pose_init_video}")
    logger.info(f"   Output frames dir: {dir_pose_detected_pose}")
    
    # Determine video source type
    if video_source is None:
        # Default: Local MP4 file
        video_source = os.path.join(dir_pose_init_video,"gym_1_h264.mp4")
        logger.info(f"📹 Using default video source: {video_source}")
        print(f"📹 Using default video source: {video_source}")
    
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
            # File not found - try to find it in alternative locations
            logger.error(f"❌ Video source not found: {video_source}")
            print(f"❌ Error: Video source does not exist: '{video_source}'")
            print(f"\n🔍 Searching for video files in alternative locations...")
            
            # Check if file exists in output directory (common mistake)
            video_filename = os.path.basename(video_source)
            alt_locations = [
                dir_pose_detected_pose,  # Output directory
                os.path.join(git_up_root, "data_dir", "pose_detected"),  # Parent directory
                os.path.join(git_up_root, "data_dir"),  # data_dir root
            ]
            
            found_alternatives = []
            for alt_dir in alt_locations:
                alt_path = os.path.join(alt_dir, video_filename)
                if os.path.isfile(alt_path):
                    found_alternatives.append(alt_path)
            
            # Search recursively for any video files
            all_videos = []
            try:
                for root, dirs, files in os.walk(os.path.join(git_up_root, "data_dir")):
                    for file in files:
                        if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                            all_videos.append(os.path.join(root, file))
            except Exception as e:
                logger.warning(f"Error scanning for videos: {e}")
            
            if found_alternatives:
                print(f"\n✅ Found '{video_filename}' in wrong location:")
                for alt in found_alternatives:
                    print(f"   📁 {alt}")
                print(f"\n💡 Solution: Move the file to the correct location:")
                print(f"   mv '{found_alternatives[0]}' '{video_source}'")
                print(f"\n   Or run from terminal:")
                print(f"   mv '{found_alternatives[0]}' '{dir_pose_init_video}/'")
            elif all_videos:
                print(f"\n📹 Found {len(all_videos)} video file(s) in data_dir:")
                for vid in all_videos[:10]:  # Show max 10
                    rel_path = os.path.relpath(vid, git_up_root)
                    print(f"   • {rel_path}")
                if len(all_videos) > 10:
                    print(f"   ... and {len(all_videos) - 10} more")
                print(f"\n💡 To use one of these videos:")
                print(f"   MediaPipeGoog().pose_media_pipe_google_2('{all_videos[0]}')")
            else:
                print(f"\n❌ No video files found in {os.path.join(git_up_root, 'data_dir')}")
                print(f"\n💡 Please place your video file in:")
                print(f"   {dir_pose_init_video}/")
                print(f"\n   Supported formats: .mp4, .avi, .mov, .mkv")
            
            return
    else:
        logger.error(f"❌ Invalid video source type: {type(video_source)}")
        print(f"❌ Error: Invalid video source type")
        return
    
    # Initialize video capture
    capture_vid_init = cv2.VideoCapture(video_source)
    print(f"TYPE---capture_vid_init-------", type(capture_vid_init))
    print("    --A " * 10)

    # Check if the video was opened successfully
    if not capture_vid_init.isOpened():
        logger.error(f"❌ Failed to open video source: {video_source}")
        print(f"❌ Error: Unable to open video source: '{video_source}'")
        print(f"💡 Troubleshooting:")
        if source_type == "LOCAL_FILE":
            abs_path = os.path.abspath(video_source)
            print(f"   - Absolute path: {abs_path}")
            print(f"   - File exists: {os.path.exists(video_source)}")
            print(f"   - Check file permissions: ls -lh '{abs_path}'")
            print(f"   - Verify codec support (install ffmpeg if needed)")
            # List available video files
            print(f"   - Available videos in {dir_pose_init_video}:")
            try:
                video_files = [f for f in os.listdir(dir_pose_init_video) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                if video_files:
                    for vf in video_files:
                        print(f"     • {vf}")
                else:
                    print(f"     (No video files found)")
            except Exception as e:
                print(f"     Error listing directory: {e}")
        elif source_type == "RTSP_STREAM" or source_type == "IP_CAMERA":
            print(f"   - Verify camera/stream is running")
            print(f"   - Check network connectivity")
            print(f"   - Verify correct URL format")
            print(f"   - Check authentication credentials if required")
        elif source_type == "USB_WEBCAM":
            print(f"   - Verify camera is connected")
            print(f"   - Try different device index (0, 1, 2...)")
            print(f"   - Check camera permissions")
        return
    
    # Get video properties
    fps = capture_vid_init.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture_vid_init.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle live streams (frame_count is 0 or -1 for live feeds)
    is_live_stream = (frame_count <= 0 or source_type in ["RTSP_STREAM", "IP_CAMERA", "USB_WEBCAM"])
    
    if is_live_stream:
        logger.info(f"🔴 LIVE STREAM MODE - Running indefinitely (Press 'q' to quit)")
        print(f"🔴 LIVE STREAM MODE")
        print(f"   FPS: {fps if fps > 0 else 'Unknown (will auto-detect)'}")
        print(f"   Duration: Continuous (press 'q' to stop)")
        # Set default FPS if not detected
        if fps <= 0:
            fps = 30.0  # Assume 30 FPS for live streams
    else:
        duration_of_video = frame_count / fps if fps > 0 else 0
        logger.info(f"📁 FILE MODE - Processing video file")
        print(f"📁 FILE MODE")
        print(f"   FPS: {fps}")
        print(f"   Total Frames: {frame_count}")
        print(f"   Video Duration: {duration_of_video:.2f} seconds")
    
    print(f"Video FPS: {fps}")
    if not is_live_stream:
        print(f"Total Frames: {frame_count}")
        print(f"Video Duration: {duration_of_video:.2f} seconds")
    
    # Frame capture interval (every N seconds)
    capture_interval = 5  # in seconds
    frame_interval = int(fps * capture_interval)  # Number of frames to skip
    
    # Initialize variables
    current_frame = 0
    captured_frame_count = 0
    
    logger.info(f"🎬 Starting frame processing (capture every {capture_interval}s)")
    
    while True:  # Loop through the video frames
        ret, frame = capture_vid_init.read()  # Read the next frame
        
        if not ret:  # Break the loop if no more frames are available
            if is_live_stream:
                logger.warning("⚠️ Lost connection to live stream, retrying...")
                print("⚠️ Stream interrupted, attempting to reconnect...")
                time.sleep(2)
                capture_vid_init.release()
                capture_vid_init = cv2.VideoCapture(video_source)
                if not capture_vid_init.isOpened():
                    logger.error("❌ Failed to reconnect to stream")
                    print("❌ Cannot reconnect to stream")
                    break
                continue
            else:
                logger.info("✅ Reached end of video file")
                print("✅ Finished processing all frames")
                break
        
        resized_frame_0 = cv2.resize(frame, (500, 650))
        
        if current_frame % frame_interval == 0:  # Capture a frame every N seconds
            resized_frame1 = cv2.resize(frame, (500, 650))
            
            # Save the captured frame
            frame_pose_save_path = os.path.join(dir_pose_detected_pose, f"frame_{captured_frame_count:04d}.jpg")
            cv2.imwrite(frame_pose_save_path, resized_frame1)
            
            logger.info(f"💾 Frame {captured_frame_count} saved to {frame_pose_save_path}")
            print(f"Frame {captured_frame_count} saved to===>> {frame_pose_save_path}")
            captured_frame_count += 1
            
            # Run pose detection
            pose_write_path = MediaPipeGoog().pose_media_pipe_google_0(frame_pose_save_path)
            print(f"INFO--pose_write_path-: {pose_write_path}")
            print("    "*100)

            
            if isinstance(pose_write_path, str):
                if pose_write_path == "EMPTY_STR":
                    continue  # TODO -- get older -- pose_write_path --
                
                image_pose_saved_last = cv2.imread(pose_write_path)
                if image_pose_saved_last is not None:
                    resized_pose_frame = cv2.resize(image_pose_saved_last, (500, 650))
                    top_row = np.hstack((resized_frame_0, resized_pose_frame))
                    cv2.imshow('OVERLANDER__GRID_VIEW', top_row)  # TODO -- grid
                    
                    # Press 'q' to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("🛑 User requested quit (pressed 'q')")
                        print("🛑 Stopping video processing...")
                        break

        # Increment the frame counter
        current_frame += 1
        
        # For live streams, show progress periodically
        if is_live_stream and current_frame % 300 == 0:  # Every 10 seconds at 30fps
            logger.info(f"🔴 Live stream running... Frames processed: {current_frame}, Poses detected: {captured_frame_count}")

    # Release the video capture object
    capture_vid_init.release()
    cv2.destroyAllWindows()
    
    logger.info(f"✅ Frame capture completed. Total frames processed: {current_frame}, Poses detected: {captured_frame_count}")
    print(f"✅ Frame capture completed.")
    print(f"   Total frames processed: {current_frame}")
    print(f"   Poses detected: {captured_frame_count}")
    

    # #while True:
    # ret1, frame1 = capture_vid_init.read() 
    # logger.warning("-pose---Abb --frame1->> %s",type(frame1))
    # current_time = time.time()
    # if current_time - last_save_time >= save_interval:
    #   logger.warning("-pose-Abb-current_time---> %s",current_time)

    #   resized_frame1 = cv2.resize(frame1, (900,850)) #TODO # 300 , 200 
    #   logger.warning("-pose---Abb --resized_frame1->> %s",type(resized_frame1))
    #   frame_pose_save_path = dir_pose_detected_pose + "frame_for_pose_"+ str(frame_counter)+"__.png" # frame_for_pose_ Frame where POSE is to BE Detected   
    #   cv2.imwrite(frame_pose_save_path, resized_frame1) 
    #   frame_counter += 1
    #   last_save_time = current_time  # Update the last save time


    

    # pose_write_path = "EMPTY_STR"
    # frame_counter = 0
    # dt_time_now = datetime.now()
    # time_minute_now  = dt_time_now.strftime("_%m_%d_%Y_%H_%M_%S")  
    # print("-time_minute_now----->> %s",time_minute_now)
    # second_now = str(time_minute_now).rsplit("_",1)[1]
    # print("-time_minute_now--min_now--->> %s",second_now)
    
    # # logger.warning("-POSE---TYPE---image_saved_ipcam->> %s",image_saved_ipcam)
    # # if "frame_for_pose" in str(image_saved_ipcam):
    # #     image_name_pose_detect = str(str(image_saved_ipcam).rsplit("frame_for_pose/",1)[1])
    # # ##../data_dir/pose_detected/frame_for_pose/frame_for_pose_0__.png

    # base_options = media_pipe_python_api.BaseOptions(model_asset_path='../data_dir/pose_detected/pose_models/pose_landmarker.task')
    # options =  media_pipe_vision_api.PoseLandmarkerOptions(
    #                             base_options=base_options,
    #                             output_segmentation_masks=True)
    #                             #running_mode=VisionRunningMode.IMAGE) ## TODO 
    # detector = media_pipe_vision_api.PoseLandmarker.create_from_options(options) ##- <class 'mediapipe.python._framework_bindings.image.Image'>
    # #image = mp.Image.create_from_file(image_saved_ipcam) #print("----mp.Image.create_from_file(image_rootDIR)----",type(image)) #
    
    # detection_result = detector.detect(image)
    # annotated_image = self.pose_draw_landmarks_on_image(image.numpy_view(), detection_result)
    
    # # WRITE TO DIR -- pose_rect_only
    # # name_to_write = image_name_pose_detect+"_frame_pose_"+str(second_now)+"__"+str(frame_counter)+"__.png"
    # # pose_write_path = dir_pose_rect_only+name_to_write
    # cv2.imwrite(pose_write_path, annotated_image)
    # frame_counter += 1
    # logger.warning("--pose_write_path-POSE FRAMES WRITTEN--- %s",pose_write_path)
    # print("----pose_write_path--AA->> %s",pose_write_path)
    # return pose_write_path

  @classmethod
  def pose_media_pipe_google_0(self,image_saved_ipcam):
    """
    This is INIT Frame -- frame_pose_save_path == image_saved_ipcam
    """
    logger.warning("-HIT-pose_media_pipe_google_0--->>")
    print("-HIT-pose_media_pipe_google_0--->>")

    dir_pose_not_ipcam = "../data_dir/pose_detected/pose_not_ipcam/"
    dir_got_pose_id_not_ipcam = "../data_dir/pose_detected/pose_id_not_ipcam/" #pose_id_not_ipcam

    pose_write_path = "EMPTY_STR"
    frame_counter = 0
    dt_time_now = datetime.now()
    time_minute_now  = dt_time_now.strftime("_%m_%d_%Y_%H_%M_%S")  
    print("-time_minute_now----->> %s",time_minute_now)
    second_now = str(time_minute_now).rsplit("_",1)[1]
    print("-time_minute_now--min_now--->> %s",second_now)
    
    logger.warning("-pose_media_pipe_google_0--image_saved_ipcam->> %s",image_saved_ipcam)
    if "detected_pose" in str(image_saved_ipcam):
        image_name_pose_detect = str(str(image_saved_ipcam).rsplit("detected_pose/",1)[1])
        print("-image_name_pose_detect--AAA-->> %s",image_name_pose_detect)
    
    # Initialize detector if not already done
    if self.detector is None:
        self.init_pose_detector()
    
    # Use class-level detector (NO recreation per frame)
    image = mp.Image.create_from_file(image_saved_ipcam)
    print("-detection_result--AAA----image-->> %s",type(image)) ## <class 'mediapipe.tasks.python.vision.core.image.Image'>

    detection_result = self.detector.detect(image)
    print("-detection_result--AAA-->> %s",detection_result)
    """   
    -detection_result--AAA-->> %s PoseLandmarkerResult(pose_landmarks=[], pose_world_landmarks=[], segmentation_masks=None)
    """
    logger.warning("---YES_LANDMARKS-detection_result-- %s",detection_result)

    annotated_image , flag_pose_landamrks_detected = self.pose_draw_landmarks_on_image(image.numpy_view(), detection_result)
    ##pose_id_not_ipcam
    
    if flag_pose_landamrks_detected == "YES_LANDMARKS":
      #dir_got_pose_id_not_ipcam
      name_to_write = image_name_pose_detect+"_frame_pose_"+str(second_now)+"__"+str(frame_counter)+"__.png"
      pose_write_path = dir_got_pose_id_not_ipcam+name_to_write
      cv2.imwrite(pose_write_path, annotated_image)
      frame_counter += 1
      logger.warning("---YES_LANDMARKS-pose_write_path-POSE FRAMES WRITTEN--- %s",pose_write_path)
      print("--YES_LANDMARKS--pose_write_path--AA->> %s",pose_write_path)
      return pose_write_path
    else:
      # WRITE TO DIR -- pose_rect_only
      name_to_write = image_name_pose_detect+"_frame_pose_"+str(second_now)+"__"+str(frame_counter)+"__.png"
      pose_write_path = dir_pose_not_ipcam+name_to_write
      cv2.imwrite(pose_write_path, annotated_image)
      frame_counter += 1
      logger.warning("--pose_write_path-POSE FRAMES WRITTEN--- %s",pose_write_path)
      print("----pose_write_path--AA->> %s",pose_write_path)
      return pose_write_path



  @classmethod
  def pose_media_pipe_google_1(self,image_saved_ipcam):
    """
    This is INIT Frame -- frame_pose_save_path == image_saved_ipcam
    """
    dir_pose_rect_only = "../data_dir/pose_detected/pose_rect_only/"
    pose_write_path = "EMPTY_STR"
    frame_counter = 0
    dt_time_now = datetime.now()
    time_minute_now  = dt_time_now.strftime("_%m_%d_%Y_%H_%M_%S")  
    print("-time_minute_now----->> %s",time_minute_now)
    second_now = str(time_minute_now).rsplit("_",1)[1]
    print("-time_minute_now--min_now--->> %s",second_now)
    
    logger.warning("-POSE---TYPE---image_saved_ipcam->> %s",image_saved_ipcam)
    if "frame_for_pose" in str(image_saved_ipcam):
        image_name_pose_detect = str(str(image_saved_ipcam).rsplit("frame_for_pose/",1)[1])
    ##../data_dir/pose_detected/frame_for_pose/frame_for_pose_0__.png
    
    # Initialize detector if not already done
    if self.detector is None:
        self.init_pose_detector()
    
    # Use class-level detector (NO recreation per frame)
    image = mp.Image.create_from_file(image_saved_ipcam)
    detection_result = self.detector.detect(image)
    annotated_image = self.pose_draw_landmarks_on_image(image.numpy_view(), detection_result)
    # WRITE TO DIR -- pose_rect_only
    name_to_write = image_name_pose_detect+"_frame_pose_"+str(second_now)+"__"+str(frame_counter)+"__.png"
    pose_write_path = dir_pose_rect_only+name_to_write
    cv2.imwrite(pose_write_path, annotated_image)
    frame_counter += 1
    logger.warning("--pose_write_path-POSE FRAMES WRITTEN--- %s",pose_write_path)
    print("----pose_write_path--AA->> %s",pose_write_path)
    return pose_write_path



#   @classmethod
#   def media_pipe_google():
#     """ 
#     """
#     # get POSE Object 
#     obj_pose_static_img = mp_pose.Pose(
#                                     static_image_mode=True,
#                                     model_complexity=2,
#                                     enable_segmentation=True,
#                                     min_detection_confidence=0.5) 
#     print("--Type---",type(obj_pose_static_img))
#     print("--Type---",obj_pose_static_img)




# # For static images:
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
#     # Convert the BGR image to RGB before processing.
#     results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     if not results.pose_landmarks:
#       continue
#     print(
#         f'Nose coordinates: ('
#         f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
#         f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
#     )

#     annotated_image = image.copy()
#     # Draw segmentation on the image.
#     # To improve segmentation around boundaries, consider applying a joint
#     # bilateral filter to "results.segmentation_mask" with "image".
#     condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
#     bg_image = np.zeros(image.shape, dtype=np.uint8)
#     bg_image[:] = BG_COLOR
#     annotated_image = np.where(condition, annotated_image, bg_image)
#     # Draw pose landmarks on the image.
#     mp_drawing.draw_landmarks(
#         annotated_image,
#         results.pose_landmarks,
#         mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#     cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
#     # Plot pose world landmarks.
#     mp_drawing.plot_landmarks(
#         results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# # For webcam input:
# cap = cv2.VideoCapture(0)
# with mp_pose.Pose(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as pose:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue

#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = pose.process(image)

#     # Draw the pose annotation on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     mp_drawing.draw_landmarks(
#         image,
#         results.pose_landmarks,
#         mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()