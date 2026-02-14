**Global Variables:**
- `CLASSES`: List of 80 COCO dataset class names
- `COLORS`: List of RGB color values for visualization
- `detr_image_processor_detr_resnet101`: DETR image processor (ResNet-101)
- `model_detr_resnet101`: DETR model (ResNet-101)
- `detr_image_processor_detr_resnet101_dc5`: DETR image processor (ResNet-101-DC5)
- `model_detr_resnet101_dc5`: DETR model (ResNet-101-DC5)
- `pipeline_rtdetr_v2`: RT-DETR v2 pipeline

### Class: `GetFramesFromVids`

#### Method: `get_static_vids_local_list(self)`
- **Input Parameters:** None
  
- **Processing:**
  - Gets initial video directory (hardcoded to `"../data_dir/init_vid_dir/"`)
  - Walks through directory to collect all video file paths using `os.walk()`
  - Logs the list of video files
  
- **Output Parameters:**
  - Returns: `ls_video_files_uploads` (list) - List of video file paths

#### Method: `get_frame_from_video(self)`
- **Input Parameters:** None
  
- **Processing:**
  - Defines frame extraction list: `[4,11,17,25,30,37,45,55,66,77,88,100,110]`
  - Gets list of video files using `get_static_vids_local_list()`
  - Iterates through each video file
  - Opens video using `cv2.VideoCapture()`
  - Extracts specific frames based on frame numbers in list
  - Cleans video filename by removing timestamps and extensions
  - Checks image size (if >= 6000000 pixels, uses .jpg format)
  - Saves frames as JPEG images to `"../data_dir/out_vid_frames_dir/"`
  - Frame filename format: `vid_short_name + "_frame_" + count + "__.jpg"`
  - Logs frame writing details and errors
  
- **Output Parameters:**
  - Returns: None (saves extracted frames to disk)
