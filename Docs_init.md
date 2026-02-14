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








---------------------

- certain hiccups 


---------------------


  ```bash
  $ pip install dlib
Collecting dlib
  Downloading dlib-20.0.0.tar.gz (3.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.3/3.3 MB 1.1 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Building wheels for collected packages: dlib
  Building wheel for dlib (pyproject.toml) ... done
  Created wheel for dlib: filename=dlib-20.0.0-cp310-cp310-linux_x86_64.whl size=4102548 sha256=c0f7cc554c95417c470faabf1b71adb36d866ec09e6f4eeef5500a37e3ea42b6
  Stored in directory: /home/dhankar/.cache/pip/wheels/97/bc/4a/1f441cf62ce4c81ad4f83f298cef0e5ff3af0577ffb4cdff2f
Successfully built dlib
Installing collected packages: dlib
Successfully installed dlib-20.0.0
(env_overlander) dhankar@dhankar-1:~/.../ipWebCam$ 
```

