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


### Codec changes 

---------------------

- codec changes with -- ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 


---------------------

```bash
$ ffmpeg -i /home/dhankar/temp/26_02/git_up/data_dir/pose_detected/detected_pose/gym_1.mp4 \
       -c:v libx264 -preset fast -crf 23 -c:a aac \
       /home/dhankar/temp/26_02/git_up/data_dir/pose_detected/init_video/gym_1_h264.mp4
ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers
  built with gcc 11 (Ubuntu 11.2.0-19ubuntu1)
  configuration: --prefix=/usr --extra-version=0ubuntu0.22.04.1 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libdav1d --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librabbitmq --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libsrt --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzimg --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --enable-pocketsphinx --enable-librsvg --enable-libmfx --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r --enable-libx264 --enable-shared
  libavutil      56. 70.100 / 56. 70.100
  libavcodec     58.134.100 / 58.134.100
  libavformat    58. 76.100 / 58. 76.100
  libavdevice    58. 13.100 / 58. 13.100
  libavfilter     7.110.100 /  7.110.100
  libswscale      5.  9.100 /  5.  9.100
  libswresample   3.  9.100 /  3.  9.100
  libpostproc    55.  9.100 / 55.  9.100
[libdav1d @ 0x564b703f33c0] libdav1d 0.9.2
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '/home/dhankar/temp/26_02/git_up/data_dir/pose_detected/detected_pose/gym_1.mp4':
  Metadata:
    major_brand     : isom
    minor_version   : 512
    compatible_brands: isomav01iso2mp41
    encoder         : Lavf58.76.100
  Duration: 00:00:15.09, start: 0.000000, bitrate: 545 kb/s
  Stream #0:0(und): Video: av1 (Main) (av01 / 0x31307661), yuv420p(tv, bt709), 720x1280 [SAR 1:1 DAR 9:16], 415 kb/s, 25 fps, 25 tbr, 12800 tbn, 12800 tbc (default)
    Metadata:
      handler_name    : ISO Media file produced by Google Inc.
      vendor_id       : [0][0][0][0]
  Stream #0:1(und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 128 kb/s (default)
    Metadata:
      handler_name    : ISO Media file produced by Google Inc.
      vendor_id       : [0][0][0][0]
[libdav1d @ 0x564b7043f600] libdav1d 0.9.2
Stream mapping:
  Stream #0:0 -> #0:0 (av1 (libdav1d) -> h264 (libx264))
  Stream #0:1 -> #0:1 (aac (native) -> aac (native))
Press [q] to stop, [?] for help
[libx264 @ 0x564b703f6ec0] using SAR=1/1
[libx264 @ 0x564b703f6ec0] using cpu capabilities: MMX2 SSE2Fast SSSE3 SSE4.2 AVX FMA3 BMI2 AVX2
[libx264 @ 0x564b703f6ec0] profile High, level 3.1, 4:2:0, 8-bit
[libx264 @ 0x564b703f6ec0] 264 - core 163 r3060 5db6aa6 - H.264/MPEG-4 AVC codec - Copyleft 2003-2021 - http://www.videolan.org/x264.html - options: cabac=1 ref=2 deblock=1:0:0 analyse=0x3:0x113 me=hex subme=6 psy=1 psy_rd=1.00:0.00 mixed_ref=1 me_range=16 chroma_me=1 trellis=1 8x8dct=1 cqm=0 deadzone=21,11 fast_pskip=1 chroma_qp_offset=-2 threads=12 lookahead_threads=2 sliced_threads=0 nr=0 decimate=1 interlaced=0 bluray_compat=0 constrained_intra=0 bframes=3 b_pyramid=2 b_adapt=1 b_bias=0 direct=1 weightb=1 open_gop=0 weightp=1 keyint=250 keyint_min=25 scenecut=40 intra_refresh=0 rc_lookahead=30 rc=crf mbtree=1 crf=23.0 qcomp=0.60 qpmin=0 qpmax=69 qpstep=4 ip_ratio=1.40 aq=1:1.00
Output #0, mp4, to '/home/dhankar/temp/26_02/git_up/data_dir/pose_detected/init_video/gym_1_h264.mp4':
  Metadata:
    major_brand     : isom
    minor_version   : 512
    compatible_brands: isomav01iso2mp41
    encoder         : Lavf58.76.100
  Stream #0:0(und): Video: h264 (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 720x1280 [SAR 1:1 DAR 9:16], q=2-31, 25 fps, 12800 tbn (default)
    Metadata:
      handler_name    : ISO Media file produced by Google Inc.
      vendor_id       : [0][0][0][0]
      encoder         : Lavc58.134.100 libx264
    Side data:
      cpb: bitrate max/min/avg: 0/0/0 buffer size: 0 vbv_delay: N/A
  Stream #0:1(und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 128 kb/s (default)
    Metadata:
      handler_name    : ISO Media file produced by Google Inc.
      vendor_id       : [0][0][0][0]
      encoder         : Lavc58.134.100 aac
frame=    1 fps=0.0 q=0.0 size=       0kB time=00:00:00.00 bitrate=N/A speed=   frame=  122 fps=0.0 q=28.0 size=     256kB time=00:00:04.96 bitrate= 422.1kbits/frame=  215 fps=193 q=28.0 size=     768kB time=00:00:08.68 bitrate= 724.5kbits/frame=  316 fps=196 q=28.0 size=    1280kB time=00:00:12.72 bitrate= 824.1kbits/frame=  374 fps=194 q=-1.0 Lsize=    1803kB time=00:00:15.06 bitrate= 980.3kbits/s speed=7.83x    
video:1552kB audio:238kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.767374%
[libx264 @ 0x564b703f6ec0] frame I:3     Avg QP:17.16  size: 63477
[libx264 @ 0x564b703f6ec0] frame P:95    Avg QP:21.35  size:  8449
[libx264 @ 0x564b703f6ec0] frame B:276   Avg QP:25.08  size:  2158
[libx264 @ 0x564b703f6ec0] consecutive B-frames:  0.8%  2.1%  0.8% 96.3%
[libx264 @ 0x564b703f6ec0] mb I  I16..4:  8.0% 71.6% 20.4%
[libx264 @ 0x564b703f6ec0] mb P  I16..4:  1.4%  4.4%  0.4%  P16..4: 32.6%  9.1%  3.6%  0.0%  0.0%    skip:48.4%
[libx264 @ 0x564b703f6ec0] mb B  I16..4:  0.8%  1.2%  0.1%  B16..8: 19.6%  1.9%  0.1%  direct: 3.4%  skip:73.1%  L0:46.2% L1:50.2% BI: 3.6%
[libx264 @ 0x564b703f6ec0] 8x8 transform intra:65.4% inter:81.8%
[libx264 @ 0x564b703f6ec0] coded y,uvDC,uvAC intra: 47.4% 62.0% 28.0% inter: 4.2% 6.6% 0.2%
[libx264 @ 0x564b703f6ec0] i16 v,h,dc,p: 30% 43% 10% 18%
[libx264 @ 0x564b703f6ec0] i8 v,h,dc,ddl,ddr,vr,hd,vl,hu: 27% 26% 16%  4%  4%  5%  6%  5%  5%
[libx264 @ 0x564b703f6ec0] i4 v,h,dc,ddl,ddr,vr,hd,vl,hu: 35% 23% 10%  4%  6%  7%  6%  6%  3%
[libx264 @ 0x564b703f6ec0] i8c dc,h,v,p: 42% 25% 21% 12%
[libx264 @ 0x564b703f6ec0] Weighted P-Frames: Y:0.0% UV:0.0%
[libx264 @ 0x564b703f6ec0] ref P L0: 65.6% 34.4%
[libx264 @ 0x564b703f6ec0] ref B L0: 82.6% 17.4%
[libx264 @ 0x564b703f6ec0] ref B L1: 94.5%  5.5%
[libx264 @ 0x564b703f6ec0] kb/s:849.52
[aac @ 0x564b7040b7c0] Qavg: 601.070
```

