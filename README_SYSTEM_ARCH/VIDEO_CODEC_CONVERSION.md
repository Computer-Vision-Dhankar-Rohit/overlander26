# Video Codec Auto-Conversion Implementation

## Overview
Implemented automatic video codec conversion to handle incompatible codecs (AV1, VP9, HEVC) on systems without GPU acceleration. The solution uses ffmpeg to convert videos to H.264 format, which is natively supported by OpenCV.

## Files Created/Modified

### 1. NEW: `/src/util_video_converter.py`
A comprehensive video codec detection and conversion utility.

**Key Features:**
- Detects video codec using ffprobe
- Identifies unsupported codecs (AV1, VP9, HEVC/H.265)
- Converts to H.264 using ffmpeg with configurable presets
- Handles output path generation and duplicate prevention
- Main entry point: `util_convert_mp4(video_path)`

**Supported Codecs:**
- ✅ H.264 (AVC)
- ✅ MPEG-4
- ✅ MJPEG
- ❌ AV1 (unsupported)
- ❌ VP9 (unsupported)
- ❌ HEVC/H.265 (unsupported)

**Usage:**
```python
from util_video_converter import util_convert_mp4

# Automatically converts if needed, returns H.264 compatible path
safe_video_path = util_convert_mp4("/path/to/video.mp4")
```

### 2. MODIFIED: `/src/analysis/media_pipe.py`

**Changes:**
1. **Line 7**: Added imports
   ```python
   from util_video_converter import util_convert_mp4, VideoCodecConverter
   ```

2. **Lines 32-50**: Updated `init_pose_detector()` method
   - Changed from hardcoded relative path to dynamic path construction
   - Now uses `os.path.join()` to build proper path to DATA_DIR

3. **Lines 262-276**: Added video codec checking block in `pose_media_pipe_google_2()`
   ```python
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
   ```

## How It Works

### Detection Flow
1. When `pose_media_pipe_google_2()` is called with a local video file
2. Before `cv2.VideoCapture()`, the video codec is checked
3. If codec is unsupported → conversion triggered
4. Converted file is saved with `_h264.mp4` suffix
5. Converted path is used for video capture

### Conversion Process
```
Input: gym_1.mp4 (AV1 codec)
↓
ffmpeg -i gym_1.mp4 \
  -c:v libx264 -preset fast -crf 23 \
  -c:a aac gym_1_h264.mp4
↓
Output: gym_1_h264.mp4 (H.264 codec)
```

## Configuration Options

### Conversion Parameters
Edit `VideoCodecConverter.convert_to_h264()`:
- **preset**: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
  - Default: `fast` (good balance)
- **crf**: Quality 0-51 (lower = better)
  - Default: `23` (reasonable quality)

Example:
```python
VideoCodecConverter.convert_to_h264(
    input_video="gym_1.mp4",
    preset="medium",  # Slower encoding, better quality
    crf=18            # Higher quality
)
```

## Installation Requirements

### ffmpeg (Required)
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### ffprobe (Included with ffmpeg)
- Used to detect video codec
- Automatically included with ffmpeg installation

## Logging Output

### Successful Conversion
```
🔍 Checking video codec compatibility...
❌ Video uses unsupported codec: av1
Converting video: /path/to/gym_1.mp4
Output: /path/to/gym_1_h264.mp4
Codec: H.264 | Preset: fast | CRF: 23
✅ Video converted successfully: /path/to/gym_1_h264.mp4
📺 Using converted video: /path/to/gym_1_h264.mp4
📹 Video source: Local File (.MP4) - /path/to/gym_1_h264.mp4
```

### No Conversion Needed
```
🔍 Checking video codec compatibility...
✅ Video codec is compatible, no conversion needed
📹 Video source: Local File (.MP4) - /path/to/gym_1_h264.mp4
```

## Error Handling

### Missing ffmpeg
```
❌ Failed to prepare video for playback
💡 Ensure ffmpeg is installed: sudo apt install ffmpeg
```

### Missing Input File
```
❌ Input video not found: /path/to/missing.mp4
```

### Conversion Timeout
- Default timeout: 3600 seconds (1 hour)
- Automatically logged and handled

## Performance Considerations

- **First Run**: Video conversion takes time (depends on file size and preset)
- **Subsequent Runs**: Converted file is reused (checks for existence first)
- **Preset Impact**:
  - `fast`: ~50% faster, slightly lower quality
  - `medium`: Balanced (default approach)
  - `slow`: ~30% slower, better quality
  - Set in config parameters as needed

## Testing

### Unit Test
```bash
cd /home/dhankar/temp/26_02/git_over/overlander26/src
python3 util_video_converter.py /path/to/video.mp4
```

### Integration Test
```bash
python3 << 'EOF'
from util_video_converter import VideoCodecConverter

# Test on actual video file
converter = VideoCodecConverter()
codec = converter.get_video_codec("video.mp4")
needs_conversion = converter.needs_conversion("video.mp4")

print(f"Codec: {codec}")
print(f"Needs conversion: {needs_conversion}")
EOF
```

## Troubleshooting

### Issue: "ffmpeg not found"
**Solution:**
```bash
sudo apt install ffmpeg ffprobe
which ffmpeg  # Verify installation
```

### Issue: Conversion takes too long
**Solution:** Adjust preset in code (use `fast` instead of `medium`)

### Issue: Audio sync issues after conversion
**Solution:** Use `-af aresample=async=1` with audio codec

### Issue: Converted file won't play
**Solution:** Verify ffmpeg installation and try manual conversion:
```bash
ffmpeg -i input.mp4 -c:v libx264 -preset fast -crf 23 -c:a aac output_h264.mp4
```

## Future Enhancements

1. **Batch Conversion**: Pre-convert all videos in DATA_DIR
2. **Quality Presets**: Add high/medium/low quality profiles
3. **Format Support**: Extend to other formats (WebM, MKV)
4. **Parallel Conversion**: Use multiprocessing for multiple videos
5. **Conversion Cache**: Track converted files in metadata
