"""
Video Codec Conversion Utility

Handles conversion of MP4 videos to H.264 codec for OpenCV compatibility.
Some systems (especially without GPU acceleration) struggle with:
- AV1 codec
- VP9 codec
- HEVC/H.265 codec

This utility detects unsupported codecs and auto-converts them using ffmpeg.
"""

import os
import subprocess
from pathlib import Path

## TODO -- /home/dhankar/temp/26_02/git_over/overlander26/src/util_logger.py
from datetime import datetime
import os, sys
sys.path.append('..')
from util_logger import setup_logger
logger = setup_logger(module_name=str(__name__))

class VideoCodecConverter:
    """
    Detects video codec and converts unsupported formats to H.264.
    
    Supported codecs for OpenCV (CPU):
    - H.264 (AVC)
    - MPEG-4
    
    Unsupported (requires GPU or special libraries):
    - AV1
    - VP9
    - HEVC/H.265
    """
    
    # Codec definitions
    SUPPORTED_CODECS = ['h264', 'mpeg4', 'mjpeg']
    UNSUPPORTED_CODECS = ['av1', 'vp9', 'hevc', 'h265']
    
    @staticmethod
    def check_ffmpeg_installed():
        """
        Check if ffmpeg is installed on the system.
        
        Returns:
            bool: True if ffmpeg is available, False otherwise
        """
        try:
            result = subprocess.run(['which', 'ffmpeg'], 
                                  capture_output=True, 
                                  timeout=5)
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Error checking ffmpeg: {e}")
            return False
    
    @staticmethod
    def get_video_codec(video_path):
        """
        Detect video codec using ffprobe.
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            str: Codec name (lowercase) or 'unknown' if detection fails
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name',
                '-of', 'default=noprint_wrappers=1:nokey=1:noinfer_type=1',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            codec = result.stdout.strip().lower()
            logger.debug(f"Detected codec for {video_path}: {codec}")
            return codec
        except Exception as e:
            logger.warning(f"Error detecting codec for {video_path}: {e}")
            return 'unknown'
    
    @staticmethod
    def needs_conversion(video_path):
        """
        Check if video needs conversion to H.264.
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            bool: True if conversion is needed, False otherwise
        """
        codec = VideoCodecConverter.get_video_codec(video_path)
        
        # If ffprobe returned empty codec, the file is likely corrupted/incomplete
        if not codec or codec == 'unknown':
            logger.warning(f"⚠️ Cannot detect video codec (file may be corrupted/incomplete): {video_path}")
            logger.info(f"Attempting H.264 conversion to recover file integrity...")
            return True
        
        # If codec is already H.264 and file has h264 in name, no conversion needed
        if codec == 'h264' and '_h264' in video_path.lower():
            return False
        
        # Check if codec is unsupported
        if codec in VideoCodecConverter.UNSUPPORTED_CODECS:
            logger.info(f"Video uses unsupported codec: {codec}")
            return True
        
        return False
    
    @staticmethod
    def convert_to_h264(input_video, output_video=None, preset='fast', crf=23):
        """
        Convert video to H.264 codec using ffmpeg.
        
        Args:
            input_video (str): Path to input video file
            output_video (str, optional): Path to output video file. 
                                         If None, creates with _h264 suffix
            preset (str): ffmpeg preset (ultrafast, superfast, veryfast, faster, 
                         fast, medium, slow, slower, veryslow)
            crf (int): Quality (0-51, lower is better, 23 is default)
            
        Returns:
            str: Path to output video file if successful, None if failed
        """
        if not os.path.exists(input_video):
            logger.error(f"Input video not found: {input_video}")
            return None
        
        if not VideoCodecConverter.check_ffmpeg_installed():
            logger.error("ffmpeg is not installed. Cannot convert video.")
            logger.info("Install ffmpeg with: sudo apt install ffmpeg")
            return None
        
        # Generate output path if not provided
        if output_video is None:
            base_path = os.path.splitext(input_video)[0]
            output_video = f"{base_path}_h264.mp4"
        
        # Skip if output already exists
        if os.path.exists(output_video):
            logger.info(f"Output file already exists: {output_video}")
            return output_video
        
        logger.info(f"Converting video: {input_video}")
        logger.info(f"Output: {output_video}")
        logger.info(f"Codec: H.264 | Preset: {preset} | CRF: {crf}")
        
        try:
            cmd = [
                'ffmpeg',
                '-i', input_video,
                '-c:v', 'libx264',           # Video codec: H.264
                '-preset', preset,           # Encoding speed/quality tradeoff
                '-crf', str(crf),           # Quality (0-51, lower is better)
                '-c:a', 'aac',              # Audio codec
                '-b:a', '128k',             # Audio bitrate
                '-y',                       # Overwrite output file without asking
                output_video
            ]
            
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                logger.info(f"✅ Video converted successfully: {output_video}")
                return output_video
            else:
                logger.error(f"❌ Conversion failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Conversion timed out (1 hour limit)")
            return None
        except Exception as e:
            logger.error(f"Conversion error: {e}")
            return None
    
    @staticmethod
    def get_safe_video_path(video_path):
        """
        Get a safe video path, converting if necessary.
        
        This is the main entry point for the converter. It checks if conversion
        is needed and performs it automatically if required.
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            str: Path to H.264 compatible video (either original if already 
                 H.264, or converted file)
        """
        if not os.path.exists(video_path):
            logger.error(f"❌ Video file not found: {video_path}")
            return None
        
        # Check file size - should be at least 1MB for a valid video
        file_size = os.path.getsize(video_path)
        if file_size < 1024 * 1024:  # Less than 1MB
            logger.error(f"❌ Video file too small ({file_size} bytes), likely incomplete download: {video_path}")
            return None
        
        logger.debug(f"Checking video compatibility: {video_path} ({file_size / (1024*1024):.1f} MB)")
        
        if VideoCodecConverter.needs_conversion(video_path):
            logger.warning(f"📺 Video needs H.264 conversion: {video_path}")
            converted_path = VideoCodecConverter.convert_to_h264(video_path)
            return converted_path
        
        logger.debug(f"✅ Video is compatible: {video_path}")
        return video_path


def util_convert_mp4(video_path):
    """
    Utility function to ensure video is H.264 compatible.
    
    This is a wrapper function for easy use in media_pipe.py
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        str: Path to H.264 compatible video file, or None if conversion failed
    """
    converter = VideoCodecConverter()
    return converter.get_safe_video_path(video_path)


if __name__ == "__main__":
    # Test the converter
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python util_video_converter.py <video_path>")
        sys.exit(1)
    
    video_file = sys.argv[1]
    print(f"\nTesting video converter with: {video_file}")
    
    converter = VideoCodecConverter()
    codec = converter.get_video_codec(video_file)
    needs_conv = converter.needs_conversion(video_file)
    
    print(f"Codec: {codec}")
    print(f"Needs conversion: {needs_conv}")
    
    if needs_conv:
        result = converter.convert_to_h264(video_file)
        if result:
            print(f"Converted successfully: {result}")
        else:
            print("Conversion failed")
