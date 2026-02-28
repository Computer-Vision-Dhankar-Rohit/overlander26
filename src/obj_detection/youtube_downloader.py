import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from util_logger import setup_logger
logger = setup_logger(module_name=str(__name__))

# Project root: overlander26/
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "DATA_DIR"
VIDEOS_DIR   = DATA_DIR / "youtube_videos"

# Auto-create on import
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


class YouTubeDownloader:
    """
    Download YouTube videos as MP4 to DATA_DIR/youtube_videos/
    Uses yt_dlp. Requires: pip install yt-dlp ffmpeg
    """

    @staticmethod
    def download(youtube_url: str, output_filename: str = "%(title)s.%(ext)s"):
        """
        Download a YouTube video as best-quality MP4.

        Args:
            youtube_url:     Full YouTube URL or Shorts URL
            output_filename: yt_dlp output template (default: video title)

        Output saved to: DATA_DIR/youtube_videos/
        """
        from yt_dlp import YoutubeDL

        logger.info(f"Starting download: {youtube_url}")
        logger.debug("--- Downloading %s output_dir %s", youtube_url, VIDEOS_DIR)

        ydl_opts = {
            # Best MP4 video + best audio
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',

            # Save location
            'outtmpl': str(VIDEOS_DIR / output_filename),

            # Merge video + audio → MP4
            'merge_output_format': 'mp4',

            # Use ffmpeg for post-processing
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],

            'noplaylist': True,
            'ignoreerrors': False,
            'quiet': False,
        }

        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        logger.info(f"Download complete → {VIDEOS_DIR}")
        logger.debug("--- Download complete %s", VIDEOS_DIR)

    @staticmethod
    def list_downloaded():
        """List all MP4 files currently in DATA_DIR/youtube_videos/"""
        mp4_files = sorted(VIDEOS_DIR.glob("*.mp4"))
        if not mp4_files:
            logger.debug("--- No MP4 files found in %s", VIDEOS_DIR)
            return []
        logger.debug("--- Downloaded videos in %s", VIDEOS_DIR)
        for f in mp4_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.debug("--- %s size_mb %s", f.name, size_mb)
        return mp4_files


if __name__ == "__main__":
    # Example usage — replace URL with any YouTube video
    yt_url = "https://www.youtube.com/shorts/qlPfknL2P9A"
    YouTubeDownloader.download(youtube_url=yt_url)
    YouTubeDownloader.list_downloaded()
