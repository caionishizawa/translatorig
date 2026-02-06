"""
Base extractor class for video sources.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Callable
import subprocess
import json

from ..utils.models import VideoData
from ..utils.file_handler import FileHandler
from ..config import settings


class BaseExtractor(ABC):
    """Abstract base class for video extractors."""

    def __init__(self):
        self.progress_callback: Optional[Callable] = None

    @abstractmethod
    async def extract(
        self,
        source: str,
        max_quality: bool = True,
        extract_audio: bool = True,
    ) -> VideoData:
        """
        Extract video from source.

        Args:
            source: Source URL or path
            max_quality: Download highest quality available
            extract_audio: Extract audio to separate file

        Returns:
            VideoData object with paths and metadata
        """
        pass

    @abstractmethod
    def supports(self, source: str) -> bool:
        """Check if this extractor supports the given source."""
        pass

    def set_progress_callback(self, callback: Callable) -> None:
        """Set callback for progress updates."""
        self.progress_callback = callback

    def _update_progress(self, progress: float, message: str = "") -> None:
        """Update progress via callback if set."""
        if self.progress_callback:
            self.progress_callback(progress, message)

    def _get_video_info(self, video_path: Path) -> dict:
        """Get video metadata using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to get video info: {e}")

    def _extract_audio_from_video(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        codec: str = "aac",
        bitrate: str = "256k",
    ) -> Path:
        """Extract audio from video file using ffmpeg."""
        if output_path is None:
            output_path = FileHandler.get_temp_path(extension=".m4a", prefix="audio_")

        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", codec if codec != "aac" else "aac",
            "-b:a", bitrate,
            "-y",  # Overwrite output
            str(output_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}")

    def _parse_video_metadata(self, info: dict, video_path: Path, audio_path: Path) -> VideoData:
        """Parse ffprobe output into VideoData."""
        video_stream = None
        audio_stream = None

        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video" and video_stream is None:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and audio_stream is None:
                audio_stream = stream

        if not video_stream:
            raise ValueError("No video stream found in file")

        format_info = info.get("format", {})

        return VideoData(
            video_path=video_path,
            audio_path=audio_path,
            duration=float(format_info.get("duration", 0)),
            resolution=(
                int(video_stream.get("width", 0)),
                int(video_stream.get("height", 0)),
            ),
            fps=self._parse_fps(video_stream.get("r_frame_rate", "30/1")),
            codec=video_stream.get("codec_name", "unknown"),
            bitrate=int(format_info.get("bit_rate", 0)) if format_info.get("bit_rate") else None,
            has_audio=audio_stream is not None,
            metadata={
                "format": format_info.get("format_name"),
                "video_codec": video_stream.get("codec_name"),
                "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
                "pixel_format": video_stream.get("pix_fmt"),
            },
        )

    @staticmethod
    def _parse_fps(fps_string: str) -> float:
        """Parse FPS from ffprobe format (e.g., '30/1' or '29.97')."""
        try:
            if "/" in fps_string:
                num, den = fps_string.split("/")
                return float(num) / float(den)
            return float(fps_string)
        except (ValueError, ZeroDivisionError):
            return 30.0
