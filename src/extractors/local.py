"""
Local file extractor for video files.
"""

import shutil
from pathlib import Path
from typing import Optional

from .base import BaseExtractor
from ..utils.models import VideoData
from ..utils.file_handler import FileHandler


class LocalExtractor(BaseExtractor):
    """Extract/process local video files."""

    def supports(self, source: str) -> bool:
        """Check if source is a local file path."""
        path = Path(source)
        return path.exists() and FileHandler.is_video_file(path)

    async def extract(
        self,
        source: str,
        max_quality: bool = True,
        extract_audio: bool = True,
    ) -> VideoData:
        """
        Process a local video file.

        Args:
            source: Path to local video file
            max_quality: Not applicable for local files
            extract_audio: Extract audio to separate file

        Returns:
            VideoData object with paths and metadata
        """
        self._update_progress(0.0, "Processing local video file")

        source_path = Path(source)

        # Validate file exists
        if not source_path.exists():
            raise FileNotFoundError(f"Video file not found: {source_path}")

        if not FileHandler.is_video_file(source_path):
            raise ValueError(f"Unsupported video format: {source_path.suffix}")

        self._update_progress(0.2, "Copying video to temp directory")

        # Copy to temp directory for processing
        video_path = FileHandler.get_temp_path(
            extension=source_path.suffix,
            prefix="local_"
        )
        shutil.copy2(source_path, video_path)

        self._update_progress(0.5, "Extracting video metadata")

        # Get video info
        info = self._get_video_info(video_path)

        # Extract audio if requested
        audio_path = None
        if extract_audio:
            self._update_progress(0.7, "Extracting audio track")
            audio_path = self._extract_audio_from_video(video_path)
        else:
            audio_path = video_path

        self._update_progress(1.0, "Processing complete")

        # Build VideoData
        video_data = self._parse_video_metadata(info, video_path, audio_path)
        video_data.original_url = str(source_path.absolute())

        return video_data

    async def validate(self, source: str) -> dict:
        """
        Validate a local video file and return its properties.

        Args:
            source: Path to local video file

        Returns:
            Dictionary with video properties
        """
        source_path = Path(source)

        if not source_path.exists():
            return {"valid": False, "error": "File not found"}

        if not FileHandler.is_video_file(source_path):
            return {"valid": False, "error": "Unsupported format"}

        try:
            info = self._get_video_info(source_path)
            video_stream = None
            audio_stream = None

            for stream in info.get("streams", []):
                if stream.get("codec_type") == "video" and video_stream is None:
                    video_stream = stream
                elif stream.get("codec_type") == "audio" and audio_stream is None:
                    audio_stream = stream

            format_info = info.get("format", {})

            return {
                "valid": True,
                "path": str(source_path.absolute()),
                "size_mb": FileHandler.get_file_size_mb(source_path),
                "duration": float(format_info.get("duration", 0)),
                "resolution": f"{video_stream.get('width', 0)}x{video_stream.get('height', 0)}" if video_stream else "unknown",
                "fps": self._parse_fps(video_stream.get("r_frame_rate", "30/1")) if video_stream else 0,
                "video_codec": video_stream.get("codec_name") if video_stream else None,
                "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
                "has_audio": audio_stream is not None,
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}
