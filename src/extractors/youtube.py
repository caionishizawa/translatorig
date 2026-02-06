"""
YouTube video extractor using yt-dlp.
"""

import re
import subprocess
import json
from pathlib import Path
from typing import Optional

from .base import BaseExtractor
from ..utils.models import VideoData
from ..utils.file_handler import FileHandler


class YouTubeExtractor(BaseExtractor):
    """Extract videos from YouTube using yt-dlp."""

    # YouTube URL patterns
    YOUTUBE_PATTERNS = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([A-Za-z0-9_-]+)",
        r"(?:https?://)?(?:www\.)?youtu\.be/([A-Za-z0-9_-]+)",
        r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([A-Za-z0-9_-]+)",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([A-Za-z0-9_-]+)",
    ]

    def supports(self, source: str) -> bool:
        """Check if source is a YouTube URL."""
        return any(re.match(pattern, source) for pattern in self.YOUTUBE_PATTERNS)

    async def extract(
        self,
        source: str,
        max_quality: bool = True,
        extract_audio: bool = True,
    ) -> VideoData:
        """
        Extract video from YouTube URL.

        Args:
            source: YouTube URL
            max_quality: Download highest quality available (up to 4K)
            extract_audio: Extract audio to separate file

        Returns:
            VideoData object with paths and metadata
        """
        self._update_progress(0.0, "Starting YouTube extraction")

        # Generate output path
        video_path = FileHandler.get_temp_path(extension=".mp4", prefix="yt_")

        # Build yt-dlp command
        cmd = self._build_ytdlp_command(source, video_path, max_quality)

        self._update_progress(0.1, "Downloading video from YouTube")

        # Execute download
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Monitor download progress
            for line in iter(process.stdout.readline, ""):
                if "[download]" in line and "%" in line:
                    try:
                        progress_match = re.search(r"(\d+\.?\d*)%", line)
                        if progress_match:
                            pct = float(progress_match.group(1))
                            self._update_progress(0.1 + (pct / 100) * 0.6, f"Downloading: {pct:.1f}%")
                    except ValueError:
                        pass

            process.wait()

            if process.returncode != 0:
                raise RuntimeError("yt-dlp download failed")

        except Exception as e:
            raise RuntimeError(f"Failed to download YouTube video: {e}")

        self._update_progress(0.7, "Processing downloaded video")

        # Find the actual downloaded file
        actual_path = self._find_downloaded_file(video_path)

        if not actual_path or not actual_path.exists():
            raise RuntimeError(f"Downloaded file not found at {video_path}")

        self._update_progress(0.8, "Extracting video metadata")

        # Get video info
        info = self._get_video_info(actual_path)

        # Extract audio if requested
        audio_path = None
        if extract_audio:
            self._update_progress(0.9, "Extracting audio track")
            audio_path = self._extract_audio_from_video(actual_path)
        else:
            audio_path = actual_path

        self._update_progress(1.0, "Extraction complete")

        # Build VideoData
        video_data = self._parse_video_metadata(info, actual_path, audio_path)
        video_data.original_url = source

        return video_data

    def _build_ytdlp_command(
        self,
        url: str,
        output_path: Path,
        max_quality: bool,
    ) -> list:
        """Build yt-dlp command with appropriate options."""
        cmd = [
            "yt-dlp",
            "--no-warnings",
            "--no-playlist",
            "-o", str(output_path),
        ]

        # Quality settings for YouTube
        if max_quality:
            # Best quality up to 4K
            cmd.extend([
                "-f", "bestvideo[height<=2160][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=2160]+bestaudio/best[height<=2160]/best",
                "--merge-output-format", "mp4",
            ])
        else:
            cmd.extend(["-f", "best[height<=1080][ext=mp4]/best[height<=1080]/best"])

        # Video processing options
        cmd.extend([
            "--embed-metadata",
            "--embed-thumbnail",
            "--no-check-certificates",
        ])

        cmd.append(url)

        return cmd

    def _find_downloaded_file(self, expected_path: Path) -> Optional[Path]:
        """Find the actual downloaded file."""
        if expected_path.exists():
            return expected_path

        for ext in [".mp4", ".mkv", ".webm", ".mov"]:
            alt_path = expected_path.with_suffix(ext)
            if alt_path.exists():
                return alt_path

        parent = expected_path.parent
        stem = expected_path.stem

        for file in parent.glob(f"{stem}*"):
            if file.is_file() and FileHandler.is_video_file(file):
                return file

        return None

    async def get_video_info(self, url: str) -> dict:
        """Get video information without downloading."""
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-warnings",
            "--no-download",
            url,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to get video info: {e}")
