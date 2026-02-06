"""
FFmpeg-based video rendering engine.
"""

from pathlib import Path
from typing import Optional, Callable, Literal, List
import subprocess
import json
import re
import tempfile

from .quality_presets import QUALITY_PRESETS, SUBTITLE_STYLES, get_preset
from ..utils.models import RenderResult
from ..utils.file_handler import FileHandler
from ..config import settings


class FFmpegRenderer:
    """
    High-quality video rendering using FFmpeg.

    Features:
    - Multiple quality presets (4K, 1080p, 720p)
    - H.264/H.265/AV1 encoding
    - Subtitle embedding (soft/hardcoded)
    - Multiple audio tracks
    - HDR preservation
    - iPhone optimization
    """

    def __init__(self):
        self.progress_callback: Optional[Callable] = None
        self._verify_ffmpeg()

    def _verify_ffmpeg(self):
        """Verify FFmpeg is installed and get version."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
            )
            # Extract version from first line
            version_line = result.stdout.split("\n")[0]
            self.ffmpeg_version = version_line
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg.")

    def set_progress_callback(self, callback: Callable) -> None:
        """Set callback for progress updates."""
        self.progress_callback = callback

    def _update_progress(self, progress: float, message: str = "") -> None:
        """Update progress via callback if set."""
        if self.progress_callback:
            self.progress_callback(progress, message)

    async def render(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Optional[Path] = None,
        subtitles: Optional[str] = None,
        subtitle_style: Literal["soft", "hardcoded"] = "soft",
        resolution: Literal["4k", "1080p", "720p", "original"] = "original",
        preset: Optional[str] = None,
        original_audio: Optional[Path] = None,
        output_format: Literal["mp4", "mov", "mkv"] = "mp4",
    ) -> RenderResult:
        """
        Render final video with all components.

        Args:
            video_path: Path to video (with lip-sync applied)
            audio_path: Path to dubbed audio
            output_path: Output file path
            subtitles: SRT subtitle content (optional)
            subtitle_style: How to embed subtitles
            resolution: Output resolution
            preset: Quality preset name
            original_audio: Path to original audio (for dual audio track)
            output_format: Output container format

        Returns:
            RenderResult with final video details
        """
        self._update_progress(0.0, "Preparing render")

        # Determine output path
        if output_path is None:
            output_path = FileHandler.get_temp_path(
                extension=f".{output_format}",
                prefix="final_"
            )

        # Get video info
        video_info = self._get_video_info(video_path)
        original_resolution = self._get_resolution(video_info)
        original_fps = self._get_fps(video_info)
        duration = self._get_duration(video_info)

        # Determine quality preset
        if preset:
            quality_preset = get_preset(preset)
        else:
            quality_preset = self._get_preset_for_resolution(resolution, original_resolution)

        self._update_progress(0.1, "Building FFmpeg command")

        # Build FFmpeg command
        cmd = await self._build_ffmpeg_command(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path,
            subtitles=subtitles,
            subtitle_style=subtitle_style,
            quality_preset=quality_preset,
            original_resolution=original_resolution,
            original_fps=original_fps,
            original_audio=original_audio,
            output_format=output_format,
        )

        self._update_progress(0.2, "Starting render")

        # Execute FFmpeg
        await self._execute_ffmpeg(cmd, duration)

        self._update_progress(0.95, "Finalizing")

        # Get output info
        output_info = self._get_video_info(output_path)
        output_resolution = self._get_resolution(output_info)
        output_fps = self._get_fps(output_info)
        output_duration = self._get_duration(output_info)
        file_size = output_path.stat().st_size

        self._update_progress(1.0, "Render complete")

        return RenderResult(
            path=output_path,
            duration=output_duration,
            resolution=output_resolution,
            fps=output_fps,
            file_size=file_size,
            video_codec=quality_preset["video_codec"].replace("lib", ""),
            audio_codec=quality_preset["audio_codec"],
            has_subtitles=subtitles is not None,
        )

    async def _build_ffmpeg_command(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        subtitles: Optional[str],
        subtitle_style: str,
        quality_preset: dict,
        original_resolution: tuple[int, int],
        original_fps: float,
        original_audio: Optional[Path],
        output_format: str,
    ) -> List[str]:
        """Build the FFmpeg command."""
        cmd = ["ffmpeg", "-y"]

        # Input files
        cmd.extend(["-i", str(video_path)])
        cmd.extend(["-i", str(audio_path)])

        if original_audio:
            cmd.extend(["-i", str(original_audio)])

        # Subtitle file if hardcoded
        subtitle_file = None
        if subtitles and subtitle_style == "hardcoded":
            subtitle_file = await self._create_subtitle_file(subtitles)
            cmd.extend(["-i", str(subtitle_file)])

        # Video encoding
        target_resolution = quality_preset.get("resolution")
        if target_resolution and target_resolution != original_resolution:
            # Scale video
            w, h = target_resolution
            cmd.extend(["-vf", f"scale={w}:{h}:flags=lanczos"])

        cmd.extend([
            "-c:v", quality_preset["video_codec"],
            "-crf", str(quality_preset["video_crf"]),
            "-preset", quality_preset["video_preset"],
        ])

        if quality_preset.get("video_profile"):
            cmd.extend(["-profile:v", quality_preset["video_profile"]])

        # Add video params
        cmd.extend(quality_preset.get("video_params", []))

        # FPS control
        target_fps = quality_preset.get("target_fps")
        if target_fps and target_fps != original_fps:
            cmd.extend(["-r", str(target_fps)])

        # Audio encoding
        cmd.extend([
            "-c:a", quality_preset["audio_codec"],
            "-b:a", quality_preset["audio_bitrate"],
            "-ar", str(quality_preset["audio_sample_rate"]),
            "-ac", str(quality_preset.get("audio_channels", 2)),
        ])

        # Stream mapping
        cmd.extend(["-map", "0:v:0"])  # Video from first input
        cmd.extend(["-map", "1:a:0"])  # Audio from second input (dubbed)

        if original_audio:
            cmd.extend(["-map", "2:a:0"])  # Original audio as second track
            cmd.extend(["-metadata:s:a:0", "title=Dubbed"])
            cmd.extend(["-metadata:s:a:1", "title=Original"])

        # Subtitles
        if subtitles:
            if subtitle_style == "soft" and output_format in ["mp4", "mkv"]:
                # Create subtitle file
                if not subtitle_file:
                    subtitle_file = await self._create_subtitle_file(subtitles)

                cmd.extend(["-i", str(subtitle_file)])
                cmd.extend(["-c:s", "mov_text" if output_format == "mp4" else "srt"])
                cmd.extend(["-map", f"{3 if original_audio else 2}:0"])

            elif subtitle_style == "hardcoded":
                # Burn subtitles into video
                vf_idx = cmd.index("-vf") if "-vf" in cmd else -1
                if vf_idx >= 0:
                    current_vf = cmd[vf_idx + 1]
                    cmd[vf_idx + 1] = f"{current_vf},subtitles={subtitle_file}"
                else:
                    cmd.extend(["-vf", f"subtitles={subtitle_file}"])

        # Output file
        cmd.append(str(output_path))

        return cmd

    async def _execute_ffmpeg(self, cmd: List[str], duration: float):
        """Execute FFmpeg with progress monitoring."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True,
        )

        # Parse FFmpeg output for progress
        for line in iter(process.stdout.readline, ""):
            # Look for time= in output
            time_match = re.search(r"time=(\d+):(\d+):(\d+\.?\d*)", line)
            if time_match:
                hours = int(time_match.group(1))
                minutes = int(time_match.group(2))
                seconds = float(time_match.group(3))
                current_time = hours * 3600 + minutes * 60 + seconds

                progress = 0.2 + (current_time / duration) * 0.7
                self._update_progress(min(progress, 0.9), f"Rendering: {current_time:.1f}s / {duration:.1f}s")

        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg failed with return code {process.returncode}")

    async def _create_subtitle_file(self, srt_content: str) -> Path:
        """Create a temporary subtitle file."""
        subtitle_path = FileHandler.get_temp_path(extension=".srt", prefix="subs_")
        with open(subtitle_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        return subtitle_path

    def _get_video_info(self, path: Path) -> dict:
        """Get video metadata using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def _get_resolution(self, info: dict) -> tuple[int, int]:
        """Extract resolution from video info."""
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                return (
                    int(stream.get("width", 0)),
                    int(stream.get("height", 0)),
                )
        return (0, 0)

    def _get_fps(self, info: dict) -> float:
        """Extract FPS from video info."""
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                fps_str = stream.get("r_frame_rate", "30/1")
                try:
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        return float(num) / float(den)
                    return float(fps_str)
                except (ValueError, ZeroDivisionError):
                    return 30.0
        return 30.0

    def _get_duration(self, info: dict) -> float:
        """Extract duration from video info."""
        return float(info.get("format", {}).get("duration", 0))

    def _get_preset_for_resolution(
        self,
        target_resolution: str,
        original_resolution: tuple[int, int],
    ) -> dict:
        """Get appropriate preset for target resolution."""
        if target_resolution == "original":
            # Choose preset based on original resolution
            width = original_resolution[0]
            if width >= 3840:
                return QUALITY_PRESETS["4k_balanced"]
            elif width >= 1920:
                return QUALITY_PRESETS["1080p_balanced"]
            else:
                return QUALITY_PRESETS["720p_fast"]

        preset_map = {
            "4k": "4k_balanced",
            "1080p": "1080p_balanced",
            "720p": "720p_fast",
        }

        return QUALITY_PRESETS.get(preset_map.get(target_resolution, "1080p_balanced"))

    async def extract_audio(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        codec: str = "aac",
        bitrate: str = "256k",
    ) -> Path:
        """Extract audio from video file."""
        if output_path is None:
            output_path = FileHandler.get_temp_path(extension=".m4a", prefix="audio_")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", codec,
            "-b:a", bitrate,
            str(output_path),
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        return output_path

    async def merge_audio_video(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Simple merge of video and audio without re-encoding."""
        if output_path is None:
            output_path = FileHandler.get_temp_path(extension=".mp4", prefix="merged_")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_path),
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
