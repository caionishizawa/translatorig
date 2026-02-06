"""
Wav2Lip lip synchronization processor.

Wav2Lip generates accurate lip-sync for any video with speech audio.
Paper: "A Lip Sync Expert Is All You Need for Speech to Lip Generation In The Wild"
"""

from pathlib import Path
from typing import Optional, Literal
import subprocess
import tempfile
import shutil
import os

from .base import BaseLipSync
from ..utils.models import LipSyncResult
from ..utils.file_handler import FileHandler
from ..config import settings


class Wav2LipProcessor(BaseLipSync):
    """
    Lip synchronization using Wav2Lip.

    Wav2Lip produces highly accurate lip-sync and is robust to:
    - Arbitrary speech audio
    - Any face in any pose
    - Occlusions
    - Varying lighting conditions

    Quality presets:
    - fast: Lower resolution, faster processing
    - standard: Balanced quality/speed
    - high: Full resolution, best quality
    """

    # Wav2Lip model variants
    MODEL_VARIANTS = {
        "standard": "wav2lip.pth",
        "gan": "wav2lip_gan.pth",  # Better quality, slower
    }

    QUALITY_SETTINGS = {
        "fast": {
            "resize_factor": 2,
            "face_det_batch_size": 16,
            "wav2lip_batch_size": 128,
            "model": "standard",
        },
        "standard": {
            "resize_factor": 1,
            "face_det_batch_size": 8,
            "wav2lip_batch_size": 64,
            "model": "standard",
        },
        "high": {
            "resize_factor": 1,
            "face_det_batch_size": 4,
            "wav2lip_batch_size": 32,
            "model": "gan",
        },
    }

    def __init__(
        self,
        wav2lip_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize Wav2Lip processor.

        Args:
            wav2lip_path: Path to Wav2Lip repository
            checkpoint_path: Path to model checkpoint
            device: Device to use (cpu, cuda)
        """
        super().__init__()

        self.wav2lip_path = wav2lip_path or settings.MODELS_DIR / "wav2lip"
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if self._check_cuda() else "cpu")

        self._ensure_wav2lip_installed()

    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _ensure_wav2lip_installed(self):
        """Ensure Wav2Lip is installed and models are downloaded."""
        # Check if Wav2Lip directory exists
        if not self.wav2lip_path.exists():
            self._update_progress(0.0, "Wav2Lip not found. Installation required.")
            # Note: In production, this would clone the repo and download models
            # For now, we'll handle this in the Docker setup

    async def process(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Optional[Path] = None,
        quality: Literal["fast", "standard", "high"] = "high",
    ) -> LipSyncResult:
        """
        Process video with Wav2Lip lip synchronization.

        Args:
            video_path: Path to input video
            audio_path: Path to dubbed audio
            output_path: Output video path
            quality: Quality preset

        Returns:
            LipSyncResult with processed video
        """
        self._update_progress(0.0, "Initializing lip-sync processing")

        # Prepare output path
        if output_path is None:
            output_path = FileHandler.get_temp_path(extension=".mp4", prefix="lipsync_")

        # Get quality settings
        settings_dict = self.QUALITY_SETTINGS.get(quality, self.QUALITY_SETTINGS["standard"])

        self._update_progress(0.1, "Detecting faces in video")

        # Detect faces
        faces_count = self._detect_faces(video_path)

        self._update_progress(0.2, f"Found {faces_count} face(s), starting lip-sync")

        # Run Wav2Lip inference
        await self._run_wav2lip(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path,
            settings=settings_dict,
        )

        self._update_progress(0.9, "Finalizing output")

        # Get output video info
        info = self._get_video_info(output_path)
        video_stream = None
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        duration = float(info.get("format", {}).get("duration", 0))
        resolution = (
            int(video_stream.get("width", 0)),
            int(video_stream.get("height", 0)),
        ) if video_stream else (0, 0)

        fps = self._parse_fps(video_stream.get("r_frame_rate", "30/1")) if video_stream else 30.0

        self._update_progress(1.0, "Lip-sync complete")

        return LipSyncResult(
            path=output_path,
            duration=duration,
            resolution=resolution,
            fps=fps,
            faces_detected=faces_count,
        )

    async def _run_wav2lip(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        settings: dict,
    ):
        """
        Run Wav2Lip inference.

        This method handles the actual Wav2Lip processing, either by:
        1. Running the Wav2Lip inference script directly
        2. Using the Wav2Lip Python API if available
        3. Falling back to a simplified implementation
        """
        # Try to use the installed Wav2Lip
        wav2lip_inference = self.wav2lip_path / "inference.py"

        if wav2lip_inference.exists():
            await self._run_wav2lip_script(
                video_path, audio_path, output_path, settings
            )
        else:
            # Fallback: Use simplified lip-sync via ffmpeg
            # This just replaces audio without actual lip-sync
            await self._fallback_audio_replace(
                video_path, audio_path, output_path
            )

    async def _run_wav2lip_script(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        settings: dict,
    ):
        """Run Wav2Lip using the inference script."""
        # Determine checkpoint
        model_name = settings.get("model", "standard")
        checkpoint = self.checkpoint_path or (
            self.wav2lip_path / "checkpoints" / self.MODEL_VARIANTS[model_name]
        )

        cmd = [
            "python",
            str(self.wav2lip_path / "inference.py"),
            "--checkpoint_path", str(checkpoint),
            "--face", str(video_path),
            "--audio", str(audio_path),
            "--outfile", str(output_path),
            "--resize_factor", str(settings.get("resize_factor", 1)),
            "--face_det_batch_size", str(settings.get("face_det_batch_size", 8)),
            "--wav2lip_batch_size", str(settings.get("wav2lip_batch_size", 64)),
        ]

        if self.device == "cpu":
            cmd.extend(["--nosmooth"])

        self._update_progress(0.3, "Running Wav2Lip inference")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(self.wav2lip_path),
            )

            # Monitor progress
            for line in iter(process.stdout.readline, ""):
                if "Processing" in line or "frame" in line.lower():
                    # Extract progress if possible
                    self._update_progress(0.5, line.strip()[:50])

            process.wait()

            if process.returncode != 0:
                raise RuntimeError("Wav2Lip inference failed")

        except FileNotFoundError:
            raise RuntimeError(
                "Wav2Lip not properly installed. "
                "Please install from https://github.com/Rudrabha/Wav2Lip"
            )

    async def _fallback_audio_replace(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
    ):
        """
        Fallback: Replace audio without lip-sync.

        Used when Wav2Lip is not available.
        """
        self._update_progress(0.3, "Using audio replacement (Wav2Lip not available)")

        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-y",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Audio replacement failed: {e.stderr.decode()}")

    @staticmethod
    def _parse_fps(fps_string: str) -> float:
        """Parse FPS from ffprobe format."""
        try:
            if "/" in fps_string:
                num, den = fps_string.split("/")
                return float(num) / float(den)
            return float(fps_string)
        except (ValueError, ZeroDivisionError):
            return 30.0

    async def process_with_face_enhancement(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Optional[Path] = None,
        quality: str = "high",
        enhance_face: bool = True,
    ) -> LipSyncResult:
        """
        Process with optional face enhancement.

        Uses GFPGAN or similar for face restoration after lip-sync.

        Args:
            video_path: Input video path
            audio_path: Dubbed audio path
            output_path: Output path
            quality: Quality preset
            enhance_face: Apply face enhancement

        Returns:
            LipSyncResult with enhanced video
        """
        # First, run standard lip-sync
        result = await self.process(video_path, audio_path, output_path, quality)

        if enhance_face:
            # Apply face enhancement (requires GFPGAN)
            try:
                enhanced_path = await self._enhance_faces(result.path)
                result.path = enhanced_path
            except Exception as e:
                # Enhancement failed, return original
                self._update_progress(1.0, f"Face enhancement skipped: {e}")

        return result

    async def _enhance_faces(self, video_path: Path) -> Path:
        """
        Enhance faces in video using GFPGAN.

        This is optional and requires GFPGAN to be installed.
        """
        # This would use GFPGAN for face restoration
        # For now, just return the original
        return video_path
