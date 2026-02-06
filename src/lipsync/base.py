"""
Base class for lip synchronization processors.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Callable
import subprocess
import json

from ..utils.models import LipSyncResult
from ..utils.file_handler import FileHandler


class BaseLipSync(ABC):
    """Abstract base class for lip-sync processors."""

    def __init__(self):
        self.progress_callback: Optional[Callable] = None

    @abstractmethod
    async def process(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Optional[Path] = None,
        quality: str = "high",
    ) -> LipSyncResult:
        """
        Process video with lip synchronization.

        Args:
            video_path: Path to input video
            audio_path: Path to dubbed audio
            output_path: Output video path (optional)
            quality: Quality preset (fast, standard, high)

        Returns:
            LipSyncResult with processed video path
        """
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

    def _detect_faces(self, video_path: Path) -> int:
        """
        Detect number of faces in video.

        Returns approximate count from first few frames.
        """
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            face_counts = []
            frame_count = 0
            sample_frames = 10

            while cap.isOpened() and frame_count < sample_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames for faster processing
                if frame_count % 5 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    face_counts.append(len(faces))

                frame_count += 1

            cap.release()

            # Return max faces detected
            return max(face_counts) if face_counts else 0

        except Exception:
            return 1  # Assume at least one face
