"""
File handling utilities for Video Translator.
"""

import os
import shutil
import hashlib
import mimetypes
from pathlib import Path
from typing import Optional
import tempfile
import uuid

from ..config import settings


class FileHandler:
    """Utility class for file operations."""

    SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}
    SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".aac", ".m4a", ".flac", ".ogg", ".wma"}

    @classmethod
    def is_video_file(cls, path: Path) -> bool:
        """Check if the file is a supported video format."""
        return path.suffix.lower() in cls.SUPPORTED_VIDEO_EXTENSIONS

    @classmethod
    def is_audio_file(cls, path: Path) -> bool:
        """Check if the file is a supported audio format."""
        return path.suffix.lower() in cls.SUPPORTED_AUDIO_EXTENSIONS

    @classmethod
    def get_temp_path(cls, extension: str = ".tmp", prefix: str = "vt_") -> Path:
        """Generate a unique temporary file path."""
        settings.ensure_directories()
        filename = f"{prefix}{uuid.uuid4().hex}{extension}"
        return settings.TEMP_DIR / filename

    @classmethod
    def get_output_path(cls, original_name: str, suffix: str = "_translated") -> Path:
        """Generate output file path based on original filename."""
        settings.ensure_directories()
        original = Path(original_name)
        new_name = f"{original.stem}{suffix}{original.suffix}"
        return settings.OUTPUT_DIR / new_name

    @classmethod
    def get_file_hash(cls, path: Path, algorithm: str = "md5") -> str:
        """Calculate hash of a file for caching/comparison."""
        hash_func = hashlib.new(algorithm)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    @classmethod
    def get_file_size(cls, path: Path) -> int:
        """Get file size in bytes."""
        return path.stat().st_size

    @classmethod
    def get_file_size_mb(cls, path: Path) -> float:
        """Get file size in megabytes."""
        return cls.get_file_size(path) / (1024 * 1024)

    @classmethod
    def cleanup_temp_files(cls, pattern: str = "vt_*") -> int:
        """Clean up temporary files. Returns number of files deleted."""
        count = 0
        for file in settings.TEMP_DIR.glob(pattern):
            try:
                if file.is_file():
                    file.unlink()
                    count += 1
                elif file.is_dir():
                    shutil.rmtree(file)
                    count += 1
            except Exception:
                pass
        return count

    @classmethod
    def ensure_directory(cls, path: Path) -> Path:
        """Ensure a directory exists, creating it if necessary."""
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def copy_file(cls, source: Path, destination: Path) -> Path:
        """Copy a file to a new location."""
        cls.ensure_directory(destination.parent)
        shutil.copy2(source, destination)
        return destination

    @classmethod
    def move_file(cls, source: Path, destination: Path) -> Path:
        """Move a file to a new location."""
        cls.ensure_directory(destination.parent)
        shutil.move(str(source), str(destination))
        return destination

    @classmethod
    def get_mime_type(cls, path: Path) -> Optional[str]:
        """Get MIME type of a file."""
        mime_type, _ = mimetypes.guess_type(str(path))
        return mime_type

    @classmethod
    def validate_video_file(cls, path: Path) -> bool:
        """Validate that a file exists and is a video."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not cls.is_video_file(path):
            raise ValueError(f"Unsupported video format: {path.suffix}")
        return True

    @classmethod
    def create_temp_directory(cls, prefix: str = "vt_dir_") -> Path:
        """Create a temporary directory."""
        settings.ensure_directories()
        temp_dir = settings.TEMP_DIR / f"{prefix}{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
