"""
Data models for Video Translator pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
from enum import Enum


class ProcessingStatus(Enum):
    """Status of a processing job."""
    QUEUED = "queued"
    EXTRACTING = "extracting"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"
    SYNTHESIZING = "synthesizing"
    LIPSYNCING = "lipsyncing"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class VideoData:
    """Container for extracted video data."""
    video_path: Path
    audio_path: Path
    duration: float  # seconds
    resolution: tuple[int, int]  # (width, height)
    fps: float
    codec: str
    bitrate: Optional[int] = None
    has_audio: bool = True
    original_url: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.resolution[0]

    @property
    def height(self) -> int:
        return self.resolution[1]

    @property
    def resolution_str(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def is_4k(self) -> bool:
        return self.width >= 3840 or self.height >= 2160

    @property
    def is_1080p(self) -> bool:
        return self.width >= 1920 or self.height >= 1080


@dataclass
class AudioData:
    """Container for audio data."""
    path: Path
    duration: float  # seconds
    sample_rate: int
    channels: int
    codec: str
    bitrate: Optional[int] = None


@dataclass
class TranscriptionSegment:
    """A single segment of transcribed text with timing."""
    id: int
    text: str
    start: float  # seconds
    end: float  # seconds
    words: Optional[List[dict]] = None  # Word-level timestamps
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def to_srt(self) -> str:
        """Convert segment to SRT format."""
        return (
            f"{self.id}\n"
            f"{self.to_srt_time(self.start)} --> {self.to_srt_time(self.end)}\n"
            f"{self.text}\n"
        )


@dataclass
class TranscriptionResult:
    """Result of speech-to-text transcription."""
    segments: List[TranscriptionSegment]
    language: str
    language_probability: float
    duration: float
    audio_path: Path

    @property
    def full_text(self) -> str:
        """Get the complete transcribed text."""
        return " ".join(segment.text for segment in self.segments)

    def to_srt(self) -> str:
        """Convert transcription to SRT subtitle format."""
        return "\n".join(segment.to_srt() for segment in self.segments)

    def to_json(self) -> dict:
        """Convert transcription to JSON-serializable dict."""
        return {
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "full_text": self.full_text,
            "segments": [
                {
                    "id": s.id,
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                    "words": s.words,
                    "confidence": s.confidence,
                }
                for s in self.segments
            ],
        }


@dataclass
class TranslationSegment:
    """A translated segment with timing preserved."""
    id: int
    original_text: str
    translated_text: str
    start: float
    end: float
    source_lang: str
    target_lang: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def to_srt(self) -> str:
        """Convert segment to SRT format."""
        return (
            f"{self.id}\n"
            f"{self.to_srt_time(self.start)} --> {self.to_srt_time(self.end)}\n"
            f"{self.translated_text}\n"
        )


@dataclass
class TranslationResult:
    """Result of text translation."""
    segments: List[TranslationSegment]
    source_lang: str
    target_lang: str
    duration: float

    @property
    def full_text(self) -> str:
        """Get the complete translated text."""
        return " ".join(segment.translated_text for segment in self.segments)

    def to_srt(self) -> str:
        """Convert translation to SRT subtitle format."""
        return "\n".join(segment.to_srt() for segment in self.segments)

    def to_json(self) -> dict:
        """Convert translation to JSON-serializable dict."""
        return {
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "duration": self.duration,
            "full_text": self.full_text,
            "segments": [
                {
                    "id": s.id,
                    "original_text": s.original_text,
                    "translated_text": s.translated_text,
                    "start": s.start,
                    "end": s.end,
                }
                for s in self.segments
            ],
        }


@dataclass
class SynthesizedAudio:
    """Result of voice synthesis."""
    path: Path
    duration: float
    sample_rate: int
    voice_cloned: bool
    target_lang: str


@dataclass
class LipSyncResult:
    """Result of lip-sync processing."""
    path: Path
    duration: float
    resolution: tuple[int, int]
    fps: float
    faces_detected: int


@dataclass
class RenderResult:
    """Result of final video rendering."""
    path: Path
    duration: float
    resolution: tuple[int, int]
    fps: float
    file_size: int  # bytes
    video_codec: str
    audio_codec: str
    has_subtitles: bool

    @property
    def file_size_mb(self) -> float:
        return self.file_size / (1024 * 1024)

    @property
    def resolution_str(self) -> str:
        return f"{self.resolution[0]}x{self.resolution[1]}"
