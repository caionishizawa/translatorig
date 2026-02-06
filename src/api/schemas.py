"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, Literal
from enum import Enum
from datetime import datetime


class JobStatus(str, Enum):
    """Status of a translation job."""
    QUEUED = "queued"
    EXTRACTING = "extracting"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"
    SYNTHESIZING = "synthesizing"
    LIPSYNCING = "lipsyncing"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


class TranslationRequest(BaseModel):
    """Request to start a video translation job."""
    source_url: Optional[HttpUrl] = Field(
        None,
        description="URL of video to translate (Instagram/YouTube)"
    )
    target_language: str = Field(
        default="por",
        description="Target language code (e.g., 'por', 'eng', 'spa')"
    )
    output_resolution: Literal["4k", "1080p", "720p", "original"] = Field(
        default="original",
        description="Output video resolution"
    )
    include_subtitles: bool = Field(
        default=True,
        description="Include subtitles in output"
    )
    subtitle_style: Literal["soft", "hardcoded"] = Field(
        default="soft",
        description="Subtitle embedding style"
    )
    preserve_original_audio: bool = Field(
        default=False,
        description="Keep original audio as secondary track"
    )
    skip_lipsync: bool = Field(
        default=False,
        description="Skip lip-sync processing"
    )


class TranslationStatus(BaseModel):
    """Status of a translation job."""
    job_id: str
    status: JobStatus
    progress: float = Field(ge=0, le=100, description="Progress percentage")
    current_step: str = ""
    message: str = ""
    created_at: datetime
    updated_at: datetime
    output_url: Optional[str] = None
    error: Optional[str] = None

    # Additional metadata
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    duration: Optional[float] = None
    resolution: Optional[str] = None


class TranslationResult(BaseModel):
    """Result of a completed translation job."""
    job_id: str
    output_url: str
    source_language: str
    target_language: str
    duration: float
    resolution: str
    file_size_mb: float
    has_subtitles: bool
    processing_time_seconds: float


class LanguageInfo(BaseModel):
    """Information about a supported language."""
    code: str
    name: str


class HealthStatus(BaseModel):
    """API health check response."""
    status: str
    version: str
    gpu_available: bool
    models_loaded: dict


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    job_id: Optional[str] = None
