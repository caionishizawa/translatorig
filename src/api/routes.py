"""
FastAPI REST API routes for Video Translator.
"""

import uuid
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch

from .schemas import (
    TranslationRequest,
    TranslationStatus,
    TranslationResult,
    LanguageInfo,
    HealthStatus,
    ErrorResponse,
    JobStatus,
)
from ..main import VideoTranslatorPipeline
from ..config import settings, SUPPORTED_LANGUAGES
from ..utils.file_handler import FileHandler


# Job storage (in production, use Redis or database)
jobs: dict[str, TranslationStatus] = {}

# Pipeline instance (reused across requests)
pipeline: Optional[VideoTranslatorPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global pipeline

    # Startup
    settings.ensure_directories()
    pipeline = VideoTranslatorPipeline(use_gpu=torch.cuda.is_available())

    yield

    # Shutdown
    FileHandler.cleanup_temp_files()


# Create FastAPI app
app = FastAPI(
    title="Video Translator API",
    description="""
    API for translating videos with AI-powered dubbing and lip-sync.

    Features:
    - Extract videos from Instagram, YouTube, or local files
    - Transcribe audio using Whisper
    - Translate text using SeamlessM4T
    - Clone voice and synthesize dubbed audio
    - Apply lip-sync using Wav2Lip
    - Render high-quality output video

    Supported languages: Portuguese, English, Spanish, French, German, Italian,
    Chinese, Japanese, Korean, Russian, Arabic, Hindi, and more.
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Health Check ====================

@app.get("/health", response_model=HealthStatus, tags=["Health"])
async def health_check():
    """Check API health and system status."""
    return HealthStatus(
        status="healthy",
        version="1.0.0",
        gpu_available=torch.cuda.is_available(),
        models_loaded={
            "transcription": pipeline._transcriber is not None if pipeline else False,
            "translation": pipeline._translator is not None if pipeline else False,
            "voice": pipeline._voice_cloner is not None if pipeline else False,
            "lipsync": pipeline._lipsync is not None if pipeline else False,
        }
    )


# ==================== Translation Jobs ====================

@app.post("/api/translate/url", response_model=dict, tags=["Translation"])
async def translate_from_url(
    request: TranslationRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start a translation job from a video URL.

    Supports Instagram Reels, YouTube videos, and other supported platforms.
    """
    if not request.source_url:
        raise HTTPException(status_code=400, detail="source_url is required")

    job_id = str(uuid.uuid4())
    now = datetime.utcnow()

    jobs[job_id] = TranslationStatus(
        job_id=job_id,
        status=JobStatus.QUEUED,
        progress=0,
        current_step="Queued for processing",
        created_at=now,
        updated_at=now,
        target_language=request.target_language,
    )

    background_tasks.add_task(
        process_translation_job,
        job_id,
        str(request.source_url),
        request.target_language,
        request.output_resolution,
        request.include_subtitles,
        request.subtitle_style,
        request.preserve_original_audio,
        request.skip_lipsync,
    )

    return {"job_id": job_id, "status": "queued"}


@app.post("/api/translate/upload", response_model=dict, tags=["Translation"])
async def translate_from_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file to translate"),
    target_language: str = Query(default="por", description="Target language code"),
    output_resolution: str = Query(default="original", description="Output resolution"),
    include_subtitles: bool = Query(default=True, description="Include subtitles"),
    subtitle_style: str = Query(default="soft", description="Subtitle style"),
    preserve_original_audio: bool = Query(default=False, description="Keep original audio"),
    skip_lipsync: bool = Query(default=False, description="Skip lip-sync"),
):
    """
    Start a translation job from an uploaded video file.

    Supports MP4, MOV, MKV, AVI, and WebM formats.
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    valid_extensions = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported: {', '.join(valid_extensions)}"
        )

    # Save uploaded file
    job_id = str(uuid.uuid4())
    temp_path = settings.TEMP_DIR / f"{job_id}_{file.filename}"

    try:
        content = await file.read()

        # Check file size
        if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB"
            )

        with open(temp_path, "wb") as f:
            f.write(content)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    now = datetime.utcnow()
    jobs[job_id] = TranslationStatus(
        job_id=job_id,
        status=JobStatus.QUEUED,
        progress=0,
        current_step="File uploaded, queued for processing",
        created_at=now,
        updated_at=now,
        target_language=target_language,
    )

    background_tasks.add_task(
        process_translation_job,
        job_id,
        str(temp_path),
        target_language,
        output_resolution,
        include_subtitles,
        subtitle_style,
        preserve_original_audio,
        skip_lipsync,
    )

    return {"job_id": job_id, "status": "queued", "filename": file.filename}


@app.get("/api/status/{job_id}", response_model=TranslationStatus, tags=["Translation"])
async def get_job_status(job_id: str):
    """Get the status of a translation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return jobs[job_id]


@app.get("/api/download/{job_id}", tags=["Translation"])
async def download_result(job_id: str):
    """Download the translated video."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job.status.value}"
        )

    if not job.output_url:
        raise HTTPException(status_code=404, detail="Output file not available")

    output_path = Path(job.output_url)

    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"translated_{job_id}.mp4"
    )


@app.delete("/api/job/{job_id}", tags=["Translation"])
async def delete_job(job_id: str):
    """Delete a translation job and its output file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # Delete output file if exists
    if job.output_url:
        output_path = Path(job.output_url)
        if output_path.exists():
            output_path.unlink()

    del jobs[job_id]

    return {"message": "Job deleted", "job_id": job_id}


# ==================== Languages ====================

@app.get("/api/languages", response_model=list[LanguageInfo], tags=["Languages"])
async def get_supported_languages():
    """Get list of supported languages."""
    return [
        LanguageInfo(code=code, name=info["name"])
        for code, info in SUPPORTED_LANGUAGES.items()
    ]


# ==================== Background Job Processing ====================

async def process_translation_job(
    job_id: str,
    source: str,
    target_language: str,
    output_resolution: str,
    include_subtitles: bool,
    subtitle_style: str,
    preserve_original_audio: bool,
    skip_lipsync: bool,
):
    """Process a translation job in the background."""

    def update_progress(status: dict):
        """Update job progress from pipeline."""
        if job_id in jobs:
            current_step = status.get("current_step", "")
            step_info = status.get("steps", {}).get(current_step, {})

            jobs[job_id].progress = status.get("overall_progress", 0) * 100
            jobs[job_id].current_step = step_info.get("name", current_step)
            jobs[job_id].message = step_info.get("message", "")
            jobs[job_id].updated_at = datetime.utcnow()

            # Map step to status
            status_map = {
                "extraction": JobStatus.EXTRACTING,
                "transcription": JobStatus.TRANSCRIBING,
                "translation": JobStatus.TRANSLATING,
                "voice_synthesis": JobStatus.SYNTHESIZING,
                "lip_sync": JobStatus.LIPSYNCING,
                "rendering": JobStatus.RENDERING,
            }

            if current_step in status_map:
                jobs[job_id].status = status_map[current_step]

    try:
        jobs[job_id].status = JobStatus.EXTRACTING
        jobs[job_id].updated_at = datetime.utcnow()

        # Process video
        result = await pipeline.process(
            source=source,
            target_language=target_language,
            include_subtitles=include_subtitles,
            subtitle_style=subtitle_style,
            output_resolution=output_resolution,
            preserve_original_audio=preserve_original_audio,
            skip_lipsync=skip_lipsync,
            progress_callback=update_progress,
        )

        # Update job with result
        jobs[job_id].status = JobStatus.COMPLETED
        jobs[job_id].progress = 100
        jobs[job_id].output_url = str(result.output_path)
        jobs[job_id].source_language = result.source_language
        jobs[job_id].duration = result.output_duration
        jobs[job_id].resolution = f"{result.output_resolution[0]}x{result.output_resolution[1]}"
        jobs[job_id].current_step = "Completed"
        jobs[job_id].updated_at = datetime.utcnow()

    except Exception as e:
        jobs[job_id].status = JobStatus.FAILED
        jobs[job_id].error = str(e)
        jobs[job_id].current_step = "Failed"
        jobs[job_id].updated_at = datetime.utcnow()


# ==================== Static Files (Web UI) ====================

# Mount web UI if directory exists
web_dir = settings.BASE_DIR / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")


# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc.detail),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
        ).model_dump(),
    )
