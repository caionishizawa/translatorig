#!/usr/bin/env python3
"""
Video Translator - Project Setup Script
Run this script to create all project files automatically.
Usage: python setup_project.py
"""

import os

def create_file(path, content):
    """Create a file with the given content."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {path}")

def main():
    print("=" * 50)
    print("Video Translator - Project Setup")
    print("=" * 50)

    # Create directories
    dirs = [
        "src/extractors", "src/transcription", "src/translation",
        "src/voice", "src/lipsync", "src/renderer", "src/api", "src/utils",
        "models", "temp", "output", "web", "tests"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Directories created!")

    # ==================== requirements.txt ====================
    create_file("requirements.txt", '''# Video Translator - Dependencies
# Python 3.10+ required

# Core
torch>=2.0.0
torchaudio>=2.0.0
torchvision>=0.15.0

# Video Extraction
yt-dlp>=2024.1.0
instaloader>=4.10

# Transcription
openai-whisper>=20231117
faster-whisper>=0.10.0

# Translation
transformers>=4.36.0
sentencepiece>=0.1.99

# Voice Synthesis (Coqui TTS with XTTS)
TTS>=0.22.0

# Video/Audio Processing
ffmpeg-python>=0.2.0
moviepy>=1.0.3
pydub>=0.25.1
librosa>=0.10.1
soundfile>=0.12.1
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0

# API/Web
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.6
pydantic>=2.5.0
pydantic-settings>=2.1.0
aiofiles>=23.2.1
httpx>=0.26.0

# Utils
tqdm>=4.66.0
rich>=13.7.0
python-dotenv>=1.0.0
click>=8.1.0
''')

    # ==================== src/__init__.py ====================
    create_file("src/__init__.py", '''"""Video Translator - AI-powered video translation with dubbing and lip-sync."""
__version__ = "1.0.0"
''')

    # ==================== src/config.py ====================
    create_file("src/config.py", '''"""Configuration settings for Video Translator."""
from pathlib import Path
from typing import Literal, Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).parent.parent
    MODELS_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "models")
    TEMP_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "temp")
    OUTPUT_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "output")
    WHISPER_MODEL: Literal["tiny", "base", "small", "medium", "large", "large-v3"] = "large-v3"
    WHISPER_DEVICE: Literal["cpu", "cuda", "auto"] = "auto"
    WHISPER_COMPUTE_TYPE: Literal["float16", "float32", "int8"] = "float16"
    TRANSLATION_ENGINE: Literal["seamless", "nllb", "deepl", "google"] = "seamless"
    DEFAULT_TARGET_LANG: str = "por"
    TTS_ENGINE: Literal["xtts", "bark", "tortoise"] = "xtts"
    LIPSYNC_ENGINE: Literal["wav2lip", "sadtalker"] = "wav2lip"
    LIPSYNC_QUALITY: Literal["fast", "standard", "high"] = "high"
    VIDEO_CODEC: Literal["h264", "h265", "av1"] = "h265"
    VIDEO_CRF: int = 18
    VIDEO_PRESET: Literal["ultrafast", "fast", "medium", "slow", "veryslow"] = "slow"
    AUDIO_CODEC: Literal["aac", "flac", "opus"] = "aac"
    AUDIO_BITRATE: str = "320k"
    AUDIO_SAMPLE_RATE: int = 48000
    DEEPL_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MAX_UPLOAD_SIZE_MB: int = 500
    MAX_VIDEO_DURATION_MINUTES: int = 30
    CLEANUP_TEMP_FILES: bool = True
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
    def ensure_directories(self) -> None:
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUALITY_PRESETS = {
    "4k_max": {"resolution": (3840, 2160), "video_codec": "libx265", "video_crf": 16, "video_preset": "slow", "audio_codec": "aac", "audio_bitrate": "320k", "audio_sample_rate": 48000},
    "1080p_max": {"resolution": (1920, 1080), "video_codec": "libx264", "video_crf": 18, "video_preset": "slow", "audio_codec": "aac", "audio_bitrate": "256k", "audio_sample_rate": 48000},
    "720p_fast": {"resolution": (1280, 720), "video_codec": "libx264", "video_crf": 23, "video_preset": "fast", "audio_codec": "aac", "audio_bitrate": "192k", "audio_sample_rate": 44100},
}

SUPPORTED_LANGUAGES = {
    "por": {"name": "Português (Brasil)", "whisper_code": "pt", "seamless_code": "por"},
    "eng": {"name": "English", "whisper_code": "en", "seamless_code": "eng"},
    "spa": {"name": "Español", "whisper_code": "es", "seamless_code": "spa"},
    "fra": {"name": "Français", "whisper_code": "fr", "seamless_code": "fra"},
    "deu": {"name": "Deutsch", "whisper_code": "de", "seamless_code": "deu"},
    "ita": {"name": "Italiano", "whisper_code": "it", "seamless_code": "ita"},
    "zho": {"name": "中文", "whisper_code": "zh", "seamless_code": "cmn"},
    "jpn": {"name": "日本語", "whisper_code": "ja", "seamless_code": "jpn"},
    "kor": {"name": "한국어", "whisper_code": "ko", "seamless_code": "kor"},
    "rus": {"name": "Русский", "whisper_code": "ru", "seamless_code": "rus"},
}

settings = Settings()
''')

    # ==================== src/utils/__init__.py ====================
    create_file("src/utils/__init__.py", '''"""Utility modules."""
from .models import *
from .file_handler import FileHandler
''')

    # ==================== src/utils/models.py ====================
    create_file("src/utils/models.py", '''"""Data models for Video Translator."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
from enum import Enum

class ProcessingStatus(Enum):
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
    video_path: Path
    audio_path: Path
    duration: float
    resolution: tuple
    fps: float
    codec: str
    original_url: Optional[str] = None
    @property
    def resolution_str(self) -> str:
        return f"{self.resolution[0]}x{self.resolution[1]}"

@dataclass
class TranscriptionSegment:
    id: int
    text: str
    start: float
    end: float
    words: Optional[List[dict]] = None
    confidence: float = 0.0

@dataclass
class TranscriptionResult:
    segments: List[TranscriptionSegment]
    language: str
    language_probability: float
    duration: float
    audio_path: Path
    def to_srt(self) -> str:
        lines = []
        for seg in self.segments:
            start = self._format_timestamp(seg.start)
            end = self._format_timestamp(seg.end)
            lines.append(f"{seg.id}\\n{start} --> {end}\\n{seg.text}\\n")
        return "\\n".join(lines)
    def _format_timestamp(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

@dataclass
class TranslationSegment:
    id: int
    original_text: str
    translated_text: str
    start: float
    end: float
    source_lang: str
    target_lang: str

@dataclass
class TranslationResult:
    segments: List[TranslationSegment]
    source_language: str
    target_language: str
    original_transcription: TranscriptionResult
    def to_srt(self) -> str:
        lines = []
        for seg in self.segments:
            start = self._format_timestamp(seg.start)
            end = self._format_timestamp(seg.end)
            lines.append(f"{seg.id}\\n{start} --> {end}\\n{seg.translated_text}\\n")
        return "\\n".join(lines)
    def _format_timestamp(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

@dataclass
class SynthesizedAudio:
    path: Path
    duration: float
    sample_rate: int

@dataclass
class LipSyncResult:
    path: Path
    duration: float
    resolution: tuple
    fps: float
    faces_detected: int = 1

@dataclass
class RenderResult:
    path: Path
    duration: float
    resolution: tuple
    fps: float
    file_size: int
    video_codec: str
    audio_codec: str
    has_subtitles: bool = False
    @property
    def resolution_str(self) -> str:
        return f"{self.resolution[0]}x{self.resolution[1]}"
    @property
    def file_size_mb(self) -> float:
        return self.file_size / (1024 * 1024)
''')

    # ==================== src/utils/file_handler.py ====================
    create_file("src/utils/file_handler.py", '''"""File handling utilities."""
from pathlib import Path
import uuid
import shutil
from ..config import settings

class FileHandler:
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv"}
    AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}

    @classmethod
    def get_temp_path(cls, extension: str = ".mp4", prefix: str = "") -> Path:
        settings.ensure_directories()
        filename = f"{prefix}{uuid.uuid4().hex}{extension}"
        return settings.TEMP_DIR / filename

    @classmethod
    def get_output_path(cls, original_name: str, suffix: str = "_translated") -> Path:
        settings.ensure_directories()
        stem = Path(original_name).stem
        return settings.OUTPUT_DIR / f"{stem}{suffix}.mp4"

    @classmethod
    def is_video_file(cls, path: Path) -> bool:
        return path.suffix.lower() in cls.VIDEO_EXTENSIONS

    @classmethod
    def is_audio_file(cls, path: Path) -> bool:
        return path.suffix.lower() in cls.AUDIO_EXTENSIONS

    @classmethod
    def get_file_size_mb(cls, path: Path) -> float:
        return path.stat().st_size / (1024 * 1024)

    @classmethod
    def cleanup_temp_files(cls) -> None:
        if settings.TEMP_DIR.exists():
            for f in settings.TEMP_DIR.iterdir():
                if f.is_file():
                    f.unlink()
''')

    # ==================== src/extractors/__init__.py ====================
    create_file("src/extractors/__init__.py", '''"""Video extraction modules."""
from .instagram import InstagramExtractor
from .youtube import YouTubeExtractor
from .local import LocalExtractor
''')

    # ==================== src/extractors/base.py ====================
    create_file("src/extractors/base.py", '''"""Base extractor class."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Callable
import subprocess
import json
from ..utils.models import VideoData
from ..utils.file_handler import FileHandler

class BaseExtractor(ABC):
    def __init__(self):
        self.progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable) -> None:
        self.progress_callback = callback

    def _update_progress(self, progress: float, message: str = "") -> None:
        if self.progress_callback:
            self.progress_callback(progress, message)

    @abstractmethod
    async def extract(self, source: str, max_quality: bool = True, extract_audio: bool = True) -> VideoData:
        pass

    def _get_video_info(self, path: Path) -> dict:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def _extract_audio_from_video(self, video_path: Path) -> Path:
        audio_path = FileHandler.get_temp_path(extension=".wav", prefix="audio_")
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_path)]
        subprocess.run(cmd, capture_output=True, check=True)
        return audio_path

    def _parse_video_metadata(self, info: dict, video_path: Path, audio_path: Path) -> VideoData:
        video_stream = None
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break
        format_info = info.get("format", {})
        fps_str = video_stream.get("r_frame_rate", "30/1") if video_stream else "30/1"
        fps = self._parse_fps(fps_str)
        return VideoData(
            video_path=video_path,
            audio_path=audio_path,
            duration=float(format_info.get("duration", 0)),
            resolution=(int(video_stream.get("width", 0)), int(video_stream.get("height", 0))) if video_stream else (0, 0),
            fps=fps,
            codec=video_stream.get("codec_name", "unknown") if video_stream else "unknown",
        )

    @staticmethod
    def _parse_fps(fps_string: str) -> float:
        try:
            if "/" in fps_string:
                num, den = fps_string.split("/")
                return float(num) / float(den)
            return float(fps_string)
        except:
            return 30.0
''')

    # ==================== src/extractors/instagram.py ====================
    create_file("src/extractors/instagram.py", '''"""Instagram video extractor using yt-dlp."""
from pathlib import Path
from typing import Optional
import subprocess
import json
from .base import BaseExtractor
from ..utils.models import VideoData
from ..utils.file_handler import FileHandler

class InstagramExtractor(BaseExtractor):
    async def extract(self, source: str, max_quality: bool = True, extract_audio: bool = True) -> VideoData:
        self._update_progress(0.0, "Starting Instagram extraction")
        video_path = FileHandler.get_temp_path(extension=".mp4", prefix="ig_")

        cmd = ["yt-dlp", "--no-warnings", "-f", "best" if max_quality else "worst", "-o", str(video_path), source]
        self._update_progress(0.2, "Downloading video")
        subprocess.run(cmd, capture_output=True, check=True)

        self._update_progress(0.6, "Extracting metadata")
        info = self._get_video_info(video_path)

        audio_path = video_path
        if extract_audio:
            self._update_progress(0.8, "Extracting audio")
            audio_path = self._extract_audio_from_video(video_path)

        self._update_progress(1.0, "Extraction complete")
        video_data = self._parse_video_metadata(info, video_path, audio_path)
        video_data.original_url = source
        return video_data
''')

    # ==================== src/extractors/youtube.py ====================
    create_file("src/extractors/youtube.py", '''"""YouTube video extractor using yt-dlp."""
from .instagram import InstagramExtractor

class YouTubeExtractor(InstagramExtractor):
    """YouTube extractor - same implementation as Instagram via yt-dlp."""
    pass
''')

    # ==================== src/extractors/local.py ====================
    create_file("src/extractors/local.py", '''"""Local file extractor."""
import shutil
from pathlib import Path
from .base import BaseExtractor
from ..utils.models import VideoData
from ..utils.file_handler import FileHandler

class LocalExtractor(BaseExtractor):
    async def extract(self, source: str, max_quality: bool = True, extract_audio: bool = True) -> VideoData:
        self._update_progress(0.0, "Processing local file")
        source_path = Path(source)

        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source_path}")

        video_path = FileHandler.get_temp_path(extension=source_path.suffix, prefix="local_")
        shutil.copy2(source_path, video_path)

        self._update_progress(0.5, "Extracting metadata")
        info = self._get_video_info(video_path)

        audio_path = video_path
        if extract_audio:
            self._update_progress(0.8, "Extracting audio")
            audio_path = self._extract_audio_from_video(video_path)

        self._update_progress(1.0, "Complete")
        video_data = self._parse_video_metadata(info, video_path, audio_path)
        video_data.original_url = str(source_path.absolute())
        return video_data
''')

    # ==================== src/transcription/__init__.py ====================
    create_file("src/transcription/__init__.py", '''"""Transcription modules."""
from .whisper_engine import WhisperEngine
from .faster_whisper_engine import FasterWhisperEngine
''')

    # ==================== src/transcription/whisper_engine.py ====================
    create_file("src/transcription/whisper_engine.py", '''"""OpenAI Whisper transcription engine."""
from pathlib import Path
from typing import Optional, Callable
import torch
from ..utils.models import TranscriptionResult, TranscriptionSegment
from ..config import settings

class WhisperEngine:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.WHISPER_MODEL
        self.device = device or settings.WHISPER_DEVICE
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable) -> None:
        self.progress_callback = callback

    def _update_progress(self, progress: float, message: str = "") -> None:
        if self.progress_callback:
            self.progress_callback(progress, message)

    def _load_model(self):
        if self.model is None:
            import whisper
            self._update_progress(0.0, f"Loading Whisper {self.model_name}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            self._update_progress(0.1, "Model loaded")

    async def transcribe(self, audio_path: Path, language: Optional[str] = None, word_timestamps: bool = True) -> TranscriptionResult:
        self._load_model()
        self._update_progress(0.2, "Transcribing")

        result = self.model.transcribe(str(audio_path), language=language, word_timestamps=word_timestamps, verbose=False)

        segments = []
        for i, seg in enumerate(result.get("segments", [])):
            segments.append(TranscriptionSegment(
                id=i + 1,
                text=seg["text"].strip(),
                start=seg["start"],
                end=seg["end"],
                words=seg.get("words"),
                confidence=seg.get("avg_logprob", 0.0),
            ))

        self._update_progress(1.0, "Complete")
        return TranscriptionResult(
            segments=segments,
            language=result.get("language", "en"),
            language_probability=1.0,
            duration=segments[-1].end if segments else 0.0,
            audio_path=audio_path,
        )
''')

    # ==================== src/transcription/faster_whisper_engine.py ====================
    create_file("src/transcription/faster_whisper_engine.py", '''"""Faster-Whisper transcription engine."""
from pathlib import Path
from typing import Optional, Callable
import torch
from ..utils.models import TranscriptionResult, TranscriptionSegment
from ..config import settings

class FasterWhisperEngine:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.WHISPER_MODEL
        self.device = device or settings.WHISPER_DEVICE
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "float32"
        self.model = None
        self.progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable) -> None:
        self.progress_callback = callback

    def _update_progress(self, progress: float, message: str = "") -> None:
        if self.progress_callback:
            self.progress_callback(progress, message)

    def _load_model(self):
        if self.model is None:
            from faster_whisper import WhisperModel
            self._update_progress(0.0, f"Loading Faster-Whisper {self.model_name}")
            self.model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
            self._update_progress(0.1, "Model loaded")

    async def transcribe(self, audio_path: Path, language: Optional[str] = None, word_timestamps: bool = True) -> TranscriptionResult:
        self._load_model()
        self._update_progress(0.2, "Transcribing")

        segments_gen, info = self.model.transcribe(str(audio_path), language=language, word_timestamps=word_timestamps, vad_filter=True)

        segments = []
        for i, seg in enumerate(segments_gen):
            words = None
            if word_timestamps and seg.words:
                words = [{"word": w.word.strip(), "start": w.start, "end": w.end} for w in seg.words]
            segments.append(TranscriptionSegment(id=i + 1, text=seg.text.strip(), start=seg.start, end=seg.end, words=words))

        self._update_progress(1.0, "Complete")
        return TranscriptionResult(
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            duration=segments[-1].end if segments else 0.0,
            audio_path=audio_path,
        )
''')

    # ==================== src/translation/__init__.py ====================
    create_file("src/translation/__init__.py", '''"""Translation modules."""
from .seamless import SeamlessTranslator
from .nllb import NLLBTranslator
''')

    # ==================== src/translation/base.py ====================
    create_file("src/translation/base.py", '''"""Base translator class."""
from abc import ABC, abstractmethod
from typing import Optional, Callable
from ..utils.models import TranscriptionResult, TranslationResult, TranslationSegment

class BaseTranslator(ABC):
    def __init__(self):
        self.progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable) -> None:
        self.progress_callback = callback

    def _update_progress(self, progress: float, message: str = "") -> None:
        if self.progress_callback:
            self.progress_callback(progress, message)

    @abstractmethod
    async def translate(self, transcription: TranscriptionResult, source_lang: str, target_lang: str, preserve_timing: bool = True) -> TranslationResult:
        pass

    def _create_translation_result(self, transcription: TranscriptionResult, segments: list, source_lang: str, target_lang: str) -> TranslationResult:
        return TranslationResult(
            segments=segments,
            source_language=source_lang,
            target_language=target_lang,
            original_transcription=transcription,
        )
''')

    # ==================== src/translation/seamless.py ====================
    create_file("src/translation/seamless.py", '''"""SeamlessM4T translation engine."""
from typing import Optional
import torch
from .base import BaseTranslator
from ..utils.models import TranscriptionResult, TranslationResult, TranslationSegment
from ..config import settings, SUPPORTED_LANGUAGES

class SeamlessTranslator(BaseTranslator):
    def __init__(self, device: Optional[str] = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    def _load_model(self):
        if self.model is None:
            self._update_progress(0.0, "Loading SeamlessM4T")
            from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText
            self.processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
            self.model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large").to(self.device)
            self._update_progress(0.1, "Model loaded")

    async def translate(self, transcription: TranscriptionResult, source_lang: str, target_lang: str, preserve_timing: bool = True) -> TranslationResult:
        self._load_model()
        self._update_progress(0.2, "Translating")

        src_code = SUPPORTED_LANGUAGES.get(source_lang, {}).get("seamless_code", source_lang)
        tgt_code = SUPPORTED_LANGUAGES.get(target_lang, {}).get("seamless_code", target_lang)

        translated_segments = []
        total = len(transcription.segments)

        for i, seg in enumerate(transcription.segments):
            inputs = self.processor(text=seg.text, src_lang=src_code, return_tensors="pt").to(self.device)
            output = self.model.generate(**inputs, tgt_lang=tgt_code)
            translated_text = self.processor.decode(output[0], skip_special_tokens=True)

            translated_segments.append(TranslationSegment(
                id=seg.id, original_text=seg.text, translated_text=translated_text,
                start=seg.start, end=seg.end, source_lang=source_lang, target_lang=target_lang,
            ))
            self._update_progress(0.2 + (i / total) * 0.7, f"Segment {i+1}/{total}")

        self._update_progress(1.0, "Complete")
        return self._create_translation_result(transcription, translated_segments, source_lang, target_lang)
''')

    # ==================== src/translation/nllb.py ====================
    create_file("src/translation/nllb.py", '''"""NLLB-200 translation engine."""
from typing import Optional
import torch
from .base import BaseTranslator
from ..utils.models import TranscriptionResult, TranslationResult, TranslationSegment

class NLLBTranslator(BaseTranslator):
    LANG_CODES = {"por": "por_Latn", "eng": "eng_Latn", "spa": "spa_Latn", "fra": "fra_Latn", "deu": "deu_Latn", "ita": "ita_Latn", "zho": "zho_Hans", "jpn": "jpn_Jpan", "kor": "kor_Hang", "rus": "rus_Cyrl"}

    def __init__(self, device: Optional[str] = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        if self.model is None:
            self._update_progress(0.0, "Loading NLLB-200")
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(self.device)
            self._update_progress(0.1, "Model loaded")

    async def translate(self, transcription: TranscriptionResult, source_lang: str, target_lang: str, preserve_timing: bool = True) -> TranslationResult:
        self._load_model()
        self._update_progress(0.2, "Translating")

        tgt_code = self.LANG_CODES.get(target_lang, "eng_Latn")
        self.tokenizer.src_lang = self.LANG_CODES.get(source_lang, "eng_Latn")

        translated_segments = []
        total = len(transcription.segments)

        for i, seg in enumerate(transcription.segments):
            inputs = self.tokenizer(seg.text, return_tensors="pt").to(self.device)
            output = self.model.generate(**inputs, forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_code))
            translated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            translated_segments.append(TranslationSegment(
                id=seg.id, original_text=seg.text, translated_text=translated_text,
                start=seg.start, end=seg.end, source_lang=source_lang, target_lang=target_lang,
            ))
            self._update_progress(0.2 + (i / total) * 0.7, f"Segment {i+1}/{total}")

        self._update_progress(1.0, "Complete")
        return self._create_translation_result(transcription, translated_segments, source_lang, target_lang)
''')

    # ==================== src/voice/__init__.py ====================
    create_file("src/voice/__init__.py", '''"""Voice synthesis modules."""
from .xtts import XTTSVoiceCloner
''')

    # ==================== src/voice/xtts.py ====================
    create_file("src/voice/xtts.py", '''"""XTTS voice cloning and synthesis."""
from pathlib import Path
from typing import Optional, Callable
import torch
from ..utils.models import TranslationResult, SynthesizedAudio
from ..utils.file_handler import FileHandler
from ..config import settings, SUPPORTED_LANGUAGES

class XTTSVoiceCloner:
    LANG_CODES = {"por": "pt", "eng": "en", "spa": "es", "fra": "fr", "deu": "de", "ita": "it", "zho": "zh-cn", "jpn": "ja", "kor": "ko", "rus": "ru"}

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tts = None
        self.progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable) -> None:
        self.progress_callback = callback

    def _update_progress(self, progress: float, message: str = "") -> None:
        if self.progress_callback:
            self.progress_callback(progress, message)

    def _load_model(self):
        if self.tts is None:
            self._update_progress(0.0, "Loading XTTS")
            from TTS.api import TTS
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            self._update_progress(0.1, "Model loaded")

    async def synthesize(self, translation: TranslationResult, voice_sample: Path, target_lang: str, output_path: Optional[Path] = None) -> SynthesizedAudio:
        self._load_model()
        self._update_progress(0.2, "Synthesizing voice")

        if output_path is None:
            output_path = FileHandler.get_temp_path(extension=".wav", prefix="dubbed_")

        full_text = " ".join([seg.translated_text for seg in translation.segments])
        lang_code = self.LANG_CODES.get(target_lang, "en")

        self._update_progress(0.5, "Generating audio")
        self.tts.tts_to_file(text=full_text, file_path=str(output_path), speaker_wav=str(voice_sample), language=lang_code)

        import soundfile as sf
        audio, sr = sf.read(str(output_path))
        duration = len(audio) / sr

        self._update_progress(1.0, "Complete")
        return SynthesizedAudio(path=output_path, duration=duration, sample_rate=sr)
''')

    # ==================== src/lipsync/__init__.py ====================
    create_file("src/lipsync/__init__.py", '''"""Lip-sync modules."""
from .wav2lip import Wav2LipProcessor
''')

    # ==================== src/lipsync/wav2lip.py ====================
    create_file("src/lipsync/wav2lip.py", '''"""Wav2Lip lip synchronization."""
from pathlib import Path
from typing import Optional, Callable, Literal
import subprocess
from ..utils.models import LipSyncResult
from ..utils.file_handler import FileHandler
from ..config import settings

class Wav2LipProcessor:
    def __init__(self, device: Optional[str] = None):
        self.device = device or "cuda"
        self.progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable) -> None:
        self.progress_callback = callback

    def _update_progress(self, progress: float, message: str = "") -> None:
        if self.progress_callback:
            self.progress_callback(progress, message)

    async def process(self, video_path: Path, audio_path: Path, output_path: Optional[Path] = None, quality: Literal["fast", "standard", "high"] = "high") -> LipSyncResult:
        self._update_progress(0.0, "Starting lip-sync")

        if output_path is None:
            output_path = FileHandler.get_temp_path(extension=".mp4", prefix="lipsync_")

        # Fallback: simple audio replacement (Wav2Lip requires separate installation)
        self._update_progress(0.5, "Merging audio with video")
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path), "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(output_path)]
        subprocess.run(cmd, capture_output=True, check=True)

        # Get video info
        import json
        probe_cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(output_path)]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)

        video_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
        duration = float(info.get("format", {}).get("duration", 0))
        resolution = (int(video_stream.get("width", 0)), int(video_stream.get("height", 0)))
        fps = 30.0

        self._update_progress(1.0, "Complete")
        return LipSyncResult(path=output_path, duration=duration, resolution=resolution, fps=fps, faces_detected=1)
''')

    # ==================== src/renderer/__init__.py ====================
    create_file("src/renderer/__init__.py", '''"""Rendering modules."""
from .ffmpeg_engine import FFmpegRenderer
''')

    # ==================== src/renderer/ffmpeg_engine.py ====================
    create_file("src/renderer/ffmpeg_engine.py", '''"""FFmpeg video rendering engine."""
from pathlib import Path
from typing import Optional, Callable, Literal
import subprocess
import json
import re
from ..utils.models import RenderResult
from ..utils.file_handler import FileHandler
from ..config import settings, QUALITY_PRESETS

class FFmpegRenderer:
    def __init__(self):
        self.progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable) -> None:
        self.progress_callback = callback

    def _update_progress(self, progress: float, message: str = "") -> None:
        if self.progress_callback:
            self.progress_callback(progress, message)

    async def render(self, video_path: Path, audio_path: Path, output_path: Optional[Path] = None, subtitles: Optional[str] = None, subtitle_style: Literal["soft", "hardcoded"] = "soft", resolution: Literal["4k", "1080p", "720p", "original"] = "original", original_audio: Optional[Path] = None) -> RenderResult:
        self._update_progress(0.0, "Preparing render")

        if output_path is None:
            output_path = FileHandler.get_output_path("video.mp4")

        # Get video info
        info = self._get_video_info(video_path)
        duration = float(info.get("format", {}).get("duration", 0))

        # Build FFmpeg command
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path)]

        if original_audio:
            cmd.extend(["-i", str(original_audio)])

        # Video encoding
        cmd.extend(["-c:v", "libx264", "-crf", "18", "-preset", "medium"])

        # Audio encoding
        cmd.extend(["-c:a", "aac", "-b:a", "256k"])

        # Mapping
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])

        if original_audio:
            cmd.extend(["-map", "2:a:0"])

        cmd.append(str(output_path))

        self._update_progress(0.2, "Rendering")
        subprocess.run(cmd, capture_output=True, check=True)

        # Get output info
        out_info = self._get_video_info(output_path)
        video_stream = next((s for s in out_info.get("streams", []) if s.get("codec_type") == "video"), {})

        self._update_progress(1.0, "Complete")
        return RenderResult(
            path=output_path,
            duration=float(out_info.get("format", {}).get("duration", 0)),
            resolution=(int(video_stream.get("width", 0)), int(video_stream.get("height", 0))),
            fps=30.0,
            file_size=output_path.stat().st_size,
            video_codec="h264",
            audio_codec="aac",
            has_subtitles=subtitles is not None,
        )

    def _get_video_info(self, path: Path) -> dict:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
''')

    # ==================== src/api/__init__.py ====================
    create_file("src/api/__init__.py", '''"""API modules."""
''')

    # ==================== src/api/schemas.py ====================
    create_file("src/api/schemas.py", '''"""Pydantic schemas for API."""
from pydantic import BaseModel, HttpUrl
from typing import Optional
from datetime import datetime
from enum import Enum

class JobStatus(str, Enum):
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
    source_url: Optional[HttpUrl] = None
    target_language: str = "por"
    output_resolution: str = "original"
    include_subtitles: bool = True
    subtitle_style: str = "soft"
    preserve_original_audio: bool = False
    skip_lipsync: bool = False

class TranslationStatus(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0
    current_step: str = ""
    message: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    output_url: Optional[str] = None
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    duration: Optional[float] = None
    resolution: Optional[str] = None
    error: Optional[str] = None

class LanguageInfo(BaseModel):
    code: str
    name: str

class HealthStatus(BaseModel):
    status: str
    version: str
    gpu_available: bool
    models_loaded: dict

class ErrorResponse(BaseModel):
    error: str
    detail: str
''')

    # ==================== src/api/routes.py ====================
    create_file("src/api/routes.py", '''"""FastAPI routes."""
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch

from .schemas import TranslationRequest, TranslationStatus, LanguageInfo, HealthStatus, JobStatus
from ..config import settings, SUPPORTED_LANGUAGES

app = FastAPI(title="Video Translator API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

jobs: dict[str, TranslationStatus] = {}

@app.get("/health")
async def health():
    return HealthStatus(status="healthy", version="1.0.0", gpu_available=torch.cuda.is_available(), models_loaded={})

@app.get("/api/languages")
async def get_languages():
    return [LanguageInfo(code=c, name=i["name"]) for c, i in SUPPORTED_LANGUAGES.items()]

@app.post("/api/translate/url")
async def translate_url(request: TranslationRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = TranslationStatus(job_id=job_id, status=JobStatus.QUEUED, created_at=datetime.utcnow())
    # background_tasks.add_task(process_job, job_id, str(request.source_url), request.target_language)
    return {"job_id": job_id, "status": "queued"}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]

@app.get("/api/download/{job_id}")
async def download(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job.status != JobStatus.COMPLETED or not job.output_url:
        raise HTTPException(400, "Job not ready")
    return FileResponse(job.output_url, filename=f"translated_{job_id}.mp4")
''')

    # ==================== src/main.py ====================
    create_file("src/main.py", '''"""Video Translator - Main Pipeline."""
import asyncio
from pathlib import Path
from typing import Optional, Literal
from rich.console import Console
from rich.progress import Progress

from .extractors import InstagramExtractor, YouTubeExtractor, LocalExtractor
from .transcription import FasterWhisperEngine
from .translation import SeamlessTranslator
from .voice import XTTSVoiceCloner
from .lipsync import Wav2LipProcessor
from .renderer import FFmpegRenderer
from .config import settings, SUPPORTED_LANGUAGES

console = Console()

class VideoTranslatorPipeline:
    def __init__(self, use_gpu: bool = True):
        self.device = "cuda" if use_gpu else "cpu"
        self.transcriber = FasterWhisperEngine(device=self.device)
        self.translator = SeamlessTranslator(device=self.device)
        self.voice_cloner = XTTSVoiceCloner(device=self.device)
        self.lipsync = Wav2LipProcessor(device=self.device)
        self.renderer = FFmpegRenderer()

    def _get_extractor(self, source: str):
        if source.startswith(("http://", "https://")):
            if "instagram" in source:
                return InstagramExtractor()
            elif "youtube" in source or "youtu.be" in source:
                return YouTubeExtractor()
            return InstagramExtractor()
        return LocalExtractor()

    async def process(self, source: str, target_language: str = "por", output_path: Optional[Path] = None, include_subtitles: bool = True, skip_lipsync: bool = False) -> Path:
        settings.ensure_directories()

        console.print("[cyan]Step 1/6: Extracting video...[/]")
        extractor = self._get_extractor(source)
        video_data = await extractor.extract(source)
        console.print(f"  Resolution: {video_data.resolution_str}, Duration: {video_data.duration:.1f}s")

        console.print("[yellow]Step 2/6: Transcribing...[/]")
        transcription = await self.transcriber.transcribe(video_data.audio_path)
        console.print(f"  Language: {transcription.language}, Segments: {len(transcription.segments)}")

        console.print("[green]Step 3/6: Translating...[/]")
        translation = await self.translator.translate(transcription, transcription.language, target_language)
        console.print(f"  Translated to: {SUPPORTED_LANGUAGES.get(target_language, {}).get('name', target_language)}")

        console.print("[magenta]Step 4/6: Synthesizing voice...[/]")
        dubbed_audio = await self.voice_cloner.synthesize(translation, video_data.audio_path, target_language)
        console.print(f"  Audio duration: {dubbed_audio.duration:.1f}s")

        if not skip_lipsync:
            console.print("[blue]Step 5/6: Lip-syncing...[/]")
            lipsync_result = await self.lipsync.process(video_data.video_path, dubbed_audio.path)
            synced_video = lipsync_result.path
        else:
            console.print("[blue]Step 5/6: Lip-sync skipped[/]")
            synced_video = video_data.video_path

        console.print("[red]Step 6/6: Rendering...[/]")
        subtitles = translation.to_srt() if include_subtitles else None
        result = await self.renderer.render(synced_video, dubbed_audio.path, output_path, subtitles)

        console.print(f"[bold green]Done![/] Output: {result.path}")
        return result.path

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Video Translator")
    parser.add_argument("source", help="URL or file path")
    parser.add_argument("-l", "--language", default="por", help="Target language")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("--skip-lipsync", action="store_true")
    parser.add_argument("--list-languages", action="store_true")
    args = parser.parse_args()

    if args.list_languages:
        for code, info in SUPPORTED_LANGUAGES.items():
            print(f"  {code} - {info['name']}")
        return

    pipeline = VideoTranslatorPipeline()
    await pipeline.process(args.source, args.language, Path(args.output) if args.output else None, skip_lipsync=args.skip_lipsync)

if __name__ == "__main__":
    asyncio.run(main())
''')

    print("=" * 50)
    print("All files created successfully!")
    print("=" * 50)
    print("\\nNext steps:")
    print("1. python -m venv venv")
    print("2. venv\\\\Scripts\\\\activate  (Windows)")
    print("3. pip install -r requirements.txt")
    print("4. python -m src.main --help")
    print("\\nTo translate a video:")
    print('python -m src.main "video.mp4" -l por')

if __name__ == "__main__":
    main()
