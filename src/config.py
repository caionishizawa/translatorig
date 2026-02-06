"""
Configuration settings for Video Translator.
"""

from pathlib import Path
from typing import Literal, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Global settings for the Video Translator system."""

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    MODELS_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "models")
    TEMP_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "temp")
    OUTPUT_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "output")

    # Whisper Configuration
    WHISPER_MODEL: Literal["tiny", "base", "small", "medium", "large", "large-v3"] = "large-v3"
    WHISPER_DEVICE: Literal["cpu", "cuda", "auto"] = "auto"
    WHISPER_COMPUTE_TYPE: Literal["float16", "float32", "int8"] = "float16"

    # Translation Configuration
    TRANSLATION_ENGINE: Literal["seamless", "nllb", "deepl", "google"] = "seamless"
    SEAMLESS_MODEL: str = "seamlessM4T_v2_large"
    DEFAULT_TARGET_LANG: str = "por"  # Portuguese (Brazil)

    # Voice Synthesis Configuration
    TTS_ENGINE: Literal["xtts", "bark", "tortoise"] = "xtts"
    XTTS_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    VOICE_SAMPLE_DURATION: int = 10  # seconds

    # Lip-sync Configuration
    LIPSYNC_ENGINE: Literal["wav2lip", "sadtalker"] = "wav2lip"
    LIPSYNC_QUALITY: Literal["fast", "standard", "high"] = "high"

    # Video Output Configuration
    VIDEO_CODEC: Literal["h264", "h265", "av1"] = "h265"
    VIDEO_CRF: int = 18  # Lower = better quality (18-23 recommended)
    VIDEO_PRESET: Literal["ultrafast", "fast", "medium", "slow", "veryslow"] = "slow"
    AUDIO_CODEC: Literal["aac", "flac", "opus"] = "aac"
    AUDIO_BITRATE: str = "320k"
    AUDIO_SAMPLE_RATE: int = 48000

    # API Keys (optional, for external services)
    DEEPL_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    INSTAGRAM_USERNAME: Optional[str] = None
    INSTAGRAM_PASSWORD: Optional[str] = None

    # Server Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MAX_UPLOAD_SIZE_MB: int = 500

    # Processing Configuration
    MAX_VIDEO_DURATION_MINUTES: int = 30
    CLEANUP_TEMP_FILES: bool = True

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Quality Presets for video rendering
QUALITY_PRESETS = {
    "4k_max": {
        "resolution": (3840, 2160),
        "video_codec": "libx265",
        "video_crf": 16,
        "video_preset": "slow",
        "video_params": ["-pix_fmt", "yuv420p10le"],
        "audio_codec": "aac",
        "audio_bitrate": "320k",
        "audio_sample_rate": 48000,
    },
    "4k_balanced": {
        "resolution": (3840, 2160),
        "video_codec": "libx265",
        "video_crf": 20,
        "video_preset": "medium",
        "video_params": ["-pix_fmt", "yuv420p"],
        "audio_codec": "aac",
        "audio_bitrate": "256k",
        "audio_sample_rate": 48000,
    },
    "1080p_max": {
        "resolution": (1920, 1080),
        "video_codec": "libx264",
        "video_crf": 18,
        "video_preset": "slow",
        "video_params": ["-pix_fmt", "yuv420p"],
        "audio_codec": "aac",
        "audio_bitrate": "256k",
        "audio_sample_rate": 48000,
    },
    "1080p_fast": {
        "resolution": (1920, 1080),
        "video_codec": "libx264",
        "video_crf": 22,
        "video_preset": "fast",
        "video_params": ["-pix_fmt", "yuv420p"],
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "audio_sample_rate": 44100,
    },
    "720p_fast": {
        "resolution": (1280, 720),
        "video_codec": "libx264",
        "video_crf": 23,
        "video_preset": "fast",
        "video_params": ["-pix_fmt", "yuv420p"],
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "audio_sample_rate": 44100,
    },
    "iphone_optimized": {
        # Optimized for iPhone 15 4K 30fps videos
        "resolution": (3840, 2160),
        "video_codec": "libx265",
        "video_crf": 18,
        "video_preset": "slow",
        "video_params": [
            "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-movflags", "+faststart"
        ],
        "audio_codec": "aac",
        "audio_bitrate": "320k",
        "audio_sample_rate": 48000,
        "fps": 30,
    }
}

# Supported languages with their codes
SUPPORTED_LANGUAGES = {
    "por": {"name": "Português (Brasil)", "whisper_code": "pt", "seamless_code": "por"},
    "eng": {"name": "English", "whisper_code": "en", "seamless_code": "eng"},
    "spa": {"name": "Español", "whisper_code": "es", "seamless_code": "spa"},
    "fra": {"name": "Français", "whisper_code": "fr", "seamless_code": "fra"},
    "deu": {"name": "Deutsch", "whisper_code": "de", "seamless_code": "deu"},
    "ita": {"name": "Italiano", "whisper_code": "it", "seamless_code": "ita"},
    "zho": {"name": "中文 (Chinese)", "whisper_code": "zh", "seamless_code": "cmn"},
    "jpn": {"name": "日本語 (Japanese)", "whisper_code": "ja", "seamless_code": "jpn"},
    "kor": {"name": "한국어 (Korean)", "whisper_code": "ko", "seamless_code": "kor"},
    "rus": {"name": "Русский", "whisper_code": "ru", "seamless_code": "rus"},
    "ara": {"name": "العربية", "whisper_code": "ar", "seamless_code": "arb"},
    "hin": {"name": "हिन्दी", "whisper_code": "hi", "seamless_code": "hin"},
    "tur": {"name": "Türkçe", "whisper_code": "tr", "seamless_code": "tur"},
    "vie": {"name": "Tiếng Việt", "whisper_code": "vi", "seamless_code": "vie"},
    "tha": {"name": "ไทย", "whisper_code": "th", "seamless_code": "tha"},
    "nld": {"name": "Nederlands", "whisper_code": "nl", "seamless_code": "nld"},
    "pol": {"name": "Polski", "whisper_code": "pl", "seamless_code": "pol"},
    "swe": {"name": "Svenska", "whisper_code": "sv", "seamless_code": "swe"},
}


# Global settings instance
settings = Settings()
