"""
Faster-Whisper transcription engine for improved performance.
"""

from pathlib import Path
from typing import Optional, Callable, List
import torch

from ..utils.models import TranscriptionResult, TranscriptionSegment
from ..config import settings


class FasterWhisperEngine:
    """
    Speech-to-text transcription using faster-whisper.

    Faster-whisper is a reimplementation of OpenAI's Whisper model
    using CTranslate2, which provides up to 4x speedup with similar accuracy.
    """

    MODEL_SIZE_MAP = {
        "tiny": "tiny",
        "base": "base",
        "small": "small",
        "medium": "medium",
        "large": "large-v2",
        "large-v3": "large-v3",
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ):
        """
        Initialize Faster-Whisper engine.

        Args:
            model_name: Model size (tiny, base, small, medium, large, large-v3)
            device: Device to use (cpu, cuda, auto)
            compute_type: Compute type (float16, float32, int8)
        """
        self.model_name = model_name or settings.WHISPER_MODEL
        self.device = device or settings.WHISPER_DEVICE
        self.compute_type = compute_type or settings.WHISPER_COMPUTE_TYPE

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Adjust compute type for CPU
        if self.device == "cpu" and self.compute_type == "float16":
            self.compute_type = "float32"

        self.model = None
        self.progress_callback: Optional[Callable] = None

    def _load_model(self):
        """Load Faster-Whisper model (lazy loading)."""
        if self.model is None:
            from faster_whisper import WhisperModel

            self._update_progress(0.0, f"Loading Faster-Whisper {self.model_name} model")

            model_size = self.MODEL_SIZE_MAP.get(self.model_name, self.model_name)

            self.model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

            self._update_progress(0.1, "Model loaded")

    def set_progress_callback(self, callback: Callable) -> None:
        """Set callback for progress updates."""
        self.progress_callback = callback

    def _update_progress(self, progress: float, message: str = "") -> None:
        """Update progress via callback if set."""
        if self.progress_callback:
            self.progress_callback(progress, message)

    async def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        word_timestamps: bool = True,
        task: str = "transcribe",
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> TranscriptionResult:
        """
        Transcribe audio file to text with timestamps.

        Args:
            audio_path: Path to audio file
            language: Source language code (None for auto-detection)
            word_timestamps: Include word-level timestamps
            task: "transcribe" for same-language, "translate" for English
            beam_size: Beam size for decoding (higher = better quality, slower)
            vad_filter: Use VAD to filter non-speech segments

        Returns:
            TranscriptionResult with segments and timestamps
        """
        self._load_model()

        self._update_progress(0.2, "Starting transcription")

        # Transcription options
        segments_gen, info = self.model.transcribe(
            str(audio_path),
            language=language,
            task=task,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
        )

        self._update_progress(0.4, "Processing segments")

        # Convert generator to list and track progress
        segments = []
        total_duration = info.duration if info.duration else 1.0

        for i, seg in enumerate(segments_gen):
            # Calculate progress based on segment timing
            progress = 0.4 + (seg.end / total_duration) * 0.5
            self._update_progress(min(progress, 0.9), f"Processing: {seg.end:.1f}s")

            words = None
            if word_timestamps and seg.words:
                words = [
                    {
                        "word": w.word.strip(),
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    }
                    for w in seg.words
                ]

            segments.append(TranscriptionSegment(
                id=i + 1,
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                words=words,
                confidence=seg.avg_logprob if hasattr(seg, 'avg_logprob') else 0.0,
            ))

        # Calculate actual duration from segments
        duration = segments[-1].end if segments else 0.0

        self._update_progress(1.0, "Transcription complete")

        return TranscriptionResult(
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            duration=duration,
            audio_path=audio_path,
        )

    async def detect_language(self, audio_path: Path) -> tuple[str, float]:
        """
        Detect the language of an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (language_code, probability)
        """
        self._load_model()

        # Use transcribe with a short segment for detection
        _, info = self.model.transcribe(
            str(audio_path),
            language=None,  # Auto-detect
        )

        return info.language, info.language_probability

    def get_available_languages(self) -> List[str]:
        """Get list of languages supported by Whisper."""
        # Faster-whisper supports same languages as Whisper
        return [
            "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo",
            "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es",
            "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw",
            "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja",
            "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo",
            "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
            "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt",
            "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq",
            "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl",
            "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh", "yue",
        ]
