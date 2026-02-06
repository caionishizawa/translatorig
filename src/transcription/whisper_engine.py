"""
OpenAI Whisper transcription engine.
"""

from pathlib import Path
from typing import Optional, Callable, List
import torch

from ..utils.models import TranscriptionResult, TranscriptionSegment
from ..config import settings


class WhisperEngine:
    """
    Speech-to-text transcription using OpenAI Whisper.

    Supports multiple model sizes for quality/speed tradeoff:
    - tiny: Fastest, lowest accuracy
    - base: Fast, low accuracy
    - small: Balanced
    - medium: Good accuracy
    - large: Best accuracy
    - large-v3: Latest, best accuracy
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize Whisper engine.

        Args:
            model_name: Whisper model name (tiny, base, small, medium, large, large-v3)
            device: Device to use (cpu, cuda, auto)
        """
        self.model_name = model_name or settings.WHISPER_MODEL
        self.device = device or settings.WHISPER_DEVICE

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.progress_callback: Optional[Callable] = None

    def _load_model(self):
        """Load Whisper model (lazy loading)."""
        if self.model is None:
            import whisper

            self._update_progress(0.0, f"Loading Whisper {self.model_name} model")
            self.model = whisper.load_model(
                self.model_name,
                device=self.device,
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
        task: str = "transcribe",  # "transcribe" or "translate"
    ) -> TranscriptionResult:
        """
        Transcribe audio file to text with timestamps.

        Args:
            audio_path: Path to audio file
            language: Source language code (None for auto-detection)
            word_timestamps: Include word-level timestamps
            task: "transcribe" for same-language, "translate" for English output

        Returns:
            TranscriptionResult with segments and timestamps
        """
        self._load_model()

        self._update_progress(0.2, "Starting transcription")

        # Transcription options
        options = {
            "task": task,
            "word_timestamps": word_timestamps,
            "verbose": False,
        }

        if language:
            options["language"] = language

        # Run transcription
        self._update_progress(0.3, "Processing audio")

        result = self.model.transcribe(
            str(audio_path),
            **options,
        )

        self._update_progress(0.8, "Processing results")

        # Convert to our format
        segments = self._convert_segments(result["segments"], word_timestamps)

        # Calculate total duration
        duration = segments[-1].end if segments else 0.0

        self._update_progress(1.0, "Transcription complete")

        return TranscriptionResult(
            segments=segments,
            language=result.get("language", "unknown"),
            language_probability=result.get("language_probability", 0.0) if "language_probability" in result else 1.0,
            duration=duration,
            audio_path=audio_path,
        )

    def _convert_segments(
        self,
        whisper_segments: List[dict],
        include_words: bool,
    ) -> List[TranscriptionSegment]:
        """Convert Whisper segments to our format."""
        segments = []

        for i, seg in enumerate(whisper_segments):
            words = None
            if include_words and "words" in seg:
                words = [
                    {
                        "word": w.get("word", "").strip(),
                        "start": w.get("start", 0),
                        "end": w.get("end", 0),
                        "probability": w.get("probability", 1.0),
                    }
                    for w in seg["words"]
                ]

            segments.append(TranscriptionSegment(
                id=i + 1,
                text=seg["text"].strip(),
                start=seg["start"],
                end=seg["end"],
                words=words,
                confidence=seg.get("avg_logprob", 0.0),
            ))

        return segments

    async def detect_language(self, audio_path: Path) -> tuple[str, float]:
        """
        Detect the language of an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (language_code, probability)
        """
        self._load_model()

        import whisper

        # Load audio and get mel spectrogram
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)

        # Detect language
        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)

        return detected_lang, probs[detected_lang]

    def get_available_languages(self) -> List[str]:
        """Get list of languages supported by Whisper."""
        import whisper
        return list(whisper.tokenizer.LANGUAGES.keys())
