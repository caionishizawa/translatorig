"""
High-level voice cloning interface supporting multiple backends.
"""

from pathlib import Path
from typing import Optional, Callable, List, Literal
import numpy as np
import soundfile as sf

from ..utils.models import TranslationResult, SynthesizedAudio
from ..utils.file_handler import FileHandler
from ..config import settings


class VoiceCloner:
    """
    High-level voice cloning interface.

    Supports multiple TTS backends:
    - xtts: Coqui XTTS-v2 (recommended)
    - bark: Suno Bark (experimental)

    Provides a unified interface for voice cloning and synthesis.
    """

    def __init__(
        self,
        engine: Literal["xtts", "bark"] = "xtts",
        device: Optional[str] = None,
    ):
        """
        Initialize voice cloner.

        Args:
            engine: TTS engine to use
            device: Device (cpu, cuda, auto)
        """
        self.engine_name = engine
        self.device = device or settings.WHISPER_DEVICE

        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.engine = None
        self.progress_callback: Optional[Callable] = None

    def _load_engine(self):
        """Load the TTS engine."""
        if self.engine is None:
            if self.engine_name == "xtts":
                from .xtts import XTTSVoiceCloner
                self.engine = XTTSVoiceCloner(device=self.device)
            elif self.engine_name == "bark":
                self.engine = self._create_bark_engine()
            else:
                raise ValueError(f"Unknown engine: {self.engine_name}")

            if self.progress_callback:
                self.engine.set_progress_callback(self.progress_callback)

    def _create_bark_engine(self):
        """Create Bark TTS engine wrapper."""
        # Bark requires special handling
        raise NotImplementedError("Bark engine not yet implemented")

    def set_progress_callback(self, callback: Callable) -> None:
        """Set callback for progress updates."""
        self.progress_callback = callback
        if self.engine:
            self.engine.set_progress_callback(callback)

    async def clone_and_synthesize(
        self,
        translation: TranslationResult,
        voice_sample: Path,
        target_lang: str,
        output_path: Optional[Path] = None,
    ) -> SynthesizedAudio:
        """
        Clone voice and synthesize translated text.

        Args:
            translation: TranslationResult with translated segments
            voice_sample: Path to voice sample for cloning
            target_lang: Target language code
            output_path: Output path (optional)

        Returns:
            SynthesizedAudio with generated audio
        """
        self._load_engine()

        return await self.engine.synthesize(
            translation=translation,
            voice_sample=voice_sample,
            target_lang=target_lang,
            output_path=output_path,
        )

    async def extract_voice_embedding(
        self,
        audio_path: Path,
    ) -> np.ndarray:
        """
        Extract voice embedding from audio sample.

        Can be used to store and reuse voice profiles.

        Args:
            audio_path: Path to audio file

        Returns:
            Voice embedding array
        """
        self._load_engine()

        if hasattr(self.engine, 'extract_embedding'):
            return await self.engine.extract_embedding(audio_path)

        # Fallback: just return the audio samples
        audio, sr = sf.read(str(audio_path))
        return audio

    async def synthesize_from_embedding(
        self,
        text: str,
        voice_embedding: np.ndarray,
        target_lang: str,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Synthesize audio using pre-extracted voice embedding.

        Args:
            text: Text to synthesize
            voice_embedding: Pre-extracted voice embedding
            target_lang: Target language code
            output_path: Output path

        Returns:
            Path to synthesized audio
        """
        self._load_engine()

        if hasattr(self.engine, 'synthesize_from_embedding'):
            return await self.engine.synthesize_from_embedding(
                text, voice_embedding, target_lang, output_path
            )

        raise NotImplementedError("Engine does not support embedding synthesis")

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        self._load_engine()
        return self.engine.get_supported_languages()

    async def validate_voice_sample(
        self,
        audio_path: Path,
    ) -> dict:
        """
        Validate a voice sample for cloning quality.

        Args:
            audio_path: Path to voice sample

        Returns:
            Dict with validation results
        """
        import librosa

        audio, sr = librosa.load(str(audio_path), sr=None)
        duration = len(audio) / sr

        # Calculate audio quality metrics
        rms = np.sqrt(np.mean(audio ** 2))
        silence_threshold = 0.01
        speech_ratio = np.sum(np.abs(audio) > silence_threshold) / len(audio)

        # Check for clipping
        clipping_ratio = np.sum(np.abs(audio) > 0.99) / len(audio)

        return {
            "valid": duration >= 3.0 and speech_ratio > 0.3,
            "duration": duration,
            "sample_rate": sr,
            "rms_level": float(rms),
            "speech_ratio": float(speech_ratio),
            "clipping_ratio": float(clipping_ratio),
            "recommendations": self._get_sample_recommendations(
                duration, speech_ratio, rms, clipping_ratio
            ),
        }

    def _get_sample_recommendations(
        self,
        duration: float,
        speech_ratio: float,
        rms: float,
        clipping_ratio: float,
    ) -> List[str]:
        """Generate recommendations for improving voice sample."""
        recommendations = []

        if duration < 6:
            recommendations.append(
                "Voice sample is short. 6-10 seconds recommended for best results."
            )
        elif duration > 30:
            recommendations.append(
                "Voice sample is long. Consider trimming to 10-15 seconds of clean speech."
            )

        if speech_ratio < 0.5:
            recommendations.append(
                "Sample contains significant silence. Use a segment with continuous speech."
            )

        if rms < 0.05:
            recommendations.append(
                "Audio level is low. Consider normalizing or recording at higher volume."
            )
        elif rms > 0.5:
            recommendations.append(
                "Audio level is high. Risk of distortion. Consider lowering input gain."
            )

        if clipping_ratio > 0.01:
            recommendations.append(
                "Audio shows clipping/distortion. Re-record at lower volume."
            )

        if not recommendations:
            recommendations.append("Voice sample quality is good!")

        return recommendations
