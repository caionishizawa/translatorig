"""
XTTS (Coqui TTS) voice cloning and synthesis engine.
"""

from pathlib import Path
from typing import Optional, Callable, List
import torch
import numpy as np
import soundfile as sf

from ..utils.models import TranslationResult, SynthesizedAudio
from ..utils.file_handler import FileHandler
from ..config import settings, SUPPORTED_LANGUAGES


class XTTSVoiceCloner:
    """
    Voice synthesis with cloning using Coqui XTTS.

    XTTS-v2 features:
    - Voice cloning from 6-10 seconds of audio
    - Multilingual synthesis (17+ languages)
    - Emotion preservation
    - High-quality 24kHz output
    """

    # XTTS supported languages
    SUPPORTED_LANGS = [
        "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
        "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"
    ]

    LANG_CODE_MAP = {
        "por": "pt",
        "eng": "en",
        "spa": "es",
        "fra": "fr",
        "deu": "de",
        "ita": "it",
        "zho": "zh-cn",
        "jpn": "ja",
        "kor": "ko",
        "rus": "ru",
        "ara": "ar",
        "hin": "hi",
        "tur": "tr",
        "nld": "nl",
        "pol": "pl",
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize XTTS voice cloner.

        Args:
            model_name: Coqui TTS model name
            device: Device to use (cpu, cuda, auto)
        """
        self.model_name = model_name or settings.XTTS_MODEL
        self.device = device or settings.WHISPER_DEVICE

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tts = None
        self.progress_callback: Optional[Callable] = None

    def _load_model(self):
        """Load XTTS model (lazy loading)."""
        if self.tts is None:
            self._update_progress(0.0, "Loading XTTS model")

            from TTS.api import TTS

            # Load XTTS-v2 model
            self.tts = TTS(self.model_name)

            # Move to device
            if self.device == "cuda":
                self.tts.to("cuda")

            self._update_progress(0.1, "Model loaded")

    def set_progress_callback(self, callback: Callable) -> None:
        """Set callback for progress updates."""
        self.progress_callback = callback

    def _update_progress(self, progress: float, message: str = "") -> None:
        """Update progress via callback if set."""
        if self.progress_callback:
            self.progress_callback(progress, message)

    async def synthesize(
        self,
        translation: TranslationResult,
        voice_sample: Path,
        target_lang: str,
        output_path: Optional[Path] = None,
    ) -> SynthesizedAudio:
        """
        Synthesize translated text using cloned voice.

        Args:
            translation: TranslationResult with segments to synthesize
            voice_sample: Path to voice sample for cloning (6-10 seconds)
            target_lang: Target language code
            output_path: Output audio path (optional)

        Returns:
            SynthesizedAudio with path to generated audio
        """
        self._load_model()

        self._update_progress(0.2, "Preparing voice sample")

        # Prepare output path
        if output_path is None:
            output_path = FileHandler.get_temp_path(extension=".wav", prefix="synth_")

        # Map language code
        lang_code = self._map_language_code(target_lang)

        # Extract a good voice sample (trim to optimal length)
        voice_sample_processed = await self._prepare_voice_sample(voice_sample)

        self._update_progress(0.3, "Starting voice synthesis")

        # Synthesize each segment and concatenate
        audio_segments = []
        total_segments = len(translation.segments)

        for i, segment in enumerate(translation.segments):
            progress = 0.3 + (i / total_segments) * 0.6
            self._update_progress(progress, f"Synthesizing segment {i + 1}/{total_segments}")

            # Synthesize segment
            segment_audio = await self._synthesize_segment(
                segment.translated_text,
                voice_sample_processed,
                lang_code,
                target_duration=segment.duration,
            )

            audio_segments.append({
                "audio": segment_audio,
                "start": segment.start,
                "end": segment.end,
            })

        self._update_progress(0.9, "Assembling final audio")

        # Assemble final audio with proper timing
        final_audio = await self._assemble_audio(
            audio_segments,
            translation.duration,
        )

        # Save output
        sf.write(str(output_path), final_audio, 24000)

        self._update_progress(1.0, "Synthesis complete")

        return SynthesizedAudio(
            path=output_path,
            duration=len(final_audio) / 24000,
            sample_rate=24000,
            voice_cloned=True,
            target_lang=target_lang,
        )

    async def _synthesize_segment(
        self,
        text: str,
        voice_sample: Path,
        lang_code: str,
        target_duration: Optional[float] = None,
    ) -> np.ndarray:
        """Synthesize a single text segment."""
        # Generate audio using XTTS
        wav = self.tts.tts(
            text=text,
            speaker_wav=str(voice_sample),
            language=lang_code,
        )

        # Convert to numpy array
        audio = np.array(wav)

        # Adjust speed if target duration specified
        if target_duration is not None:
            actual_duration = len(audio) / 24000
            if abs(actual_duration - target_duration) > 0.5:
                audio = self._adjust_audio_speed(audio, actual_duration, target_duration)

        return audio

    async def _prepare_voice_sample(
        self,
        voice_sample: Path,
        target_duration: float = 10.0,
    ) -> Path:
        """
        Prepare voice sample for cloning.

        Extracts a clean segment of the specified duration.
        """
        import librosa

        # Load audio
        audio, sr = librosa.load(str(voice_sample), sr=24000)

        # Get duration
        duration = len(audio) / sr

        if duration > target_duration:
            # Find a good segment (avoid silence at start/end)
            # Use voice activity detection to find speech regions
            intervals = librosa.effects.split(audio, top_db=30)

            if len(intervals) > 0:
                # Use the longest speech interval
                longest_idx = np.argmax([end - start for start, end in intervals])
                start, end = intervals[longest_idx]

                # Ensure we have enough audio
                segment_length = int(target_duration * sr)
                if end - start >= segment_length:
                    audio = audio[start:start + segment_length]
                else:
                    # Extend with some context
                    center = (start + end) // 2
                    half_len = segment_length // 2
                    audio = audio[max(0, center - half_len):min(len(audio), center + half_len)]
            else:
                # Just take the first N seconds
                audio = audio[:int(target_duration * sr)]

        # Save processed sample
        output_path = FileHandler.get_temp_path(extension=".wav", prefix="voice_sample_")
        sf.write(str(output_path), audio, 24000)

        return output_path

    async def _assemble_audio(
        self,
        segments: List[dict],
        total_duration: float,
    ) -> np.ndarray:
        """
        Assemble audio segments with proper timing.

        Creates a final audio array with segments placed at their
        correct timestamps, filling gaps with silence.
        """
        sample_rate = 24000
        total_samples = int(total_duration * sample_rate)

        # Initialize with silence
        final_audio = np.zeros(total_samples, dtype=np.float32)

        for seg in segments:
            start_sample = int(seg["start"] * sample_rate)
            audio = seg["audio"]

            # Ensure we don't exceed bounds
            end_sample = min(start_sample + len(audio), total_samples)
            audio_len = end_sample - start_sample

            if audio_len > 0:
                final_audio[start_sample:end_sample] = audio[:audio_len]

        return final_audio

    def _adjust_audio_speed(
        self,
        audio: np.ndarray,
        actual_duration: float,
        target_duration: float,
    ) -> np.ndarray:
        """Adjust audio speed to match target duration."""
        import librosa

        # Calculate speed ratio
        speed_ratio = actual_duration / target_duration

        # Limit speed adjustment to reasonable range (0.7x to 1.3x)
        speed_ratio = max(0.7, min(1.3, speed_ratio))

        # Time stretch
        audio_stretched = librosa.effects.time_stretch(audio, rate=speed_ratio)

        # Trim or pad to exact target length
        target_samples = int(target_duration * 24000)
        if len(audio_stretched) > target_samples:
            audio_stretched = audio_stretched[:target_samples]
        elif len(audio_stretched) < target_samples:
            padding = np.zeros(target_samples - len(audio_stretched))
            audio_stretched = np.concatenate([audio_stretched, padding])

        return audio_stretched

    def _map_language_code(self, code: str) -> str:
        """Map our language code to XTTS format."""
        if code in self.LANG_CODE_MAP:
            return self.LANG_CODE_MAP[code]

        # Check SUPPORTED_LANGUAGES config
        if code in SUPPORTED_LANGUAGES:
            whisper_code = SUPPORTED_LANGUAGES[code].get("whisper_code", code)
            if whisper_code in self.SUPPORTED_LANGS:
                return whisper_code

        # Default to English if unsupported
        return "en"

    async def synthesize_text(
        self,
        text: str,
        voice_sample: Path,
        target_lang: str,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Synthesize a single text string with cloned voice.

        Args:
            text: Text to synthesize
            voice_sample: Path to voice sample
            target_lang: Target language code
            output_path: Output path (optional)

        Returns:
            Path to synthesized audio file
        """
        self._load_model()

        if output_path is None:
            output_path = FileHandler.get_temp_path(extension=".wav", prefix="synth_text_")

        lang_code = self._map_language_code(target_lang)
        voice_sample_processed = await self._prepare_voice_sample(voice_sample)

        # Generate audio
        wav = self.tts.tts(
            text=text,
            speaker_wav=str(voice_sample_processed),
            language=lang_code,
        )

        # Save
        sf.write(str(output_path), np.array(wav), 24000)

        return output_path

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return list(self.LANG_CODE_MAP.keys())
