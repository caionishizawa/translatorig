"""
Base translator class.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, List

from ..utils.models import TranscriptionResult, TranslationResult, TranslationSegment


class BaseTranslator(ABC):
    """Abstract base class for translation engines."""

    def __init__(self):
        self.progress_callback: Optional[Callable] = None

    @abstractmethod
    async def translate(
        self,
        transcription: TranscriptionResult,
        source_lang: str,
        target_lang: str,
        preserve_timing: bool = True,
    ) -> TranslationResult:
        """
        Translate transcription to target language.

        Args:
            transcription: TranscriptionResult with segments to translate
            source_lang: Source language code
            target_lang: Target language code
            preserve_timing: Try to preserve original timing

        Returns:
            TranslationResult with translated segments
        """
        pass

    @abstractmethod
    async def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """
        Translate a single text string.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translated text
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        pass

    def set_progress_callback(self, callback: Callable) -> None:
        """Set callback for progress updates."""
        self.progress_callback = callback

    def _update_progress(self, progress: float, message: str = "") -> None:
        """Update progress via callback if set."""
        if self.progress_callback:
            self.progress_callback(progress, message)

    def _create_translation_result(
        self,
        transcription: TranscriptionResult,
        translated_segments: List[TranslationSegment],
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        """Create TranslationResult from translated segments."""
        return TranslationResult(
            segments=translated_segments,
            source_lang=source_lang,
            target_lang=target_lang,
            duration=transcription.duration,
        )
