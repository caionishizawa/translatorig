"""
SeamlessM4T translation engine from Meta.
High-quality multilingual translation with speech support.
"""

from pathlib import Path
from typing import Optional, Callable, List
import torch

from .base import BaseTranslator
from ..utils.models import TranscriptionResult, TranslationResult, TranslationSegment
from ..config import settings, SUPPORTED_LANGUAGES


class SeamlessTranslator(BaseTranslator):
    """
    Translation using Meta's SeamlessM4T model.

    SeamlessM4T is a massively multilingual model that supports:
    - Speech-to-speech translation
    - Speech-to-text translation
    - Text-to-speech translation
    - Text-to-text translation

    Supports 100+ languages with high quality.
    """

    # Map our language codes to SeamlessM4T codes
    LANG_CODE_MAP = {
        "por": "por",  # Portuguese
        "eng": "eng",  # English
        "spa": "spa",  # Spanish
        "fra": "fra",  # French
        "deu": "deu",  # German
        "ita": "ita",  # Italian
        "zho": "cmn",  # Chinese Mandarin
        "jpn": "jpn",  # Japanese
        "kor": "kor",  # Korean
        "rus": "rus",  # Russian
        "ara": "arb",  # Arabic
        "hin": "hin",  # Hindi
        "tur": "tur",  # Turkish
        "vie": "vie",  # Vietnamese
        "tha": "tha",  # Thai
        "nld": "nld",  # Dutch
        "pol": "pol",  # Polish
        "swe": "swe",  # Swedish
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize SeamlessM4T translator.

        Args:
            model_name: Model name (seamlessM4T_v2_large, seamlessM4T_medium, etc.)
            device: Device to use (cpu, cuda, auto)
        """
        super().__init__()

        self.model_name = model_name or settings.SEAMLESS_MODEL
        self.device = device or settings.WHISPER_DEVICE

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.processor = None

    def _load_model(self):
        """Load SeamlessM4T model (lazy loading)."""
        if self.model is None:
            self._update_progress(0.0, "Loading SeamlessM4T model")

            try:
                from transformers import AutoProcessor, SeamlessM4TModel

                # Try loading from Hugging Face
                model_id = f"facebook/{self.model_name}"

                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model = SeamlessM4TModel.from_pretrained(model_id)
                self.model.to(self.device)

            except ImportError:
                # Fallback to seamless_communication library
                self._load_seamless_communication()

            self._update_progress(0.1, "Model loaded")

    def _load_seamless_communication(self):
        """Load using Meta's seamless_communication library."""
        try:
            import seamless_communication
            from seamless_communication.models.inference import Translator

            self.model = Translator(
                model_name_or_card=self.model_name,
                vocoder_name_or_card="vocoder_36langs",
                device=torch.device(self.device),
            )
        except ImportError:
            raise ImportError(
                "Please install seamless_communication: "
                "pip install git+https://github.com/facebookresearch/seamless_communication.git"
            )

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
            transcription: TranscriptionResult with segments
            source_lang: Source language code
            target_lang: Target language code
            preserve_timing: Preserve original timing

        Returns:
            TranslationResult with translated segments
        """
        self._load_model()

        self._update_progress(0.2, "Starting translation")

        # Map language codes
        src_code = self._map_language_code(source_lang)
        tgt_code = self._map_language_code(target_lang)

        translated_segments = []
        total_segments = len(transcription.segments)

        for i, segment in enumerate(transcription.segments):
            progress = 0.2 + (i / total_segments) * 0.7
            self._update_progress(progress, f"Translating segment {i + 1}/{total_segments}")

            # Translate the segment text
            translated_text = await self.translate_text(
                segment.text,
                src_code,
                tgt_code,
            )

            translated_segments.append(TranslationSegment(
                id=segment.id,
                original_text=segment.text,
                translated_text=translated_text,
                start=segment.start,
                end=segment.end,
                source_lang=source_lang,
                target_lang=target_lang,
            ))

        self._update_progress(1.0, "Translation complete")

        return self._create_translation_result(
            transcription,
            translated_segments,
            source_lang,
            target_lang,
        )

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
        self._load_model()

        src_code = self._map_language_code(source_lang)
        tgt_code = self._map_language_code(target_lang)

        try:
            # Try using transformers model
            if self.processor is not None:
                inputs = self.processor(text=text, src_lang=src_code, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                output_tokens = self.model.generate(
                    **inputs,
                    tgt_lang=tgt_code,
                    generate_speech=False,
                )

                translated_text = self.processor.decode(
                    output_tokens[0].tolist()[0],
                    skip_special_tokens=True,
                )
            else:
                # Use seamless_communication Translator
                translated_text, _, _ = self.model.predict(
                    text,
                    "t2tt",  # text-to-text translation
                    tgt_code,
                    src_lang=src_code,
                )

            return translated_text.strip()

        except Exception as e:
            raise RuntimeError(f"Translation failed: {e}")

    def _map_language_code(self, code: str) -> str:
        """Map our language code to SeamlessM4T code."""
        # Check our map first
        if code in self.LANG_CODE_MAP:
            return self.LANG_CODE_MAP[code]

        # Check SUPPORTED_LANGUAGES
        if code in SUPPORTED_LANGUAGES:
            return SUPPORTED_LANGUAGES[code].get("seamless_code", code)

        # Return as-is
        return code

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return list(self.LANG_CODE_MAP.keys())

    async def translate_with_audio(
        self,
        audio_path: Path,
        target_lang: str,
    ) -> tuple[str, Optional[Path]]:
        """
        Translate audio directly (speech-to-text translation).

        Args:
            audio_path: Path to audio file
            target_lang: Target language code

        Returns:
            Tuple of (translated_text, audio_path or None)
        """
        self._load_model()

        tgt_code = self._map_language_code(target_lang)

        try:
            # Use seamless_communication for S2TT
            if hasattr(self.model, 'predict'):
                translated_text, _, _ = self.model.predict(
                    str(audio_path),
                    "s2tt",  # speech-to-text translation
                    tgt_code,
                )
                return translated_text, None
            else:
                raise NotImplementedError(
                    "Direct audio translation requires seamless_communication library"
                )
        except Exception as e:
            raise RuntimeError(f"Audio translation failed: {e}")
