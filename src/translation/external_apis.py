"""
External API-based translation engines (DeepL, Google Translate).
"""

from typing import Optional, List
import httpx

from .base import BaseTranslator
from ..utils.models import TranscriptionResult, TranslationResult, TranslationSegment
from ..config import settings


class DeepLTranslator(BaseTranslator):
    """
    Translation using DeepL API.

    DeepL offers high-quality neural machine translation
    with support for 30+ languages.

    Requires DEEPL_API_KEY in settings.
    """

    API_URL = "https://api-free.deepl.com/v2/translate"  # Free tier
    API_URL_PRO = "https://api.deepl.com/v2/translate"  # Pro tier

    LANG_CODE_MAP = {
        "por": "PT-BR",  # Portuguese (Brazil)
        "eng": "EN",     # English
        "spa": "ES",     # Spanish
        "fra": "FR",     # French
        "deu": "DE",     # German
        "ita": "IT",     # Italian
        "zho": "ZH",     # Chinese
        "jpn": "JA",     # Japanese
        "kor": "KO",     # Korean (Pro only)
        "rus": "RU",     # Russian
        "nld": "NL",     # Dutch
        "pol": "PL",     # Polish
        "swe": "SV",     # Swedish
        "tur": "TR",     # Turkish (Pro only)
        "ukr": "UK",     # Ukrainian
        "ces": "CS",     # Czech
        "rom": "RO",     # Romanian
        "hun": "HU",     # Hungarian
        "ell": "EL",     # Greek
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_pro: bool = False,
    ):
        """
        Initialize DeepL translator.

        Args:
            api_key: DeepL API key (uses settings if not provided)
            use_pro: Use Pro API endpoint
        """
        super().__init__()

        self.api_key = api_key or settings.DEEPL_API_KEY
        if not self.api_key:
            raise ValueError("DeepL API key required. Set DEEPL_API_KEY in .env")

        self.api_url = self.API_URL_PRO if use_pro else self.API_URL
        self.client = httpx.AsyncClient(timeout=30.0)

    async def translate(
        self,
        transcription: TranscriptionResult,
        source_lang: str,
        target_lang: str,
        preserve_timing: bool = True,
    ) -> TranslationResult:
        """
        Translate transcription using DeepL API.

        Args:
            transcription: TranscriptionResult with segments
            source_lang: Source language code
            target_lang: Target language code
            preserve_timing: Preserve original timing

        Returns:
            TranslationResult with translated segments
        """
        self._update_progress(0.2, "Starting DeepL translation")

        translated_segments = []
        total_segments = len(transcription.segments)

        # Batch translation (DeepL supports up to 50 texts per request)
        batch_size = 50
        for batch_start in range(0, total_segments, batch_size):
            batch_end = min(batch_start + batch_size, total_segments)
            batch = transcription.segments[batch_start:batch_end]

            progress = 0.2 + (batch_start / total_segments) * 0.7
            self._update_progress(progress, f"Translating segments {batch_start + 1}-{batch_end}")

            # Translate batch
            texts = [seg.text for seg in batch]
            translated_texts = await self._translate_batch(texts, source_lang, target_lang)

            # Create translated segments
            for seg, translated_text in zip(batch, translated_texts):
                translated_segments.append(TranslationSegment(
                    id=seg.id,
                    original_text=seg.text,
                    translated_text=translated_text,
                    start=seg.start,
                    end=seg.end,
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
        """Translate a single text string."""
        results = await self._translate_batch([text], source_lang, target_lang)
        return results[0] if results else text

    async def _translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
    ) -> List[str]:
        """Translate a batch of texts via DeepL API."""
        src_code = self._map_language_code(source_lang)
        tgt_code = self._map_language_code(target_lang)

        response = await self.client.post(
            self.api_url,
            data={
                "auth_key": self.api_key,
                "text": texts,
                "source_lang": src_code,
                "target_lang": tgt_code,
            },
        )

        if response.status_code != 200:
            raise RuntimeError(f"DeepL API error: {response.status_code} - {response.text}")

        data = response.json()
        return [t["text"] for t in data.get("translations", [])]

    def _map_language_code(self, code: str) -> str:
        """Map our language code to DeepL format."""
        return self.LANG_CODE_MAP.get(code, code.upper())

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return list(self.LANG_CODE_MAP.keys())

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class GoogleTranslator(BaseTranslator):
    """
    Translation using Google Cloud Translation API.

    Requires GOOGLE_API_KEY in settings or Google Cloud credentials.
    """

    API_URL = "https://translation.googleapis.com/language/translate/v2"

    LANG_CODE_MAP = {
        "por": "pt",   # Portuguese
        "eng": "en",   # English
        "spa": "es",   # Spanish
        "fra": "fr",   # French
        "deu": "de",   # German
        "ita": "it",   # Italian
        "zho": "zh",   # Chinese
        "jpn": "ja",   # Japanese
        "kor": "ko",   # Korean
        "rus": "ru",   # Russian
        "ara": "ar",   # Arabic
        "hin": "hi",   # Hindi
        "tur": "tr",   # Turkish
        "vie": "vi",   # Vietnamese
        "tha": "th",   # Thai
        "nld": "nl",   # Dutch
        "pol": "pl",   # Polish
        "swe": "sv",   # Swedish
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Google Translator.

        Args:
            api_key: Google Cloud API key (uses settings if not provided)
        """
        super().__init__()

        self.api_key = api_key or settings.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY in .env")

        self.client = httpx.AsyncClient(timeout=30.0)

    async def translate(
        self,
        transcription: TranscriptionResult,
        source_lang: str,
        target_lang: str,
        preserve_timing: bool = True,
    ) -> TranslationResult:
        """
        Translate transcription using Google Translate API.

        Args:
            transcription: TranscriptionResult with segments
            source_lang: Source language code
            target_lang: Target language code
            preserve_timing: Preserve original timing

        Returns:
            TranslationResult with translated segments
        """
        self._update_progress(0.2, "Starting Google translation")

        translated_segments = []
        total_segments = len(transcription.segments)

        # Batch translation
        batch_size = 128  # Google allows up to 128 segments
        for batch_start in range(0, total_segments, batch_size):
            batch_end = min(batch_start + batch_size, total_segments)
            batch = transcription.segments[batch_start:batch_end]

            progress = 0.2 + (batch_start / total_segments) * 0.7
            self._update_progress(progress, f"Translating segments {batch_start + 1}-{batch_end}")

            # Translate batch
            texts = [seg.text for seg in batch]
            translated_texts = await self._translate_batch(texts, source_lang, target_lang)

            # Create translated segments
            for seg, translated_text in zip(batch, translated_texts):
                translated_segments.append(TranslationSegment(
                    id=seg.id,
                    original_text=seg.text,
                    translated_text=translated_text,
                    start=seg.start,
                    end=seg.end,
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
        """Translate a single text string."""
        results = await self._translate_batch([text], source_lang, target_lang)
        return results[0] if results else text

    async def _translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
    ) -> List[str]:
        """Translate a batch of texts via Google API."""
        src_code = self._map_language_code(source_lang)
        tgt_code = self._map_language_code(target_lang)

        response = await self.client.post(
            self.API_URL,
            params={"key": self.api_key},
            json={
                "q": texts,
                "source": src_code,
                "target": tgt_code,
                "format": "text",
            },
        )

        if response.status_code != 200:
            raise RuntimeError(f"Google API error: {response.status_code} - {response.text}")

        data = response.json()
        translations = data.get("data", {}).get("translations", [])
        return [t.get("translatedText", "") for t in translations]

    def _map_language_code(self, code: str) -> str:
        """Map our language code to Google format."""
        return self.LANG_CODE_MAP.get(code, code)

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return list(self.LANG_CODE_MAP.keys())

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
