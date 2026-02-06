"""
NLLB-200 (No Language Left Behind) translation engine from Meta.
Supports 200+ languages for text translation.
"""

from typing import Optional, List
import torch

from .base import BaseTranslator
from ..utils.models import TranscriptionResult, TranslationResult, TranslationSegment
from ..config import settings


class NLLBTranslator(BaseTranslator):
    """
    Translation using Meta's NLLB-200 model.

    NLLB-200 supports 200+ languages for text-to-text translation.
    Available model sizes:
    - nllb-200-distilled-600M: Smallest, fastest
    - nllb-200-distilled-1.3B: Medium size
    - nllb-200-1.3B: Full 1.3B model
    - nllb-200-3.3B: Largest, highest quality
    """

    # NLLB language code format: xxx_Xxxx (language_Script)
    LANG_CODE_MAP = {
        "por": "por_Latn",  # Portuguese
        "eng": "eng_Latn",  # English
        "spa": "spa_Latn",  # Spanish
        "fra": "fra_Latn",  # French
        "deu": "deu_Latn",  # German
        "ita": "ita_Latn",  # Italian
        "zho": "zho_Hans",  # Chinese Simplified
        "jpn": "jpn_Jpan",  # Japanese
        "kor": "kor_Hang",  # Korean
        "rus": "rus_Cyrl",  # Russian
        "ara": "arb_Arab",  # Arabic
        "hin": "hin_Deva",  # Hindi
        "tur": "tur_Latn",  # Turkish
        "vie": "vie_Latn",  # Vietnamese
        "tha": "tha_Thai",  # Thai
        "nld": "nld_Latn",  # Dutch
        "pol": "pol_Latn",  # Polish
        "swe": "swe_Latn",  # Swedish
        "ukr": "ukr_Cyrl",  # Ukrainian
        "ces": "ces_Latn",  # Czech
        "ron": "ron_Latn",  # Romanian
        "hun": "hun_Latn",  # Hungarian
        "ell": "ell_Grek",  # Greek
        "heb": "heb_Hebr",  # Hebrew
        "ind": "ind_Latn",  # Indonesian
        "msa": "zsm_Latn",  # Malay
        "fil": "tgl_Latn",  # Filipino/Tagalog
    }

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: Optional[str] = None,
    ):
        """
        Initialize NLLB translator.

        Args:
            model_name: HuggingFace model name
            device: Device to use (cpu, cuda, auto)
        """
        super().__init__()

        self.model_name = model_name
        self.device = device or settings.WHISPER_DEVICE

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.tokenizer = None

    def _load_model(self):
        """Load NLLB model (lazy loading)."""
        if self.model is None:
            self._update_progress(0.0, "Loading NLLB model")

            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)

            self._update_progress(0.1, "Model loaded")

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

        translated_segments = []
        total_segments = len(transcription.segments)

        # Batch translation for efficiency
        batch_size = 8
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

        # Set source language
        self.tokenizer.src_lang = src_code

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate translation
        forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_code]

        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=512,
        )

        # Decode
        translated_text = self.tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True,
        )[0]

        return translated_text.strip()

    async def _translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
    ) -> List[str]:
        """Translate a batch of texts."""
        self._load_model()

        src_code = self._map_language_code(source_lang)
        tgt_code = self._map_language_code(target_lang)

        # Set source language
        self.tokenizer.src_lang = src_code

        # Tokenize batch
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate translations
        forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_code]

        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=512,
        )

        # Decode batch
        translated_texts = self.tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True,
        )

        return [t.strip() for t in translated_texts]

    def _map_language_code(self, code: str) -> str:
        """Map our language code to NLLB format."""
        if code in self.LANG_CODE_MAP:
            return self.LANG_CODE_MAP[code]

        # Try to find a match with script suffix
        for nllb_code in self.LANG_CODE_MAP.values():
            if nllb_code.startswith(code):
                return nllb_code

        # Return with Latin script as default
        return f"{code}_Latn"

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return list(self.LANG_CODE_MAP.keys())
