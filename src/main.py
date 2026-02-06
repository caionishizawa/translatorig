"""
Video Translator - Main Pipeline Orchestrator

Complete pipeline for extracting, transcribing, translating,
dubbing, and rendering videos with lip-sync.
"""

import asyncio
from pathlib import Path
from typing import Optional, Callable, Literal
from dataclasses import dataclass, field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table

from .extractors import InstagramExtractor, YouTubeExtractor, LocalExtractor
from .transcription import WhisperEngine, FasterWhisperEngine
from .translation import SeamlessTranslator, NLLBTranslator
from .voice import XTTSVoiceCloner
from .lipsync import Wav2LipProcessor
from .renderer import FFmpegRenderer
from .utils.models import VideoData, TranscriptionResult, TranslationResult, SynthesizedAudio, RenderResult
from .utils.file_handler import FileHandler
from .utils.progress import ProgressTracker
from .config import settings, SUPPORTED_LANGUAGES

console = Console()


@dataclass
class PipelineResult:
    """Result of the complete translation pipeline."""
    output_path: Path
    original_duration: float
    output_duration: float
    original_resolution: tuple[int, int]
    output_resolution: tuple[int, int]
    source_language: str
    target_language: str
    file_size_mb: float
    has_subtitles: bool
    steps_completed: list = field(default_factory=list)


class VideoTranslatorPipeline:
    """
    Complete pipeline for video translation with:
    - Extraction from Instagram/YouTube/local files
    - Transcription with Whisper
    - Translation with SeamlessM4T/NLLB
    - Voice cloning with XTTS
    - Lip-sync with Wav2Lip
    - High-quality rendering with FFmpeg
    """

    def __init__(
        self,
        transcription_engine: Literal["whisper", "faster-whisper"] = "faster-whisper",
        translation_engine: Literal["seamless", "nllb"] = "seamless",
        use_gpu: bool = True,
    ):
        """
        Initialize the video translation pipeline.

        Args:
            transcription_engine: Speech-to-text engine
            translation_engine: Translation engine
            use_gpu: Use GPU acceleration if available
        """
        self.device = "cuda" if use_gpu else "cpu"

        # Initialize components (lazy loading)
        self._transcriber = None
        self._translator = None
        self._voice_cloner = None
        self._lipsync = None
        self._renderer = None

        self.transcription_engine = transcription_engine
        self.translation_engine_name = translation_engine

        self.progress_tracker = ProgressTracker()

    @property
    def transcriber(self):
        """Lazy load transcription engine."""
        if self._transcriber is None:
            if self.transcription_engine == "faster-whisper":
                self._transcriber = FasterWhisperEngine(device=self.device)
            else:
                self._transcriber = WhisperEngine(device=self.device)
        return self._transcriber

    @property
    def translator(self):
        """Lazy load translation engine."""
        if self._translator is None:
            if self.translation_engine_name == "nllb":
                self._translator = NLLBTranslator(device=self.device)
            else:
                self._translator = SeamlessTranslator(device=self.device)
        return self._translator

    @property
    def voice_cloner(self):
        """Lazy load voice cloner."""
        if self._voice_cloner is None:
            self._voice_cloner = XTTSVoiceCloner(device=self.device)
        return self._voice_cloner

    @property
    def lipsync(self):
        """Lazy load lip-sync processor."""
        if self._lipsync is None:
            self._lipsync = Wav2LipProcessor(device=self.device)
        return self._lipsync

    @property
    def renderer(self):
        """Lazy load renderer."""
        if self._renderer is None:
            self._renderer = FFmpegRenderer()
        return self._renderer

    def _get_extractor(self, source: str):
        """Get appropriate extractor for source."""
        if source.startswith(("http://", "https://")):
            if "instagram.com" in source or "instagr.am" in source:
                return InstagramExtractor()
            elif "youtube.com" in source or "youtu.be" in source:
                return YouTubeExtractor()
            else:
                # Try Instagram extractor for other URLs
                return InstagramExtractor()
        else:
            return LocalExtractor()

    async def process(
        self,
        source: str,
        target_language: str = "por",
        output_path: Optional[Path] = None,
        include_subtitles: bool = True,
        subtitle_style: Literal["soft", "hardcoded"] = "soft",
        output_resolution: Literal["4k", "1080p", "720p", "original"] = "original",
        preserve_original_audio: bool = False,
        skip_lipsync: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> PipelineResult:
        """
        Process video through complete translation pipeline.

        Args:
            source: URL or path to video
            target_language: Target language code (e.g., 'por', 'eng', 'spa')
            output_path: Output file path (auto-generated if not provided)
            include_subtitles: Include subtitles in output
            subtitle_style: Subtitle embedding style ('soft' or 'hardcoded')
            output_resolution: Output resolution ('4k', '1080p', '720p', 'original')
            preserve_original_audio: Keep original audio as secondary track
            skip_lipsync: Skip lip-sync processing (faster but no lip sync)
            progress_callback: Callback for progress updates

        Returns:
            PipelineResult with output details
        """
        settings.ensure_directories()

        if progress_callback:
            self.progress_tracker.callback = progress_callback

        steps_completed = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:

            # Create progress tasks
            tasks = {
                "extraction": progress.add_task("[cyan]Extracting video...", total=100),
                "transcription": progress.add_task("[yellow]Transcribing...", total=100),
                "translation": progress.add_task("[green]Translating...", total=100),
                "synthesis": progress.add_task("[magenta]Synthesizing voice...", total=100),
                "lipsync": progress.add_task("[blue]Lip-syncing...", total=100),
                "rendering": progress.add_task("[red]Rendering...", total=100),
            }

            # ==================== STEP 1: EXTRACTION ====================
            console.print("\n[bold cyan]📥 Step 1/6: Video Extraction[/]")
            self.progress_tracker.start_step("extraction", "Extracting video")

            extractor = self._get_extractor(source)
            extractor.set_progress_callback(
                lambda p, m: progress.update(tasks["extraction"], completed=p * 100)
            )

            video_data = await extractor.extract(
                source,
                max_quality=True,
                extract_audio=True,
            )

            progress.update(tasks["extraction"], completed=100)
            self.progress_tracker.complete_step("extraction")
            steps_completed.append("extraction")

            console.print(f"  ✓ Resolution: {video_data.resolution_str}")
            console.print(f"  ✓ Duration: {video_data.duration:.1f}s")
            console.print(f"  ✓ FPS: {video_data.fps}")

            # ==================== STEP 2: TRANSCRIPTION ====================
            console.print("\n[bold yellow]🎤 Step 2/6: Transcription[/]")
            self.progress_tracker.start_step("transcription", "Transcribing audio")

            self.transcriber.set_progress_callback(
                lambda p, m: progress.update(tasks["transcription"], completed=p * 100)
            )

            transcription = await self.transcriber.transcribe(
                video_data.audio_path,
                word_timestamps=True,
            )

            progress.update(tasks["transcription"], completed=100)
            self.progress_tracker.complete_step("transcription")
            steps_completed.append("transcription")

            console.print(f"  ✓ Detected language: {transcription.language}")
            console.print(f"  ✓ Segments: {len(transcription.segments)}")

            # ==================== STEP 3: TRANSLATION ====================
            console.print("\n[bold green]🌍 Step 3/6: Translation[/]")
            self.progress_tracker.start_step("translation", "Translating text")

            self.translator.set_progress_callback(
                lambda p, m: progress.update(tasks["translation"], completed=p * 100)
            )

            translation = await self.translator.translate(
                transcription,
                source_lang=transcription.language,
                target_lang=target_language,
                preserve_timing=True,
            )

            progress.update(tasks["translation"], completed=100)
            self.progress_tracker.complete_step("translation")
            steps_completed.append("translation")

            target_lang_name = SUPPORTED_LANGUAGES.get(target_language, {}).get("name", target_language)
            console.print(f"  ✓ Translated to: {target_lang_name}")

            # ==================== STEP 4: VOICE SYNTHESIS ====================
            console.print("\n[bold magenta]🗣️ Step 4/6: Voice Synthesis[/]")
            self.progress_tracker.start_step("voice_synthesis", "Cloning voice and synthesizing")

            self.voice_cloner.set_progress_callback(
                lambda p, m: progress.update(tasks["synthesis"], completed=p * 100)
            )

            dubbed_audio = await self.voice_cloner.synthesize(
                translation,
                voice_sample=video_data.audio_path,
                target_lang=target_language,
            )

            progress.update(tasks["synthesis"], completed=100)
            self.progress_tracker.complete_step("voice_synthesis")
            steps_completed.append("voice_synthesis")

            console.print(f"  ✓ Generated audio: {dubbed_audio.duration:.1f}s")

            # ==================== STEP 5: LIP-SYNC ====================
            if not skip_lipsync:
                console.print("\n[bold blue]👄 Step 5/6: Lip Synchronization[/]")
                self.progress_tracker.start_step("lip_sync", "Synchronizing lips")

                self.lipsync.set_progress_callback(
                    lambda p, m: progress.update(tasks["lipsync"], completed=p * 100)
                )

                lipsync_result = await self.lipsync.process(
                    video_path=video_data.video_path,
                    audio_path=dubbed_audio.path,
                    quality=settings.LIPSYNC_QUALITY,
                )

                synced_video_path = lipsync_result.path
                progress.update(tasks["lipsync"], completed=100)
                self.progress_tracker.complete_step("lip_sync")
                steps_completed.append("lip_sync")

                console.print(f"  ✓ Faces detected: {lipsync_result.faces_detected}")
            else:
                console.print("\n[bold blue]👄 Step 5/6: Lip Sync (Skipped)[/]")
                synced_video_path = video_data.video_path
                progress.update(tasks["lipsync"], completed=100)

            # ==================== STEP 6: RENDERING ====================
            console.print("\n[bold red]🎬 Step 6/6: Final Rendering[/]")
            self.progress_tracker.start_step("rendering", "Rendering final video")

            self.renderer.set_progress_callback(
                lambda p, m: progress.update(tasks["rendering"], completed=p * 100)
            )

            # Prepare subtitles
            subtitles_srt = translation.to_srt() if include_subtitles else None

            # Determine output path
            if output_path is None:
                output_path = FileHandler.get_output_path(
                    Path(source).name if not source.startswith("http") else "video.mp4",
                    suffix=f"_{target_language}"
                )

            render_result = await self.renderer.render(
                video_path=synced_video_path,
                audio_path=dubbed_audio.path,
                output_path=output_path,
                subtitles=subtitles_srt,
                subtitle_style=subtitle_style,
                resolution=output_resolution,
                original_audio=video_data.audio_path if preserve_original_audio else None,
            )

            progress.update(tasks["rendering"], completed=100)
            self.progress_tracker.complete_step("rendering")
            steps_completed.append("rendering")

            console.print(f"  ✓ Output: {render_result.resolution_str}")
            console.print(f"  ✓ Size: {render_result.file_size_mb:.1f} MB")

        # Clean up temporary files
        if settings.CLEANUP_TEMP_FILES:
            FileHandler.cleanup_temp_files()

        # Print summary
        self._print_summary(video_data, render_result, transcription.language, target_language)

        return PipelineResult(
            output_path=render_result.path,
            original_duration=video_data.duration,
            output_duration=render_result.duration,
            original_resolution=video_data.resolution,
            output_resolution=render_result.resolution,
            source_language=transcription.language,
            target_language=target_language,
            file_size_mb=render_result.file_size_mb,
            has_subtitles=include_subtitles,
            steps_completed=steps_completed,
        )

    def _print_summary(
        self,
        video_data: VideoData,
        render_result: RenderResult,
        source_lang: str,
        target_lang: str,
    ):
        """Print processing summary."""
        table = Table(title="Processing Summary", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Source Language", SUPPORTED_LANGUAGES.get(source_lang, {}).get("name", source_lang))
        table.add_row("Target Language", SUPPORTED_LANGUAGES.get(target_lang, {}).get("name", target_lang))
        table.add_row("Original Resolution", video_data.resolution_str)
        table.add_row("Output Resolution", render_result.resolution_str)
        table.add_row("Duration", f"{render_result.duration:.1f}s")
        table.add_row("Output Size", f"{render_result.file_size_mb:.1f} MB")
        table.add_row("Video Codec", render_result.video_codec)
        table.add_row("Audio Codec", render_result.audio_codec)

        console.print("\n")
        console.print(table)
        console.print(f"\n[bold green]✅ Video saved to:[/] {render_result.path}\n")


# CLI Interface
async def main():
    """Command-line interface for Video Translator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Video Translator - Translate videos with AI dubbing and lip-sync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate Instagram reel to Portuguese
  python -m src.main "https://www.instagram.com/reel/ABC123/" -l por

  # Translate local video to English with 4K output
  python -m src.main ~/video.mp4 -l eng -r 4k

  # Translate without lip-sync (faster)
  python -m src.main video.mp4 -l spa --skip-lipsync

  # Keep original audio as secondary track
  python -m src.main video.mp4 -l fra --keep-original

Supported Languages:
  por - Portuguese (Brazil)    eng - English
  spa - Spanish                fra - French
  deu - German                 ita - Italian
  zho - Chinese                jpn - Japanese
  kor - Korean                 rus - Russian
        """
    )

    parser.add_argument(
        "source",
        help="URL (Instagram/YouTube) or path to video file"
    )
    parser.add_argument(
        "-l", "--language",
        default="por",
        help="Target language code (default: por)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path"
    )
    parser.add_argument(
        "-r", "--resolution",
        default="original",
        choices=["4k", "1080p", "720p", "original"],
        help="Output resolution (default: original)"
    )
    parser.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Don't include subtitles"
    )
    parser.add_argument(
        "--hardcoded-subs",
        action="store_true",
        help="Burn subtitles into video"
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep original audio as secondary track"
    )
    parser.add_argument(
        "--skip-lipsync",
        action="store_true",
        help="Skip lip-sync processing (faster)"
    )
    parser.add_argument(
        "--transcription-engine",
        default="faster-whisper",
        choices=["whisper", "faster-whisper"],
        help="Transcription engine (default: faster-whisper)"
    )
    parser.add_argument(
        "--translation-engine",
        default="seamless",
        choices=["seamless", "nllb"],
        help="Translation engine (default: seamless)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU"
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List supported languages and exit"
    )

    args = parser.parse_args()

    # List languages
    if args.list_languages:
        console.print("\n[bold]Supported Languages:[/]\n")
        for code, info in SUPPORTED_LANGUAGES.items():
            console.print(f"  [cyan]{code}[/] - {info['name']}")
        console.print()
        return

    # Validate language
    if args.language not in SUPPORTED_LANGUAGES:
        console.print(f"[red]Error:[/] Unknown language code: {args.language}")
        console.print(f"Use --list-languages to see available options")
        return

    # Banner
    console.print(Panel.fit(
        "[bold blue]Video Translator[/]\n"
        "[dim]AI-powered video translation with dubbing and lip-sync[/]",
        border_style="blue"
    ))

    # Initialize pipeline
    pipeline = VideoTranslatorPipeline(
        transcription_engine=args.transcription_engine,
        translation_engine=args.translation_engine,
        use_gpu=not args.cpu,
    )

    # Process video
    try:
        result = await pipeline.process(
            source=args.source,
            target_language=args.language,
            output_path=Path(args.output) if args.output else None,
            include_subtitles=not args.no_subtitles,
            subtitle_style="hardcoded" if args.hardcoded_subs else "soft",
            output_resolution=args.resolution,
            preserve_original_audio=args.keep_original,
            skip_lipsync=args.skip_lipsync,
        )

        console.print(f"\n[bold green]Success![/] Video translated and saved to:")
        console.print(f"  {result.output_path}")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/] {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
