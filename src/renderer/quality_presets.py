"""
Video quality presets for FFmpeg rendering.
"""

from typing import Optional, Literal

# Quality preset definitions
QUALITY_PRESETS = {
    "4k_max": {
        "name": "4K Maximum Quality",
        "resolution": (3840, 2160),
        "video_codec": "libx265",
        "video_crf": 16,
        "video_preset": "slow",
        "video_profile": "main10",
        "video_params": [
            "-pix_fmt", "yuv420p10le",
            "-x265-params", "aq-mode=3:rd=4",
        ],
        "audio_codec": "aac",
        "audio_bitrate": "320k",
        "audio_sample_rate": 48000,
        "audio_channels": 2,
    },
    "4k_balanced": {
        "name": "4K Balanced",
        "resolution": (3840, 2160),
        "video_codec": "libx265",
        "video_crf": 20,
        "video_preset": "medium",
        "video_profile": "main",
        "video_params": ["-pix_fmt", "yuv420p"],
        "audio_codec": "aac",
        "audio_bitrate": "256k",
        "audio_sample_rate": 48000,
        "audio_channels": 2,
    },
    "1080p_max": {
        "name": "1080p Maximum Quality",
        "resolution": (1920, 1080),
        "video_codec": "libx264",
        "video_crf": 18,
        "video_preset": "slow",
        "video_profile": "high",
        "video_params": ["-pix_fmt", "yuv420p"],
        "audio_codec": "aac",
        "audio_bitrate": "256k",
        "audio_sample_rate": 48000,
        "audio_channels": 2,
    },
    "1080p_balanced": {
        "name": "1080p Balanced",
        "resolution": (1920, 1080),
        "video_codec": "libx264",
        "video_crf": 20,
        "video_preset": "medium",
        "video_profile": "high",
        "video_params": ["-pix_fmt", "yuv420p"],
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "audio_sample_rate": 44100,
        "audio_channels": 2,
    },
    "1080p_fast": {
        "name": "1080p Fast",
        "resolution": (1920, 1080),
        "video_codec": "libx264",
        "video_crf": 22,
        "video_preset": "fast",
        "video_profile": "main",
        "video_params": ["-pix_fmt", "yuv420p"],
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "audio_sample_rate": 44100,
        "audio_channels": 2,
    },
    "720p_fast": {
        "name": "720p Fast",
        "resolution": (1280, 720),
        "video_codec": "libx264",
        "video_crf": 23,
        "video_preset": "fast",
        "video_profile": "main",
        "video_params": ["-pix_fmt", "yuv420p"],
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "audio_sample_rate": 44100,
        "audio_channels": 2,
    },
    "iphone_optimized": {
        "name": "iPhone Optimized (4K 30fps)",
        "resolution": (3840, 2160),
        "video_codec": "libx265",
        "video_crf": 18,
        "video_preset": "slow",
        "video_profile": "main",
        "video_params": [
            "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-movflags", "+faststart",
            "-tag:v", "hvc1",  # Apple HEVC compatibility
        ],
        "audio_codec": "aac",
        "audio_bitrate": "320k",
        "audio_sample_rate": 48000,
        "audio_channels": 2,
        "target_fps": 30,
    },
    "social_media": {
        "name": "Social Media (1080p)",
        "resolution": (1080, 1920),  # Vertical for stories/reels
        "video_codec": "libx264",
        "video_crf": 20,
        "video_preset": "medium",
        "video_profile": "high",
        "video_params": [
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ],
        "audio_codec": "aac",
        "audio_bitrate": "256k",
        "audio_sample_rate": 48000,
        "audio_channels": 2,
    },
    "web_streaming": {
        "name": "Web Streaming",
        "resolution": (1920, 1080),
        "video_codec": "libx264",
        "video_crf": 21,
        "video_preset": "medium",
        "video_profile": "main",
        "video_params": [
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-tune", "film",
        ],
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "audio_sample_rate": 48000,
        "audio_channels": 2,
    },
}

# Subtitle style presets
SUBTITLE_STYLES = {
    "default": {
        "fontname": "Arial",
        "fontsize": 24,
        "primary_color": "&HFFFFFF",  # White
        "outline_color": "&H000000",  # Black
        "outline_width": 2,
        "shadow_depth": 1,
        "margin_v": 30,
        "alignment": 2,  # Bottom center
    },
    "netflix": {
        "fontname": "Netflix Sans",
        "fontsize": 26,
        "primary_color": "&HFFFFFF",
        "outline_color": "&H000000",
        "outline_width": 3,
        "shadow_depth": 0,
        "margin_v": 50,
        "alignment": 2,
    },
    "youtube": {
        "fontname": "Roboto",
        "fontsize": 22,
        "primary_color": "&HFFFFFF",
        "back_color": "&H80000000",  # Semi-transparent black
        "outline_width": 0,
        "shadow_depth": 0,
        "margin_v": 20,
        "alignment": 2,
    },
    "bold": {
        "fontname": "Arial Black",
        "fontsize": 28,
        "primary_color": "&H00FFFF",  # Yellow
        "outline_color": "&H000000",
        "outline_width": 4,
        "shadow_depth": 2,
        "margin_v": 40,
        "alignment": 2,
    },
}


def get_preset(
    name: str,
    custom_resolution: Optional[tuple[int, int]] = None,
    custom_fps: Optional[float] = None,
) -> dict:
    """
    Get a quality preset with optional customizations.

    Args:
        name: Preset name
        custom_resolution: Override resolution (width, height)
        custom_fps: Override target FPS

    Returns:
        Preset dictionary
    """
    if name not in QUALITY_PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(QUALITY_PRESETS.keys())}")

    preset = QUALITY_PRESETS[name].copy()

    if custom_resolution:
        preset["resolution"] = custom_resolution

    if custom_fps:
        preset["target_fps"] = custom_fps

    return preset


def get_resolution_preset(
    resolution: Literal["4k", "1080p", "720p", "original"],
    quality: Literal["max", "balanced", "fast"] = "balanced",
) -> dict:
    """
    Get preset based on resolution and quality level.

    Args:
        resolution: Target resolution
        quality: Quality level

    Returns:
        Preset dictionary
    """
    preset_map = {
        ("4k", "max"): "4k_max",
        ("4k", "balanced"): "4k_balanced",
        ("4k", "fast"): "4k_balanced",
        ("1080p", "max"): "1080p_max",
        ("1080p", "balanced"): "1080p_balanced",
        ("1080p", "fast"): "1080p_fast",
        ("720p", "max"): "720p_fast",
        ("720p", "balanced"): "720p_fast",
        ("720p", "fast"): "720p_fast",
    }

    preset_name = preset_map.get((resolution, quality))

    if preset_name:
        return QUALITY_PRESETS[preset_name]

    # Default to 1080p balanced
    return QUALITY_PRESETS["1080p_balanced"]
