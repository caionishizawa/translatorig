"""Video extraction modules."""

from .base import BaseExtractor
from .instagram import InstagramExtractor
from .youtube import YouTubeExtractor
from .local import LocalExtractor

__all__ = [
    "BaseExtractor",
    "InstagramExtractor",
    "YouTubeExtractor",
    "LocalExtractor",
]
