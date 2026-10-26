"""Source adapters for the free-only research pipeline."""

from scout.sources.base import Source, SourceHit, SourceSearch, SourceStatus, render_hits
from scout.sources.registry import SourceRegistry, build_registry

__all__ = [
    "Source",
    "SourceHit",
    "SourceSearch",
    "SourceStatus",
    "SourceRegistry",
    "build_registry",
    "render_hits",
]
