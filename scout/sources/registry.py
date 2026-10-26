"""Registry of available research sources and their capability descriptions."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

from scout.errors import ConfigurationError, SourceError
from scout.sources.base import Source, SourceStatus
from scout.sources.bluesky import BlueskySource
from scout.sources.github import GitHubSource
from scout.sources.hn import HackerNewsSource
from scout.sources.opencli import OpenCLISource
from scout.sources.stackexchange import StackExchangeSource
from scout.sources.web import WebSource

_BLOCKED = {"reddit", "redd.it"}


class SourceRegistry:
    def __init__(self, sources: Iterable[Source]):
        self._sources = {source.name: source for source in sources}

    def names(self) -> list[str]:
        return list(self._sources)

    def get(self, name: str) -> Source:
        key = name.strip().lower()
        if key in _BLOCKED:
            raise SourceError(
                "Reddit is not available as an automated source. Use manual import instead."
            )
        source = self._sources.get(key)
        if source is None:
            raise ConfigurationError(
                f"Unknown source '{name}'. Available: {', '.join(self.names())}."
            )
        return source

    def describe(self) -> list[dict[str, object]]:
        output = []
        for source in self._sources.values():
            status: SourceStatus = source.status()
            output.append(
                {
                    "name": source.name,
                    "label": source.label,
                    "description": source.description,
                    "available": status.available,
                    "reason": status.reason,
                }
            )
        return output

    def select(self, names: Iterable[str] | None = None) -> list[Source]:
        if names is None:
            return [source for source in self._sources.values() if source.status().available]
        requested = [name.strip().lower() for name in names if name.strip()]
        if not requested:
            return [source for source in self._sources.values() if source.status().available]
        return [self.get(name) for name in requested]

    def close(self) -> None:
        for source in self._sources.values():
            source.close()


def build_registry(
    *,
    search_provider: str = "auto",
    user_agent: str = "NemotronScout/0.1",
    env: Mapping[str, str] | None = None,
) -> SourceRegistry:
    values = os.environ if env is None else env
    return SourceRegistry(
        [
            HackerNewsSource(user_agent=user_agent),
            StackExchangeSource(
                site=values.get("STACKEXCHANGE_SITE", "stackoverflow"),
                api_key=values.get("STACKEXCHANGE_KEY", ""),
                user_agent=user_agent,
            ),
            GitHubSource(
                token=values.get("GITHUB_TOKEN", "") or values.get("GH_TOKEN", ""),
                user_agent=user_agent,
            ),
            BlueskySource(user_agent=user_agent),
            WebSource(search_provider=search_provider, env=values),
            OpenCLISource.from_env(values),
        ]
    )
