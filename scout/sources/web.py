"""General web search as a source, using TinyFish first and Tavily as fallback."""

from __future__ import annotations

import os
from collections.abc import Mapping

from scout.discovery import TavilySearchClient
from scout.errors import ConfigurationError
from scout.sources.base import Source, SourceSearch, SourceStatus


class WebSource(Source):
    name = "web"
    label = "Web"
    description = (
        "General web search results (title, URL, snippet); full pages are read separately."
    )

    def __init__(
        self,
        *,
        search_provider: str = "auto",
        env: Mapping[str, str] | None = None,
        client: object | None = None,
    ):
        self._provider = search_provider
        self._env = os.environ if env is None else env
        self._client = client
        self._owned = client is None

    def _keys(self) -> tuple[str, str]:
        return (
            self._env.get("TINYFISH_API_KEY", "").strip(),
            self._env.get("TAVILY_API_KEY", "").strip(),
        )

    def status(self) -> SourceStatus:
        if self._client is not None:
            return SourceStatus(True)
        tinyfish, tavily = self._keys()
        if self._provider == "tavily":
            return (
                SourceStatus(True)
                if tavily
                else SourceStatus(False, "Set TAVILY_API_KEY to use the Tavily web source.")
            )
        if self._provider == "tinyfish":
            return (
                SourceStatus(True)
                if tinyfish
                else SourceStatus(False, "Set TINYFISH_API_KEY to use the TinyFish web source.")
            )
        if self._provider != "auto":
            return SourceStatus(False, "Search provider must be auto, tinyfish, or tavily.")
        if tinyfish or tavily:
            return SourceStatus(True)
        return SourceStatus(
            False, "Set TINYFISH_API_KEY or TAVILY_API_KEY to use the general web source."
        )

    def _make_client(self) -> object:
        tinyfish, tavily = self._keys()
        if self._provider == "tavily" or (self._provider == "auto" and not tinyfish):
            if not tavily:
                raise ConfigurationError(
                    "Set TAVILY_API_KEY in the project .env to use the Tavily web source."
                )
            return TavilySearchClient(tavily)
        if not tinyfish:
            raise ConfigurationError(
                "Set TINYFISH_API_KEY in the project .env to use the TinyFish web source."
            )
        from scout.tinyfish import TinyFishClient

        return TinyFishClient(tinyfish)

    def search(self, query: str, *, limit: int = 5) -> SourceSearch:
        if self._client is None:
            self._client = self._make_client()
        results = self._client.search(query)  # type: ignore[attr-defined]
        hits = [
            self.hit(result.title, result.url, result.description) for result in results[:limit]
        ]
        return SourceSearch(
            hits,
            ["Web results are search snippets; full pages are read only when selected directly."],
        )

    def close(self) -> None:
        if self._owned and self._client is not None:
            self._client.close()  # type: ignore[attr-defined]
