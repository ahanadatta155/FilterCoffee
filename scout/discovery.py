"""Turn a natural-language request into one safe page URL and an extract task."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from scout.config import Settings
from scout.errors import ConfigurationError, DiscoveryError, ScoutError
from scout.llm import LLMClient, LLMResponse, ModelClient, parse_json

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_SEARCH_RESULTS = 5


class RequestIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: Literal["direct_url", "site_search", "web_search"]
    url: str | None = None
    site_domain: str | None = None
    query: str = Field(default="", max_length=300)
    instruction: str = Field(min_length=1, max_length=500)

    @field_validator("instruction")
    @classmethod
    def nonblank_instruction(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instruction must not be blank")
        return value.strip()

    @field_validator("query")
    @classmethod
    def normalized_query(cls, value: str) -> str:
        return value.strip()


class CandidateSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_index: int = Field(ge=0, strict=True)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    description: str


@dataclass(frozen=True)
class DiscoveryResult:
    route: str
    query: str
    title: str
    url: str
    instruction: str
    search_result_count: int
    search_latency_ms: int
    intent_response: LLMResponse
    selection_response: LLMResponse | None
    search_provider: str = "none"


class TavilySearchClient:
    """Small adapter for Tavily's basic-search free tier."""

    def __init__(self, api_key: str, *, http_client: httpx.Client | None = None):
        if not api_key.strip():
            raise ConfigurationError("Set TAVILY_API_KEY in the project .env to search the web.")
        self.api_key = api_key.strip()
        self._owned = http_client is None
        self._client = http_client or httpx.Client(
            timeout=15,
            follow_redirects=False,
            trust_env=False,
        )

    def search(self, query: str, *, include_domains: list[str] | None = None) -> list[SearchResult]:
        payload: dict[str, Any] = {
            "query": query,
            "search_depth": "basic",
            "max_results": MAX_SEARCH_RESULTS,
            "include_answer": False,
        }
        if include_domains:
            payload["include_domains"] = include_domains
            payload["include_domains_mode"] = "restrict"
        try:
            response = self._client.post(
                TAVILY_SEARCH_URL,
                json=payload,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
        except httpx.HTTPError:
            raise DiscoveryError(
                "Web search request failed; check connectivity and try again."
            ) from None
        if response.status_code in {401, 403}:
            raise DiscoveryError("Tavily rejected the search key; check TAVILY_API_KEY.")
        if response.status_code == 429:
            raise DiscoveryError("Tavily search rate limit reached; wait before trying again.")
        if response.status_code == 432:
            raise DiscoveryError(
                "Tavily's free search credits are exhausted; they reset monthly. "
                "No paid search was attempted."
            )
        if response.status_code == 433:
            raise DiscoveryError("Tavily rejected the request because its usage limit was reached.")
        if not 200 <= response.status_code < 300:
            raise DiscoveryError(f"Tavily search failed with HTTP {response.status_code}.")
        try:
            payload = response.json()
        except ValueError:
            raise DiscoveryError("Tavily returned an unreadable search response.") from None
        rows = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise DiscoveryError("Tavily returned no usable web search results.")
        results = []
        for row in rows[:MAX_SEARCH_RESULTS]:
            if not isinstance(row, dict):
                continue
            title, url = row.get("title"), row.get("url")
            if not isinstance(title, str) or not isinstance(url, str) or not _http_url(url):
                continue
            description = row.get("content")
            results.append(
                SearchResult(
                    title[:300],
                    url,
                    description[:1200] if isinstance(description, str) else "",
                )
            )
        return results

    def close(self) -> None:
        if self._owned:
            self._client.close()


def _http_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not any(char.isspace() for char in value)
        )
    except ValueError:
        return False


def _domain(value: str | None) -> str:
    if not value or any(char in value for char in "/:@?#"):
        raise DiscoveryError("The model did not identify a valid site domain.")
    try:
        normalized = value.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError:
        raise DiscoveryError("The model returned an invalid site domain.") from None
    labels = normalized.split(".")
    if len(labels) < 2 or any(
        not label
        or len(label) > 63
        or label.startswith("-")
        or label.endswith("-")
        or not re.fullmatch(r"[a-z0-9-]+", label)
        for label in labels
    ):
        raise DiscoveryError("The model returned an invalid site domain.")
    return normalized


def _matches_domain(url: str, domain: str) -> bool:
    if not _http_url(url):
        return False
    host = urlsplit(url).hostname or ""
    try:
        host = host.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError:
        return False
    return host == domain or host.endswith("." + domain)


def resolve_request(
    request: str,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    search: Any = None,
    search_provider: str = "tinyfish",
) -> DiscoveryResult:
    """Classify one request, search if needed, and select one real result URL."""
    if not request.strip():
        raise DiscoveryError("Enter a URL or describe what you want to find.")
    owned_llm = llm is None
    if llm is None:
        if settings is None:
            raise ConfigurationError("Model settings are required to interpret a request.")
        llm = ModelClient(settings)
    owned_search = search is None
    try:
        intent_response: LLMResponse
        prompt = (PROMPTS / "discover.md").read_text(encoding="utf-8")
        try:
            intent_response = llm.chat(
                "discover",
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": request},
                ],
                role="strong",
                max_tokens=4096,
            )
            intent = RequestIntent.model_validate(parse_json(intent_response.text))
        except (ScoutError, ValidationError, ValueError) as error:
            raise DiscoveryError(f"Could not understand that request: {error}") from None

        if intent.route == "direct_url":
            if not intent.url or intent.url not in request or not _http_url(intent.url):
                raise DiscoveryError(
                    "A direct URL must be a public HTTP(S) URL included in your request."
                )
            return DiscoveryResult(
                intent.route, "", "", intent.url, intent.instruction, 0, 0, intent_response, None
            )

        if not intent.query:
            raise DiscoveryError("Could not form a useful search query from that request.")
        query = intent.query
        domain = None
        if intent.route == "site_search":
            domain = _domain(intent.site_domain)
            if domain in {"www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}:
                domain = "youtube.com"

        if search is None:
            if domain == "youtube.com":
                from scout.youtube import YouTubeClient

                search = YouTubeClient()
                search_provider = "youtube"
            elif search_provider in {"tinyfish", "auto"}:
                tinyfish_key = os.getenv("TINYFISH_API_KEY", "").strip()
                if tinyfish_key:
                    from scout.tinyfish import TinyFishClient

                    search = TinyFishClient(tinyfish_key)
                    search_provider = "tinyfish"
                elif os.getenv("TAVILY_API_KEY", "").strip():
                    search = TavilySearchClient(os.getenv("TAVILY_API_KEY", ""))
                    search_provider = "tavily"
                elif search_provider == "auto":
                    from scout.tinyfish import TinyFishClient

                    search = TinyFishClient("")
                else:
                    from scout.tinyfish import TinyFishClient

                    search = TinyFishClient("")
            elif search_provider == "tavily":
                search = TavilySearchClient(os.getenv("TAVILY_API_KEY", ""))
            else:
                raise ConfigurationError("Search provider must be tinyfish, tavily, or auto.")

        started = time.perf_counter()
        results = search.search(query, include_domains=[domain] if domain else None)
        search_latency_ms = int((time.perf_counter() - started) * 1000)
        candidates = [result for result in results if _http_url(result.url)]
        if domain:
            candidates = [result for result in candidates if _matches_domain(result.url, domain)]
        if not candidates:
            raise DiscoveryError("No suitable page was found for that request.")

        selection_prompt = (PROMPTS / "select_page.md").read_text(encoding="utf-8")
        candidate_data = [
            {
                "index": index,
                "title": result.title,
                "url": result.url,
                "description": result.description,
            }
            for index, result in enumerate(candidates)
        ]
        try:
            selection_response = llm.chat(
                "discover_select",
                [
                    {"role": "system", "content": selection_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"request": request, "candidates": candidate_data},
                            ensure_ascii=False,
                        ),
                    },
                ],
                role="fast",
                max_tokens=1024,
            )
            selection = CandidateSelection.model_validate(parse_json(selection_response.text))
        except (ScoutError, ValidationError, ValueError) as error:
            raise DiscoveryError(f"Could not choose a search result: {error}") from None
        if selection.candidate_index >= len(candidates):
            raise DiscoveryError("The model could not choose a search result from the candidates.")
        chosen = candidates[selection.candidate_index]
        return DiscoveryResult(
            intent.route,
            f"site:{domain} {query}" if domain else query,
            chosen.title,
            chosen.url,
            intent.instruction,
            len(candidates),
            search_latency_ms,
            intent_response,
            selection_response,
            search_provider,
        )
    finally:
        if owned_search and search is not None:
            search.close()
        if owned_llm:
            llm.close()
