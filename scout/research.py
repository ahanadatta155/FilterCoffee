"""Multi-source research: plan, search free APIs, then reuse the extraction pipeline."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from urllib.parse import quote, urlencode

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from scout.config import Settings, load_settings
from scout.contracts import ExtractResult, ModelCall, SourceReport
from scout.errors import DiscoveryError, FetchError, ModelError, SourceError
from scout.fetch import FetchedPage
from scout.gemini import ExtractionRouter, GeminiExtractionClient
from scout.llm import ChatMessages, LLMClient, ModelClient, parse_json
from scout.pipeline import RunOptions, run
from scout.sources.base import Source, render_hits
from scout.sources.registry import SourceRegistry, build_registry

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
_ALIASES = {
    "hackernews": "hn",
    "hacker_news": "hn",
    "stackoverflow": "stackexchange",
    "stack_overflow": "stackexchange",
    "so": "stackexchange",
    "bsky": "bluesky",
}
_QUERY_NOISE = {
    "a",
    "about",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "best",
    "but",
    "by",
    "can",
    "could",
    "data",
    "did",
    "do",
    "does",
    "find",
    "for",
    "from",
    "get",
    "github",
    "give",
    "had",
    "has",
    "have",
    "how",
    "in",
    "info",
    "information",
    "into",
    "is",
    "it",
    "just",
    "latest",
    "list",
    "may",
    "might",
    "more",
    "most",
    "must",
    "new",
    "newest",
    "no",
    "not",
    "of",
    "on",
    "only",
    "or",
    "over",
    "page",
    "pages",
    "please",
    "popular",
    "project",
    "projects",
    "recent",
    "recently",
    "repo",
    "repos",
    "repository",
    "repositories",
    "result",
    "results",
    "show",
    "site",
    "sites",
    "some",
    "star",
    "starred",
    "stars",
    "tell",
    "the",
    "their",
    "to",
    "tool",
    "tools",
    "top",
    "trending",
    "under",
    "very",
    "was",
    "web",
    "website",
    "websites",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "without",
    "would",
    "yes",
}


class ResearchIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=300)
    instruction: str = Field(min_length=1, max_length=600)
    sources: list[str] = Field(default_factory=list, max_length=4)

    @field_validator("query", "instruction")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @field_validator("sources")
    @classmethod
    def clean_sources(cls, values: list[str]) -> list[str]:
        return list(
            dict.fromkeys(
                part.strip().lower()
                for value in values
                for part in value.split(",")
                if part.strip()
            )
        )


class ResearchFetcher:
    """Fetcher-compatible adapter that turns source searches into one evidence page."""

    source_name = "research"
    trusted_url = True

    def __init__(
        self,
        sources: Iterable[Source],
        query: str,
        *,
        limit: int,
        plan_warnings: Iterable[str] = (),
        prefix_calls: Iterable[ModelCall] = (),
    ):
        self._sources = list(sources)
        self._query = query
        self._limit = limit
        self._plan_warnings = list(plan_warnings)
        self.prefix_calls = list(prefix_calls)
        self.reports: list[SourceReport] = []
        names = ",".join(source.name for source in self._sources)
        self.url = "https://scout.local/research?" + urlencode(
            {"q": query, "sources": names, "v": "1"}, quote_via=quote
        )

    def fetch(self, url: str, *, ignore_robots: bool = False) -> FetchedPage:
        del url, ignore_robots
        hits = []
        warnings = list(self._plan_warnings)
        self.reports = []
        for source in self._sources:
            started = time.perf_counter()
            try:
                status = source.status()
                if not status.available:
                    self.reports.append(
                        SourceReport(
                            name=source.name,
                            status="unavailable",
                            warnings=[status.reason],
                        )
                    )
                    warnings.append(f"{source.label} unavailable: {status.reason}")
                    continue
                found = source.search(self._query, limit=self._limit)
            except SourceError as error:
                latency = int((time.perf_counter() - started) * 1000)
                self.reports.append(
                    SourceReport(
                        name=source.name,
                        status="error",
                        latency_ms=latency,
                        warnings=[str(error)],
                    )
                )
                warnings.append(f"{source.label} failed: {error}")
                continue
            except Exception as error:
                latency = int((time.perf_counter() - started) * 1000)
                message = f"{source.label} failed unexpectedly ({type(error).__name__})."
                self.reports.append(
                    SourceReport(
                        name=source.name,
                        status="error",
                        latency_ms=latency,
                        warnings=[message],
                    )
                )
                warnings.append(message)
                continue
            latency = int((time.perf_counter() - started) * 1000)
            state = "ok" if found.hits else "empty"
            self.reports.append(
                SourceReport(
                    name=source.name,
                    status=state,
                    hits=len(found.hits),
                    latency_ms=latency,
                    warnings=list(found.warnings),
                )
            )
            warnings.extend(f"{source.label}: {warning}" for warning in found.warnings)
            hits.extend(found.hits)
        if not hits:
            detail = "; ".join(warnings) if warnings else "Try a broader query."
            raise FetchError(f"No source returned usable results. {detail}")
        return FetchedPage(
            url=self.url,
            html=render_hits(self._query, hits),
            status_code=200,
            content_type="text/html",
            tier="research",
            warnings=warnings,
        )


def compact_query(query: str, *, limit: int = 6) -> str:
    """Keep subject terms and drop search boilerplate from a planner query."""
    tokens = query.split()
    meaningful = [token for token in tokens if _query_word(token) not in _QUERY_NOISE]
    selected = (meaningful or tokens)[:limit]
    return " ".join(selected).strip() or query.strip()


def _query_word(token: str) -> str:
    return token.strip(".,;:!?()[]{}\"'").casefold()


def research(
    request: str,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    registry: SourceRegistry | None = None,
    source_names: Iterable[str] | None = None,
    search_provider: str = "auto",
    limit: int = 5,
    options: RunOptions | None = None,
    cache_path: str | Path = "data/cache.db",
) -> ExtractResult:
    """Plan sources, search them, and extract verified rows from the combined page."""
    if settings is None:
        settings = load_settings()
    owned_registry = registry is None
    if registry is None:
        registry = build_registry(
            search_provider=search_provider,
            user_agent=settings.user_agent,
        )
    owned_llm = llm is None
    if llm is None:
        llm = ModelClient(settings)
    close_llm: LLMClient = llm
    limit = max(1, min(int(limit), 10))
    try:
        available = [item for item in registry.describe() if item["available"]]
        if not available:
            reasons = "; ".join(
                f"{item['name']}: {item['reason']}"
                for item in registry.describe()
                if not item["available"]
            )
            raise DiscoveryError(f"No research sources are available. {reasons}")
        call_start = len(getattr(llm, "calls", []))
        intent, plan_warnings = plan_research(request, available, llm)
        planner_calls = list(getattr(llm, "calls", []))[call_start:]
        requested = _names(source_names) if source_names is not None else intent.sources
        sources = registry.select(requested)
        if not sources:
            raise DiscoveryError("No research sources were selected.")
        fetcher = ResearchFetcher(
            sources,
            compact_query(intent.query),
            limit=limit,
            plan_warnings=plan_warnings,
            prefix_calls=planner_calls,
        )
        routing = options.routing if options is not None else "gemini"
        if owned_llm and routing == "gemini":
            gemini_key = os.getenv("GEMINI_KEY", "").strip()
            gemini = GeminiExtractionClient(gemini_key) if gemini_key else None
            llm = ExtractionRouter(llm, gemini)
            close_llm = llm
        return run(
            fetcher.url,
            intent.instruction,
            schema=None,
            options=options,
            fetcher=fetcher,
            llm=llm,
            cache_path=cache_path,
            settings=settings,
        )
    finally:
        if owned_registry:
            registry.close()
        if owned_llm:
            close_llm.close()


def plan_research(
    request: str, available: Sequence[Mapping[str, object]], llm: LLMClient
) -> tuple[ResearchIntent, list[str]]:
    system = (PROMPTS / "research.md").read_text(encoding="utf-8")
    payload = json.dumps(
        {
            "request": request,
            "available_sources": [
                {
                    "name": item["name"],
                    "label": item["label"],
                    "description": item["description"],
                }
                for item in available
            ],
        },
        ensure_ascii=False,
    )
    warnings: list[str] = []
    try:
        response = llm.chat(
            "research_plan",
            _messages(system, payload),
            role="strong",
            max_tokens=1024,
        )
    except ModelError as error:
        if "empty or truncated" not in str(error):
            raise
        warnings.append("Planner retried with the fast model after an empty response.")
        response = llm.chat(
            "research_plan",
            _messages(system, payload),
            role="fast",
            max_tokens=2048,
        )
    try:
        intent = ResearchIntent.model_validate(parse_json(response.text))
    except (ValidationError, ModelError) as error:
        raise DiscoveryError(
            f"The research planner returned an invalid plan ({error}); choose --sources explicitly."
        ) from None
    known = {str(item["name"]) for item in available}
    valid = []
    unknown = []
    for name in intent.sources:
        name = _ALIASES.get(name, name)
        if name in known:
            valid.append(name)
        else:
            unknown.append(name)
    if unknown:
        warnings.append(f"Dropped unavailable sources: {', '.join(unknown)}.")
    if not valid:
        valid = [str(available[0]["name"])]
        warnings.append(f"Planner selected no available source; using {valid[0]}.")
    return intent.model_copy(update={"sources": valid}), warnings


def _names(values: Iterable[str]) -> list[str]:
    if isinstance(values, str):
        values = values.split(",")
    return [value.strip().lower() for value in values if value.strip()]


def _messages(system: str, payload: str) -> ChatMessages:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": payload},
    ]
