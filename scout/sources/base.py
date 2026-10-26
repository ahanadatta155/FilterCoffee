"""Source adapter contract and deterministic HTML rendering for research hits."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape, unescape
from typing import Any
from urllib.parse import urlsplit

from scout.clean import normalize_text
from scout.errors import SourceError

_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class SourceHit:
    """One normalized result from a source, ready to render as evidence."""

    source: str
    title: str
    url: str
    snippet: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    source_label: str = ""


@dataclass(frozen=True)
class SourceSearch:
    hits: list[SourceHit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SourceStatus:
    available: bool
    reason: str = ""


class Source:
    """Minimal interface every research source implements."""

    name = "source"
    label = "Source"
    description = ""

    def status(self) -> SourceStatus:
        return SourceStatus(True)

    def search(self, query: str, *, limit: int = 5) -> SourceSearch:
        raise SourceError(f"{self.label} does not support search.")

    def close(self) -> None:
        return None

    def hit(
        self,
        title: str,
        url: str,
        snippet: str = "",
        data: dict[str, Any] | None = None,
    ) -> SourceHit:
        return SourceHit(
            source=self.name,
            source_label=self.label,
            title=normalize_text(title) or self.label,
            url=url,
            snippet=normalize_text(snippet),
            data={key: value for key, value in (data or {}).items() if value not in (None, "")},
        )


def plain_text(value: Any) -> str:
    """Strip API-supplied HTML without executing it."""
    if not isinstance(value, str):
        return ""
    return normalize_text(unescape(_TAG.sub(" ", value)))


def http_url(value: Any) -> str:
    """Return a safe HTTP(S) URL, or an empty string."""
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        parsed = urlsplit(value.strip())
        if (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and not any(char.isspace() for char in value)
        ):
            return value.strip()
    except ValueError:
        pass
    return ""


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_value(item) for item in value if item not in (None, ""))
    return str(value)


def render_hits(
    query: str,
    hits: list[SourceHit],
    *,
    title: str | None = None,
) -> str:
    """Render hits as plain HTML so the existing cleaner can treat them as a page.

    Every value is escaped. Each source becomes a section and every hit an
    article, which lets the existing candidate-group detector see list results.
    """
    grouped: dict[str, list[SourceHit]] = {}
    for hit in hits:
        grouped.setdefault(hit.source, []).append(hit)
    sections = []
    for source, source_hits in grouped.items():
        label = escape(source_hits[0].source_label or source)
        sections.append(f'<section data-source="{escape(source)}"><h1>{label} results</h1>')
        for hit in source_hits:
            safe_url = http_url(hit.url)
            link = (
                f'<p><strong>URL:</strong> <a href="{escape(safe_url, quote=True)}">'
                f"{escape(safe_url)}</a></p>"
                if safe_url
                else f"<p><strong>URL:</strong> {escape(hit.url)}</p>"
            )
            parts = [
                f'<article data-source="{escape(source)}">',
                f"<h2>{escape(hit.title)}</h2>",
                f"<p><strong>Source:</strong> {label}</p>",
                link,
            ]
            for key, raw in hit.data.items():
                value = _value(raw).strip()
                if value:
                    parts.append(f"<p><strong>{escape(str(key))}:</strong> {escape(value)}</p>")
            if hit.snippet:
                parts.append(f"<p><strong>Excerpt:</strong> {escape(hit.snippet)}</p>")
            parts.append("</article>")
            sections.append("".join(parts))
        sections.append("</section>")
    page_title = escape(title or f"Research results for: {query}")
    body = "".join(sections) or "<p>No results were returned by the selected sources.</p>"
    return (
        "<html><head>"
        f"<title>{page_title}</title>"
        f'<meta name="description" content="{escape(query, quote=True)}">'
        "</head><body><main>"
        f"{body}"
        "</main></body></html>"
    )
