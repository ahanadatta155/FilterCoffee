"""Manual local-file import for sources that do not allow automated access."""

from __future__ import annotations

import html
from pathlib import Path
from urllib.parse import urlsplit

from scout.errors import ConfigurationError, FetchError
from scout.fetch import FetchedPage

_MAX_BYTES = 5 * 1024 * 1024
_TEXT_SUFFIXES = {".txt", ".text", ".md", ".markdown"}
_HTML_SUFFIXES = {".html", ".htm"}


class ManualFetcher:
    """Serve a user-saved document to the normal extraction pipeline.

    The file is never fetched from the network. A public citation URL can be
    attached so evidence links remain traceable, including for Reddit threads
    the user saved themselves.
    """

    source_name = "manual"
    trusted_url = True

    def __init__(
        self,
        path: str | Path,
        *,
        citation_url: str | None = None,
        title: str | None = None,
    ):
        self.path = Path(path)
        suffix = self.path.suffix.casefold()
        if suffix not in _TEXT_SUFFIXES | _HTML_SUFFIXES:
            raise ConfigurationError(
                "Manual import supports .txt, .md, .markdown, .html, and .htm files."
            )
        if not self.path.is_file():
            raise FetchError(f"Manual import file not found: {self.path}")
        self.is_html = suffix in _HTML_SUFFIXES
        self.citation_url = _public_url(citation_url) if citation_url else None
        self.title = (title or "").strip() or None
        self.url = self.citation_url or self.path.resolve().as_uri()

    def fetch(self, url: str, *, ignore_robots: bool = False) -> FetchedPage:
        del url, ignore_robots
        raw = self.path.read_bytes()
        if len(raw) > _MAX_BYTES:
            raise FetchError(
                f"Manual import file exceeds {_MAX_BYTES // (1024 * 1024)} MiB: {self.path}"
            )
        text = raw.decode("utf-8-sig", errors="replace")
        warnings = ["Manual import: no network request was made for this file."]
        if "\ufffd" in text:
            warnings.append(
                "Some characters could not be decoded as UTF-8; re-save the file as UTF-8."
            )
        if not text.strip():
            raise FetchError(f"Manual import file is empty: {self.path}")

        if self.is_html:
            return FetchedPage(
                url=self.url,
                html=text,
                status_code=200,
                content_type="text/html",
                tier="manual",
                warnings=warnings,
            )

        title = self.title or _first_heading(text) or self.path.stem
        return FetchedPage(
            url=self.url,
            html=_wrap_text(title, text, self.citation_url),
            status_code=200,
            content_type="text/html",
            tier="manual",
            warnings=warnings,
        )


def _public_url(value: str) -> str:
    text = value.strip()
    parsed = urlsplit(text)
    try:
        hostname = parsed.hostname
    except ValueError:
        hostname = None
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ConfigurationError("--manual-url must be a public HTTP(S) URL.")
    return text


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or None
        if stripped:
            return None
    return None


def _wrap_text(title: str, text: str, citation_url: str | None) -> str:
    source = (
        f"<p><strong>Source:</strong> "
        f'<a href="{html.escape(citation_url, quote=True)}">{html.escape(citation_url)}</a></p>'
        if citation_url
        else ""
    )
    paragraphs = "\n".join(
        f"<p>{html.escape(line)}</p>" for line in text.splitlines() if line.strip()
    )
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>{html.escape(title)}</title></head>
<body><main><article>
<h1>{html.escape(title)}</h1>
{source}
{paragraphs}
</article></main></body>
</html>"""
