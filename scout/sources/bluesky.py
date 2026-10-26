"""Bluesky post search through the free public AT Protocol API."""

from __future__ import annotations

import re

import httpx

from scout.sources._http import get_json, make_client
from scout.sources.base import Source, SourceSearch, http_url, plain_text

_API = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
_HANDLE = re.compile(r"^[A-Za-z0-9.-]+$")
_RKEY = re.compile(r"^[A-Za-z0-9]+$")


class BlueskySource(Source):
    name = "bluesky"
    label = "Bluesky"
    description = "Public posts from Bluesky, via the AT Protocol public API."

    def __init__(
        self,
        *,
        user_agent: str = "NemotronScout/0.1",
        http_client: httpx.Client | None = None,
    ):
        self._owned = http_client is None
        self._client = http_client or make_client(user_agent)

    def close(self) -> None:
        if self._owned:
            self._client.close()

    def search(self, query: str, *, limit: int = 5) -> SourceSearch:
        limit = max(1, min(limit, 10))
        data = get_json(
            self._client,
            _API,
            label=self.label,
            params={"q": query, "limit": limit, "sort": "latest"},
        )
        rows = data.get("posts")
        if not isinstance(rows, list):
            return SourceSearch([], ["Bluesky returned no posts array."])
        hits = []
        for row in rows[:limit]:
            if not isinstance(row, dict):
                continue
            record_value = row.get("record")
            record = record_value if isinstance(record_value, dict) else {}
            author_value = row.get("author")
            author = author_value if isinstance(author_value, dict) else {}
            text = plain_text(record.get("text"))
            handle_value = author.get("handle")
            handle = handle_value if isinstance(handle_value, str) else ""
            uri_value = row.get("uri")
            uri = uri_value if isinstance(uri_value, str) else ""
            rkey = uri.rsplit("/", 1)[-1] if uri else ""
            if not text or not _HANDLE.fullmatch(handle) or not _RKEY.fullmatch(rkey):
                continue
            url = http_url(f"https://bsky.app/profile/{handle}/post/{rkey}")
            if not url:
                continue
            title = text.splitlines()[0][:120] or f"Bluesky post by {handle}"
            hits.append(
                self.hit(
                    title,
                    url,
                    text[:1200],
                    {
                        "Author": author.get("displayName") or handle,
                        "Handle": handle,
                        "Likes": row.get("likeCount"),
                        "Replies": row.get("replyCount"),
                        "Reposts": row.get("repostCount"),
                        "Quotes": row.get("quoteCount"),
                        "Posted": _date(record.get("createdAt") or row.get("indexedAt")),
                        "Languages": record.get("langs"),
                    },
                )
            )
        return SourceSearch(hits)


def _date(value: object) -> str:
    return value[:10] if isinstance(value, str) and len(value) >= 10 else ""
