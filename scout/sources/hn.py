"""Hacker News stories through the free Algolia Search API."""

from __future__ import annotations

import re

import httpx

from scout.sources._http import get_json, make_client
from scout.sources.base import Source, SourceSearch, http_url, plain_text

_API = "https://hn.algolia.com/api/v1/search"
_ID = re.compile(r"^\d+$")


class HackerNewsSource(Source):
    name = "hn"
    label = "Hacker News"
    description = "Stories and discussions from Hacker News, via the Algolia API."

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
            params={"query": query, "tags": "story", "hitsPerPage": limit, "page": 0},
        )
        rows = data.get("hits")
        if not isinstance(rows, list):
            return SourceSearch([], ["Hacker News returned no hits array."])
        hits = []
        for row in rows[:limit]:
            if not isinstance(row, dict):
                continue
            object_id = str(row.get("objectID") or "")
            if not _ID.fullmatch(object_id):
                continue
            title = plain_text(row.get("title")) or plain_text(row.get("story_title"))
            snippet = plain_text(row.get("story_text") or row.get("comment_text"))
            if not title:
                title = snippet[:120] or f"Hacker News item {object_id}"
            external = http_url(row.get("url"))
            data_fields = {
                "Author": row.get("author"),
                "Points": row.get("points"),
                "Comments": row.get("num_comments"),
                "Posted": row.get("created_at"),
                "Type": _kind(row.get("_tags")),
                "Link": external,
            }
            hits.append(
                self.hit(
                    title,
                    f"https://news.ycombinator.com/item?id={object_id}",
                    snippet,
                    data_fields,
                )
            )
        return SourceSearch(hits)


def _kind(tags: object) -> str:
    if not isinstance(tags, list):
        return "story"
    for tag in tags:
        if isinstance(tag, str) and tag in {"story", "ask_hn", "show_hn", "job", "poll"}:
            return tag
    return "story"
