"""Stack Exchange questions through the free public API."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from scout.sources._http import get_json, make_client
from scout.sources.base import Source, SourceSearch, http_url, plain_text

_API = "https://api.stackexchange.com/2.3/search/advanced"


class StackExchangeSource(Source):
    name = "stackexchange"
    label = "Stack Exchange"
    description = (
        "Programming questions and answers from Stack Overflow and other Stack Exchange sites."
    )

    def __init__(
        self,
        *,
        site: str = "stackoverflow",
        api_key: str = "",
        user_agent: str = "NemotronScout/0.1",
        http_client: httpx.Client | None = None,
    ):
        self.site = site.strip() or "stackoverflow"
        self._key = api_key.strip()
        self._owned = http_client is None
        self._client = http_client or make_client(user_agent)

    def close(self) -> None:
        if self._owned:
            self._client.close()

    def search(self, query: str, *, limit: int = 5) -> SourceSearch:
        limit = max(1, min(limit, 10))
        params: dict[str, Any] = {
            "order": "desc",
            "sort": "relevance",
            "q": query,
            "site": self.site,
            "pagesize": limit,
            "filter": "withbody",
        }
        if self._key:
            params["key"] = self._key
        data = get_json(self._client, _API, label=self.label, params=params)
        rows = data.get("items")
        if not isinstance(rows, list):
            return SourceSearch([], ["Stack Exchange returned no items array."])
        warnings = []
        quota = data.get("quota_remaining")
        if isinstance(quota, int) and quota <= 10:
            warnings.append(f"Stack Exchange API quota is low ({quota} requests remaining).")
        hits = []
        for row in rows[:limit]:
            if not isinstance(row, dict):
                continue
            title = plain_text(row.get("title"))
            url = http_url(row.get("link"))
            if not title or not url:
                continue
            owner_value = row.get("owner")
            owner = owner_value if isinstance(owner_value, dict) else {}
            accepted = row.get("accepted_answer_id") is not None
            hits.append(
                self.hit(
                    title,
                    url,
                    plain_text(row.get("body"))[:1200],
                    {
                        "Site": self.site,
                        "Score": row.get("score"),
                        "Answers": row.get("answer_count"),
                        "Accepted answer": accepted,
                        "Views": row.get("view_count"),
                        "Tags": row.get("tags"),
                        "Asked": _date(row.get("creation_date")),
                        "Owner": owner.get("display_name"),
                    },
                )
            )
        return SourceSearch(hits, warnings)


def _date(value: Any) -> str:
    try:
        stamp = int(value)
    except (TypeError, ValueError, OverflowError):
        return ""
    try:
        return datetime.fromtimestamp(stamp, tz=UTC).date().isoformat()
    except (OSError, OverflowError, ValueError):
        return ""
