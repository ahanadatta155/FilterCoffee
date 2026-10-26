"""GitHub repository search through the free REST API."""

from __future__ import annotations

import httpx

from scout.sources._http import get_json, make_client
from scout.sources.base import Source, SourceSearch, http_url, plain_text

_API = "https://api.github.com/search/repositories"


class GitHubSource(Source):
    name = "github"
    label = "GitHub"
    description = "Repositories, issues, and code hosted on GitHub, via the REST search API."

    def __init__(
        self,
        *,
        token: str = "",
        user_agent: str = "NemotronScout/0.1",
        http_client: httpx.Client | None = None,
    ):
        self._token = token.strip()
        self._owned = http_client is None
        self._client = http_client or make_client(user_agent)

    def close(self) -> None:
        if self._owned:
            self._client.close()

    def search(self, query: str, *, limit: int = 5) -> SourceSearch:
        limit = max(1, min(limit, 10))
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        data = get_json(
            self._client,
            _API,
            label=self.label,
            params={"q": query, "sort": "stars", "order": "desc", "per_page": limit},
            headers=headers,
        )
        rows = data.get("items")
        if not isinstance(rows, list):
            return SourceSearch([], ["GitHub returned no items array."])
        warnings = []
        if data.get("incomplete_results") is True:
            warnings.append("GitHub reported incomplete search results.")
        hits = []
        for row in rows[:limit]:
            if not isinstance(row, dict):
                continue
            name = row.get("full_name")
            url = http_url(row.get("html_url"))
            if not isinstance(name, str) or not name or not url:
                continue
            license_value = row.get("license")
            license_info = license_value if isinstance(license_value, dict) else {}
            hits.append(
                self.hit(
                    name,
                    url,
                    plain_text(row.get("description"))[:1200],
                    {
                        "Stars": row.get("stargazers_count"),
                        "Forks": row.get("forks_count"),
                        "Language": row.get("language"),
                        "Open issues": row.get("open_issues_count"),
                        "Updated": _date(row.get("updated_at")),
                        "License": license_info.get("spdx_id") or license_info.get("name"),
                        "Topics": row.get("topics"),
                    },
                )
            )
        return SourceSearch(hits, warnings)


def _date(value: object) -> str:
    return value[:10] if isinstance(value, str) and len(value) >= 10 else ""
