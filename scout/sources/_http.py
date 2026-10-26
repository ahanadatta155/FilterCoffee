"""Shared HTTP client and safe error normalization for public API sources."""

from __future__ import annotations

from typing import Any

import httpx

from scout.errors import SourceError
from scout.security import PinnedTransport


def make_client(user_agent: str) -> httpx.Client:
    return httpx.Client(
        transport=PinnedTransport(),
        headers={"User-Agent": user_agent, "Accept": "application/json"},
        trust_env=False,
        follow_redirects=False,
        timeout=20,
    )


def get_json(
    client: httpx.Client,
    url: str,
    *,
    label: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Fetch one JSON object and turn provider failures into safe messages.

    Provider response bodies are never echoed: they can contain account details
    or key fragments.
    """
    try:
        response = client.get(url, params=params, headers=headers)
    except httpx.TimeoutException:
        raise SourceError(f"{label} timed out; try again or choose another source.") from None
    except httpx.HTTPError:
        raise SourceError(f"{label} request failed; check connectivity.") from None
    if response.status_code in {401, 403}:
        raise SourceError(
            f"{label} rejected the request (HTTP {response.status_code}); "
            "check its key, permissions, or usage terms."
        )
    if response.status_code == 429:
        retry = response.headers.get("retry-after", "")
        wait = f" Retry after {retry} seconds." if retry.isdigit() else ""
        raise SourceError(f"{label} rate limit reached; wait before trying again.{wait}")
    if response.status_code >= 500:
        raise SourceError(f"{label} is unavailable (HTTP {response.status_code}); try again later.")
    if not 200 <= response.status_code < 300:
        raise SourceError(f"{label} returned HTTP {response.status_code}.")
    try:
        value = response.json()
    except ValueError:
        raise SourceError(f"{label} returned unreadable JSON.") from None
    if not isinstance(value, dict):
        raise SourceError(f"{label} returned an invalid response.")
    return value
