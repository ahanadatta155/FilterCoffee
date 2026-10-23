"""Expose only OpenRouter account quotas that its key endpoint documents."""

import math
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _count(value: object) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


@dataclass(frozen=True)
class OpenRouterKeyStatus:
    free_requests_used: int | None
    free_requests_limit: int | None
    free_requests_remaining: int | None
    credit_limit: float | None
    credit_remaining: float | None
    daily_credit_usage: float | None

    def summary(self) -> list[str]:
        if self.free_requests_limit is None:
            lines = ["Free-model daily requests: not reported for this key"]
        else:
            used = "unknown" if self.free_requests_used is None else str(self.free_requests_used)
            remaining = (
                "unknown"
                if self.free_requests_remaining is None
                else str(self.free_requests_remaining)
            )
            lines = [
                f"Free-model daily requests: {used} used / {self.free_requests_limit} limit / "
                f"{remaining} remaining (UTC day)"
            ]
        if self.credit_limit is None:
            lines.append("Key credits: unlimited or not reported")
        elif self.credit_remaining is None:
            lines.append(f"Key credit limit: USD {self.credit_limit:.2f}; remaining not reported")
        else:
            lines.append(
                f"Key credits: USD {self.credit_remaining:.2f} remaining / "
                f"USD {self.credit_limit:.2f} limit"
            )
        if self.daily_credit_usage is not None:
            lines.append(f"Credit usage today: USD {self.daily_credit_usage:.2f}")
        lines.append("RPM: current remaining/minute count is not exposed by the key endpoint")
        lines.append("TPM/TPD: token-per-minute/day quotas are not exposed by the key endpoint")
        return lines


def get_openrouter_key_status(
    api_key: str, base_url: str = "https://openrouter.ai/api/v1/"
) -> OpenRouterKeyStatus:
    """Query the account-key endpoint; never return its label or raw response."""
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "openrouter.ai"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("OpenRouter API base URL must use https://openrouter.ai/.")
    endpoint = base_url.rstrip("/") + "/key"
    try:
        response = httpx.get(
            endpoint, headers={"Authorization": f"Bearer {api_key}"}, timeout=8, trust_env=False
        )
    except httpx.HTTPError:
        raise RuntimeError("OpenRouter quota status is temporarily unavailable.") from None
    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter quota status returned HTTP {response.status_code}.")
    try:
        data = response.json().get("data")
    except (ValueError, AttributeError):
        data = None
    if not isinstance(data, dict):
        raise RuntimeError("OpenRouter returned an invalid quota response.")
    daily = data.get("free_model_daily_requests")
    if not isinstance(daily, dict):
        daily = {}
    return OpenRouterKeyStatus(
        free_requests_used=_count(daily.get("used")),
        free_requests_limit=_count(daily.get("limit")),
        free_requests_remaining=_count(daily.get("remaining")),
        credit_limit=_number(data.get("limit")),
        credit_remaining=_number(data.get("limit_remaining")),
        daily_credit_usage=_number(data.get("usage_daily")),
    )
