"""Bounded HTML fetching with robots, throttling, and guarded manual redirects."""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx

from scout.errors import FetchError
from scout.robots import RobotsChecker
from scout.security import PinnedTransport, Resolver, resolve_host, validate_url


class RateLimiter:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._clock, self._sleep = clock, sleep
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, domain: str) -> None:
        with self._lock:
            previous = self._last.get(domain)
            delay = 0 if previous is None else 1 - (self._clock() - previous)
            if delay > 0:
                self._sleep(delay)
            self._last[domain] = self._clock()


_LIMITER = RateLimiter()


@dataclass(frozen=True)
class FetchedPage:
    url: str
    html: str
    status_code: int
    content_type: str
    tier: str = "http"
    warnings: list[str] = field(default_factory=list)


class Fetcher:
    def __init__(
        self,
        *,
        user_agent: str = "NemotronScout/0.1",
        allow_localhost: bool = False,
        client: httpx.Client | None = None,
        resolver: Resolver = resolve_host,
        limiter: RateLimiter | None = None,
        robots: RobotsChecker | None = None,
        max_bytes: int = 5 * 1024 * 1024,
    ):
        self.user_agent, self.allow_localhost = user_agent, allow_localhost
        self.resolver, self.max_bytes = resolver, max_bytes
        self.limiter = limiter if limiter is not None else _LIMITER
        self.robots = robots if robots is not None else RobotsChecker()
        self._owned = client is None
        self.client = (
            client
            if client is not None
            else httpx.Client(
                transport=PinnedTransport(allow_localhost=allow_localhost, resolver=resolver),
                trust_env=False,
                follow_redirects=False,
                timeout=20,
            )
        )

    def close(self) -> None:
        if self._owned:
            self.client.close()

    def __enter__(self) -> "Fetcher":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def fetch(self, url: str, *, ignore_robots: bool = False) -> FetchedPage:
        final, status, content_type, text = self._download(url, check_robots=not ignore_robots)
        if status == 403:
            raise FetchError(
                "Site blocked access (403). Choose another public URL; bypass is not supported."
            )
        if status == 429:
            raise FetchError("Site rate limit reached (429); wait before trying again.")
        if not 200 <= status < 300:
            raise FetchError(f"Page returned HTTP {status}; check the URL or choose another page.")
        if content_type not in {"text/html", "application/xhtml+xml"}:
            raise FetchError("Unsupported content: only HTML pages are supported.")
        warnings = []
        if self.allow_localhost:
            warnings.append("Localhost access enabled for loopback fixtures only.")
        if ignore_robots:
            warnings.append("robots.txt check explicitly disabled; use only on sites you own.")
        return FetchedPage(final, text, status, content_type, warnings=warnings)

    def check_robots(self, url: str, *, ignore_robots: bool = False) -> None:
        """Apply the same robots policy to an alternate renderer such as CloakBrowser."""
        if ignore_robots:
            return
        target = validate_url(url, allow_localhost=self.allow_localhost, resolver=self.resolver)
        self.robots.check(target.url, self.user_agent, self._robots_download)

    def _robots_download(self, url: str) -> tuple[int, str]:
        _, status, _, text = self._download(url, check_robots=False)
        return status, text

    def _download(self, url: str, *, check_robots: bool) -> tuple[str, int, str, str]:
        for hop in range(6):
            target = validate_url(url, allow_localhost=self.allow_localhost, resolver=self.resolver)
            if check_robots:
                self.robots.check(target.url, self.user_agent, self._robots_download)
            self.limiter.wait(target.hostname)
            headers = {
                "User-Agent": f"Mozilla/5.0 (compatible; {self.user_agent})",
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "identity",
            }
            try:
                with self.client.stream(
                    "GET", target.url, headers=headers, timeout=20, follow_redirects=False
                ) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location or hop == 5:
                            raise FetchError(
                                "Missing redirect target or too many redirects (maximum 5)."
                            )
                        url = urljoin(target.url, location)
                        continue
                    length = response.headers.get("content-length", "")
                    if length.isdigit() and int(length) > self.max_bytes:
                        raise FetchError("Response exceeds the allowed body size limit.")
                    chunks = bytearray()
                    started = time.monotonic()
                    for chunk in response.iter_bytes(chunk_size=65_536):
                        if len(chunks) + len(chunk) > self.max_bytes:
                            raise FetchError("Response exceeds the allowed body size limit.")
                        if time.monotonic() - started > 20:
                            raise FetchError("Page download timed out; choose a smaller page.")
                        chunks.extend(chunk)
                    content_type = (
                        response.headers.get("content-type", "").split(";", 1)[0].lower().strip()
                    )
                    text = chunks.decode(response.encoding or "utf-8", errors="replace")
                    return target.url, response.status_code, content_type, text
            except httpx.TimeoutException:
                raise FetchError(
                    "Page request timed out; try again or choose another URL."
                ) from None
            except httpx.HTTPError:
                raise FetchError(
                    "Page request failed; check connectivity or choose another URL."
                ) from None
        raise FetchError("Too many redirects.")
