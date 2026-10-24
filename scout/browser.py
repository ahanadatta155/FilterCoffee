"""Bounded Playwright renderer for JavaScript-dependent public pages."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any
from urllib.parse import urlsplit

from scout.errors import BrowserError, SecurityError
from scout.proxy import GuardedProxy
from scout.security import Resolver, resolve_host, validate_url


@dataclass(frozen=True)
class RenderedPage:
    html: str
    final_url: str
    status_code: int | None
    screenshot: bytes | None = None
    warnings: list[str] = field(default_factory=list)
    screenshot_truncated: bool = False


class BrowserRenderer:
    """Render through an IP-pinning loopback proxy; never reuse browser state."""

    def __init__(
        self,
        *,
        user_agent: str = "NemotronScout/0.1",
        allow_localhost: bool = False,
        resolver: Resolver = resolve_host,
        timeout_ms: int = 20_000,
    ):
        self.user_agent = user_agent
        self.allow_localhost = allow_localhost
        self.resolver = resolver
        self.timeout_ms = timeout_ms

    def render(self, url: str, *, capture_screenshot: bool = False) -> RenderedPage:
        try:
            target = validate_url(url, allow_localhost=self.allow_localhost, resolver=self.resolver)
        except SecurityError:
            raise
        if find_spec("playwright") is None:
            raise BrowserError(
                "Browser rendering needs Playwright. Run `uv sync` and "
                "`uv run playwright install chromium`."
            )
        try:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise BrowserError(
                "Browser rendering needs Playwright. Run `uv sync` and "
                "`uv run playwright install chromium`."
            ) from None

        try:
            with (
                GuardedProxy(allow_localhost=self.allow_localhost, resolver=self.resolver) as proxy,
                sync_playwright() as playwright,
            ):
                browser = playwright.chromium.launch(
                    headless=True,
                    proxy={"server": proxy.url},
                    args=[
                        "--proxy-bypass-list=<-loopback>",
                        "--disable-quic",
                        "--disable-dns-prefetch",
                        "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1",
                        "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
                    ],
                )
                try:
                    context = browser.new_context(
                        user_agent=self.user_agent,
                        locale="en-US",
                        color_scheme="light",
                        viewport={"width": 1280, "height": 1600},
                        service_workers="block",
                        accept_downloads=False,
                    )
                    context.route("**/*", _restrict_browser_request)
                    page = context.new_page()
                    page.on("dialog", lambda dialog: dialog.dismiss())
                    page.on("popup", lambda popup: popup.close())
                    response = page.goto(
                        target.url,
                        wait_until="domcontentloaded",
                        timeout=self.timeout_ms,
                    )
                    try:
                        page.wait_for_load_state("networkidle", timeout=5_000)
                    except PlaywrightTimeoutError:
                        pass
                    page.wait_for_timeout(350)
                    html = page.content()
                    final_url = page.url
                    blocked_navigation = False
                    try:
                        final = validate_url(
                            final_url,
                            allow_localhost=self.allow_localhost,
                            resolver=self.resolver,
                        )
                        final_url = final.url
                    except SecurityError:
                        final_url = target.url
                        blocked_navigation = True
                    status_code = (
                        403 if blocked_navigation else response.status if response else None
                    )
                    screenshot, screenshot_truncated = (
                        _screenshot(page) if capture_screenshot else (None, False)
                    )
                    warnings = (
                        [
                            f"Blocked {proxy.blocked_requests} browser request(s) "
                            "to unsafe destinations."
                        ]
                        if proxy.blocked_requests
                        else []
                    )
                    context.close()
                    return RenderedPage(
                        html, final_url, status_code, screenshot, warnings, screenshot_truncated
                    )
                finally:
                    browser.close()
        except BrowserError:
            raise
        except SecurityError:
            raise
        except Exception as error:
            message = str(error).lower()
            if "executable doesn't exist" in message or "browser has been closed" in message:
                raise BrowserError(
                    "Chromium is not installed. Run `uv run playwright install chromium`."
                ) from None
            raise BrowserError(
                "Browser render failed; check the URL, connectivity, and Chromium installation."
            ) from None


def _restrict_browser_request(route: Any) -> None:
    request = route.request
    parsed = urlsplit(request.url)
    if parsed.scheme not in {"http", "https", "data", "about"}:
        route.abort()
    elif request.method not in {"GET", "HEAD"}:
        route.abort()
    else:
        route.continue_()


def _screenshot(page: Any) -> tuple[bytes, bool]:
    canvas = page.locator("canvas").first
    if canvas.count():
        try:
            image = canvas.screenshot(type="jpeg", quality=75, timeout=3_000)
            if len(image) <= 4 * 1024 * 1024:
                return image, False
        except Exception:
            pass
    dimensions = page.evaluate(
        "() => ({width: document.documentElement.scrollWidth, "
        "height: document.documentElement.scrollHeight})"
    )
    full_page = dimensions["height"] <= 6000 and dimensions["width"] <= 1920
    image = page.screenshot(type="jpeg", quality=70, full_page=full_page, timeout=5_000)
    truncated = not full_page
    if len(image) > 4 * 1024 * 1024:
        image = page.screenshot(type="jpeg", quality=45, full_page=False, timeout=5_000)
        truncated = True
    if len(image) > 4 * 1024 * 1024:
        raise BrowserError("Page screenshot exceeds the 4 MiB vision input limit.")
    return image, truncated
