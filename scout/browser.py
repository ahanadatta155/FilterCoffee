"""Bounded Playwright and optional CloakBrowser renderers for public pages."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from importlib.util import find_spec
from typing import Any, NoReturn
from urllib.parse import urlsplit

from scout.errors import BrowserError, RobotsError, SecurityError
from scout.proxy import GuardedProxy
from scout.security import Resolver, resolve_host, validate_url

_BROWSER_ARGS = [
    "--proxy-bypass-list=<-loopback>",
    "--disable-quic",
    "--disable-dns-prefetch",
    "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1",
    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
]


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
        target = validate_url(url, allow_localhost=self.allow_localhost, resolver=self.resolver)
        if find_spec("playwright") is None:
            raise BrowserError(
                "Browser rendering needs Playwright. Run `uv sync` and "
                "`uv run playwright install chromium`."
            )
        try:
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
                    args=_BROWSER_ARGS,
                )
                try:
                    return _render_loaded_browser(
                        browser,
                        target,
                        proxy,
                        user_agent=self.user_agent,
                        allow_localhost=self.allow_localhost,
                        resolver=self.resolver,
                        timeout_ms=self.timeout_ms,
                        capture_screenshot=capture_screenshot,
                    )
                finally:
                    browser.close()
        except (BrowserError, SecurityError):
            raise
        except Exception as error:
            _raise_browser_error(
                error,
                missing=("Chromium is not installed. Run `uv run playwright install chromium`."),
                generic=(
                    "Browser render failed; check the URL, connectivity, and Chromium installation."
                ),
            )


class CloakBrowserRenderer:
    """Optional CloakBrowser renderer with the same proxy and request bounds."""

    def __init__(
        self,
        *,
        user_agent: str = "NemotronScout/0.1",
        allow_localhost: bool = False,
        resolver: Resolver = resolve_host,
        timeout_ms: int = 20_000,
        headless: bool = True,
        humanize: bool = False,
        license_key: str | None = None,
        robots_check: Callable[[str], None] | None = None,
    ):
        self.user_agent = user_agent
        self.allow_localhost = allow_localhost
        self.resolver = resolver
        self.timeout_ms = timeout_ms
        self.headless = headless
        self.humanize = humanize
        self.license_key = license_key
        self.robots_check = robots_check

    def render(self, url: str, *, capture_screenshot: bool = False) -> RenderedPage:
        target = validate_url(url, allow_localhost=self.allow_localhost, resolver=self.resolver)
        if find_spec("cloakbrowser") is None:
            raise BrowserError(
                "CloakBrowser is optional and not installed. Run `uv sync --extra cloak`, "
                "then set SCOUT_CLOAKBROWSER=1."
            )
        if self.robots_check is not None:
            self.robots_check(target.url)
        try:
            launch = getattr(import_module("cloakbrowser"), "launch")
        except (ImportError, AttributeError):
            raise BrowserError(
                "CloakBrowser is optional and not installed. Run `uv sync --extra cloak`."
            ) from None

        launch_options: dict[str, Any] = {
            "headless": self.headless,
            "humanize": self.humanize,
            "args": _BROWSER_ARGS,
        }
        if self.license_key:
            launch_options["license_key"] = self.license_key
        try:
            with GuardedProxy(
                allow_localhost=self.allow_localhost, resolver=self.resolver
            ) as proxy:
                launch_options["proxy"] = {"server": proxy.url}
                browser = launch(**launch_options)
                try:
                    return _render_loaded_browser(
                        browser,
                        target,
                        proxy,
                        user_agent=self.user_agent,
                        allow_localhost=self.allow_localhost,
                        resolver=self.resolver,
                        timeout_ms=self.timeout_ms,
                        capture_screenshot=capture_screenshot,
                    )
                finally:
                    browser.close()
        except (BrowserError, RobotsError, SecurityError):
            raise
        except Exception as error:
            _raise_browser_error(
                error,
                missing=(
                    "CloakBrowser could not start its Chromium build. "
                    "Run `uv sync --extra cloak` and try again."
                ),
                generic="CloakBrowser render failed; check the URL and connectivity.",
            )


def make_renderer(
    *,
    user_agent: str = "NemotronScout/0.1",
    allow_localhost: bool = False,
    resolver: Resolver = resolve_host,
    timeout_ms: int = 20_000,
    robots_check: Callable[[str], None] | None = None,
) -> BrowserRenderer | CloakBrowserRenderer:
    """Use CloakBrowser only when explicitly enabled; otherwise use Playwright."""
    if os.getenv("SCOUT_CLOAKBROWSER", "0").strip().casefold() in {"1", "true", "yes"}:
        return CloakBrowserRenderer(
            user_agent=user_agent,
            allow_localhost=allow_localhost,
            resolver=resolver,
            timeout_ms=timeout_ms,
            headless=os.getenv("SCOUT_CLOAKBROWSER_HEADLESS", "1").strip().casefold()
            not in {"0", "false", "no"},
            humanize=os.getenv("SCOUT_CLOAKBROWSER_HUMANIZE", "0").strip().casefold()
            in {"1", "true", "yes"},
            license_key=os.getenv("CLOAKBROWSER_LICENSE_KEY", "").strip() or None,
            robots_check=robots_check,
        )
    return BrowserRenderer(
        user_agent=user_agent,
        allow_localhost=allow_localhost,
        resolver=resolver,
        timeout_ms=timeout_ms,
    )


def _render_loaded_browser(
    browser: Any,
    target: Any,
    proxy: GuardedProxy,
    *,
    user_agent: str,
    allow_localhost: bool,
    resolver: Resolver,
    timeout_ms: int,
    capture_screenshot: bool,
) -> RenderedPage:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    context = browser.new_context(
        user_agent=user_agent,
        locale="en-US",
        color_scheme="light",
        viewport={"width": 1280, "height": 1600},
        service_workers="block",
        accept_downloads=False,
    )
    try:
        context.route("**/*", _restrict_browser_request)
        page = context.new_page()
        page.on("dialog", lambda dialog: dialog.dismiss())
        page.on("popup", lambda popup: popup.close())
        response = page.goto(
            target.url,
            wait_until="domcontentloaded",
            timeout=timeout_ms,
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
                allow_localhost=allow_localhost,
                resolver=resolver,
            )
            final_url = final.url
        except SecurityError:
            final_url = target.url
            blocked_navigation = True
        status_code = 403 if blocked_navigation else response.status if response else None
        screenshot, screenshot_truncated = (
            _screenshot(page) if capture_screenshot else (None, False)
        )
        warnings = (
            [f"Blocked {proxy.blocked_requests} browser request(s) to unsafe destinations."]
            if proxy.blocked_requests
            else []
        )
        return RenderedPage(
            html, final_url, status_code, screenshot, warnings, screenshot_truncated
        )
    finally:
        context.close()


def _raise_browser_error(error: Exception, *, missing: str, generic: str) -> NoReturn:
    message = str(error).lower()
    if "executable doesn't exist" in message or "browser has been closed" in message:
        raise BrowserError(missing) from None
    raise BrowserError(generic) from None


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
