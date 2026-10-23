"""Robots rules are fetched through the same guarded downloader as pages."""

import time
from collections.abc import Callable
from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

from scout.errors import RobotsError


class RobotsChecker:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic):
        self._cache: dict[str, tuple[float, RobotFileParser]] = {}
        self._clock = clock

    def check(self, url: str, user_agent: str, download: Callable[[str], tuple[int, str]]) -> None:
        parsed = urlsplit(url)
        origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        cached = self._cache.get(origin)
        if cached is None or self._clock() - cached[0] >= 3600:
            status, text = download(origin + "/robots.txt")
            if status == 404:
                lines = []
            elif status in {401, 403}:
                lines = ["User-agent: *", "Disallow: /"]
            elif 200 <= status < 300:
                lines = text.splitlines()
            else:
                raise RobotsError("Could not verify robots.txt; try again later.")
            parser = RobotFileParser(origin + "/robots.txt")
            parser.parse(lines)
            cached = (self._clock(), parser)
            self._cache[origin] = cached
        if not cached[1].can_fetch(user_agent, url):
            raise RobotsError(
                "robots.txt disallows this path. Use --ignore-robots only if you own the site."
            )
