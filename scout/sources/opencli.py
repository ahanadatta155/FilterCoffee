"""Opt-in bridge to an allowlisted OpenCLI site command.

Scout never imports Agent Reach, shares its browser profile, or accepts Reddit
as a site. The subprocess runs with the caller's local OpenCLI installation.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from scout.errors import SourceError
from scout.sources.base import Source, SourceSearch, SourceStatus, http_url, plain_text


class OpenCLISource(Source):
    name = "opencli"
    label = "OpenCLI"
    description = "An allowlisted OpenCLI site command executed in the local browser session."

    def __init__(
        self,
        *,
        site: str = "",
        command: str = "",
        allowlist: Iterable[str] = (),
        binary: str = "opencli",
        timeout: int = 60,
        env: Mapping[str, str] | None = None,
    ):
        self.site = site.strip().lower()
        self.command = command.strip()
        self.allowlist = {item.strip().lower() for item in allowlist if item.strip()}
        self.binary = binary.strip() or "opencli"
        self.timeout = timeout
        self._env = os.environ if env is None else env

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> OpenCLISource:
        try:
            timeout = max(1, int(env.get("SCOUT_OPENCLI_TIMEOUT", "60") or 60))
        except ValueError:
            timeout = 60
        return cls(
            site=env.get("SCOUT_OPENCLI_SITE", ""),
            command=env.get("SCOUT_OPENCLI_COMMAND", ""),
            allowlist=env.get("SCOUT_OPENCLI_ALLOWLIST", "").split(","),
            binary=env.get("SCOUT_OPENCLI_BIN", "opencli"),
            timeout=timeout,
            env=env,
        )

    def status(self) -> SourceStatus:
        if self._env.get("SCOUT_ENABLE_OPENCLI", "0").strip() != "1":
            return SourceStatus(False, "Set SCOUT_ENABLE_OPENCLI=1 to enable OpenCLI.")
        if not self.site or not self.command:
            return SourceStatus(
                False, "Set SCOUT_OPENCLI_SITE and SCOUT_OPENCLI_COMMAND to use OpenCLI."
            )
        if _blocked_site(self.site):
            return SourceStatus(False, "This site is not available as an automated OpenCLI source.")
        if self.site not in self.allowlist:
            return SourceStatus(False, f"Add '{self.site}' to SCOUT_OPENCLI_ALLOWLIST to use it.")
        if _which_binary(self.binary) is None:
            return SourceStatus(
                False, "OpenCLI is not installed; install it or set SCOUT_OPENCLI_BIN."
            )
        return SourceStatus(True)

    def search(self, query: str, *, limit: int = 5) -> SourceSearch:
        status = self.status()
        if not status.available:
            raise SourceError(status.reason)
        limit = max(1, min(limit, 10))
        try:
            args = shlex.split(self.command)
        except ValueError:
            raise SourceError("SCOUT_OPENCLI_COMMAND is not a valid argument list.") from None
        if _blocked_url(args):
            raise SourceError("Reddit URLs are not available as an automated source.")
        args = [part.replace("{query}", query).replace("{limit}", str(limit)) for part in args]
        if _blocked_url(args):
            raise SourceError("Reddit URLs are not available as an automated source.")
        binary = _which_binary(self.binary)
        if binary is None:
            raise SourceError("OpenCLI is not installed; install it or set SCOUT_OPENCLI_BIN.")
        command = _command(binary, self.site, args)
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                check=False,
                stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            raise SourceError("OpenCLI timed out; no retry attempted.") from None
        except OSError:
            raise SourceError("Could not start OpenCLI; check SCOUT_OPENCLI_BIN.") from None
        if process.returncode == 66:
            return SourceSearch([], ["OpenCLI returned no results."])
        if process.returncode != 0:
            raise SourceError(_failure(process.returncode, process.stdout))
        try:
            payload = json.loads(process.stdout)
        except (ValueError, RecursionError):
            raise SourceError("OpenCLI returned unreadable JSON.") from None
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise SourceError(_failure(process.returncode, process.stdout))
        rows = _rows(payload.get("data"))
        hits = []
        for row in rows[:limit]:
            if not isinstance(row, dict):
                continue
            title = _first(row, "title", "name", "headline", "text")
            url = http_url(_first(row, "url", "link", "href"))
            if not title or not url:
                continue
            snippet = _first(row, "description", "snippet", "summary", "text", "content")
            data = {
                str(key): value
                for key, value in row.items()
                if isinstance(value, (str, int, float, bool))
                and key not in {"title", "name", "headline", "url", "link", "href"}
            }
            hits.append(self.hit(title, url, plain_text(snippet)[:1200], data))
        return SourceSearch(
            hits,
            ["OpenCLI uses your local browser session; Scout did not fetch the page itself."],
        )


def _first(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _rows(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("items", "results", "posts", "rows", "list", "data"):
            if isinstance(value.get(key), list):
                return value[key]
        if any(key in value for key in ("title", "url", "text")):
            return [value]
    return []


def _which_binary(binary: str) -> str | None:
    """Resolve Windows command shims before the extensionless npm shell script."""
    candidates = [binary]
    if os.name == "nt" and not Path(binary).suffix:
        candidates = [f"{binary}.exe", f"{binary}.cmd", f"{binary}.bat", binary]
    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
    return None


def _command(binary: str, site: str, args: list[str]) -> list[str]:
    if os.name == "nt" and Path(binary).suffix.casefold() in {".cmd", ".bat"}:
        return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", binary, site, *args]
    return [binary, site, *args]


def _blocked_site(site: str) -> bool:
    return site in {"browser", "reddit", "redd.it"} or _is_reddit_url(site)


def _blocked_url(args: Iterable[str]) -> bool:
    return any(_is_reddit_url(value) for value in args)


def _is_reddit_url(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    candidates = [text]
    if "://" not in text:
        candidates.append("//" + text)
    for candidate in candidates:
        try:
            host = (urlsplit(candidate).hostname or "").lower().rstrip(".")
        except ValueError:
            continue
        if (
            host in {"reddit.com", "redd.it"}
            or host.endswith(".reddit.com")
            or host.endswith(".redd.it")
        ):
            return True
    return False


def _failure(returncode: int, stdout: str) -> str:
    code = ""
    try:
        payload = json.loads(stdout)
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict) and isinstance(error.get("code"), str):
                code = error["code"]
    except (ValueError, RecursionError):
        pass
    if code in {"NEED_LOGIN", "AUTH_REQUIRED"} or returncode == 77:
        return "OpenCLI needs a logged-in browser session for this site."
    if code in {"NEED_SETUP", "NOT_CONFIGURED"}:
        return "OpenCLI site setup is incomplete; run its setup command first."
    if code in {"PLATFORM_BLOCKED", "UPSTREAM_BROKEN"}:
        return "OpenCLI could not read the site; do not retry in a loop."
    return "OpenCLI failed; check its local browser session and site adapter."
