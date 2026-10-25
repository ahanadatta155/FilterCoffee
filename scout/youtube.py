"""Read one YouTube video's metadata and available captions, without media downloads."""

import json
import re
import subprocess
import sys
import tempfile
from html import escape
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx

from scout.discovery import SearchResult
from scout.errors import FetchError
from scout.fetch import FetchedPage
from scout.proxy import GuardedProxy
from scout.security import PinnedTransport

_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}


def is_youtube_url(url: str) -> bool:
    try:
        return urlsplit(url).hostname in _HOSTS
    except ValueError:
        return False


def video_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in _HOSTS
            or parsed.username
            or parsed.password
            or parsed.port not in {None, 80, 443}
        ):
            raise ValueError
        parts = parsed.path.strip("/").split("/")
        if parsed.hostname in {"youtu.be", "www.youtu.be"} and len(parts) == 1:
            identity = parts[0]
        elif parsed.path == "/watch":
            identity = parse_qs(parsed.query).get("v", [""])[0]
        elif len(parts) == 2 and parts[0] in {"shorts", "embed", "live"}:
            identity = parts[1]
        else:
            raise ValueError
        if not _ID.fullmatch(identity):
            raise ValueError
    except ValueError:
        raise FetchError(
            "Use a single public YouTube video URL, not a channel or playlist."
        ) from None
    return f"https://www.youtube.com/watch?v={identity}"


class YouTubeClient:
    def __init__(self, *, http_client: httpx.Client | None = None):
        self._owned = http_client is None
        self._client = http_client or httpx.Client(
            transport=PinnedTransport(), trust_env=False, follow_redirects=False, timeout=15
        )

    def close(self) -> None:
        if self._owned:
            self._client.close()

    def _info(self, target: str, *, search: bool = False) -> dict[str, Any]:
        # A fresh subprocess supplies a hard deadline; no user config/plugins/cookies.
        with GuardedProxy() as proxy, tempfile.TemporaryFile() as output:
            command = [
                sys.executable,
                "-m",
                "yt_dlp",
                "--ignore-config",
                "--no-plugin-dirs",
                "--skip-download",
                "--dump-single-json",
                "--no-playlist",
                "--no-warnings",
                "--no-js-runtimes",
                "--no-cache-dir",
                "--ignore-no-formats-error",
                "--socket-timeout",
                "15",
                "--retries",
                "0",
                "--extractor-retries",
                "0",
                "--proxy",
                proxy.url,
            ]
            if search:
                command.append("--flat-playlist")
            command.extend(["--", target])
            try:
                process = subprocess.run(
                    command,
                    stdout=output,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    timeout=60,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                raise FetchError(
                    "YouTube metadata request timed out; no retry attempted."
                ) from None
            except OSError:
                raise FetchError("Could not start yt-dlp; run `uv sync`.") from None
            if process.returncode:
                raise FetchError(
                    "yt-dlp could not read YouTube (unavailable, verification, rate limit, or "
                    "unsupported extractor). No login or alternate provider was attempted."
                )
            output.seek(0)
            raw = output.read(5 * 1024 * 1024 + 1)
            if len(raw) > 5 * 1024 * 1024:
                raise FetchError("YouTube metadata exceeds the 5 MiB limit.")
        try:
            value = json.loads(raw)
        except (ValueError, RecursionError):
            raise FetchError("yt-dlp returned unreadable metadata.") from None
        if not isinstance(value, dict):
            raise FetchError("yt-dlp returned invalid metadata.")
        return value

    def search(self, query: str, *, include_domains: list[str] | None = None) -> list[SearchResult]:
        data = self._info(f"ytsearch5:{query}", search=True)
        rows = data.get("entries")
        if not isinstance(rows, list):
            raise FetchError("YouTube returned invalid search results.")
        results = []
        for row in rows[:5]:
            if not isinstance(row, dict) or not _ID.fullmatch(str(row.get("id", ""))):
                continue
            results.append(
                SearchResult(
                    str(row.get("title") or "YouTube video")[:300],
                    f"https://www.youtube.com/watch?v={row['id']}",
                    str(row.get("description") or "")[:1200],
                )
            )
        return results

    def _captions(self, data: dict[str, Any]) -> tuple[str, str]:
        for kind, label in (("subtitles", "manual"), ("automatic_captions", "automatic")):
            tracks = data.get(kind)
            if not isinstance(tracks, dict):
                continue
            languages = sorted(tracks, key=lambda lang: (not lang.startswith("en"), lang))
            for lang in languages:
                formats = tracks[lang]
                if not isinstance(formats, list):
                    continue
                for track in formats:
                    if not isinstance(track, dict) or track.get("ext") != "json3":
                        continue
                    url = track.get("url")
                    if not isinstance(url, str):
                        continue
                    parsed = urlsplit(url)
                    # Caption links come from untrusted metadata: only a known caption endpoint.
                    if (
                        parsed.scheme != "https"
                        or parsed.hostname != "www.youtube.com"
                        or parsed.path != "/api/timedtext"
                        or parsed.username
                        or parsed.password
                        or parsed.port not in {None, 443}
                    ):
                        continue
                    with self._client.stream("GET", url, timeout=15) as response:
                        if response.status_code != 200:
                            raise FetchError(
                                f"YouTube subtitles returned HTTP {response.status_code}; stopped."
                            )
                        body = bytearray()
                        for chunk in response.iter_bytes(chunk_size=65_536):
                            body.extend(chunk)
                            if len(body) > 2 * 1024 * 1024:
                                raise FetchError("YouTube subtitles exceed the 2 MiB limit.")
                    payload = json.loads(body)
                    events = payload.get("events") if isinstance(payload, dict) else None
                    if not isinstance(events, list):
                        raise FetchError("YouTube subtitles have an invalid structure.")
                    lines = []
                    for event in events:
                        segments = event.get("segs") if isinstance(event, dict) else None
                        if not isinstance(segments, list):
                            continue
                        line = "".join(
                            s["utf8"]
                            for s in segments
                            if isinstance(s, dict) and isinstance(s.get("utf8"), str)
                        ).strip()
                        if line and (not lines or lines[-1] != line):
                            lines.append(line)
                    if lines:
                        return "\n".join(lines), f"YouTube subtitles: {label}, language {lang}."
                    return "", "Selected YouTube subtitles were empty; metadata only, no retry."
        return "", "No usable subtitles returned; metadata only, not a video transcript."

    def fetch(self, url: str) -> FetchedPage:
        canonical = video_url(url)
        data = self._info(canonical)
        if data.get("id") != parse_qs(urlsplit(canonical).query)["v"][0]:
            raise FetchError("yt-dlp returned a different video; extraction stopped.")
        warnings = [
            "Source: yt-dlp metadata and captions; no audio transcription or visual analysis."
        ]
        try:
            transcript, notice = self._captions(data)
        except (FetchError, httpx.HTTPError, ValueError, RecursionError) as error:
            transcript = ""
            notice = (
                str(error)
                if isinstance(error, FetchError)
                else "YouTube subtitles could not be read; metadata only, no retry attempted."
            )
        warnings.append(notice)
        title = str(data.get("title") or "YouTube video")
        fields = {
            "Title": title,
            "Channel": data.get("uploader") or data.get("channel"),
            "Description": data.get("description"),
            "Duration seconds": data.get("duration"),
        }
        date = data.get("upload_date")
        if isinstance(date, str) and re.fullmatch(r"\d{8}", date):
            fields["Upload date"] = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        sections = [
            f"<h2>{escape(name)}</h2><pre>{escape(str(value))}</pre>"
            for name, value in fields.items()
            if value is not None
        ]
        if transcript:
            sections.append(f"<h2>Subtitles</h2><pre>{escape(transcript)}</pre>")
        html = (
            f"<html><head><title>{escape(title)}</title></head>"
            f"<body><main>{''.join(sections)}</main></body></html>"
        )
        return FetchedPage(canonical, html, 200, "text/html", tier="youtube", warnings=warnings)
