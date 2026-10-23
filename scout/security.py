"""Validate destinations and pin connections to checked DNS addresses."""

import ipaddress
import socket
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx

from scout.errors import SecurityError

Resolver = Callable[[str, int], Iterable[str]]


def resolve_host(hostname: str, port: int) -> list[str]:
    return [
        str(entry[4][0]) for entry in socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    ]


@dataclass(frozen=True)
class ValidatedURL:
    url: str
    hostname: str
    port: int
    addresses: tuple[str, ...]
    local: bool


def validate_url(
    url: str, *, allow_localhost: bool = False, resolver: Resolver = resolve_host
) -> ValidatedURL:
    """Reject unsafe URLs and mixed public/private answers; localhost is opt-in."""
    try:
        if not isinstance(url, str) or not url or "\\" in url:
            raise ValueError
        if any(char.isspace() for char in url) or any(
            ord(char) < 32 or ord(char) == 127 for char in unquote(url)
        ):
            raise ValueError
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError
        if parsed.username is not None or parsed.password is not None or "%" in parsed.hostname:
            raise ValueError
        port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
        if not 1 <= port <= 65535:
            raise ValueError
        hostname = parsed.hostname.rstrip(".").encode("idna").decode("ascii").lower()
        if not hostname:
            raise ValueError
    except (ValueError, UnicodeError):
        raise SecurityError(
            "Use a valid HTTP(S) URL without credentials or control characters."
        ) from None
    hostname_local = hostname == "localhost" or hostname.endswith(".localhost")
    if hostname_local and not allow_localhost:
        raise SecurityError("Localhost is disabled; enable it only for local fixtures.")
    literal = None
    try:
        literal = ipaddress.ip_address(hostname)
        addresses = [str(literal)]
    except ValueError:
        try:
            addresses = list(resolver(hostname, port))
        except OSError:
            raise SecurityError("Could not resolve the page hostname; check the URL.") from None
    if not addresses:
        raise SecurityError("Could not resolve the page hostname; check the URL.")
    checked = []
    local = False
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            raise SecurityError("DNS returned an invalid address.") from None
        effective = getattr(ip, "ipv4_mapped", None) or ip
        local_name = hostname_local or (literal is not None and literal.is_loopback)
        if hostname_local and not effective.is_loopback:
            raise SecurityError("Localhost must resolve to a loopback address.")
        if allow_localhost and local_name and effective.is_loopback:
            local = True
        elif not effective.is_global or effective.is_multicast or effective.is_reserved:
            raise SecurityError("Private, local, reserved, and metadata destinations are blocked.")
        checked.append(str(ip))
    host = f"[{hostname}]" if ":" in hostname else hostname
    authority = f"{host}:{port}" if parsed.port is not None else host
    normalized = urlunsplit((parsed.scheme, authority, parsed.path or "/", parsed.query, ""))
    return ValidatedURL(normalized, hostname, port, tuple(dict.fromkeys(checked)), local)


class PinnedTransport(httpx.BaseTransport):
    """Preserve Host/TLS identity while connecting only to a validated IP."""

    def __init__(
        self,
        *,
        allow_localhost: bool = False,
        resolver: Resolver = resolve_host,
        inner: httpx.BaseTransport | None = None,
    ):
        self.allow_localhost = allow_localhost
        self.resolver = resolver
        self.inner = inner if inner is not None else httpx.HTTPTransport(trust_env=False, retries=0)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        target = validate_url(
            str(request.url), allow_localhost=self.allow_localhost, resolver=self.resolver
        )
        ip = sorted(
            target.addresses, key=lambda value: (ipaddress.ip_address(value).version, value)
        )[0]
        headers = request.headers.copy()
        headers["host"] = request.url.netloc.decode("ascii")
        pinned = httpx.Request(
            request.method,
            request.url.copy_with(host=ip),
            headers=headers,
            stream=request.stream,
            extensions={**request.extensions, "sni_hostname": target.hostname},
        )
        return self.inner.handle_request(pinned)

    def close(self) -> None:
        self.inner.close()
