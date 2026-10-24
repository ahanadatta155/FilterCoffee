"""Loopback HTTP proxy that validates and pins browser network destinations."""

from __future__ import annotations

import http.client
import ipaddress
import select
import socket
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler
from typing import Any, cast
from urllib.parse import urlsplit

from scout.errors import SecurityError
from scout.security import Resolver, resolve_host, validate_url

MAX_RESPONSE_BYTES = 5 * 1024 * 1024


class GuardedProxy:
    """Forward browser HTTP(S) through a private-address rejecting loopback proxy."""

    def __init__(
        self,
        *,
        allow_localhost: bool = False,
        resolver: Resolver = resolve_host,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
    ):
        self.allow_localhost = allow_localhost
        self.resolver = resolver
        self.max_response_bytes = max_response_bytes
        self._blocked_requests = 0
        self._counter_lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_CONNECT(self) -> None:
                try:
                    authority = urlsplit(f"//{self.path}")
                    host, port = authority.hostname, authority.port
                    if not host or port is None:
                        raise ValueError("Invalid CONNECT authority")
                    target = validate_url(
                        f"https://{self.path}/",
                        allow_localhost=owner.allow_localhost,
                        resolver=owner.resolver,
                    )
                except (ValueError, SecurityError):
                    owner._mark_blocked()
                    self.send_error(403, "Destination blocked")
                    return
                address = min(
                    target.addresses,
                    key=lambda item: (ipaddress.ip_address(item).version, item),
                )
                try:
                    upstream = socket.create_connection((address, target.port), timeout=10)
                except OSError:
                    self.send_error(502, "Upstream unavailable")
                    return
                self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                self.wfile.flush()
                self.tunnel(self.connection, upstream)

            def do_GET(self) -> None:
                self.forward_http()

            def do_HEAD(self) -> None:
                self.forward_http()

            def do_POST(self) -> None:
                self.send_error(405, "Read-only browser requests only")

            def forward_http(self) -> None:
                try:
                    target = validate_url(
                        self.path,
                        allow_localhost=owner.allow_localhost,
                        resolver=owner.resolver,
                    )
                except SecurityError:
                    owner._mark_blocked()
                    self.send_error(403, "Destination blocked")
                    return
                parsed = urlsplit(target.url)
                address = min(
                    target.addresses,
                    key=lambda item: (ipaddress.ip_address(item).version, item),
                )
                connection = http.client.HTTPConnection(address, target.port, timeout=20)
                headers = {
                    key: value
                    for key, value in self.headers.items()
                    if key.lower() not in {"connection", "proxy-connection", "host"}
                }
                headers["Host"] = parsed.netloc
                headers["Connection"] = "close"
                path = parsed.path or "/"
                if parsed.query:
                    path += "?" + parsed.query
                try:
                    connection.request(self.command, path, headers=headers)
                    response = connection.getresponse()
                    body = response.read(owner.max_response_bytes + 1)
                    if len(body) > owner.max_response_bytes:
                        self.send_error(413, "Upstream response too large")
                        return
                    self.wfile.write(f"HTTP/1.1 {response.status} {response.reason}\r\n".encode())
                    for key, value in response.getheaders():
                        if key.lower() not in {"connection", "transfer-encoding", "content-length"}:
                            self.wfile.write(f"{key}: {value}\r\n".encode("latin-1"))
                    self.wfile.write(
                        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                    )
                    if self.command != "HEAD":
                        self.wfile.write(body)
                    self.wfile.flush()
                except (OSError, http.client.HTTPException):
                    self.send_error(502, "Upstream unavailable")
                finally:
                    connection.close()

            @staticmethod
            def tunnel(client: socket.socket, upstream: socket.socket) -> None:
                deadline = time.monotonic() + 90
                client.setblocking(False)
                upstream.setblocking(False)
                sockets = (client, upstream)
                try:
                    while time.monotonic() < deadline:
                        readable, _, _ = select.select(sockets, [], [], 1)
                        if not readable:
                            continue
                        for source in readable:
                            destination = upstream if source is client else client
                            data = source.recv(65_536)
                            if not data:
                                return
                            destination.settimeout(15)
                            destination.setblocking(True)
                            destination.sendall(data)
                            destination.setblocking(False)
                except OSError:
                    return
                finally:
                    upstream.close()

            def log_message(self, format: str, *args: Any) -> None:
                return

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = Server(("127.0.0.1", 0), Handler)
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = cast(tuple[str, int], self._server.server_address)
        return f"http://{host}:{port}"

    @property
    def blocked_requests(self) -> int:
        with self._counter_lock:
            return self._blocked_requests

    def _mark_blocked(self) -> None:
        with self._counter_lock:
            self._blocked_requests += 1

    def __enter__(self) -> GuardedProxy:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread:
            self._thread.join(timeout=2)

    def __exit__(self, *_: object) -> None:
        self.close()
