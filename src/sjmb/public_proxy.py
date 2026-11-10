"""A small local forward proxy that only connects to public internet addresses.

The capture browser sends all its traffic through this proxy. Each connection is resolved
here, refused if any resolved address is private, loopback, link-local or reserved, and then
made to the address that was checked. Because the check happens where the connection is made,
it also covers redirects, subresources, WebSockets and DNS rebinding, which a browser-side
request filter cannot.
"""

from __future__ import annotations

import contextlib
import ipaddress
import select
import socket
import socketserver
import threading
from urllib.parse import urlsplit

ALLOWED_PORTS = {80, 443, 8080, 8443}


def public_address(host: str, port: int) -> str:
    """Resolve host and return one address to connect to. Raises ValueError unless every
    resolved address is public."""
    if port not in ALLOWED_PORTS:
        raise ValueError(f"port {port} is not allowed")
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"cannot resolve {host}") from e
    addrs = [info[4][0] for info in infos]
    for a in addrs:
        if not ipaddress.ip_address(a).is_global:
            raise ValueError(f"{host} resolves to non-public address {a}")
    if not addrs:
        raise ValueError(f"no address for {host}")
    return addrs[0]


def _relay(a: socket.socket, b: socket.socket) -> None:
    socks = [a, b]
    while True:
        ready, _, _ = select.select(socks, [], [], 60)
        if not ready:
            return
        for s in ready:
            data = s.recv(65536)
            if not data:
                return
            (b if s is a else a).sendall(data)


class _Handler(socketserver.StreamRequestHandler):
    timeout = 30

    def handle(self) -> None:
        try:
            line = self.rfile.readline(8192).decode("latin-1").strip()
            method, target, _ = line.split(" ", 2)
            headers = []
            while (h := self.rfile.readline(8192)) not in (b"\r\n", b"\n", b""):
                headers.append(h)
            if method.upper() == "CONNECT":
                host, _, port = target.rpartition(":")
                addr = public_address(host.strip("[]"), int(port))
                with socket.create_connection((addr, int(port)), timeout=30) as up:
                    self.wfile.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
                    self.wfile.flush()
                    _relay(self.connection, up)
                return
            u = urlsplit(target)
            if u.scheme != "http" or not u.hostname:
                raise ValueError("only http and https are proxied")
            port = u.port or 80
            addr = public_address(u.hostname, port)
            path = (u.path or "/") + (f"?{u.query}" if u.query else "")
            kept = [h for h in headers if not h.lower().startswith((b"proxy-", b"connection:"))]
            with socket.create_connection((addr, port), timeout=30) as up:
                up.sendall(
                    f"{method} {path} HTTP/1.1\r\n".encode("latin-1")
                    + b"".join(kept)
                    + b"Connection: close\r\n\r\n"
                )
                _relay(self.connection, up)
        except (ValueError, OSError):
            with contextlib.suppress(OSError):
                self.wfile.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")


class PublicProxy:
    """Context manager: `with PublicProxy() as url:` gives a proxy URL on 127.0.0.1."""

    def __enter__(self) -> str:
        socketserver.ThreadingTCPServer.daemon_threads = True
        self.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()
