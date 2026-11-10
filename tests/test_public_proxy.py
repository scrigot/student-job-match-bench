import contextlib
import socket

import pytest

from sjmb.public_proxy import PublicProxy, public_address


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "10.0.0.5", "169.254.169.254", "::1"])
def test_private_addresses_are_refused(host):
    with pytest.raises(ValueError):
        public_address(host, 443)


def test_odd_ports_are_refused():
    with pytest.raises(ValueError, match="port"):
        public_address("example.com", 22)


def _ask(proxy_url: str, request: bytes) -> bytes:
    port = int(proxy_url.rsplit(":", 1)[1])
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall(request)
        return s.recv(200)


def test_proxy_refuses_private_targets_for_https_http_and_other_schemes():
    with PublicProxy() as url:
        assert b"403" in _ask(url, b"CONNECT 127.0.0.1:443 HTTP/1.1\r\n\r\n")
        assert b"403" in _ask(url, b"CONNECT 169.254.169.254:80 HTTP/1.1\r\n\r\n")
        assert b"403" in _ask(url, b"GET http://localhost/admin HTTP/1.1\r\nHost: x\r\n\r\n")
        assert b"403" in _ask(url, b"GET ftp://example.com/x HTTP/1.1\r\n\r\n")


def test_plain_http_forwards_one_request_only(monkeypatch):
    """A second pipelined request (to a private host) must never reach any upstream."""
    import threading

    from sjmb import public_proxy

    got = []
    upstream = socket.socket()
    upstream.bind(("127.0.0.1", 0))
    upstream.listen(1)
    up_port = upstream.getsockname()[1]

    def serve():
        conn, _ = upstream.accept()
        conn.settimeout(2)
        data = b""
        try:
            while b"\r\n\r\n" not in data:
                data += conn.recv(4096)
            with contextlib.suppress(TimeoutError):
                data += conn.recv(4096)  # anything pipelined after the first request
        finally:
            got.append(data)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    # Pretend the test upstream is a public host on an allowed port.
    monkeypatch.setattr(public_proxy, "public_address", lambda host, port: "127.0.0.1")
    monkeypatch.setattr(public_proxy, "ALLOWED_PORTS", {up_port})
    with PublicProxy() as url:
        port = int(url.rsplit(":", 1)[1])
        with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
            s.sendall(
                f"GET http://public.example:{up_port}/a HTTP/1.1\r\nHost: public.example\r\n\r\n"
                "GET http://169.254.169.254/latest/meta-data/ HTTP/1.1\r\nHost: x\r\n\r\n".encode()
            )
            reply = s.recv(200)
    assert b"200 OK" in reply
    assert got and b"GET /a HTTP/1.1" in got[0] and b"169.254.169.254" not in got[0]
