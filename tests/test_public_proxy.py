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
