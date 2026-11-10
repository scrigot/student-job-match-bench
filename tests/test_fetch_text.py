import pytest

from sjmb.data.fetch_text import UnsafeURL, _host_is, check_public_url, fetch_posting_text


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "http://127.0.0.1/admin",
        "http://localhost:8080/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/",
        "http://[::1]/",
    ],
)
def test_private_and_non_http_urls_are_refused(url):
    with pytest.raises(UnsafeURL):
        check_public_url(url)


def test_fetch_never_raises_on_unsafe_url(tmp_path, monkeypatch):
    monkeypatch.setenv("SJMB_CACHE_DIR", str(tmp_path))
    got = fetch_posting_text("http://127.0.0.1/job/1")
    assert got.text == "" and got.method.startswith("error")


def test_ats_domains_match_exactly():
    assert _host_is("https://acme.wd1.myworkdayjobs.com/x/job/y", "myworkdayjobs.com")
    assert not _host_is("https://myworkdayjobs.com.evil.example/x", "myworkdayjobs.com")
    assert not _host_is("https://evil.example/?u=jobs.lever.co", "lever.co")
