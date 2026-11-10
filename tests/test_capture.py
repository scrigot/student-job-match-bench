import json

from conftest import make_posting

from sjmb import capture
from sjmb.capture import Capture, apply_captures, best_text, classify, load_capture


def test_classify_page_text():
    assert classify("Verify you are human to continue") == "blocked"
    assert classify("Just a moment... checking your browser") == "blocked"
    assert classify("short") == "thin"
    assert classify("Investment Banking Summer Analyst. " * 20) == "ok"
    # A long real posting that merely mentions a captcha is not a block page.
    assert (
        classify("We build CAPTCHA systems. " + "Responsibilities and requirements. " * 120) == "ok"
    )


def test_capture_cache_and_best_text(tmp_path, monkeypatch):
    monkeypatch.setenv("SJMB_CACHE_DIR", str(tmp_path))
    url = "https://example.com/job/1"
    assert load_capture(url) is None
    capture._save(Capture(url, "ok", text="full posting text " * 40, final_url=url))
    assert load_capture(url).status == "ok"
    assert best_text(url, "thin")[1] == "capture"
    assert best_text("https://example.com/other", "stored text") == ("stored text", "stored")
    raw = json.loads(next((tmp_path / "raw" / "captures").glob("*.json")).read_text())
    assert raw["url"] == url


def test_apply_captures_only_touches_thin_postings(tmp_path, monkeypatch):
    monkeypatch.setenv("SJMB_CACHE_DIR", str(tmp_path))
    thin = make_posting(id="a", url="https://example.com/a", description="Software role.")
    full = make_posting(id="b", url="https://example.com/b", description="x" * 500)
    text = "Must graduate by August 2028. We are unable to sponsor visas. " + "Details. " * 60
    capture._save(Capture(thin.url, "ok", text=text))
    capture._save(Capture(full.url, "ok", text="replacement " * 100))
    assert apply_captures([thin, full]) == 1
    assert thin.description.startswith("Must graduate") and 2028 in thin.grad_years
    assert thin.sponsorship == "no" and full.description == "x" * 500


def test_blocked_capture_is_never_used(tmp_path, monkeypatch):
    monkeypatch.setenv("SJMB_CACHE_DIR", str(tmp_path))
    url = "https://example.com/blocked"
    capture._save(Capture(url, "blocked"))
    assert best_text(url, "stored") == ("stored", "stored")
