from types import SimpleNamespace

import pytest
from conftest import AS_OF, make_posting, make_profile

from sjmb import capture
from sjmb.capture import Capture, view_url
from sjmb.judge.judges import parse_judgment
from sjmb.reference import ReferenceGrader, ReferenceUnavailable

REPLY = (
    'Here is my answer:\n```json\n{"checks": "a: ok", "disqualifier": "b", "fit": 3, '
    '"reason": "window excludes 2029", "evidence": "Must graduate by August 2028"}\n```'
)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SJMB_CACHE_DIR", str(tmp_path))
    return tmp_path


def test_parse_judgment_tolerates_prose_around_json():
    assert parse_judgment(REPLY).grade == 0
    assert parse_judgment('Sure: {"disqualifier": "none", "fit": 2, "reason": "x"} done').grade == 2


def test_claude_backend_grades_caches_and_keeps_evidence(cache):
    calls = []

    def fake(prompt):
        calls.append(prompt)
        return REPLY

    g = ReferenceGrader("claude", AS_OF, llm=fake)
    j = g.grade(make_profile(), make_posting())
    assert (j.grade, j.disqualifier, j.fit) == (0, "b", 3)
    assert "evidence" in calls[0] and "STEP 1 - ELIGIBILITY" in calls[0]
    assert g.grade(make_profile(), make_posting()).cached and len(calls) == 1
    row = (cache / "reference.jsonl").read_text()
    assert "Must graduate by August 2028" in row and "claude:sonnet" in row
    assert ReferenceGrader("claude", AS_OF, llm=fake).is_cached(make_profile(), make_posting())


def test_reference_uses_captured_text_and_budget(cache):
    post = make_posting(url="https://example.com/j", description="Software role.")
    capture._save(Capture(post.url, "ok", text="Full captured posting. " * 40))
    seen = []
    g = ReferenceGrader("claude", AS_OF, max_calls=1, llm=lambda p: seen.append(p) or REPLY)
    g.grade(make_profile(), post)
    assert "Full captured posting." in seen[0]
    with pytest.raises(ReferenceUnavailable):
        g.grade(make_profile(), make_posting(id="other"))


def test_azure_backend_attaches_screenshot_for_thin_text(cache, monkeypatch):
    monkeypatch.setenv("AZURE_REFERENCE_DEPLOYMENT", "gpt-4.1")
    shot = cache / "page.png"
    shot.write_bytes(b"\x89PNG fake")
    post = make_posting(url="https://example.com/thin", description="Thin.")
    capture._save(Capture(post.url, "thin", text="Thin.", screenshot=str(shot)))
    sent = {}

    def create(**kw):
        sent.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=REPLY))])

    llm = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    g = ReferenceGrader("azure", AS_OF, llm=llm)
    assert g.grade(make_profile(), post).grade == 0
    parts = sent["messages"][0]["content"]
    assert sent["model"] == "gpt-4.1" and parts[1]["image_url"]["url"].startswith("data:image/png")


def test_unconfigured_backends_say_so(cache, monkeypatch):
    monkeypatch.delenv("AZURE_REFERENCE_DEPLOYMENT", raising=False)
    with pytest.raises(ReferenceUnavailable):
        ReferenceGrader("azure", AS_OF)


def test_view_url_strips_workable_apply():
    assert view_url("https://apply.workable.com/acme/j/ABC123/apply") == (
        "https://apply.workable.com/acme/j/ABC123"
    )
    assert view_url("https://example.com/jobs/1/apply") == "https://example.com/jobs/1/apply"
