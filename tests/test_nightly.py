from conftest import AS_OF

from sjmb.data import load_fixture
from sjmb.judge import StubJudge
from sjmb.nightly import make_batch, run_nightly, split_of, used_pairs
from sjmb.profiles import generate_profiles
from sjmb.reference import ReferenceGrader

REPLY = '{"disqualifier": "none", "fit": 2, "reason": "adjacent", "evidence": ""}'


def test_batch_is_fresh_half_finance_and_split_by_posting():
    posts, profiles = load_fixture(), generate_profiles(200, as_of=AS_OF)
    first = make_batch(profiles, posts, 60, used=set(), seed=1)
    assert len(first) == 60 and len({p.key for p in first}) == 60
    assert sum(p.group == "finance" for p in first) == 30
    again = make_batch(profiles, posts, 60, used={p.key for p in first}, seed=1)
    assert not {p.key for p in again} & {p.key for p in first}  # never reuses a pair
    assert [p.key for p in make_batch(profiles, posts, 60, set(), seed=1)] == [p.key for p in first]
    assert all(p.split == split_of(p.posting.id) for p in first)
    assert {split_of(f"post{i}") for i in range(50)} == {"dev", "test"}


def test_nightly_end_to_end_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("SJMB_CACHE_DIR", str(tmp_path / "cache"))
    golden = tmp_path / "golden"
    golden.mkdir()
    posts, profiles = load_fixture(), generate_profiles(200, as_of=AS_OF)
    ref = ReferenceGrader("claude", AS_OF, llm=lambda prompt: REPLY)
    out = golden / "batches" / "night1"
    report = run_nightly(profiles, posts, StubJudge(AS_OF), ref, AS_OF, out, golden, n=40, seed=3)
    text = report.read_text()
    assert "Reference grades are AI labels" in text and "Test split (never inspected)" in text
    assert "## Where they disagree (dev split only)" in text
    assert len(used_pairs(golden)) == 40  # the batch itself now counts as used
    for name in ("sheet.csv", "audit_sheet.csv", "manifest.json"):
        assert (out / name).exists()
    # Test-split pairs are never listed in the disagreement section.
    import json

    by_id = {p.id: p for p in posts}
    listed = text.split("## Where they disagree")[1].splitlines()
    for m in json.loads((out / "manifest.json").read_text()):
        if m["split"] == "test":
            post = by_id[m["posting"]]
            tail = f"× {post.company}: {post.title[:60]}"
            assert not any(ln.startswith(f"- {m['profile']} ") and tail in ln for ln in listed)
    # A second night never repeats the first night's pairs, and refuses to overwrite.
    out2 = golden / "batches" / "night2"
    run_nightly(profiles, posts, StubJudge(AS_OF), None, AS_OF, out2, golden, n=40, seed=3)
    assert "none configured" in (out2 / "report.md").read_text()
    assert len(used_pairs(golden)) == 80
    import pytest

    with pytest.raises(SystemExit):
        run_nightly(profiles, posts, StubJudge(AS_OF), None, AS_OF, out2, golden, n=10)


def test_spend_cap_stops_before_paying(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setenv("SJMB_CACHE_DIR", str(tmp_path / "cache"))
    posts, profiles = load_fixture(), generate_profiles(200, as_of=AS_OF)

    class Pricey(StubJudge):
        def is_cached(self, *a):
            return False

        def estimate_cost(self, pairs):
            return 9.0

    judge = Pricey(AS_OF)
    with pytest.raises(SystemExit, match="over the"):
        run_nightly(
            profiles,
            posts,
            judge,
            None,
            AS_OF,
            tmp_path / "g" / "b",
            tmp_path / "g",
            n=20,
            max_spend=5.0,
        )
    assert judge.calls == 0 and not (tmp_path / "g" / "b").exists()
