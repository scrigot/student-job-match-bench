import math

import pytest
from conftest import make_posting, make_profile

from sjmb.data import load_fixture
from sjmb.eval.metrics import ndcg_at_k, poor_match_rate_at_k, recall_at_k, violation_rate_at_k
from sjmb.eval.runner import run_bench, to_markdown
from sjmb.judge import StubJudge, cohen_kappa
from sjmb.judge.judges import make_judge, parse_judgment, render_prompt
from sjmb.profiles import generate_profiles


def test_stub_judge_follows_rubric(as_of):
    j = StubJudge(as_of)
    prof = make_profile(skills=["Python", "SQL"])
    assert j.grade(prof, make_posting()).grade == 4
    assert j.grade(prof, make_posting(description="Rust")).grade == 3
    assert j.grade(prof, make_posting(track="data")).grade == 2
    assert j.grade(prof, make_posting(track="ib")).grade == 1
    assert j.grade(prof, make_posting(role_type="experienced")).grade == 0
    assert j.calls == 5
    # One-letter skills tokenize to nothing and must not count as a match.
    assert j.grade(make_profile(skills=["R", "Go"]), make_posting(description="Java")).grade == 3


def test_parse_judgment_variants():
    assert parse_judgment('{"grade": 3, "reason": "ok"}').grade == 3
    assert parse_judgment('```json\n{"grade": 0, "reason": "closed"}\n```').grade == 0
    assert parse_judgment('Sure. "grade": 2 because...').grade == 2
    # v4 format: the grade comes from the disqualifier and fit fields, not the model.
    assert parse_judgment('{"disqualifier": "a", "fit": 4, "reason": "intern vs FT"}').grade == 0
    assert parse_judgment('{"disqualifier": "none", "fit": 1, "reason": "off track"}').grade == 1
    assert parse_judgment('{"disqualifier": "None", "fit": 3, "reason": "ok"}').grade == 3
    assert parse_judgment('{"disqualifier": "d) no sponsor", "fit": 3, "reason": "x"}').grade == 0
    # Seen live: a disqualified reply with fit 0 must still parse, as a 0.
    assert (
        parse_judgment('{"checks": "c: PhD", "disqualifier": "c", "fit": 0, "reason": "PhD"}').grade
        == 0
    )
    assert parse_judgment('{"disqualifier": "none", "fit": 0, "reason": "odd"}').grade == 1
    with pytest.raises(ValueError):
        parse_judgment("no idea")


def test_prompt_contains_rubric_and_fields(as_of):
    p = render_prompt(make_profile(), make_posting(title="Quant Intern", grad_years=[2028]), as_of)
    assert "STEP 1 - ELIGIBILITY" in p and "Quant Intern" in p and "graduation year: 2028" in p
    assert "TODAY: 2026-11-09" in p and "extracted eligible graduation years: 2028" in p


def test_azure_judge_requires_env(monkeypatch, as_of):
    for v in ("AZURE_AI_ENDPOINT", "AZURE_AI_KEY", "AZURE_JUDGE_DEPLOYMENT"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(SystemExit):
        make_judge("azure", as_of)


def test_cohen_kappa_known_values():
    assert cohen_kappa([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) == pytest.approx(1.0)
    # Classic 2x2 example: po=0.7, pe=0.5 -> kappa 0.4.
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cohen_kappa(a, b, n_labels=2) == pytest.approx(0.4)
    # Near misses cost less under quadratic weights.
    h, m = [0, 1, 2, 3, 4, 4], [1, 1, 2, 3, 3, 4]
    assert cohen_kappa(h, m, weights="quadratic") > cohen_kappa(h, m)
    with pytest.raises(ValueError):
        cohen_kappa([1], [1, 2])
    with pytest.raises(ValueError):
        cohen_kappa([-1, 2], [0, 2])


def test_metrics():
    qrels = {"a": 4, "b": 3, "c": 0, "d": 1}
    assert ndcg_at_k(["a", "b", "d", "c"], qrels) == pytest.approx(1.0)
    assert ndcg_at_k(["c", "d", "b", "a"], qrels) < 0.7
    assert recall_at_k(["a", "x"], qrels, 50) == 0.5
    assert math.isnan(recall_at_k(["a"], {"a": 1}, 50))
    assert poor_match_rate_at_k(["a", "c", "d", "unjudged"], qrels, 10) == 0.75
    assert violation_rate_at_k(["a", "b"], {"b"}, 10) == 0.5
    assert math.isnan(poor_match_rate_at_k([], qrels))
    assert math.isnan(violation_rate_at_k([], set()))


def test_bench_end_to_end_on_sample(tmp_path, as_of):
    posts = load_fixture()
    profiles = generate_profiles(30, as_of=as_of)
    r = run_bench(
        posts, profiles, StubJudge(as_of), as_of, encoder="lsa", pool_depth=10, out_dir=tmp_path
    )
    assert len(r.systems) == 5
    for name, row in r.overall.items():
        assert 0 <= row["ndcg@10"] <= 1 and row["n"] == 30, name
    # The filter rungs never surface an ineligible posting.
    assert r.overall["hybrid-rrf + eligibility"]["elig_viol@10"] == 0
    assert r.overall["bm25 + eligibility"]["elig_viol@10"] == 0
    assert set(r.by_track) <= {"ib", "markets", "quant", "consulting", "swe", "data", "product"}
    md = to_markdown(r)
    assert "| hybrid-rrf + eligibility |" in md and "## By class year" in md
    assert (tmp_path / "results.json").exists() and (tmp_path / "judgments.jsonl").exists()
    with pytest.raises(ValueError):
        run_bench(posts, profiles[:1], StubJudge(as_of), as_of, encoder="lsa", pool_depth=5)
    with pytest.raises(ValueError):
        run_bench(posts, profiles[:1], StubJudge(as_of), as_of, encoder="lsa", pool_depth=60)


def test_hard_check_grad_window():
    from sjmb.judge.judges import hard_check

    student = make_profile(grad_year=2029)
    assert hard_check(student, make_posting(grad_years=[2028])) is not None  # outside window
    assert hard_check(student, make_posting(grad_years=[2028, 2029])) is None  # inside window
    assert hard_check(student, make_posting(grad_years=[])) is None  # not stated: never disqualify
