import csv
import zlib

import pytest
from conftest import AS_OF

from sjmb.data import load_fixture
from sjmb.golden import (
    COLUMNS,
    FILTER_OK_JUDGE_ZERO,
    FILTER_OUT_JUDGE_OK,
    _cell,
    agreement_report,
    sample_pairs,
    write_sheet,
)
from sjmb.judge import StubJudge
from sjmb.judge.agreement import read_golden
from sjmb.profiles import generate_profiles


def _judgments(n_profiles=20):
    posts = load_fixture()
    profiles = generate_profiles(n_profiles, as_of=AS_OF)
    judge = StubJudge(AS_OF)
    rows = []
    for prof in profiles:
        for post in posts[::4]:
            g = judge.grade(prof, post).grade
            # Flip a few grades so both disagreement directions exist, like a real LLM judge.
            if zlib.crc32(f"{prof.id}{post.id}".encode()) % 7 == 0:
                g = 3 if g == 0 else 0
            rows.append({"profile": prof.id, "posting": post.id, "grade": g})
    return rows, {p.id: p for p in profiles}, {p.id: p for p in posts}


def test_sample_is_deterministic_stratified_and_spread():
    rows, profiles, posts = _judgments()
    a = sample_pairs(rows, profiles, posts, AS_OF, n=100, seed=7)
    b = sample_pairs(list(reversed(rows)), profiles, posts, AS_OF, n=100, seed=7)
    key = [(r["profile"].id, r["posting"].id) for r in a]
    assert key == [(r["profile"].id, r["posting"].id) for r in b]
    assert len(set(key)) == 100
    strata = [r["stratum"] for r in a]
    assert strata.count(FILTER_OUT_JUDGE_OK) >= 10 and strata.count(FILTER_OK_JUDGE_ZERO) >= 10
    assert all(sum(r["band"] == band for r in a) >= 15 for band in ("0-1", "2", "3-4"))
    assert len({r["cell"] for r in a}) == len({(p.track, p.class_label) for p in profiles.values()})


def test_sheet_is_blind_and_round_trips(tmp_path):
    rows, profiles, posts = _judgments()
    pairs = sample_pairs(rows, profiles, posts, AS_OF, n=30, seed=7)
    path = tmp_path / "sheet.csv"
    write_sheet(pairs, path)
    text = path.read_text()
    assert text.startswith("# ") and "0 Unusable" in text
    body = [ln for ln in text.splitlines() if not ln.startswith("#")]
    header = next(csv.reader(body))
    assert header == COLUMNS
    assert "stratum" not in text and "judge" not in header
    assert read_golden(path) == []  # nothing graded yet
    # Grade it, with the rubric lines re-quoted the way a spreadsheet exports them.
    lines = text.splitlines()
    graded = [f'"{ln}",,,,,,' if ln.startswith("#") else ln for ln in lines]
    graded = [ln[:-2] + ",3," if ln[0].isdigit() else ln for ln in graded]
    path.write_text("\n".join(graded) + "\n")
    got = read_golden(path)
    assert len(got) == 30 and all(r["grade"] == 3 for r in got)


def test_agreement_report_sections():
    human = [0, 1, 2, 3, 4, 0, 4]
    model = [0, 1, 2, 3, 4, 3, 0]
    strata = [None, None, None, None, None, FILTER_OUT_JUDGE_OK, FILTER_OK_JUDGE_ZERO]
    out = agreement_report(human, model, strata)
    assert "cohen kappa (quadratic)" in out and "human \\ judge" in out
    assert "human sides with the filter: 1/1" in out  # human 0 where the filter excluded
    assert f"## Filter-judge disagreement: {FILTER_OK_JUDGE_ZERO}" in out


def test_ambiguous_grades_are_reported_not_guessed(tmp_path):
    path = tmp_path / "sheet.csv"
    path.write_text(
        "# rubric\npair_no,profile_id,posting_id,student,job,your_grade,notes\n"
        "1,p000,x,s,j,3,\n2,p001,y,s,j,3/2,between 3 and 2\n3,p002,z,s,j,5,\n4,p003,w,s,j,,\n"
    )
    invalid = []
    rows = read_golden(path, invalid)
    assert [r["grade"] for r in rows] == [3]
    assert len(invalid) == 2 and invalid[0].startswith("pair 2") and "3/2" in invalid[0]
    with pytest.raises(ValueError):
        read_golden(path)


def test_formula_like_cells_are_neutralized():
    assert _cell("=HYPERLINK(1)") == "'=HYPERLINK(1)"
    assert _cell("@SUM") == "'@SUM" and _cell("-1+2") == "'-1+2"
    assert _cell("Jane Street | Quant") == "Jane Street | Quant" and _cell(3) == 3
