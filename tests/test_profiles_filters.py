from collections import Counter
from datetime import date

from conftest import make_posting, make_profile

from sjmb.filters import violations
from sjmb.profiles import generate_profiles
from sjmb.profiles.generator import grad_year_for
from sjmb.types import TRACKS


def test_profiles_are_deterministic_and_stratified():
    a = generate_profiles(200, seed=42)
    b = generate_profiles(200, seed=42)
    assert [p.to_dict() for p in a] == [p.to_dict() for p in b]
    assert len({p.id for p in a}) == 200
    tracks = Counter(p.track for p in a)
    assert set(tracks) == set(TRACKS) and min(tracks.values()) >= 24
    assert set(Counter(p.class_label for p in a)) == {"freshman", "sophomore", "junior", "senior"}
    assert set(Counter(p.work_auth for p in a)) == {
        "citizen",
        "permanent_resident",
        "needs_sponsorship",
    }
    assert all(p.seeking == ("new_grad" if p.class_label == "senior" else "internship") for p in a)


def test_grad_year_rolls_over_in_august():
    assert grad_year_for("senior", date(2026, 11, 9)) == 2027
    assert grad_year_for("freshman", date(2026, 11, 9)) == 2030
    assert grad_year_for("senior", date(2026, 3, 1)) == 2026


def test_eligibility_rules(as_of):
    prof = make_profile()
    assert violations(prof, make_posting(), as_of) == []
    assert "experienced_role" in violations(prof, make_posting(role_type="experienced"), as_of)
    assert "role_mismatch" in violations(prof, make_posting(role_type="new_grad"), as_of)
    assert "grad_year" in violations(prof, make_posting(grad_years=[2027]), as_of)
    assert "deadline_passed" in violations(prof, make_posting(deadline="2026-10-01"), as_of)
    assert "closed" in violations(prof, make_posting(active=False), as_of)
    visa = make_profile(work_auth="needs_sponsorship")
    assert "sponsorship" in violations(visa, make_posting(sponsorship="no"), as_of)
    assert violations(visa, make_posting(sponsorship="unknown"), as_of) == []
    pr = make_profile(work_auth="permanent_resident")
    assert violations(pr, make_posting(sponsorship="us_persons"), as_of) == []
    assert "sponsorship" in violations(visa, make_posting(sponsorship="us_persons"), as_of)
    assert "sponsorship" in violations(pr, make_posting(sponsorship="citizens_only"), as_of)
    # Unknown fields never exclude a student.
    assert violations(prof, make_posting(role_type="unknown", grad_years=[]), as_of) == []
