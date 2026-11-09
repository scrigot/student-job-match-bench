"""Hard eligibility rules for a (student, posting) pair.

Used two ways:
- as a pre-retrieval filter in the ladder (the ablation), and
- as the ground truth for the eligibility-violation rate, which every system is scored on.

A field the posting does not state never makes it ineligible: unknown means allowed.
Policy: students seek one level. Seniors look for new-grad roles and everyone else for
internships, so a senior is not matched to internships (some seniors do apply to them).
"""

from __future__ import annotations

from datetime import date

import numpy as np

from sjmb.types import Posting, Profile

REASONS = (
    "closed",
    "deadline_passed",
    "experienced_role",
    "role_mismatch",
    "grad_year",
    "sponsorship",
)


def violations(profile: Profile, posting: Posting, as_of: date) -> list[str]:
    out: list[str] = []
    if not posting.active:
        out.append("closed")
    deadline = posting.deadline_date()
    if deadline is not None and deadline < as_of:
        out.append("deadline_passed")
    if posting.role_type == "experienced":
        out.append("experienced_role")
    elif posting.role_type in ("internship", "new_grad") and posting.role_type != profile.seeking:
        out.append("role_mismatch")
    if posting.grad_years and profile.grad_year not in posting.grad_years:
        out.append("grad_year")
    if profile.work_auth == "needs_sponsorship" and posting.sponsorship in (
        "no",
        "us_persons",
        "citizens_only",
    ):
        out.append("sponsorship")
    if profile.work_auth == "permanent_resident" and posting.sponsorship == "citizens_only":
        out.append("sponsorship")
    return out


def is_eligible(profile: Profile, posting: Posting, as_of: date) -> bool:
    return not violations(profile, posting, as_of)


def eligibility_mask(profile: Profile, postings: list[Posting], as_of: date) -> np.ndarray:
    return np.array([is_eligible(profile, p, as_of) for p in postings], dtype=bool)
