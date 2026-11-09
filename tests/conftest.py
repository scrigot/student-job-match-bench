from datetime import date

import pytest

from sjmb.types import Posting, Profile

AS_OF = date(2026, 11, 9)


def make_posting(**kw) -> Posting:
    base = dict(
        id="x",
        source="greenhouse",
        company="Acme",
        title="Software Engineer Intern",
        location="New York, NY",
        url="",
        description="Python SQL distributed systems",
        track="swe",
        role_type="internship",
    )
    base.update(kw)
    return Posting(**base)


def make_profile(**kw) -> Profile:
    base = dict(
        id="p",
        track="swe",
        grad_year=2028,
        class_label="junior",
        seeking="internship",
        major="Computer Science",
        skills=["Python", "SQL"],
        interests="software",
        work_auth="citizen",
        locations=["New York, NY"],
    )
    base.update(kw)
    return Profile(**base)


@pytest.fixture
def as_of() -> date:
    return AS_OF
