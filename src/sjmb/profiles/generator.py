"""Synthetic student profiles. No real people: every field comes from fixed lists and a seeded RNG.

Profiles are stratified over track x class year x work authorization so every slice in the
results has members.
"""

from __future__ import annotations

import itertools
import random
from datetime import date

from sjmb.types import TRACKS, Profile, Track, WorkAuth

CLASS_LABELS = ("senior", "junior", "sophomore", "freshman")

_MAJORS: dict[Track, list[str]] = {
    "ib": ["Economics", "Finance", "Business Administration", "Accounting"],
    "markets": ["Economics", "Finance", "Mathematics", "Statistics"],
    "quant": ["Mathematics", "Statistics", "Computer Science", "Physics"],
    "consulting": ["Economics", "Business Administration", "Public Policy", "Engineering"],
    "swe": ["Computer Science", "Computer Engineering", "Mathematics"],
    "data": ["Statistics", "Data Science", "Computer Science", "Economics"],
    "product": ["Computer Science", "Business Administration", "Cognitive Science"],
}
_SKILLS: dict[Track, list[str]] = {
    "ib": [
        "financial modeling",
        "DCF",
        "LBO",
        "Excel",
        "PowerPoint",
        "valuation",
        "accounting",
        "pitch books",
        "comparable companies",
    ],
    "markets": [
        "Excel",
        "Python",
        "fixed income",
        "equities",
        "options",
        "macro research",
        "Bloomberg",
        "risk management",
        "market making",
    ],
    "quant": [
        "Python",
        "C++",
        "probability",
        "statistics",
        "stochastic calculus",
        "linear algebra",
        "time series",
        "machine learning",
        "numpy",
    ],
    "consulting": [
        "Excel",
        "PowerPoint",
        "market sizing",
        "case interviews",
        "SQL",
        "stakeholder management",
        "research",
        "Tableau",
    ],
    "swe": [
        "Python",
        "Java",
        "C++",
        "TypeScript",
        "React",
        "SQL",
        "distributed systems",
        "Go",
        "AWS",
        "Kubernetes",
    ],
    "data": [
        "Python",
        "SQL",
        "pandas",
        "machine learning",
        "statistics",
        "A/B testing",
        "Tableau",
        "PyTorch",
        "R",
    ],
    "product": [
        "user research",
        "SQL",
        "roadmapping",
        "Figma",
        "A/B testing",
        "analytics",
        "Python",
        "product specs",
    ],
}
_INTERESTS: dict[Track, list[str]] = {
    "ib": [
        "investment banking analyst in M&A",
        "investment banking coverage group",
        "restructuring and leveraged finance",
        "private equity and corporate finance",
    ],
    "markets": [
        "sales and trading on a markets desk",
        "trading at a proprietary trading firm",
        "investment analyst at an asset manager",
        "macro research and markets",
    ],
    "quant": [
        "quantitative trading",
        "quantitative research at a hedge fund",
        "systematic trading research",
        "quant developer at a trading firm",
    ],
    "consulting": [
        "management consulting",
        "strategy consulting",
        "business analyst at a consulting firm",
        "healthcare strategy consulting",
    ],
    "swe": [
        "software engineering",
        "backend software engineer",
        "full-stack engineering",
        "infrastructure and distributed systems engineering",
    ],
    "data": [
        "data science",
        "machine learning engineering",
        "data analyst",
        "applied AI and machine learning",
    ],
    "product": [
        "product management",
        "associate product manager",
        "product analyst",
        "technical product management",
    ],
}
_CITIES = [
    "New York, NY",
    "Chicago, IL",
    "San Francisco, CA",
    "Boston, MA",
    "Charlotte, NC",
    "Seattle, WA",
    "Austin, TX",
    "Remote",
]
_AUTH: tuple[WorkAuth, ...] = ("citizen", "permanent_resident", "needs_sponsorship")


def grad_year_for(class_label: str, as_of: date) -> int:
    # Academic years roll over in August: in Nov 2026 a senior graduates in 2027.
    senior_year = as_of.year + 1 if as_of.month >= 8 else as_of.year
    return senior_year + CLASS_LABELS.index(class_label)


def generate_profiles(n: int = 200, seed: int = 42, as_of: date | None = None) -> list[Profile]:
    as_of = as_of or date(2026, 11, 9)
    rng = random.Random(seed)
    cells = list(itertools.product(TRACKS, CLASS_LABELS, _AUTH))
    rng.shuffle(cells)
    profiles: list[Profile] = []
    for i in range(n):
        track, label, auth = cells[i % len(cells)]
        skills = rng.sample(_SKILLS[track], k=rng.randint(3, 5))
        # A quarter of students list one off-track skill, like real resumes do.
        if rng.random() < 0.25:
            other = rng.choice([t for t in TRACKS if t != track])
            skills.append(rng.choice(_SKILLS[other]))
        profiles.append(
            Profile(
                id=f"p{i:03d}",
                track=track,
                grad_year=grad_year_for(label, as_of),
                class_label=label,
                seeking="new_grad" if label == "senior" else "internship",
                major=rng.choice(_MAJORS[track]),
                skills=skills,
                interests=rng.choice(_INTERESTS[track]),
                work_auth=auth,
                locations=rng.sample(_CITIES, k=2),
            )
        )
    return profiles
