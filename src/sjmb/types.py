"""Shared record types. Postings are public; profiles are synthetic."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Literal

# The first seven are the student tracks. hardware, sales and design are posting-only tracks:
# known fields that no profile targets. "other" means the extractor could not tell.
Track = Literal[
    "ib",
    "markets",
    "quant",
    "consulting",
    "swe",
    "data",
    "product",
    "hardware",
    "sales",
    "design",
    "other",
]
RoleType = Literal["internship", "new_grad", "experienced", "unknown"]
# us_persons = U.S. citizens or permanent residents only (no visa sponsorship).
Sponsorship = Literal["offers", "no", "us_persons", "citizens_only", "unknown"]
WorkAuth = Literal["citizen", "permanent_resident", "needs_sponsorship"]

TRACKS: tuple[Track, ...] = ("ib", "markets", "quant", "consulting", "swe", "data", "product")


@dataclass
class Posting:
    id: str
    source: str  # greenhouse | lever | simplify
    company: str
    title: str
    location: str
    url: str
    description: str
    posted_at: str | None = None  # ISO date
    deadline: str | None = None  # ISO date
    active: bool = True
    # Fields filled by extraction at ingest (sjmb.data.extract).
    track: Track = "other"
    role_type: RoleType = "unknown"
    grad_years: list[int] = field(default_factory=list)  # empty = not stated
    sponsorship: Sponsorship = "unknown"
    terms: list[str] = field(default_factory=list)

    def text(self) -> str:
        return f"{self.title}. {self.company}. {self.location}. {self.description}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Posting:
        return cls(**d)

    def deadline_date(self) -> date | None:
        return date.fromisoformat(self.deadline) if self.deadline else None


@dataclass
class Profile:
    id: str
    track: Track
    grad_year: int
    class_label: str  # freshman | sophomore | junior | senior
    seeking: RoleType  # internship | new_grad
    major: str
    skills: list[str]
    interests: str
    work_auth: WorkAuth
    locations: list[str]

    def query(self) -> str:
        """The text a retriever sees. Built only from synthetic fields."""
        role = "internship" if self.seeking == "internship" else "new grad full-time analyst role"
        return (
            f"{self.interests} {role}. {self.major} student graduating {self.grad_year}. "
            f"Skills: {', '.join(self.skills)}. Locations: {', '.join(self.locations)}."
        )

    def to_dict(self) -> dict:
        return asdict(self)
