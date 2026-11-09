"""Rule-based field extraction at ingest: track, role type, grad years, sponsorship, terms.

This is the v0 baseline. An LLM extractor (cached per posting) is the planned upgrade; the
rules here are deliberately simple so their failure modes are easy to read in the error slices.
"""

from __future__ import annotations

import html
import re

from sjmb.types import Posting, RoleType, Sponsorship, Track

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def strip_html(raw: str) -> str:
    # Greenhouse double-escapes content (&lt;p&gt;), so unescape before stripping tags.
    text = html.unescape(html.unescape(raw or ""))
    return _WS.sub(" ", _TAG.sub(" ", text)).strip()


# Order matters: the first track whose pattern hits the title wins, then the description.
# Role words (quant, data, engineer) are checked before desk words (banking, trading) so
# "Software Engineer, Trading Systems" lands in swe and "Capital Markets Technology" is not IB.
_TRACK_PATTERNS: list[tuple[Track, re.Pattern[str]]] = [
    (
        "quant",
        re.compile(
            r"quant|\balpha\b|systematic|algorithmic trad|"
            r"research(er)?\b[^.,]{0,30}\b(trading|alpha)",
            re.I,
        ),
    ),
    (
        "data",
        re.compile(
            r"data scien|data analy|machine learning|\bML\b|\bAI\b|analytics|"
            r"data engineer",
            re.I,
        ),
    ),
    ("product", re.compile(r"product manag|\bAPM\b|product design|product analyst", re.I)),
    (
        "swe",
        re.compile(
            r"software|engineer|developer|SWE|backend|frontend|full[- ]stack|"
            r"infrastructure|platform|security|devops|\bSRE\b",
            re.I,
        ),
    ),
    (
        "ib",
        re.compile(
            r"investment bank|\bIBD\b|M&A|mergers|(equity|debt) capital markets|"
            r"\bECM\b|\bDCM\b|leveraged finance|restructuring|corporate finance|"
            r"private equity|coverage group",
            re.I,
        ),
    ),
    ("consulting", re.compile(r"consult|strategy|advisory|business analyst", re.I)),
    (
        "markets",
        re.compile(
            r"\btrad(er|ing)\b|sales ?& ?trading|markets|portfolio|"
            r"investment analyst|economist|asset management|hedge",
            re.I,
        ),
    ),
]

_INTERN = re.compile(
    r"\bintern(ship)?s?\b|\bco-?op\b|summer analyst|summer associate|\bsummer 20\d\d\b",
    re.I,
)
# In descriptions, only phrases that describe *this* role count; "mentor our interns" does not.
_INTERN_DESC = re.compile(
    r"\b(this|our|the) (\d+-week )?(summer )?internship\b|\binternship program\b|"
    r"\bsummer analyst program\b|\bco-?op (role|position|program)\b",
    re.I,
)
_NEW_GRAD = re.compile(
    r"new grad|\bgraduates?\b(?!\s+degree)|early career|entry[- ]level|university grad|"
    r"\bcampus (recruit|hire|hiring)|full[- ]time analyst|\banalyst program|rotational|"
    r"class of 20\d\d",
    re.I,
)
_NEW_GRAD_TITLE = re.compile(
    r"associate product manager|associate consultant|\bAPM\b|rotational", re.I
)
_SENIOR = re.compile(
    r"\b(senior|sr\.?|staff|principal|lead|manager|director|head of|vp|vice president|"
    r"managing|partner|associate|ii|iii|iv|[2-4])\b",
    re.I,
)
_YEARS_EXPERIENCE = re.compile(
    r"\b([2-9]|1\d)\+?\s*(-\s*\d+\s*)?years?\b[^.]{0,40}experience", re.I
)

_MONTHS = (
    r"(?:spring|summer|fall|autumn|winter|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
_YEAR = rf"(?:{_MONTHS}\s+)?(20[2-3]\d)"
_GRAD_TRIGGER = re.compile(
    r"class of|graduat(?:e|es|ing|ion)\b|grad(?:uation)? date|expected to graduate", re.I
)
_YEAR_LIST = re.compile(
    rf"\D{{0,40}}?{_YEAR}((?:\s*(?:,|-|–|to|and|or|through)\s*(?:or\s+|and\s+)?{_YEAR})*)",
    re.I,
)
_YEAR_ANY = re.compile(r"20[2-3]\d")
_RANGE_WORD = re.compile(r"\b(between|from)\b|-|–|\bto\b|\bthrough\b", re.I)
MIN_GRAD_YEAR = 2025  # older years in a posting are founding dates, not eligibility

_NO_SPONSOR = re.compile(
    r"(?<!whether or )(?<!whether )(not|unable to|cannot|can't|will not|won't)\s+(\w+\s+){0,3}"
    r"sponsor|"
    r"without (the )?(need for |requiring )?(current or future )?(visa )?sponsorship|"
    r"sponsorship (is )?not (available|offered|provided)",
    re.I,
)
_US_PERSON = re.compile(
    r"U\.?S\.? citizens? or (lawful )?permanent residents?|"
    r"U\.?S\.? citizens?, (lawful )?permanent residents?|green card holders?",
    re.I,
)
_CITIZEN = re.compile(
    r"U\.?S\.? citizen(ship)? (is )?required|must be a U\.?S\.? citizen(?!\s*,?\s*or\b)|"
    r"(requires?|required|must (have|hold|obtain)|active)\s+(\w+\s+){0,3}security clearance|"
    r"security clearance (is )?required",
    re.I,
)
_TERM = re.compile(r"\b(summer|fall|winter|spring)\s+(20\d\d)\b", re.I)


def classify_track(title: str, description: str = "") -> Track:
    for text in (title, description[:600]):
        for track, pat in _TRACK_PATTERNS:
            if pat.search(text):
                return track
    return "other"


def classify_role(title: str, description: str = "") -> RoleType:
    if _INTERN.search(title):
        return "internship"
    if _NEW_GRAD_TITLE.search(title) or _NEW_GRAD.search(title):
        return "new_grad"
    if _SENIOR.search(title):
        return "experienced"
    head = description[:1500]
    if _INTERN_DESC.search(head):
        return "internship"
    if _YEARS_EXPERIENCE.search(description):
        return "experienced"
    if _NEW_GRAD.search(head):
        return "new_grad"
    return "unknown"


def extract_grad_years(text: str) -> list[int]:
    years: set[int] = set()
    for trig in _GRAD_TRIGGER.finditer(text):
        m = _YEAR_LIST.match(text, trig.end())
        if not m:
            continue
        found = [int(y) for y in _YEAR_ANY.findall(m.group(0))]
        found = [y for y in found if y >= MIN_GRAD_YEAR]
        if not found:
            continue
        # "between Fall 2026 and Summer 2027" or "2026-2028" is a range; "2026, 2027, or 2028"
        # is a list. Both come out as the same set when the list is contiguous.
        if len(found) == 2 and _RANGE_WORD.search(text[trig.start() : m.end()]):
            lo, hi = min(found), max(found)
            if hi - lo <= 4:
                years.update(range(lo, hi + 1))
        else:
            years.update(found)
    return sorted(years)


def extract_sponsorship(text: str) -> Sponsorship:
    text = text.replace("’", "'")
    if _US_PERSON.search(text):
        return "us_persons"
    if _CITIZEN.search(text):
        return "citizens_only"
    if _NO_SPONSOR.search(text):
        return "no"
    return "unknown"


def extract_terms(text: str) -> list[str]:
    seen: list[str] = []
    for season, year in _TERM.findall(text):
        term = f"{season.title()} {year}"
        if term not in seen:
            seen.append(term)
    return seen


def enrich(p: Posting) -> Posting:
    """Fill extracted fields that the source did not already provide."""
    full = f"{p.title}\n{p.description}"
    if p.track == "other":
        p.track = classify_track(p.title, p.description)
    if p.role_type == "unknown":
        p.role_type = classify_role(p.title, p.description)
    if not p.grad_years:
        p.grad_years = extract_grad_years(full)
    if p.sponsorship == "unknown":
        p.sponsorship = extract_sponsorship(full)
    if not p.terms:
        p.terms = extract_terms(full)
    return p
