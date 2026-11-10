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
    r"class of|graduat(?:e|es|ing|ion)\b|grad(?:uation)? date|expected to graduate|"
    r"degree expected",
    re.I,
)
_YEAR_LIST = re.compile(
    rf"\D{{0,40}}?{_YEAR}((?:\s*(?:,|-|–|/|to|and|or|through)\s*(?:or\s+|and\s+)?{_YEAR})*)",
    re.I,
)
_CLASS_YEAR = re.compile(
    r"((?:spring|summer)\s+or\s+)?"
    r"((?:fall|autumn|winter|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+)?"
    r"(20[2-3]\d)",
    re.I,
)
# "Targeting a 2028 full-time start": graduating in the fall before or the spring of that year.
_FT_START = re.compile(r"\b(20[2-3]\d)\s+full[- ]time start", re.I)
_RANGE_WORD = re.compile(r"\b(between|from)\b|-|–|\bto\b|\bthrough\b", re.I)
MIN_GRAD_YEAR = 2025  # older years in a posting are founding dates, not eligibility
OPEN_SPAN = 4  # "2027 or later" / "by 2028" cover this many years past / before the bound
_OR_LATER = re.compile(r"^\W{0,3}(or|and)\s+(later|after|beyond)|^\W{0,3}onwards?", re.I)
_BY_CUTOFF = re.compile(r"\b(by|before|no later than|prior to)\s+(\w+\.?\s+)?$", re.I)

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


# Fields no student profile targets. Checked on the title only, and first, so that
# "AI Hardware Physical Design Engineer" is hardware and not data.
_OFF_TRACK_TITLES: list[tuple[Track, re.Pattern[str]]] = [
    (
        "hardware",
        re.compile(
            r"hardware|FPGA|ASIC|physical design|firmware|embedded|silicon|circuit|"
            r"electrical engineer|mechanical engineer|validation engineer|\bRF\b",
            re.I,
        ),
    ),
    (
        "sales",
        re.compile(
            r"\bsales\b(?!\s*(?:&|and)\s*trading)|account executive|business development rep|"
            r"\bSDR\b|\bBDR\b",
            re.I,
        ),
    ),
    ("design", re.compile(r"product design|designer|\bUX\b|\bUI\b", re.I)),
]

_ADVANCED_DEGREE_TITLE = re.compile(
    r"\b(MBA|Ph\.?D|Postdoc\w*|Post-?Doc\w*|Doctoral|Master'?s|M\.?D\.?|J\.?D\.?)\b", re.I
)
_ADVANCED_DEGREE_TEXT = re.compile(
    r"(?:currently |must be )?(?:pursuing|enrolled in|candidate for|working toward)s? (?:an? )?"
    r"(?:\d-year )?(MBA|Ph\.?D\.?|Master'?s|J\.?D\.?|doctoral|graduate degree)"
    r"|(MBA|J\.?D\.?) (?:candidate|student)s?\b",
    re.I,
)
_UNDERGRAD_OK = re.compile(r"bachelor|undergraduate|\bB\.?[SA]\.?\b|\bBS/MS\b", re.I)
_PREFERRED = re.compile(r"prefer|a plus|nice to have|ideal|bonus|desired", re.I)
_VETERAN = re.compile(r"veterans?\b|military spouse", re.I)
# Programs for one school's students. A short list of names seen in campus postings; a posting
# that names a school in its title is treated as restricted to that school.
_SCHOOL_TITLE = re.compile(
    r"\b(Villanova|Howard|Spelman|Morehouse|Notre Dame|Georgetown|Wharton|Harvard|Stanford|"
    r"Princeton|Yale|Cornell|Columbia|Dartmouth|Duke|Vanderbilt|Baruch|Rutgers|Fordham|"
    r"Penn State|Georgia Tech|Carnegie Mellon|MIT|NYU|UCLA|USC|UVA|UNC|BYU)\b"
    r"|\b(University|College) of [A-Z][a-z]+",
)


def restriction(title: str, description: str = "") -> str | None:
    """A reason this posting is closed to an ordinary undergraduate, or None.

    Covers what the title or an explicit sentence states: an advanced-degree requirement, a
    veterans-only program, or a program for one named school. Deliberately conservative: a
    sentence that also names a bachelor's degree is not treated as advanced-degree-only.
    """
    if m := _ADVANCED_DEGREE_TITLE.search(title):
        return f"advanced degree in title ({m.group(0)})"
    if re.search(r"summer associate", title, re.I) and re.search(
        r"\bMBA\b|\bJ\.?D\.?\b|business school|law school", description, re.I
    ):
        return "summer associate program for MBA or JD candidates"
    for m in _ADVANCED_DEGREE_TEXT.finditer(description):
        start = description.rfind(".", 0, m.start()) + 1
        end = description.find(".", m.end())
        sentence = description[start : end if end != -1 else m.end() + 200]
        lead = description[max(0, m.start() - 80) : m.start()]
        if _PREFERRED.search(lead) or _PREFERRED.search(sentence[: m.start() - start]):
            continue  # "preferred: pursuing a Master's" is not a requirement
        if not _UNDERGRAD_OK.search(sentence):
            return f"requires an advanced degree ({m.group(0).strip()[:40]})"
    if _VETERAN.search(title) or (
        _VETERAN.search(description[:800]) and re.search(r"program", description[:800], re.I)
    ):
        return "program for veterans or military spouses"
    if m := _SCHOOL_TITLE.search(title):
        return f"program for students of one school ({m.group(0)})"
    return None


def classify_track(title: str, description: str = "") -> Track:
    for track, pat in _OFF_TRACK_TITLES:
        if pat.search(title):
            return track
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


_RISING = re.compile(r"\brising\s+(?:college\s+|undergraduate\s+)?(senior|junior|sophomore)", re.I)
_RISING_OFFSET = {"senior": 1, "junior": 2, "sophomore": 3}
_SUMMER_YEAR = re.compile(r"\bSummer\s+(20\d\d)\b|\b(20\d\d)\b[^.\n]{0,80}?\bSummer\b", re.I)


def extract_grad_years(text: str) -> list[int]:
    years: set[int] = set()
    for trig in _GRAD_TRIGGER.finditer(text):
        m = _YEAR_LIST.match(text, trig.end())
        if not m:
            continue
        # Years are class years: "December 2028" belongs with the class of spring 2029.
        # "Summer or Fall 2027" keeps both classes.
        found = []
        for both, late, y in _CLASS_YEAR.findall(m.group(0)):
            found += [int(y), int(y) + 1] if both else [int(y) + (1 if late else 0)]
        found = [y for y in found if y >= MIN_GRAD_YEAR]
        if not found:
            continue
        # "between Fall 2026 and Summer 2027" or "2026-2028" is a range; "2026, 2027, or 2028"
        # is a list. Both come out as the same set when the list is contiguous.
        before_year = text[trig.end() : m.start(1)] if m.lastindex else ""
        if len(found) == 1 and _OR_LATER.search(text[m.end() : m.end() + 20]):
            # "graduating in Fall 2027 or later": an open window upward, not exactly 2027.
            years.update(range(found[0], found[0] + OPEN_SPAN + 1))
        elif len(found) == 1 and _BY_CUTOFF.search(before_year):
            # "must graduate by August 2028": anyone graduating that year or earlier.
            years.update(range(found[0] - OPEN_SPAN, found[0] + 1))
        elif len(found) == 2 and _RANGE_WORD.search(text[trig.start() : m.end()]):
            lo, hi = min(found), max(found)
            if hi - lo <= 4:
                years.update(range(lo, hi + 1))
        else:
            years.update(found)
    for m in _FT_START.finditer(text):
        y = int(m.group(1))
        years.update((y - 1, y))
    # "Summer 2028 ... rising seniors": a rising senior that summer graduates in 2029.
    # Skipped when recent graduates are also welcome: the window is then open below.
    if (
        (rising := _RISING.findall(text))
        and (s := _SUMMER_YEAR.search(text))
        and not re.search(r"recent grad", text, re.I)
    ):
        summer = int(s.group(1) or s.group(2))
        years.update(summer + _RISING_OFFSET[r.lower()] for r in rising)
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
