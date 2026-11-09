from sjmb.data.extract import (
    classify_role,
    classify_track,
    extract_grad_years,
    extract_sponsorship,
    strip_html,
)
from sjmb.data.loaders import (
    MAX_DESCRIPTION_CHARS,
    load_fixture,
    parse_greenhouse,
    parse_lever,
    parse_simplify,
)


def test_strip_html_handles_greenhouse_double_escape():
    assert strip_html("&amp;lt;p&amp;gt;Hello&amp;lt;/p&amp;gt;") == "Hello"
    assert strip_html("&lt;p&gt;Hi &lt;b&gt;there&lt;/b&gt;&lt;/p&gt;") == "Hi there"


def test_track_prefers_role_words_over_desk_words():
    assert classify_track("Software Engineer, Trading Systems") == "swe"
    assert classify_track("Quantitative Researcher Intern") == "quant"
    assert classify_track("Investment Banking Summer Analyst") == "ib"
    assert classify_track("Application Developer - Capital Markets Technology") == "swe"
    assert classify_track("Sales & Trading Summer Analyst") == "markets"
    assert classify_track("Strategy Consultant") == "consulting"
    assert classify_track("UX Researcher") != "quant"


def test_role_type():
    assert classify_role("Software Engineer Intern (Summer 2027)") == "internship"
    assert classify_role("Product Manager, New Grad (2027)") == "new_grad"
    assert classify_role("Senior Data Scientist") == "experienced"
    assert classify_role("Data Scientist") == "unknown"
    # Review regressions.
    assert classify_role("Software Engineer", "work in close cooperation with traders") == "unknown"
    assert classify_role("Engineering Co-op") == "internship"
    assert classify_role("Investment Banking Analyst 2", "mentor summer interns") == "experienced"
    assert classify_role("Software Engineer", "Undergraduate degree, 5+ years of experience") == (
        "experienced"
    )
    assert classify_role("Associate Product Manager") == "new_grad"
    assert classify_role("Associate Consultant") == "new_grad"
    assert classify_role("Analyst", "Join our 10-week summer internship program") == "internship"


def test_grad_year_ranges_with_seasons():
    text = "You will graduate between Fall 2026 and Summer 2027 with a bachelors"
    assert extract_grad_years(text) == [2026, 2027]
    assert extract_grad_years("Class of 2028 students") == [2028]
    assert extract_grad_years("no years here") == []
    # Review regressions: lists, all month names, unrelated years.
    assert extract_grad_years("Open to the Class of 2026, 2027, or 2028") == [2026, 2027, 2028]
    assert extract_grad_years("graduating between December 2026 and August 2027") == [2026, 2027]
    assert extract_grad_years("Bachelor's degree required. Founded in 2021, we are") == []
    assert extract_grad_years("Class of 2027-2029") == [2027, 2028, 2029]


def test_sponsorship():
    assert extract_sponsorship("We are unable to sponsor visas for this role") == "no"
    assert extract_sponsorship("We can’t sponsor visas") == "no"
    assert extract_sponsorship("U.S. citizenship is required") == "citizens_only"
    assert extract_sponsorship("An active security clearance is required") == "citizens_only"
    assert extract_sponsorship("We offer great benefits") == "unknown"
    # Review regressions.
    assert extract_sponsorship("You must be a U.S. citizen or permanent resident.") == "us_persons"
    assert extract_sponsorship("A security clearance is a plus but not required") == "unknown"
    assert extract_sponsorship("Whether or not you need sponsorship, apply") == "unknown"


def test_parsers_normalize_each_source():
    gh = parse_greenhouse(
        "acme",
        {
            "jobs": [
                {
                    "id": 1,
                    "title": "Quant Intern",
                    "location": {"name": "Chicago"},
                    "absolute_url": "u",
                    "content": "&lt;p&gt;Hi&lt;/p&gt;",
                    "company_name": "Acme ",
                    "first_published": "2026-11-01T10:00:00-05:00",
                    "application_deadline": "2026-12-01",
                }
            ]
        },
    )[0]
    assert (gh.id, gh.company, gh.description, gh.deadline) == (
        "gh-acme-1",
        "Acme",
        "Hi",
        "2026-12-01",
    )
    lv = parse_lever(
        "beta",
        [
            {
                "id": "z",
                "text": "Analyst",
                "categories": {"location": "NY"},
                "hostedUrl": "h",
                "descriptionPlain": "Desc",
                "createdAt": 1790110014068,
                "lists": [{"text": "Do", "content": "<li>x</li>"}],
            }
        ],
    )[0]
    assert lv.id == "lv-beta-z" and lv.posted_at.startswith("2026") and "Do: x" in lv.description
    sp = parse_simplify(
        "simplify-intern-2027",
        [
            {
                "id": "s1",
                "company_name": "C",
                "title": "SWE Intern",
                "locations": ["NY"],
                "url": "u",
                "date_posted": 1794235407,
                "active": False,
                "terms": ["Summer 2027"],
                "sponsorship": "Does Not Offer Sponsorship",
                "category": "Software",
            }
        ],
    )[0]
    assert (sp.role_type, sp.sponsorship, sp.active, sp.terms) == (
        "internship",
        "no",
        False,
        ["Summer 2027"],
    )


def test_fixture_is_small_and_mixed():
    posts = load_fixture()
    assert 100 <= len(posts) <= 400
    assert {p.source for p in posts} == {"greenhouse", "lever", "simplify"}
    assert len({p.id for p in posts}) == len(posts)
    assert all(len(p.description) <= MAX_DESCRIPTION_CHARS for p in posts)
