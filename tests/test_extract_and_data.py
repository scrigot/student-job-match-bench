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
    # Open-ended windows (golden pairs 15, 23, 88): a bound, not a single year.
    assert extract_grad_years("graduating in Fall 2027 or later") == [2027, 2028, 2029, 2030, 2031]
    assert extract_grad_years("Must graduate by August 2028") == [2024, 2025, 2026, 2027, 2028]
    assert extract_grad_years("Bachelor's degree expected between Fall 2027 and Spring 2028") == [
        2027,
        2028,
    ]
    assert extract_grad_years("Targeting a 2028 full-time start date") == [2027, 2028]


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
    assert 100 <= len(posts) <= 500
    assert {p.source for p in posts} == {"greenhouse", "lever", "simplify", "workday"}
    assert len({p.id for p in posts}) == len(posts)
    assert all(len(p.description) <= MAX_DESCRIPTION_CHARS for p in posts)


def test_parse_workday():
    from sjmb.data.loaders import parse_workday

    p = parse_workday(
        {
            "company": "Acme Bank",
            "tenant": "acme",
            "site": "Campus",
            "jobs": [
                {
                    "id": "R1",
                    "title": "2027 Investment Banking Summer Analyst",
                    "location": "New York",
                    "posted": "2026-10-01",
                    "url": "https://acme.wd1.myworkdayjobs.com/Campus/job/NY/x_R1",
                    "description": "<p>Join our M&A team.</p>",
                }
            ],
        }
    )[0]
    assert p.id.startswith("wd-acme-R1-") and len(p.id) == len("wd-acme-R1-") + 6
    assert (p.source, p.company, p.description) == (
        "workday",
        "Acme Bank",
        "Join our M&A team.",
    )


def test_restricted_programs():
    from sjmb.data.extract import restriction

    assert "one school" in restriction("Villanova Summer 2028 Internship - Investment Banking")
    assert "MBA" in restriction("2027 Investment Banking Summer Associate", "For MBA students.")
    assert "advanced degree" in restriction("Quant Research Intern - PhD/Postdoctoral")
    assert "advanced degree" in restriction(
        "Hardware Intern", "Currently pursuing a Masters Degree in Electrical Engineering."
    )
    assert "veterans" in restriction("Data Intern", "Veteran Internship Program for veterans.")
    # Not restrictions: undergraduates named, a preference, or no statement at all.
    assert restriction("SWE Intern", "Currently pursuing a Bachelor's or Master's degree.") is None
    assert (
        restriction(
            "Algorithm Intern",
            "Preferred Qualifications: Students pursuing a Master's OR PhD degree.",
        )
        is None
    )
    assert restriction("Investment Banking Summer Analyst", "Join our M&A team.") is None
    assert restriction("Summer Associate", "Join our consulting team as an undergraduate.") is None


def test_off_track_fields():
    assert classify_track("AI Hardware Physical Design Engineer Intern") == "hardware"
    assert classify_track("Early Career, Inside Sales Representative") == "sales"
    assert classify_track("Sales & Trading Summer Analyst") == "markets"
    assert classify_track("Product Design Intern (2027)") == "design"
    assert classify_track("Machine Learning Engineer") == "data"
