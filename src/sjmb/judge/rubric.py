"""The written relevance policy both judges follow. Bump RUBRIC_VERSION whenever the text changes;
it is part of the judgment cache key, so old grades are never reused under a new policy."""

RUBRIC_VERSION = "v5"
# v2: adds today's date and extracted eligibility fields.
# v3: two-step grading with explicit disqualifiers, taken from the reasons in the human golden set
#     (golden/grading_sheet.csv); full posting text instead of a truncated excerpt.
# v4: the judge reports a disqualifier and a fit score separately and code derives the grade, so
#     a track mismatch can no longer become 0; a non-US location is a fit issue, not a
#     disqualifier, matching how the golden set grades it.
# v5: (found by hand in the Foundry playground) rule a names the career stage only, one made-up
#     worked example, and the reply lists a check per rule before it picks the disqualifier.

RUBRIC = """\
You grade how well one job posting fits one student. Work in two steps.

STEP 1 - ELIGIBILITY. The grade is 0 if ANY of these holds. Check each one against the full
posting text, not only the extracted fields:
  a. Wrong career stage (ONLY internship vs full-time). This is never about the field or
     track. A student seeking an internship cannot take a full-time, new-grad, entry-level
     or apprenticeship role. A student seeking new-grad (a full-time role after graduation)
     is not looking for an internship or co-op, including internships that convert.
  b. Graduation window. The posting limits who may apply by graduation date or class year
     ("graduating Dec 2027 - Jun 2028", "within one year of the internship", "class of 2027")
     and the student's graduation year is outside it.
  c. Degree. The role requires a PhD, MD, JD or master's (or current enrollment in one). Every
     student here is an undergraduate pursuing a bachelor's degree.
  d. Work authorization. The posting explicitly requires citizenship, a security clearance or a
     permit the student lacks, or explicitly says it does not sponsor and the student needs
     sponsorship. Silence about sponsorship is NOT a disqualifier, and neither is a location
     outside the United States: treat those as a weaker location fit in step 2.
  e. Experience. The role requires years of professional experience (for example "3+ years").
  f. Restricted program. The program is only for a group the student is not in: veterans or
     military spouses, students of one named school, residents of one country.
  g. Closed. The posting is inactive or past its deadline.
Only these seven reasons disqualify. Track, skills, major and location are never disqualifiers.
If none holds, the student is eligible. A field the posting does not state never disqualifies.

STEP 2 - FIT. Always score fit 1-4, even when step 1 found a disqualifier:
4 Excellent: target track and level, and most of the posting's core skills match the student's.
3 Good: target track and level, but partial skill overlap or a weaker location fit.
2 Fair: an adjacent track (e.g. data science for a quant student) or thin skill overlap.
1 Poor: a clear track or skill mismatch. The student would not apply.

The final grade is 0 when step 1 names a disqualifier, otherwise the fit score.

Rules:
- "Extracted" fields come from rules and can be wrong; the posting text wins when they disagree.
- Use the date given as today for deadlines and graduation years.
- Company prestige does not change the grade.
"""

PROMPT = """\
{rubric}
TODAY: {as_of}

STUDENT
track: {track}
seeking: {seeking}
degree: bachelor's (undergraduate)
graduation year: {grad_year} ({class_label})
major: {major}
skills: {skills}
interests: {interests}
work authorization: {work_auth}
preferred locations: {locations}

POSTING
company: {company}
title: {title}
location: {location}
terms: {terms}
posted: {posted_at}  deadline: {deadline}  active: {active}
extracted level: {role_type}
extracted eligible graduation years: {grad_years}
extracted work authorization: {sponsorship}
description: {description}

Example (made up, not one of the graded pairs): a data-track student seeking an internship;
the posting is a Graphic Design Intern. Same career stage, different field, so it is eligible:
{{"checks": "a: both internship, ok; b: no window; c: bachelor's ok; d: nothing stated; \
e: none; f: none; g: open", "disqualifier": "none", "fit": 1, \
"reason": "Eligible, but design is off-track for a data student."}}

Reply with JSON only, in this order. Fill "checks" first, one short phrase per rule, and
only then choose the disqualifier from your checks:
{{"checks": "a: ... b: ... c: ... d: ... e: ... f: ... g: ...", \
"disqualifier": "<one letter a-g whose check failed, or none>", "fit": <1-4>, \
"reason": "<one short sentence>"}}
"""

POOR_MATCH_MAX = 1  # grades at or below this count as a poor match
RELEVANT_MIN = 3  # grades at or above this count as relevant for recall
