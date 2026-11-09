"""The written relevance policy both judges follow. Bump RUBRIC_VERSION whenever the text changes;
it is part of the judgment cache key, so old grades are never reused under a new policy."""

RUBRIC_VERSION = "v2"  # v2: adds today's date and extracted eligibility fields

RUBRIC = """\
You grade how well one job posting fits one student. Return a single integer grade 0-4.

4 Excellent: the student is eligible, the role is in the student's target track and level
  (internship vs new-grad), and most of the posting's core skills match the student's.
3 Good: eligible, right track and level, but only partial skill overlap or a weaker location fit.
2 Fair: eligible, but an adjacent track (e.g. data science for a quant student) or thin overlap.
1 Poor: eligible on paper but a clear mismatch in track or skills. The student would not apply.
0 Unusable: the student cannot or should not apply: the role is senior or experienced-only,
  the wrong level (internship vs full-time), the posting excludes the student's graduation year,
  requires work authorization the student lacks, or is closed or past its deadline.

Rules:
- Judge only from the fields given. Do not guess facts the posting does not state.
- "Extracted" fields were read from the full posting, which the description below may truncate.
  Trust them unless the text clearly contradicts them.
- Use the date given as today when checking deadlines and graduation years.
- If eligibility is not stated, assume the student is eligible.
- Company prestige does not change the grade.
"""

PROMPT = """\
{rubric}
TODAY: {as_of}

STUDENT
track: {track}
seeking: {seeking}
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

Reply with JSON only: {{"grade": <0-4>, "reason": "<one short sentence>"}}
"""

POOR_MATCH_MAX = 1  # grades at or below this count as a poor match
RELEVANT_MIN = 3  # grades at or above this count as relevant for recall
