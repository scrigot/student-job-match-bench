# Nightly report, 2026-11-10

- Pairs: 100 ({'finance': 50, 'other': 50}; {'test': 32, 'dev': 68})
- Judge: `azure:gpt-4.1-mini`
- Reference grader: `claude:sonnet`
- Estimated spend: $0.10

> Reference grades are AI labels, not human ones. Grade `audit_sheet.csv` by hand and run `bench-compare` to check them.

## Judge grades

| grade | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| pairs | 77 | 2 | 12 | 6 | 3 |

## Judge vs reference

| Slice | Agreement |
|---|---|
| All | n=100, exact 0.84, κ 0.60, quadratic κ 0.74 |
| Test split (never inspected) | n=32, exact 0.84, κ 0.66, quadratic κ 0.75 |
| Dev split | n=68, exact 0.84, κ 0.56, quadratic κ 0.72 |
| Finance students (IB, markets) | n=50, exact 0.86, κ 0.58, quadratic κ 0.78 |
| Other students | n=50, exact 0.82, κ 0.61, quadratic κ 0.70 |
| Track: consulting | n=11, exact 0.91, κ 0.82, quadratic κ 0.54 |
| Track: data | n=7, exact 0.57, κ 0.40, quadratic κ 0.73 |
| Track: ib | n=27, exact 0.89, κ 0.36, quadratic κ 0.62 |
| Track: markets | n=23, exact 0.83, κ 0.64, quadratic κ 0.86 |
| Track: product | n=8, exact 0.75, κ 0.56, quadratic κ 0.76 |
| Track: quant | n=13, exact 1.00, κ 1.00, quadratic κ 1.00 |
| Track: swe | n=11, exact 0.73, κ 0.20, quadratic κ 0.48 |

## Where they disagree (dev split only)

### Judge too generous on fit: 5

- p151 (data, junior) × Tesla: AI Hardware Physical Design Engineer Intern - AI Hardware (on-track). Reference 1: Eligible, but a hardware physical design role is a clear mismatch for a data science student with R/Python/sta Judge 2: Eligible, but role is hardware physical design, which is an adjacent track with limited skill overlap for a da
- p024 (product, sophomore) × Tesla: AI Hardware Physical Design Engineer Intern - AI Hardware (off-track). Reference 1: Eligible, but a physical design hardware engineering role is a clear mismatch for a product/PM student with SQ Judge 2: Eligible but engineering intern role is adjacent to product track with limited core skill overlap.
- p073 (swe, sophomore) × Hudson River Trading: Software Engineering Internship (C++ or Python) – Summer 202 (on-track). Reference 3: Eligible SWE internship in preferred NY/Austin; Python matches and math fits, but C/C++ focus and trading doma Judge 4: Eligible internship matching student track and level with strong Python skill overlap and preferred locations.
- p082 (markets, senior) × Vigil Labs: Founding Member of Technical Staff — Markets (on-track). Reference 2: Eligible and markets-themed, but it is a software/quant-dev role needing Python/C++ engineering, while the stu Judge 3: [capped: not a preferred location] Eligible and a strong match on track, level, and skills for a markets new-g
- p039 (markets, junior) × Akuna Capital: Quantitative Development & Strategy Intern (on-track). Reference 2: Eligible, but a quant dev role requiring strong Python and a technical major is an adjacent track for an econo Judge 3: Eligible internship in target track with good skill overlap and location match, but some skill requirements ar

### Judge disqualified a pair the reference allows: 3

- p172 (swe, sophomore) × Cohesity: Software Engineer Intern (on-track). Reference 3: Eligible SWE internship with matching languages (Python, Java, C++, Go, AWS), but Santa Clara is outside the p Judge 0: [code] grad year 2029 not in posting window [2027, 2028]
- p143 (swe, junior) × MongoDB: 2027 - Software Engineer Intern, Dublin - 6 Month Internship (on-track). Reference 3: Eligible SWE internship matching the student's languages (Java, Go) and graduation window, but the location is Judge 0: [d] Requires work authorization in Ireland, student lacks stated Ireland authorization.
- p038 (ib, senior) × Houlihan Lokey: 2027 Graduate Financial Analyst - Corporate Finance (Industr (on-track). Reference 2: Eligible and on-track IB analyst role, but Frankfurt location and required German fluency make it a weak pract Judge 0: [a] Posting is a summer internship role, student seeks a new-grad full-time role.

### Judge missed a disqualifier (b): 1

- p098 (markets, junior) × BMO: Villanova Summer 2028 Internship - Investment Banking (off-track). Reference 0: [b] Investment banking is adjacent to the markets track, with some finance overlap but little skill match, and Judge 2: [capped: ib is not the markets track] Eligible with good fit in career stage and partial skill overlap; track 

### Judge missed a disqualifier (c): 1

- p123 (markets, junior) × Guggenheim Securities: 2027 Guggenheim Securities Investment Banking Summer Associa (off-track). Reference 0: [c] Requires MBA/JD candidacy, and it is an IB role in LA, not a markets desk role in a preferred city. Judge 2: Eligible internship but investment banking Industrials role is an adjacent track to markets; some skills overl

### Judge too harsh on fit: 1

- p020 (data, sophomore) × IMC: Quantitative Research Intern 2027 (off-track). Reference 3: Quant research internship fits a statistics/Python student, but the Hong Kong location is far from the preferr Judge 2: [capped: quant is not the data track] Eligible internship for data track student with strong Python and statis

## Proposed next steps

These are suggestions for a person to review. Nothing was changed.

- **Judge too generous on fit** (5 dev pairs): read the examples above and look for one general rule; add it with a unit test, then re-score dev before test.
- **Judge disqualified a pair the reference allows** (3 dev pairs): read the examples above and look for one general rule; add it with a unit test, then re-score dev before test.
- **Judge missed a disqualifier (b)** (1 dev pairs): read the examples above and look for one general rule; add it with a unit test, then re-score dev before test.
