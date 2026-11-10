# Nightly report, 2026-11-10

- Pairs: 10 ({'finance': 5, 'other': 5}; {'dev': 6, 'test': 4})
- Judge: `azure:gpt-4.1-mini`
- Reference grader: `claude:sonnet`
- Estimated spend: $0.01

> Reference grades are AI labels, not human ones. Grade `audit_sheet.csv` by hand and run `bench-compare` to check them.

## Judge grades

| grade | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| pairs | 6 | 0 | 2 | 1 | 1 |

## Judge vs reference

| Slice | Agreement |
|---|---|
| All | n=10, exact 0.90, κ 0.84, quadratic κ 0.80 |
| Test split (never inspected) | n=4, exact 1.00, κ 1.00, quadratic κ 1.00 |
| Dev split | n=6, exact 0.83 |
| Finance students (IB, markets) | n=5, exact 1.00, κ 1.00, quadratic κ 1.00 |
| Other students | n=5, exact 0.80, κ 0.69, quadratic κ 0.53 |
| Track: ib | n=4, exact 1.00, κ 1.00, quadratic κ 1.00 |
| Track: markets | n=1, exact 1.00 |
| Track: product | n=2, exact 1.00, κ 1.00, quadratic κ 1.00 |
| Track: quant | n=2, exact 1.00, κ 1.00, quadratic κ 1.00 |
| Track: swe | n=1, exact 0.00 |

## Where they disagree (dev split only)

### Judge disqualified a pair the reference allows: 1

- p106 (swe, junior) × palantir: Software Engineer, Internship - Production Infrastructure (on-track). Reference 3: Eligible SWE internship with Java, TypeScript, Kubernetes, React overlap, but Seattle is not a preferred locat Judge 0: [d] Disqualified due to work authorization requiring US citizenship while student is a permanent resident.

## Proposed next steps

These are suggestions for a person to review. Nothing was changed.

- **Judge disqualified a pair the reference allows** (1 dev pairs): read the examples above and look for one general rule; add it with a unit test, then re-score dev before test.
