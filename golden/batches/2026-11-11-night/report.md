# Nightly report, 2026-11-10

- Pairs: 100 ({'other': 50, 'finance': 50}; {'dev': 58, 'test': 42})
- Judge: `azure:gpt-4.1-mini`
- Reference grader: `claude:sonnet`
- Estimated spend: $0.10

> Reference grades are AI labels, not human ones. Grade `audit_sheet.csv` by hand and run `bench-compare` to check them.

## Judge grades

| grade | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| pairs | 53 | 11 | 16 | 16 | 4 |

## Judge vs reference

| Slice | Agreement |
|---|---|
| All | n=100, exact 0.78, κ 0.66, quadratic κ 0.84 |
| Test split (never inspected) | n=42, exact 0.76, κ 0.64, quadratic κ 0.90 |
| Dev split | n=58, exact 0.79, κ 0.68, quadratic κ 0.80 |
| Finance students (IB, markets) | n=50, exact 0.74, κ 0.62, quadratic κ 0.85 |
| Other students | n=50, exact 0.82, κ 0.71, quadratic κ 0.83 |
| Track: consulting | n=11, exact 0.91, κ 0.83, quadratic κ 0.95 |
| Track: data | n=9, exact 0.89, κ 0.81, quadratic κ 0.97 |
| Track: ib | n=18, exact 0.89, κ 0.83, quadratic κ 0.97 |
| Track: markets | n=32, exact 0.66, κ 0.50, quadratic κ 0.77 |
| Track: product | n=11, exact 0.55, κ 0.34, quadratic κ 0.67 |
| Track: quant | n=10, exact 0.80, κ 0.66, quadratic κ 0.69 |
| Track: swe | n=9, exact 1.00, κ 1.00, quadratic κ 1.00 |

## Where they disagree (dev split only)

### Judge too harsh on fit: 5

- p098 (markets, junior) × IMC: Deep Learning Research Intern - Summer 2027 - New York (off-track). Reference 2: Eligible, but a deep learning research role is adjacent to a finance markets student, with no deep learning fr Judge 1: [capped: swe is unrelated to markets] Eligible internship, but deep learning research role is off-track for a 
- p115 (markets, freshman) × Jump Trading: Campus Systems Engineer (Intern) (off-track). Reference 2: Eligible, but a systems/networking ops role in Chicago is adjacent to a markets trading track and needs Linux/ Judge 1: [capped: swe is unrelated to markets] Eligible internship but technical systems engineering role only partiall
- p042 (markets, freshman) × Jump Trading: Campus Systems Engineer (Intern) (off-track). Reference 2: Eligible, but a systems/network ops role is adjacent to a markets track and needs Linux/Python/networking skil Judge 1: [capped: swe is unrelated to markets] Eligible internship but track is systems engineering, not markets; parti
- p012 (markets, sophomore) × RBC: 2028 Capital Markets, Global Investment Banking Summer Analy (off-track). Reference 3: Eligible IB M&A summer analyst in San Francisco; adjacent to markets track with limited skill overlap beyond f Judge 2: [capped: ib is adjacent to markets] Eligible internship in markets-related capital markets role with very stro
- p014 (markets, junior) × IMC: Machine Learning Engineer Intern - Summer 2027 (off-track). Reference 2: Eligible and at a prop trading firm in New York, but it's a machine learning engineering role needing CS/engin Judge 1: [capped: data is unrelated to markets] Eligible and excellent fit for markets intern seeking strong Python and

### Judge disqualified a pair the reference allows: 3

- p152 (quant, junior) × Akuna Capital: Quantitative Research Intern, Summer 2027 (on-track). Reference 4: Quant research internship in Chicago matching the student's stats, ML and math background and systematic tradi Judge 0: [d] Eligible internship with good track and skills match, but disqualified due to required US work authorizati
- p123 (markets, junior) × BMO: Villanova Summer 2028 Internship - Investment Banking (off-track). Reference 2: Eligible, but investment banking is adjacent to the student's markets/sales and trading track, and New York is Judge 0: [code] program for students of one school (Villanova)
- p123 (markets, junior) × BMO: San Francisco - Technology Summer 2028 Internship - Investme (off-track). Reference 2: Eligible and in a preferred location, but investment banking technology group is adjacent to the student's sal Judge 0: [d] Disqualified due to work authorization: the posting does not sponsor visas and the student needs sponsorsh

### Judge too generous on fit: 2

- p070 (product, senior) × Stripe: Product Manager: New Grad Accelerator (on-track). Reference 3: Product Manager new-grad program in New York matches track and level, but the posting asks for a CS or enginee Judge 4: Eligible new-grad product management role with strong skill and level alignment for the student.
- p192 (product, sophomore) × PDT Partners: Summer 2027 Software Engineering Intern (off-track). Reference 1: Eligible, but a quant software engineering internship is off-track for a product-track student seeking APM rol Judge 2: Eligible internship aligned with software engineering, partial skill overlap but track mismatch for product st

### Judge missed a disqualifier (b): 2

- p182 (markets, junior) × BMO: UT @ Austin Summer 2028 Internship - Investment Banking (off-track). Reference 0: [b] Investment banking is adjacent to markets track, and the stated graduation window of Dec 2028 or Spring 20 Judge 2: Eligible internship but track is investment banking, not markets; some skill mismatch.
- p126 (markets, freshman) × RBC: 2028 Capital Markets, Global Investment Banking Summer Analy (off-track). Reference 0: [b] Summer 2028 program is for rising seniors (class of 2029), but a 2030 graduate would be a rising sophomore Judge 2: [capped: ib is adjacent to markets] Eligible internship in markets track with strong skill and location alignm

## Proposed next steps

These are suggestions for a person to review. Nothing was changed.

- **Judge too harsh on fit** (5 dev pairs): read the examples above and look for one general rule; add it with a unit test, then re-score dev before test.
- **Judge disqualified a pair the reference allows** (3 dev pairs): read the examples above and look for one general rule; add it with a unit test, then re-score dev before test.
- **Judge too generous on fit** (2 dev pairs): read the examples above and look for one general rule; add it with a unit test, then re-score dev before test.
