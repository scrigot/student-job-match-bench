# Golden set (reference labels)

> **Who graded what (corrected 2026-11-10).** Sam graded pairs 1–10 of `grading_sheet.csv` by
> hand. Pairs 11–100, and all 50 pairs in `heldout_sheet.csv`, were graded by a Claude browser
> agent that Sam ran, which opened each listing and applied the rubric. So wherever this file
> says "human" or "you", read "the reference grader": for 140 of 150 pairs that is an AI agent.
> The κ values below measure agreement between the gpt-4.1-mini judge and that reference, not
> agreement with a person. A human-graded sample is still needed to validate the reference
> grader itself.

`grading_sheet.csv` holds 100 (synthetic student, real posting) pairs to grade by hand. The
grades measure how far the LLM judge can be trusted (Cohen's κ), and they settle whether the
eligibility filter or the judge is right where the two disagree.

## How to grade

1. Open `grading_sheet.csv` in a spreadsheet. The `#` lines at the top repeat the rubric; leave
   them in place.
2. For each row, read `student` and `job`, and open the posting URL at the end of `job` if the
   excerpt is not enough. Write one grade, 0–4, in `your_grade`. Use `notes` for anything odd,
   such as wrong extracted fields or a closed posting.
3. Grade from the rubric alone. The sheet hides the judge's grade and the filter's verdict on
   purpose, so look at neither.
4. Save as CSV with the same columns. Partial sheets work: blank rows are skipped.
5. Score it:

   ```bash
   uv run bench-kappa golden/grading_sheet.csv
   ```

   It prints exact agreement, unweighted and quadratic-weighted κ, a 5×5 confusion matrix, and a
   breakdown of the filter–judge disagreement pairs. It uses cached judgments only and makes no
   Azure calls.

## Rubric

| Grade | Meaning |
|---|---|
| 4 Excellent | Eligible, right track and level (internship vs new-grad), most of the core skills match. |
| 3 Good | Eligible, right track and level, but only partial skill overlap or a weaker location fit. |
| 2 Fair | Eligible, but an adjacent track (e.g. data science for a quant student) or thin overlap. |
| 1 Poor | Eligible on paper, but a clear track or skill mismatch: the student would not apply. |
| 0 Unusable | The student cannot or should not apply: senior or experienced-only, the wrong level, excludes their graduation year, needs work authorization they lack, or closed or past its deadline. |

Judge only from what the posting states. If eligibility is not stated, assume the student is
eligible. Company prestige does not change the grade. Today is 2026-11-09.

The students are synthetic and have no school; class year, major, skills and target roles are
what matter. "Auto-extracted" fields in `job` come from rules and can be wrong. If the posting
says otherwise, grade from the posting and note it.

## How the 100 pairs were chosen

`uv run bench-golden` (seed 7) samples from `results/judgments.jsonl`, the Azure run with 20
profiles and pool depth 10 (403 judged pairs):

- **33 filter–judge disagreements**, both directions:
  - 16 where the eligibility filter excludes the posting but the judge grades it 2 or higher.
  - 17 where the filter allows it but the judge grades it 0.
- **About 20 per judge grade band** from pairs where filter and judge agree (0–1, 2, 3–4), then
  filled to 100.
- Every stage picks round-robin over track × class-year cells, so all 17 cells in the run appear.
- Rows are shuffled so the strata do not show up as blocks.

Final judge-grade mix: 42 in 0–1, 35 at 2, 23 in 3–4.

**What the disagreement breakdown answers.** The Azure run showed hybrid RRF + eligibility
slightly *below* hybrid RRF (NDCG@10 0.554 vs 0.568), with recall dropping to 0.73. Either the
judge is lenient on ineligible roles, or the filter drops jobs that really fit:

- **"Filter excludes, judge ≥ 2."** If you mostly grade these ≤ 1, the judge is too lenient and
  the rubric or prompt needs tightening. If you grade them ≥ 2, the filter or extraction is too
  strict.
- **"Filter allows, judge 0."** If you grade these 0, the judge is catching ineligibility the
  rules miss, which points at extraction. If you grade them ≥ 1, the judge is too harsh.

The early cases are mostly seniors (seeking new-grad roles) matched to internships. The filter
excludes these by policy; the judge often accepts them.

## Judge versions against this golden set

All 100 pairs graded (2026-11-09; see "Who graded what" above). Judge deployment: `gpt-4.1-mini`.

| Rubric | Change | Exact | κ unweighted | κ quadratic | Your 0s the judge also gives 0 | Your 1–3s the judge gives 0 |
|---|---|---:|---:|---:|---:|---:|
| v2 | Baseline: 2,500-char excerpt, rubric as above | 0.36 | 0.18 | 0.42 | 15 / 54 | 8 / 41 |
| v3 | Explicit disqualifiers a–g taken from the golden notes, full posting text | 0.53 | 0.22 | 0.46 | 44 / 54 | 24 / 41 |
| v4 | Code derives the grade from disqualifier + fit; a non-US location is fit, not eligibility | 0.55 | 0.26 | 0.48 | 43 / 52 | 23 / 40 |
| v5 | Found by hand in the Foundry playground: rule (a) is career stage only, one made-up example, the reply lists a check per rule before it picks a disqualifier | 0.60 | 0.45 | 0.55 | 39 / 54 | 2 / 41 |
| v5 + code check | Graduation windows checked in code before any LLM call (`hard_check`); the extractor reads "2027 or later" and "by 2028" as open windows | 0.61 | 0.46 | 0.58 | 40 / 54 | 1 / 41 |
| + location cap | A 4 needs one of the student's preferred locations, checked in code | 0.68 | 0.55 | 0.64 | 40 / 54 | 1 / 41 |
| + employer pages | Simplify rows get the employer page's text (93 of 110) | 0.72 | 0.59 | 0.72 | 46 / 54 | 3 / 41 |
| + track cap | Off-track roles cap at 2, as the rubric defines | 0.76 | 0.64 | 0.78 | 46 / 54 | 3 / 41 |
| + 8,000-char text | Eligibility lines at the end of postings no longer cut off; 2 more grad-window phrasings | 0.78 | 0.66 | 0.79 | 51 / 54 | 5 / 41 |
| + trading family | Quant trading counts as on-track for markets students | 0.79 | 0.67 | 0.82 | 51 / 54 | 5 / 41 |
| **+ US work-auth fix** | A [d] citing only US work authorization never excludes a citizen or permanent resident | **0.80** | **0.69** | **0.85** | **51 / 54** | **4 / 41** |

v4 scored 97 pairs: 3 replies could not be parsed and are cached as unparsable. v5 scored all 100.
Its first run lost 27 replies to a parser bug: the parser rejected `"fit": 0` on disqualified pairs.
The bug is fixed, and failed replies now keep their raw text so they can be re-parsed for free.

**Where v5 still disagrees with the human grades.**
- **Misses 15 human zeros, and they are mechanical checks:**
  - 8 graduation-year windows, e.g. "Dec 2027 - Jun 2028" for a class-of-2029 student, or
    "within one year of the internship".
  - 3 sponsorship limits phrased indirectly, e.g. "must be authorized to work for any employer".
  - 2 veteran-only programs.
  - 1 citizenship plus clearance requirement.
  - 1 experienced-only role.
- **Grades fit one step high:** 9 of the 10 human 3s get a 4.

**Why the code check.** The judge saw "must graduate by August 2028" next to "graduation year:
2029" and still got pair 15 right only 2 times in 6 at default temperature, and 0 in 6 at
temperature 0. Comparing years is arithmetic, so code does it. The LLM reads text and judges fit.
The code check fires on 5 golden pairs and agrees with the human grade on all 5.

**Where the gains came from.** Each step from v5 on moved a fact out of the LLM and into
code or data: dates, locations, tracks, and the employer's full posting text. The LLM now does
what only it can: read the posting and judge eligibility and fit. Every code rule is a general
policy (the rubric's own definitions, or a legal fact about work authorization) and has a unit
test. None is keyed to individual pairs.

**Caveat.** The v3 and v4 rules were written from these same 100 pairs, so their κ is optimistic. Report a final κ on a fresh, held-out set of pairs.

## Held-out check

`heldout_sheet.csv` has 50 more pairs from the same Azure run, sampled with seed 11, with no
overlap with `grading_sheet.csv`. The judge was tuned on the first 100 pairs, so the 0.80 exact /
0.85 quadratic κ above is optimistic. The held-out κ is the number to report.

```bash
uv run bench-kappa golden/heldout_sheet.csv
```

The judge's answers for these 50 are already cached, so scoring costs nothing. Grade the sheet
the same way: one whole number 0–4 per row, `notes` optional, saved as CSV.

### Held-out result (graded 2026-11-09 by the Claude browser agent, not by hand)

| Judge | Pairs | Exact | κ unweighted | κ quadratic |
|---|---:|---:|---:|---:|
| Original v2 judge (no tuning) | 50 | 0.44 | 0.27 | 0.45 |
| **Final judge (v5 + code checks + page text)** | 50 | **0.58** | **0.41** | **0.65** |
| Final judge, on the 100 pairs it was tuned on | 100 | 0.80 | 0.69 | 0.85 |

The held-out gain is real but smaller than on the tuning set. The gap (0.80 vs 0.58 exact) is
how much of the tuning-set score came from fitting those 100 pairs. With 50 pairs, the held-out
κ is uncertain by roughly ±0.1.

Remaining held-out errors:
- 6 human 1s graded 2 (adjacent vs mismatch).
- 5 human 2s graded 3–4 (fit too generous).
- 6 human 0s graded 1–2 (missed disqualifiers).

Where the filter allows a posting but the judge gives 0, the human sided with the judge 6 of 7
times. The judge catches eligibility rules the extractor misses.
