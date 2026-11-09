# Human golden set

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
