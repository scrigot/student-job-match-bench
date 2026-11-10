# Nightly loop

One command grades a fresh batch overnight and leaves a report for the morning.

```bash
scripts/nightly.sh --n 100 --max-spend 5
```

It keeps the Mac awake (`caffeinate`), logs to `results/nightly/`, and writes
`golden/batches/<date>-auto/`:

| File | What it is |
|---|---|
| `report.md` | Judge vs reference agreement (overall, by split, by track), dev-split disagreements, proposed next steps |
| `sheet.csv` | The batch, blind, for any other grader (a person or a browser agent) |
| `audit_sheet.csv` | 25 random pairs, blind, for a person to grade by hand |
| `manifest.json` | Every pair with its split, track and both grades |

## Steps

1. **Batch.** `--n` fresh pairs, half for IB and markets students. A pair already in any sheet under `golden/` is never reused.
2. **Capture** (`--capture`). Postings with thin text are opened in a real browser (`uv sync --extra capture`).
3. **Judge.** The gpt-4.1-mini judge plus code checks. Cached.
4. **Reference grader.** A stronger model grades the same pairs, blind to the judge.
5. **Report.**

## Reference grader

| `--reference` | Needs |
|---|---|
| `azure` | `AZURE_REFERENCE_DEPLOYMENT` in `.env` (for example a `gpt-4.1` deployment). Attaches the page screenshot when text is thin. |
| `claude` | The `claude` CLI signed in (`claude` then `/login` in a terminal). Runs Sonnet headless on your subscription, text only, tools disabled. |
| `auto` (default) | Uses `azure` if configured, otherwise runs with no reference and reports judge grades only. |

Reference grades are **AI labels**. To check them against a person, grade `audit_sheet.csv` by
hand and compare it with a sheet of reference grades:

```bash
uv run bench-compare golden/batches/<name>/audit_sheet.csv <sheet with reference grades>
```

## Rules the loop keeps

- **Dev and test.** Each posting is assigned to dev (60%) or test (40%) by a hash of its id. The report lists disagreements for dev pairs only. Tune on dev; read the test line only to check.
- **It proposes, it does not change.** The run never edits rules, prompts or code.
- **Spend cap.** It estimates the cost before any paid call and stops if the estimate is over `--max-spend`.
- **Honest browsing.** Capture uses plain Chrome with no stealth and no CAPTCHA solving. A page with a bot check or sign-in wall is recorded as blocked and skipped. All browser traffic goes through a local proxy that refuses private network addresses.

## Using a browser agent as the grader

Give `sheet.csv` to the agent, have it fill in `your_grade`, save the result over the file, then:

```bash
uv run bench-kappa golden/batches/<name>/sheet.csv
```
