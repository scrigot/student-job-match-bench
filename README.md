# student-job-match-bench

An offline benchmark for one question: **given a student, which job postings should they see
first?** It ranks real public postings against synthetic student profiles with a ladder of
retrieval methods, grades the rankings with an LLM judge that follows a written rubric, and
reports ranking quality, poor-match rate and eligibility violations, sliced by career track and
class year.

It is the research harness behind the job ranking in KithNode, a
recruiting CRM for students. This repo holds **no student data**: postings are public, and
profiles are generated.

## Quickstart

```bash
uv sync                          # Python 3.12, numpy, httpx, openai
uv run pytest                    # unit and end-to-end tests, offline (fake Azure client)
uv run bench --judge stub        # full ladder on the committed sample, no network, no cost
uv run bench --profiles 20 --pool-depth 10   # same, graded by the Azure judge (needs .env)
```

The Azure judge is the default backend; `--judge stub` exists for CI and tests.

`bench` prints a Markdown results table and writes `results/results.md`, `results.json` and
`judgments.jsonl` (one graded profile–posting pair per line).

| Command | What it does |
|---|---|
| `uv run bench-fetch` | Download all sources into `~/.cache/student-job-match-bench/raw` (outside git). |
| `uv run bench --data cache` | Run on everything fetched (~40k postings) instead of the sample. |
| `uv sync --extra dense` then `uv run bench --encoder st` | Dense rung with sentence-transformers (all-MiniLM-L6-v2). |
| `uv run bench` | Grade with the Azure AI Foundry judge (default). See [docs/azure-setup.md](docs/azure-setup.md). |
| `uv run bench --encoder azure` | Dense rung with Azure `text-embedding-3-small`. |
| `scripts/nightly.sh --n 100` | Overnight loop: fresh batch, judge, reference grader, morning report. See [docs/nightly-loop.md](docs/nightly-loop.md). |
| `uv run bench-capture` | Read thin postings in a real browser (`uv sync --extra capture`). |
| `uv run bench-golden` | Sample judged pairs into a blind 100-pair grading sheet (`golden/grading_sheet.csv`). |
| `uv run bench-kappa golden/grading_sheet.csv` | Judge–reference agreement: exact, κ (unweighted and quadratic), confusion matrix, filter–judge disagreement breakdown. Cached judgments only. |

## The method ladder

Each rung is scored on the same profiles and the same judged pool.

| Rung | Method | Status |
|---|---|---|
| 1 | **BM25** (Okapi, title-boosted) | done |
| 2 | **Dense**: exact cosine over sentence-transformers, Azure embeddings, or a numpy TF-IDF+SVD fallback | done |
| 3 | **Hybrid**: reciprocal rank fusion of 1 and 2 (k = 60) | done |
| 4 | **Eligibility filter ablation**: hard filters applied before ranking (rungs 1 and 3) | done |
| 5 | LightGBM **LambdaMART** over ranking features | planned (KIT-254) |
| 6 | Cross-encoder / LLM **reranker** of the top 20–30 | planned (KIT-255) |
| 7 | Small model **distilled from the judge** | planned |

## Metrics

| Metric | Definition |
|---|---|
| **NDCG@10** | Graded gain `2^g − 1` on the 0–4 rubric, ideal ranking from the judged pool. |
| **Recall@50** | Share of the pool's relevant postings (grade ≥ 3) found in the top 50. Profiles with no relevant posting are skipped. |
| **Poor Match Rate@10** | Share of the top 10 graded ≤ 1 (the student would not apply). |
| **Eligibility violation rate@10** | Share of the top 10 that break a hard rule (wrong level, grad year, sponsorship, closed, past deadline). Computed from the rules, not the judge. |
| **ms/query** | Scoring latency per profile, exact search over every posting. |

Judging uses **pooling**: each profile's pool is the union of every system's top `--pool-depth`
(default 20); unjudged postings count as grade 0. Scores are relative to that pool, as in TREC.

## Results

Measured 2026-11-12 on the sample fixture: 352 postings, 200 synthetic profiles, pool depth 10,
4,110 judged pairs, judge `azure:gpt-4.1-mini` (rubric v5 plus code checks). Grades come from the
LLM judge, not from people; see "Judge card" for how far the judge has been validated.

| System | NDCG@10 | Recall@50 | PMR@10 | Elig. viol.@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.280 | 0.953 | 0.782 | 52.8% |
| Dense (LSA, 128 dims) | 0.329 | 0.926 | 0.775 | 50.7% |
| Hybrid RRF | 0.289 | 0.974 | 0.781 | 51.9% |
| BM25 + eligibility | 0.572 | 0.989 | 0.561 | 0% |
| Hybrid RRF + eligibility | 0.591 | 0.994 | 0.555 | 0% |

NDCG@10 difference from BM25, paired bootstrap 95% CI over the 200 profiles:

| System | Difference | 95% CI |
|---|---:|---|
| Dense (LSA) | +0.048 | [+0.019, +0.078] |
| Hybrid RRF | +0.008 | [−0.009, +0.025] |
| BM25 + eligibility | +0.292 | [+0.267, +0.317] |
| Hybrid RRF + eligibility | +0.311 | [+0.283, +0.338] |

What this says: text matching alone puts an ineligible job in about half of every top 10, and
the eligibility filter is worth far more than the choice of retriever. Hybrid fusion does not
beat BM25 at this size. Part of the filter's gain is by construction, because the judge and the
filter share the graduation-window code; against the 450 AI-graded reference pairs, 173 of the
184 pairs the filter rules out were graded 0 (94%).

### Neural embeddings

The same 200 profiles with Azure `text-embedding-3-small` as the dense retriever
(`--encoder azure`). The pool changes with the retrievers, so compare systems within a table,
not across tables.

| System | NDCG@10 | Recall@50 | PMR@10 | Elig. viol.@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.235 | 0.791 | 0.782 | 52.8% |
| Dense (text-embedding-3-small) | 0.471 | 0.876 | 0.621 | 38.7% |
| Hybrid RRF | 0.375 | 0.947 | 0.684 | 46.5% |
| BM25 + eligibility | 0.488 | 0.912 | 0.561 | 0% |
| Hybrid RRF + eligibility | 0.628 | 0.994 | 0.461 | 0% |

Difference from BM25 (95% CI over profiles): dense +0.237 [+0.204, +0.270], hybrid +0.140
[+0.118, +0.162], hybrid + eligibility +0.393 [+0.364, +0.423]. With neural embeddings the
retriever matters as much as the filter, unlike with LSA.

### Learned reranker (LambdaMART)

`bench-ltr` trains a LightGBM LambdaMART model on the judge's grades from 17 cheap features
(retrieval scores, track and location match, level, skill overlap, restricted-program flag,
description length) and reranks the eligible hybrid top 50. It needs no LLM call at query time.
Evaluated with 5-fold cross-validation by profile; each list's new top 10 is judged before
scoring, so every system is scored on the same, larger set of judgments (which is why the
baseline is lower here than in the table above).

| System | NDCG@10 |
|---|---:|
| Hybrid RRF + eligibility | 0.399 |
| + restricted-program rule | 0.484 |
| + LambdaMART rerank | 0.868 |

LambdaMART vs the rule baseline: +0.384, 95% CI [+0.360, +0.407]. The largest features by gain
are the restricted-program flag (34%), description length (15%) and track match (15%). Caveat:
the model learns from the same judge that scores it, so this measures how well a cheap model
reproduces the judge, including the judge's code-enforced track caps, not agreement with people.

## Data card

| | |
|---|---|
| **Postings** | Public Greenhouse boards (26 firms), Lever boards (4 firms) and Workday campus boards of 11 banks, across trading, asset management, banking, consulting and tech, plus the SimplifyJobs Summer 2027 internship and new-grad lists. Board tokens are in `src/sjmb/data/loaders.py`. |
| **Sample fixture** | `src/sjmb/data/fixtures/sample_postings.json`: 352 postings (board samples, a track-stratified Simplify slice and 122 bank campus postings), descriptions kept up to 8,000 characters so the judge reads the full posting. The only data in git. |
| **Extraction** | Rule-based at ingest (`sjmb/data/extract.py`): track, internship vs new-grad vs experienced, eligible grad years, sponsorship, terms. Greenhouse supplies deadlines; Simplify supplies sponsorship and active status. |
| **Profiles** | 200 synthetic students from a seeded generator, stratified over 7 tracks × 4 class years × 3 work-authorization statuses. No real people. |
| **Eligibility policy** | Each student seeks one level: seniors are matched to new-grad roles, everyone else to internships. Unstated fields never exclude a student. `us_persons` (citizens or permanent residents) is separated from `citizens_only`. |
| **Known gaps** | Rule extraction misses grad years stated in unusual ways and misreads some titles. Simplify rows have no description text, so lexical methods see only title, company and category. |

## Judge card

- **Rubric**: 0–4, written in `src/sjmb/judge/rubric.py` and versioned (`RUBRIC_VERSION`). 0 means
  the student cannot or should not apply; ≤ 1 counts as a poor match; ≥ 3 counts as relevant.
- **Backends**: `azure` (default: any chat deployment on Azure AI Foundry through the `openai` SDK
  and the OpenAI v1 route) and `stub` (deterministic rules, offline, for CI and tests).
- **Cost controls**: judge calls and embeddings are cached on disk by content hash, so re-runs
  only pay for new pairs. Before judging, the runner counts uncached calls, prints an estimated
  cost when there are more than 500, and stops before any call if the count exceeds
  `--max-judge-calls` (default 500).
- **Validation**: a 100-pair blind golden set in [golden/](golden/README.md), stratified to
  include filter–judge disagreements. `bench-kappa` reports exact agreement, unweighted and
  quadratic-weighted Cohen's κ, a confusion matrix and the disagreement breakdown.
- **Where it stands (2026-11-12)**: on 50 held-out pairs graded by an AI reference grader, the
  judge agrees exactly on 62% (quadratic κ 0.66), up from 44% (0.45) before tuning. No
  human-validated number exists yet; blind human sheets are in `golden/human_check_sheet.csv`
  and `golden/anthony/`.

## Layout

```
src/sjmb/
  data/        loaders (Greenhouse, Lever, Simplify), rule-based extraction, sample fixture
  profiles/    synthetic profile generator
  retrievers/  BM25, dense encoders, hybrid RRF, exact top-k
  filters/     eligibility rules (filter and ground truth for violations)
  judge/       rubric, stub and Azure judges, Cohen's kappa
  eval/        metrics, ladder runner, Markdown and JSON reports
  cli.py       bench, bench-fetch, bench-kappa
```

## References

- Cormack, Clarke, Büttcher. *Reciprocal rank fusion outperforms Condorcet and individual rank
  learning methods.* SIGIR 2009.
- Thomas et al. *Large language models can accurately predict searcher preferences.* arXiv
  2309.10621, 2023.
- LinkedIn job search ranking: arXiv 2402.13435 (LiRank) and 2602.07309.
- Järvelin, Kekäläinen. *Cumulated gain-based evaluation of IR techniques.* TOIS 2002.
