# Azure AI Foundry setup

The Azure judge is the default backend for `bench` and `bench-kappa`. Hosted embeddings
(`--encoder azure`) are optional. `--judge stub` runs fully offline and is meant for CI and tests.

## 1. Get the subscription

1. Sign up for **Azure for Students** at <https://azure.microsoft.com/free/students> with your
   university email. No credit card; it comes with a yearly credit.
2. Open the portal at <https://portal.azure.com> and confirm the subscription shows as
   *Azure for Students*.

## 2. Create a Foundry resource and deploy models

1. Go to <https://ai.azure.com> and create a project. It creates the Foundry resource for you.
   Pick a region that lists the models below (East US 2 and Sweden Central usually do).
2. **Models + endpoints → Deploy model**:
   - Judge: a small chat model, for example `gpt-4.1-mini`. Use *Global Standard*. The
     deployment name is what goes in `AZURE_JUDGE_DEPLOYMENT`.
   - Embeddings (optional): `text-embedding-3-small`. Its deployment name goes in
     `AZURE_EMBED_DEPLOYMENT`.
3. On the resource's overview page, copy the **endpoint** and one **key**.

## 3. Configure the repo

```bash
cp .env.example .env   # .env is git-ignored
# fill in AZURE_AI_ENDPOINT, AZURE_AI_KEY, AZURE_JUDGE_DEPLOYMENT (and AZURE_EMBED_DEPLOYMENT)
```

The endpoint can be given with or without the `/openai/v1` suffix. The code uses the `openai`
SDK with `base_url` set to `<endpoint>/openai/v1`, the deployment name as `model`, and the key
sent both as `api_key` and as an `api-key` header (`src/sjmb/azure.py`).

## 4. Smoke test (about 20 calls)

```bash
uv run bench --profiles 1 --pool-depth 10 --max-judge-calls 40 --out results/azure-smoke
```

You should see a results table whose header says `Judge: azure:<deployment>`, and
`results/azure-smoke/judgments.jsonl` should hold one graded pair per line with a short reason.

## Cost controls

- **Budget cap.** `--max-judge-calls` (default 500) stops the run once that many *uncached*
  calls have been made.
- **Preflight.** Before the first judge call, the runner counts uncached pairs. Above 500 it
  prints an estimated cost; above `--max-judge-calls` it stops without making any call. Large
  embedding runs print their estimate the same way. Prices are in `PRICES` in
  `src/sjmb/azure.py`; update them if the deployments change.
- **Caches.** Judgments go to `~/.cache/student-job-match-bench/judgments.jsonl`, keyed by a hash
  of rubric version, deployment, profile and posting. Embeddings go to `embeddings.sqlite`,
  keyed by a hash of deployment and text. Re-runs only pay for new items. Changing the rubric
  text means bumping `RUBRIC_VERSION` in `src/sjmb/judge/rubric.py`. Keys are never cached.
- **Pool size.** Calls per run ≈ profiles × (unique postings in the systems' top `--pool-depth`).
  With five systems that overlap heavily, `--pool-depth 10` is about 20 calls per profile, so
  `--profiles 50 --pool-depth 10` is about 1,000 calls.
- The subscription has a **$100 credit with a $20/month budget alert** (Cost Management → Budgets).

## Validating the judge

Before trusting judge numbers, grade a golden set by hand (see the README) and run:

```bash
uv run bench-kappa golden.csv
```

Report the quadratic-weighted κ next to every results table.
