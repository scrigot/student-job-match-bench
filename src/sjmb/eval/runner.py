"""Runs the method ladder over every profile, judges the pooled results, and scores each rung."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from sjmb.azure import COST_WARN_CALLS
from sjmb.eval.metrics import ndcg_at_k, poor_match_rate_at_k, recall_at_k, violation_rate_at_k
from sjmb.filters.eligibility import eligibility_mask
from sjmb.judge.judges import Judge
from sjmb.retrievers import BM25, Dense, Hybrid, make_encoder, top_k
from sjmb.retrievers.dense import AzureEmbeddingEncoder, Encoder
from sjmb.types import Posting, Profile

METRICS = ("ndcg@10", "recall@50", "pmr@10", "elig_viol@10", "ms/query")
RETRIEVE_DEPTH = 50


@dataclass
class System:
    name: str
    scores: Callable[[str], np.ndarray]
    use_eligibility: bool = False


@dataclass
class BenchResult:
    systems: list[str]
    overall: dict[str, dict[str, float]]
    by_track: dict[str, dict[str, dict[str, float]]]
    by_class: dict[str, dict[str, dict[str, float]]]
    meta: dict = field(default_factory=dict)


def build_ladder(postings: list[Posting], encoder: str | Encoder = "auto") -> list[System]:
    bm25 = BM25().fit(postings)
    dense = Dense(make_encoder(encoder) if isinstance(encoder, str) else encoder).fit(postings)
    hybrid = Hybrid([bm25, dense]).fit(postings)
    return [
        System("bm25", bm25.scores),
        System(dense.name, dense.scores),
        System("hybrid-rrf", hybrid.scores),
        System("bm25 + eligibility", bm25.scores, use_eligibility=True),
        System("hybrid-rrf + eligibility", hybrid.scores, use_eligibility=True),
    ]


def preflight(
    judge: Judge, pairs: list[tuple[Profile, Posting]], progress: Callable[[str], None]
) -> None:
    """Before any paid call: count uncached judgments, print a cost estimate for large runs,
    and stop if the run would exceed the judge's call budget."""
    if not hasattr(judge, "is_cached"):
        return  # the stub is free
    todo = [(pr, po) for pr, po in pairs if not judge.is_cached(pr, po)]
    if len(todo) > COST_WARN_CALLS:
        progress(f"judge: {len(todo)} uncached calls, estimated ${judge.estimate_cost(todo):.2f}")
    if len(todo) > judge.max_calls:
        raise SystemExit(
            f"this run needs {len(todo)} uncached judge calls "
            f"(est. ${judge.estimate_cost(todo):.2f}), above --max-judge-calls "
            f"{judge.max_calls}. Raise the cap or use fewer --profiles / a smaller --pool-depth."
        )


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    out = {}
    for m in METRICS:
        vals = [r[m] for r in rows if not math.isnan(r[m])]
        out[m] = float(np.mean(vals)) if vals else math.nan
    out["n"] = len(rows)
    out["n_recall"] = sum(1 for r in rows if not math.isnan(r["recall@50"]))
    return out


def run_bench(
    postings: list[Posting],
    profiles: list[Profile],
    judge: Judge,
    as_of: date,
    encoder: str = "auto",
    pool_depth: int = 20,
    out_dir: Path | None = None,
    progress: Callable[[str], None] = lambda _: None,
) -> BenchResult:
    if not 10 <= pool_depth <= RETRIEVE_DEPTH:
        raise ValueError(f"pool_depth must be in [10, {RETRIEVE_DEPTH}]")
    enc = make_encoder(encoder)
    if isinstance(enc, AzureEmbeddingEncoder):
        texts = [p.text() for p in postings] + [prof.query() for prof in profiles]
        calls, cost = enc.estimate(texts)
        if calls > COST_WARN_CALLS:
            progress(f"embeddings: {calls} uncached calls, estimated ${cost:.2f}")
        enc.encode([prof.query() for prof in profiles])  # batch the queries into the cache
    progress(f"fitting retrievers on {len(postings)} postings")
    ladder = build_ladder(postings, enc)
    ids = [p.id for p in postings]

    # 1. Retrieve.
    runs: dict[str, dict[str, list[str]]] = {s.name: {} for s in ladder}
    latency: dict[str, dict[str, float]] = {s.name: {} for s in ladder}
    ineligible: dict[str, set[str]] = {}
    for prof in profiles:
        mask = eligibility_mask(prof, postings, as_of)
        ineligible[prof.id] = {ids[i] for i in np.flatnonzero(~mask)}
        q = prof.query()
        for s in ladder:
            t0 = time.perf_counter()
            ranked = top_k(s.scores(q), RETRIEVE_DEPTH, mask if s.use_eligibility else None)
            latency[s.name][prof.id] = (time.perf_counter() - t0) * 1000
            runs[s.name][prof.id] = [ids[i] for i in ranked]

    # 2. Judge the pool: the union of every system's top `pool_depth` per profile.
    by_id = {p.id: p for p in postings}
    pools = {
        prof.id: sorted({d for s in ladder for d in runs[s.name][prof.id][:pool_depth]})
        for prof in profiles
    }
    preflight(judge, [(prof, by_id[d]) for prof in profiles for d in pools[prof.id]], progress)
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    pool_size = 0
    unparsed = 0
    judgments = []
    for n_done, prof in enumerate(profiles, 1):
        pool = pools[prof.id]
        pool_size += len(pool)
        for d in pool:
            j = judge.grade(prof, by_id[d])
            if j.grade is None:
                unparsed += 1
                continue  # excluded from scoring, reported in meta
            qrels[prof.id][d] = j.grade
            judgments.append(
                {"profile": prof.id, "posting": d, "grade": j.grade, "reason": j.reason}
            )
        if n_done % 25 == 0:
            progress(f"judged {n_done}/{len(profiles)} profiles")

    # 3. Score.
    per_profile: dict[str, dict[str, dict[str, float]]] = {}
    for s in ladder:
        per_profile[s.name] = {}
        for prof in profiles:
            ranked, qr = runs[s.name][prof.id], qrels[prof.id]
            per_profile[s.name][prof.id] = {
                "ndcg@10": ndcg_at_k(ranked, qr, 10),
                "recall@50": recall_at_k(ranked, qr, 50),
                "pmr@10": poor_match_rate_at_k(ranked, qr, 10),
                "elig_viol@10": violation_rate_at_k(ranked, ineligible[prof.id], 10),
                "ms/query": latency[s.name][prof.id],
            }

    names = [s.name for s in ladder]
    overall = {n: _mean(list(per_profile[n].values())) for n in names}

    def sliced(key: Callable[[Profile], str]) -> dict[str, dict[str, dict[str, float]]]:
        groups: dict[str, list[Profile]] = defaultdict(list)
        for prof in profiles:
            groups[key(prof)].append(prof)
        return {
            g: {n: _mean([per_profile[n][p.id] for p in members]) for n in names}
            for g, members in sorted(groups.items())
        }

    result = BenchResult(
        systems=names,
        overall=overall,
        by_track=sliced(lambda p: p.track),
        by_class=sliced(lambda p: f"{p.class_label} ({p.grad_year})"),
        meta={
            "judge": judge.name,
            "judge_calls": judge.calls,
            "as_of": as_of.isoformat(),
            "postings": len(postings),
            "profiles": len(profiles),
            "pool_depth": pool_depth,
            "judged_pairs": pool_size,
            "unparsed_judgments": unparsed,
        },
    )
    if out_dir is not None:
        write_outputs(result, judgments, out_dir)
    return result


def _fmt(v: float, metric: str) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "n/a"
    return f"{v:.1f}" if metric == "ms/query" else f"{v:.3f}"


def results_table(rows: dict[str, dict[str, float]], metrics: tuple[str, ...] = METRICS) -> str:
    head = "| system | " + " | ".join(metrics) + " |"
    sep = "|---|" + "---:|" * len(metrics)
    body = [
        f"| {n} | " + " | ".join(_fmt(r[m], m) for m in metrics) + " |" for n, r in rows.items()
    ]
    return "\n".join([head, sep, *body])


def to_markdown(r: BenchResult) -> str:
    m = r.meta
    parts = [
        "# Results",
        "",
        f"Judge: `{m['judge']}` · as of {m['as_of']} · {m['postings']} postings · "
        f"{m['profiles']} profiles · pool depth {m['pool_depth']} · "
        f"{m['judged_pairs']} judged pairs ({m['judge_calls']} judge calls this run)",
        "",
    ]
    if m["judge"] == "stub":
        parts += [
            "> Stub judge: grades come from the same rules as the eligibility filter. "
            "These numbers test the pipeline, not ranking quality.",
            "",
        ]
    parts += ["## Overall", "", results_table(r.overall), ""]
    short = ("ndcg@10", "pmr@10", "elig_viol@10")
    for title, slices in (("By track", r.by_track), ("By class year", r.by_class)):
        parts += [f"## {title}", ""]
        for g, rows in slices.items():
            parts += [
                f"### {g} (n={next(iter(rows.values()))['n']})",
                "",
                results_table(rows, short),
                "",
            ]
    return "\n".join(parts)


def write_outputs(r: BenchResult, judgments: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.md").write_text(to_markdown(r))

    def clean(x):  # JSON has no NaN; write null instead
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        return None if isinstance(x, float) and math.isnan(x) else x

    payload = {"meta": r.meta, "overall": r.overall, "by_track": r.by_track, "by_class": r.by_class}
    (out_dir / "results.json").write_text(json.dumps(clean(payload), indent=1))
    with (out_dir / "judgments.jsonl").open("w") as f:
        for j in judgments:
            f.write(json.dumps(j) + "\n")
