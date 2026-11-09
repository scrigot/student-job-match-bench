"""Command-line entry points: `bench`, `bench-fetch`, `bench-kappa`."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from sjmb.data.loaders import (
    FIXTURE,
    build_sample,
    fetch_all,
    load_cache,
    load_fixture,
    save_postings,
)
from sjmb.eval.runner import run_bench, to_markdown
from sjmb.judge.agreement import cohen_kappa, read_golden
from sjmb.judge.judges import make_judge
from sjmb.profiles import generate_profiles


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="bench", description="Run the retrieval ladder and score it.")
    ap.add_argument(
        "--judge",
        choices=["stub", "azure"],
        default="azure",
        help="azure = the LLM judge (default); stub = offline rules, for CI and tests only",
    )
    ap.add_argument(
        "--data",
        choices=["sample", "cache"],
        default="sample",
        help="sample = committed fixture; cache = everything bench-fetch downloaded",
    )
    ap.add_argument("--encoder", choices=["auto", "st", "lsa", "azure"], default="auto")
    ap.add_argument("--profiles", type=int, default=200, help="number of synthetic profiles")
    ap.add_argument(
        "--pool-depth", type=int, default=20, help="judge each system's top-N per profile (>=10)"
    )
    ap.add_argument(
        "--max-judge-calls",
        type=int,
        default=500,
        help="hard cap on uncached Azure judge calls this run",
    )
    ap.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=date(2026, 11, 9),
        help="reference date for deadlines and class years",
    )
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    postings = load_fixture() if args.data == "sample" else load_cache()
    if not postings:
        raise SystemExit("no postings: run `uv run bench-fetch` first or use --data sample")
    profiles = generate_profiles(args.profiles, seed=args.seed, as_of=args.as_of)
    judge = make_judge(args.judge, args.as_of, args.max_judge_calls)
    result = run_bench(
        postings,
        profiles,
        judge,
        args.as_of,
        encoder=args.encoder,
        pool_depth=args.pool_depth,
        out_dir=args.out,
        progress=_log,
    )
    print(to_markdown(result))
    _log(f"wrote {args.out}/results.md, results.json, judgments.jsonl")


def fetch_main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="bench-fetch", description="Download public postings into the local cache."
    )
    ap.add_argument("--max-age-hours", type=float, default=24.0)
    ap.add_argument(
        "--write-fixture",
        action="store_true",
        help=f"also rebuild the committed sample fixture at {FIXTURE}",
    )
    args = ap.parse_args(argv)
    paths = fetch_all(args.max_age_hours)
    posts = load_cache(paths)
    by_source: dict[str, int] = {}
    for p in posts:
        by_source[p.source] = by_source.get(p.source, 0) + 1
    _log(f"cached {len(paths)} sources, {len(posts)} postings: {by_source}")
    if args.write_fixture:
        sample = build_sample(posts)
        save_postings(sample, FIXTURE)
        _log(f"wrote {len(sample)} postings to {FIXTURE}")


def kappa_main(argv: list[str] | None = None) -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(
        prog="bench-kappa",
        description="Agreement between a judge and a human golden set "
        "(CSV: profile_id,posting_id,grade).",
    )
    ap.add_argument("golden", type=Path)
    ap.add_argument("--judge", choices=["stub", "azure"], default="azure")
    ap.add_argument("--data", choices=["sample", "cache"], default="sample")
    ap.add_argument("--profiles", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--as-of", type=date.fromisoformat, default=date(2026, 11, 9))
    args = ap.parse_args(argv)

    postings = {p.id: p for p in (load_fixture() if args.data == "sample" else load_cache())}
    profiles = {p.id: p for p in generate_profiles(args.profiles, args.seed, args.as_of)}
    judge = make_judge(args.judge, args.as_of)
    human, model = [], []
    for row in read_golden(args.golden):
        prof, post = profiles.get(row["profile_id"]), postings.get(row["posting_id"])
        if prof is None or post is None:
            _log(f"skip unknown pair {row['profile_id']}/{row['posting_id']}")
            continue
        human.append(row["grade"])
        model.append(judge.grade(prof, post).grade)
    if not human:
        raise SystemExit("no usable rows in the golden set")
    exact = sum(h == m for h, m in zip(human, model, strict=True)) / len(human)
    print(f"pairs: {len(human)}")
    print(f"exact agreement: {exact:.3f}")
    print(f"cohen kappa (unweighted): {cohen_kappa(human, model):.3f}")
    print(f"cohen kappa (quadratic): {cohen_kappa(human, model, weights='quadratic'):.3f}")
