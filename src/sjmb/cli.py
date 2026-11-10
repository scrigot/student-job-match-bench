"""Command-line entry points: `bench`, `bench-fetch`, `bench-kappa`."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
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
from sjmb.golden import agreement_report, read_judgments, sample_pairs, stratum, write_sheet
from sjmb.judge.agreement import read_golden, read_sheet_rows
from sjmb.judge.judges import BudgetExceeded, make_judge, render_prompt
from sjmb.judge.rubric import RUBRIC_VERSION
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
        description="Agreement between the judge and a human golden set. Reads the sheet from "
        "bench-golden as-is (grades in `your_grade`; `#` rubric lines are skipped).",
    )
    ap.add_argument("golden", type=Path)
    ap.add_argument("--judge", choices=["stub", "azure"], default="azure")
    ap.add_argument("--data", choices=["sample", "cache"], default="sample")
    ap.add_argument("--profiles", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--as-of", type=date.fromisoformat, default=date(2026, 11, 9))
    ap.add_argument(
        "--max-judge-calls",
        type=int,
        default=0,
        help="uncached judge calls allowed (default 0: use cached judgments only)",
    )
    args = ap.parse_args(argv)

    postings = {p.id: p for p in (load_fixture() if args.data == "sample" else load_cache())}
    profiles = {p.id: p for p in generate_profiles(args.profiles, args.seed, args.as_of)}
    judge = make_judge(args.judge, args.as_of, args.max_judge_calls)
    human, model, strata = [], [], []
    invalid: list[str] = []
    graded = read_golden(args.golden, invalid)
    for msg in invalid:
        _log(f"skip {msg}")
    for row in graded:
        prof, post = profiles.get(row["profile_id"]), postings.get(row["posting_id"])
        if prof is None or post is None:
            _log(f"skip unknown pair {row['profile_id']}/{row['posting_id']}")
            continue
        try:
            j = judge.grade(prof, post)
        except BudgetExceeded:
            raise SystemExit(
                f"no cached judgment for {prof.id}/{post.id}; rerun with --max-judge-calls N "
                "to allow new judge calls"
            ) from None
        if j.grade is None:
            _log(f"skip {prof.id}/{post.id}: the judge reply was unparsable")
            continue
        human.append(row["grade"])
        model.append(j.grade)
        strata.append(stratum(prof, post, j.grade, args.as_of))
    if not human:
        raise SystemExit("no graded rows yet: fill in your_grade (0-4) first")
    print(f"judge: {judge.name}")
    print(agreement_report(human, model, strata))


def golden_main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="bench-golden",
        description="Sample judged pairs into a blind human grading sheet.",
    )
    ap.add_argument("--judgments", type=Path, default=Path("results/judgments.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("golden/grading_sheet.csv"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--sample-seed", type=int, default=7)
    ap.add_argument("--data", choices=["sample", "cache"], default="sample")
    ap.add_argument("--profiles", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42, help="profile generator seed")
    ap.add_argument("--as-of", type=date.fromisoformat, default=date(2026, 11, 9))
    ap.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        help="an existing sheet whose pairs must not be sampled again (repeatable)",
    )
    args = ap.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite a sheet that may hold grades")

    postings = {p.id: p for p in (load_fixture() if args.data == "sample" else load_cache())}
    profiles = {p.id: p for p in generate_profiles(args.profiles, args.seed, args.as_of)}
    used = {(r["profile_id"], r["posting_id"]) for s in args.exclude for r in read_sheet_rows(s)}
    pairs = sample_pairs(
        [j for j in read_judgments(args.judgments) if (j["profile"], j["posting"]) not in used],
        profiles,
        postings,
        args.as_of,
        n=args.n,
        seed=args.sample_seed,
    )
    write_sheet(pairs, args.out)
    strata = Counter(r["stratum"] or "filter and judge agree" for r in pairs)
    bands = Counter(r["band"] for r in pairs)
    cells = Counter(r["cell"] for r in pairs)
    _log(f"wrote {len(pairs)} pairs to {args.out}")
    _log(f"strata: {dict(strata)}")
    _log(f"judge grade bands: {dict(sorted(bands.items()))}")
    _log(f"track x class cells: {len(cells)}, largest {max(cells.values())}")


def judge_one_main(argv: list[str] | None = None) -> None:
    """Show one golden pair the way the judge sees it, for prompt work in a playground."""
    load_dotenv()
    ap = argparse.ArgumentParser(
        prog="bench-judge-one",
        description="Write the exact judge prompt for one golden pair to a file, and show the "
        "judge's cached answer next to your grade. Makes no call unless --call is given.",
    )
    ap.add_argument("pair_no", type=int)
    ap.add_argument("--sheet", type=Path, default=Path("golden/grading_sheet.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/prompts"))
    ap.add_argument("--call", action="store_true", help="ask the judge if no cached answer")
    ap.add_argument("--as-of", type=date.fromisoformat, default=date(2026, 11, 9))
    args = ap.parse_args(argv)

    rows = {int(r["pair_no"]): r for r in read_sheet_rows(args.sheet)}
    if args.pair_no not in rows:
        raise SystemExit(f"pair {args.pair_no} is not in {args.sheet}")
    row = rows[args.pair_no]
    prof = {p.id: p for p in generate_profiles(200, 42, args.as_of)}[row["profile_id"]]
    post = {p.id: p for p in load_fixture()}[row["posting_id"]]
    prompt = render_prompt(prof, post, args.as_of)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"pair_{args.pair_no}.txt"
    path.write_text(prompt)

    judge = make_judge("azure", args.as_of, max_calls=1 if args.call else 0)
    if judge.is_cached(prof, post) or args.call:
        j = judge.grade(prof, post)
        answer = f"{j.grade}  ({j.reason})"
    else:
        answer = "not cached for this rubric version (rerun with --call)"
    print(f"pair {args.pair_no}: {post.company} | {post.title}")
    print(f"student:   {prof.class_label}, {prof.track} track, seeking {prof.seeking}")
    print(f"your grade: {row.get('your_grade') or '-'}  {row.get('notes', '')[:160]}")
    print(f"judge ({judge.name}, rubric {RUBRIC_VERSION}): {answer}")
    print(f"prompt written to {path}  (copy it with: pbcopy < {path})")
