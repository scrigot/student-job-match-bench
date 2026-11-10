"""Command-line entry points: `bench`, `bench-fetch`, `bench-kappa`."""

from __future__ import annotations

import argparse
import hashlib
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


def blind_main(argv: list[str] | None = None) -> None:
    """Copy a random sample of already-graded sheets with grades and notes removed, so a second
    grader (a person) can grade the same pairs without seeing the first grader's answers."""
    import random

    from sjmb.golden import COLUMNS, RUBRIC_LINES, _cell

    ap = argparse.ArgumentParser(prog="bench-blind", description=blind_main.__doc__)
    ap.add_argument("sheets", type=Path, nargs="+", help="graded sheets to sample from")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument(
        "--skip-first", type=int, default=0, help="skip this many leading rows of the FIRST sheet"
    )
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite a sheet that may hold grades")
    pool = []
    for i, sheet in enumerate(args.sheets):
        rows = [r for r in read_sheet_rows(sheet) if (r.get("your_grade") or "").strip()]
        pool += rows[args.skip_first :] if i == 0 else rows
    rng = random.Random(args.seed)
    by_grade: dict[str, list[dict]] = {}
    for r in pool:
        by_grade.setdefault(r["your_grade"].strip(), []).append(r)
    picked: list[dict] = []
    while len(picked) < min(args.n, len(pool)):  # round-robin over grades so each is covered
        for g in sorted(by_grade):
            if by_grade[g] and len(picked) < args.n:
                picked.append(by_grade[g].pop(rng.randrange(len(by_grade[g]))))
    rng.shuffle(picked)
    import csv
    import io

    buf = io.StringIO()
    buf.write("# Grade each pair 0-4 in your_grade, by hand, without looking at other grades.\n")
    for line in RUBRIC_LINES:
        buf.write(f"# {line}\n")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COLUMNS)
    for i, r in enumerate(picked, 1):
        cells = [r["profile_id"], r["posting_id"], r["student"], r["job"]]
        w.writerow([i, *(_cell(c) for c in cells), "", ""])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(buf.getvalue())
    _log(f"wrote {len(picked)} blind pairs to {args.out}")


def compare_main(argv: list[str] | None = None) -> None:
    """Agreement between two graders' sheets, matched on (profile_id, posting_id)."""
    from sjmb.golden import confusion_matrix
    from sjmb.judge.agreement import cohen_kappa

    ap = argparse.ArgumentParser(prog="bench-compare", description=compare_main.__doc__)
    ap.add_argument("a", type=Path, help="first grader (rows of the table)")
    ap.add_argument("b", type=Path, nargs="+", help="second grader: one or more sheets")
    args = ap.parse_args(argv)
    invalid: list[str] = []
    a = {(r["profile_id"], r["posting_id"]): r["grade"] for r in read_golden(args.a, invalid)}
    b: dict[tuple[str, str], int] = {}
    for sheet in args.b:
        b.update(
            {(r["profile_id"], r["posting_id"]): r["grade"] for r in read_golden(sheet, invalid)}
        )
    for msg in invalid:
        _log(f"skip {msg}")
    keys = sorted(set(a) & set(b))
    if not keys:
        raise SystemExit("the sheets share no graded pairs")
    x, y = [a[k] for k in keys], [b[k] for k in keys]
    exact = sum(i == j for i, j in zip(x, y, strict=True)) / len(keys)
    within1 = sum(abs(i - j) <= 1 for i, j in zip(x, y, strict=True)) / len(keys)
    print(f"pairs graded by both: {len(keys)}")
    print(f"exact agreement: {exact:.3f}")
    print(f"within one grade: {within1:.3f}")
    if len(set(x)) > 1 and len(set(y)) > 1:
        print(f"cohen kappa (unweighted): {cohen_kappa(x, y):.3f}")
        print(f"cohen kappa (quadratic): {cohen_kappa(x, y, weights='quadratic'):.3f}")
    print("\nConfusion matrix (rows = first sheet, columns = second):")
    print(confusion_matrix(x, y).replace("human \\ judge", "first \\ second"))


def capture_main(argv: list[str] | None = None) -> None:
    """Open posting pages in a real browser and save their text and a screenshot."""
    from collections import Counter as _Counter

    from sjmb.capture import MIN_TEXT, apply_captures, capture_urls

    ap = argparse.ArgumentParser(prog="bench-capture", description=capture_main.__doc__)
    ap.add_argument("--sheet", type=Path, help="capture the postings in this grading sheet")
    ap.add_argument("--all", action="store_true", help="every sample posting, not only thin ones")
    ap.add_argument("--headed", action="store_true", help="show the browser window")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument(
        "--apply",
        action="store_true",
        help="write captured text into the sample for postings with thin text",
    )
    args = ap.parse_args(argv)

    posts = load_fixture()
    if args.sheet:
        ids = {r["posting_id"] for r in read_sheet_rows(args.sheet)}
        targets = [p for p in posts if p.id in ids]
    else:
        targets = [p for p in posts if args.all or len(p.description) < MIN_TEXT]
    urls = [p.url for p in targets if p.url]
    _log(f"{len(urls)} postings to capture")
    caps = capture_urls(urls, headed=args.headed, progress=_log, retry_failed=args.retry_failed)
    _log(f"capture results: {dict(_Counter(c.status for c in caps))}")
    if args.apply:
        n = apply_captures(posts)
        save_postings(posts, FIXTURE)
        _log(f"updated {n} sample postings with captured text")


def nightly_main(argv: list[str] | None = None) -> None:
    """Grade a fresh batch with the judge and a reference grader, and write a morning report."""
    import os

    from sjmb.nightly import run_nightly
    from sjmb.reference import ReferenceGrader, ReferenceUnavailable

    load_dotenv()
    ap = argparse.ArgumentParser(prog="bench-nightly", description=nightly_main.__doc__)
    ap.add_argument("--n", type=int, default=100, help="pairs in the batch")
    ap.add_argument("--judge", choices=["azure", "stub"], default="azure")
    ap.add_argument(
        "--reference",
        choices=["auto", "azure", "claude", "none"],
        default="auto",
        help="auto = the Azure reference deployment if configured, else none",
    )
    ap.add_argument("--max-spend", type=float, default=5.0, help="stop if the estimate is higher")
    ap.add_argument("--capture", action="store_true", help="browser-capture thin postings first")
    ap.add_argument("--name", help="batch folder name (default: today's date)")
    ap.add_argument("--seed", type=int, help="default: derived from the batch name")
    ap.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    args = ap.parse_args(argv)

    name = args.name or f"{date.today().isoformat()}-auto"
    seed = (
        args.seed
        if args.seed is not None
        else int(hashlib.sha256(name.encode()).hexdigest()[:6], 16)
    )
    judge = make_judge(args.judge, args.as_of, max_calls=args.n + 50)
    reference = None
    kind = args.reference
    if kind == "auto":
        kind = "azure" if os.environ.get("AZURE_REFERENCE_DEPLOYMENT") else "none"
    if kind != "none":
        try:
            reference = ReferenceGrader(kind, args.as_of, max_calls=args.n + 50)
        except ReferenceUnavailable as e:
            raise SystemExit(f"reference grader unavailable: {e}") from None
    report = run_nightly(
        generate_profiles(200, 42, args.as_of),
        load_fixture(),
        judge,
        reference,
        args.as_of,
        out_dir=Path("golden/batches") / name,
        golden_dir=Path("golden"),
        n=args.n,
        seed=seed,
        max_spend=args.max_spend,
        capture=args.capture,
        progress=_log,
    )
    print(report.read_text())
    _log(f"wrote {report}")
