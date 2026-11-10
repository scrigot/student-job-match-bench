"""The overnight loop: pick a fresh batch, grade it twice, and write a morning report.

    batch -> (capture) -> judge -> reference grader -> report + blind audit sheet

Rules the loop keeps:
- A pair is never reused: every sheet under `golden/` is excluded.
- Each posting is assigned to `dev` or `test` by a hash of its id, so the split is stable and a
  posting never sits on both sides. The report shows disagreements for dev pairs only; test
  pairs are scored in aggregate and never listed.
- Nothing here edits rules, prompts or code. The report proposes; a person decides.
- Reference labels are AI labels. The audit sheet is how a person checks them.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sjmb.golden import write_sheet
from sjmb.judge.agreement import cohen_kappa, read_sheet_rows
from sjmb.judge.judges import same_track
from sjmb.retrievers import BM25, Dense, Hybrid, make_encoder, top_k
from sjmb.types import Posting, Profile

FINANCE_TRACKS = ("ib", "markets")
DEV_SHARE = 6  # of 10


def split_of(posting_id: str) -> str:
    bucket = int(hashlib.sha256(posting_id.encode()).hexdigest()[:8], 16) % 10
    return "dev" if bucket < DEV_SHARE else "test"


def used_pairs(golden_dir: Path) -> set[tuple[str, str]]:
    used: set[tuple[str, str]] = set()
    for sheet in golden_dir.rglob("*.csv"):
        used |= {
            (r["profile_id"], r["posting_id"])
            for r in read_sheet_rows(sheet)
            if r.get("profile_id") and r.get("posting_id")
        }
    return used


@dataclass
class Pair:
    profile: Profile
    posting: Posting

    @property
    def key(self) -> tuple[str, str]:
        return (self.profile.id, self.posting.id)

    @property
    def split(self) -> str:
        return split_of(self.posting.id)

    @property
    def group(self) -> str:
        return "finance" if self.profile.track in FINANCE_TRACKS else "other"


def make_batch(
    profiles: list[Profile],
    postings: list[Posting],
    n: int,
    used: set[tuple[str, str]],
    finance_share: float = 0.5,
    seed: int = 0,
    per_posting: int = 4,
    encoder: str = "lsa",
) -> list[Pair]:
    """n fresh pairs: each student's top hybrid results plus a few from further down, with
    `finance_share` of them for IB and markets students."""
    rng = random.Random(seed)
    bm25 = BM25().fit(postings)
    dense = Dense(make_encoder(encoder)).fit(postings)
    hybrid = Hybrid([bm25, dense]).fit(postings)
    order = profiles[:]
    rng.shuffle(order)
    pools: dict[str, list[Pair]] = {"finance": [], "other": []}
    for prof in order:
        ranked = top_k(hybrid.scores(prof.query()), 60)
        picks = ranked[:12] + rng.sample(ranked[12:], k=min(4, max(0, len(ranked) - 12)))
        for i in picks:
            pair = Pair(prof, postings[i])
            if pair.key not in used:
                pools[pair.group].append(pair)
    want = {"finance": round(n * finance_share)}
    want["other"] = n - want["finance"]
    chosen: list[Pair] = []
    for group, pool in pools.items():
        rng.shuffle(pool)
        per_post: Counter[str] = Counter()
        per_prof: Counter[str] = Counter()
        cap_prof = max(4, -(-want[group] // 8))  # spread over at least ~8 students
        for pair in pool:
            if len([c for c in chosen if c.group == group]) >= want[group]:
                break
            if per_post[pair.posting.id] >= per_posting or per_prof[pair.profile.id] >= cap_prof:
                continue
            per_post[pair.posting.id] += 1
            per_prof[pair.profile.id] += 1
            chosen.append(pair)
    rng.shuffle(chosen)
    return chosen


def _quad(a: list[int], b: list[int]) -> float:
    return cohen_kappa(a, b, weights="quadratic")


def _agree(a: list[int], b: list[int]) -> str:
    if not a:
        return "n=0"
    exact = sum(x == y for x, y in zip(a, b, strict=True)) / len(a)
    out = f"n={len(a)}, exact {exact:.2f}"
    if len(set(a)) > 1 and len(set(b)) > 1:
        out += f", κ {cohen_kappa(a, b):.2f}, quadratic κ {_quad(a, b):.2f}"
    return out


def run_nightly(
    profiles: list[Profile],
    postings: list[Posting],
    judge,
    reference,
    as_of: date,
    out_dir: Path,
    golden_dir: Path,
    n: int = 100,
    seed: int = 0,
    audit_n: int = 25,
    max_spend: float = 5.0,
    capture: bool = False,
    progress=lambda _: None,
) -> Path:
    """Run one night. Returns the path of the report."""
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; refusing to overwrite a batch")
    batch = make_batch(profiles, postings, n, used_pairs(golden_dir), seed=seed)
    if not batch:
        raise SystemExit("no fresh pairs left: fetch more postings or add profiles")
    progress(
        f"batch: {len(batch)} pairs, {dict(Counter(p.group for p in batch))}, "
        f"{dict(Counter(p.split for p in batch))}"
    )

    cap_stats: dict[str, int] = {}
    if capture:
        from sjmb.capture import capture_urls

        urls = [p.posting.url for p in batch if p.posting.url and len(p.posting.description) < 300]
        cap_stats = dict(Counter(c.status for c in capture_urls(urls, progress=progress)))

    # Spend check before any paid call.
    todo_j = [
        (p.profile, p.posting)
        for p in batch
        if hasattr(judge, "is_cached") and not judge.is_cached(p.profile, p.posting)
    ]
    todo_r = [
        (p.profile, p.posting)
        for p in batch
        if reference is not None and not reference.is_cached(p.profile, p.posting)
    ]
    est = (judge.estimate_cost(todo_j) if todo_j else 0.0) + (
        reference.estimate_cost(todo_r) if todo_r else 0.0
    )
    progress(f"estimated spend: ${est:.2f} ({len(todo_j)} judge, {len(todo_r)} reference calls)")
    if est > max_spend:
        raise SystemExit(f"estimated ${est:.2f} is over the ${max_spend:.2f} cap; nothing spent")

    rows = []
    for i, p in enumerate(batch, 1):
        j = judge.grade(p.profile, p.posting)
        r = reference.grade(p.profile, p.posting) if reference is not None else None
        rows.append({"pair": p, "judge": j, "ref": r})
        if i % 25 == 0:
            progress(f"graded {i}/{len(batch)}")

    out_dir.mkdir(parents=True)
    sheet_rows = [{"profile": p.profile, "posting": p.posting} for p in batch]
    write_sheet(sheet_rows, out_dir / "sheet.csv")  # blind, for any other grader
    rng = random.Random(seed + 1)
    write_sheet(
        rng.sample(sheet_rows, k=min(audit_n, len(sheet_rows))), out_dir / "audit_sheet.csv"
    )
    manifest = [
        {
            "profile": r["pair"].profile.id,
            "posting": r["pair"].posting.id,
            "split": r["pair"].split,
            "group": r["pair"].group,
            "track": r["pair"].profile.track,
            "judge": r["judge"].grade,
            "judge_reason": r["judge"].reason,
            "reference": r["ref"].grade if r["ref"] else None,
            "reference_reason": r["ref"].reason if r["ref"] else None,
        }
        for r in rows
    ]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    report = out_dir / "report.md"
    report.write_text(build_report(rows, judge, reference, as_of, est, cap_stats))
    return report


def build_report(rows, judge, reference, as_of, est, cap_stats) -> str:
    batch = [r["pair"] for r in rows]
    out = [f"# Nightly report, {as_of.isoformat()}", ""]
    out += [
        f"- Pairs: {len(batch)} ({dict(Counter(p.group for p in batch))}; "
        f"{dict(Counter(p.split for p in batch))})",
        f"- Judge: `{judge.name}`",
        f"- Reference grader: `{reference.name}`"
        if reference
        else "- Reference grader: **none configured**, so this report has judge grades only",
        f"- Estimated spend: ${est:.2f}",
    ]
    if cap_stats:
        out.append(f"- Browser capture of thin postings: {cap_stats}")
    out += [
        "",
        "> Reference grades are AI labels, not human ones. Grade `audit_sheet.csv` by hand "
        "and run `bench-compare` to check them.",
        "",
    ]

    judged = [r for r in rows if r["judge"].grade is not None]
    out += [
        "## Judge grades",
        "",
        "| grade | 0 | 1 | 2 | 3 | 4 |",
        "|---|---|---|---|---|---|",
        "| pairs | "
        + " | ".join(str(sum(r["judge"].grade == g for r in judged)) for g in range(5))
        + " |",
        "",
    ]
    if reference is None:
        out += [
            "## Next step",
            "",
            "Configure a reference grader to get agreement numbers: set "
            "`AZURE_REFERENCE_DEPLOYMENT` in `.env`, or sign in to the `claude` CLI and pass "
            "`--reference claude`. Or have another grader fill in `sheet.csv` and run "
            "`uv run bench-kappa <that sheet>`.",
            "",
        ]
        return "\n".join(out)

    both = [r for r in judged if r["ref"].grade is not None]

    def cut(pred) -> str:
        sel = [r for r in both if pred(r["pair"])]
        return _agree([r["ref"].grade for r in sel], [r["judge"].grade for r in sel])

    out += [
        "## Judge vs reference",
        "",
        "| Slice | Agreement |",
        "|---|---|",
        f"| All | {cut(lambda p: True)} |",
        f"| Test split (never inspected) | {cut(lambda p: p.split == 'test')} |",
        f"| Dev split | {cut(lambda p: p.split == 'dev')} |",
        f"| Finance students (IB, markets) | {cut(lambda p: p.group == 'finance')} |",
        f"| Other students | {cut(lambda p: p.group == 'other')} |",
    ]
    for track in sorted({p.profile.track for p in batch}):
        out.append(f"| Track: {track} | {cut(lambda p, t=track: p.profile.track == t)} |")
    out.append("")

    dev = [r for r in both if r["pair"].split == "dev" and r["ref"].grade != r["judge"].grade]
    kinds: dict[str, list] = defaultdict(list)
    for r in dev:
        ref, jud = r["ref"].grade, r["judge"].grade
        if ref == 0:
            kinds[f"Judge missed a disqualifier ({r['ref'].disqualifier or '?'})"].append(r)
        elif jud == 0:
            kinds["Judge disqualified a pair the reference allows"].append(r)
        elif jud > ref:
            kinds["Judge too generous on fit"].append(r)
        else:
            kinds["Judge too harsh on fit"].append(r)
    out += ["## Where they disagree (dev split only)", ""]
    if not dev:
        out.append("No disagreements on dev pairs.")
    for kind, rs in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
        out += [f"### {kind}: {len(rs)}", ""]
        for r in sorted(rs, key=lambda r: -abs(r["ref"].grade - r["judge"].grade))[:6]:
            p = r["pair"]
            on = "on-track" if same_track(p.posting.track, p.profile.track) else "off-track"
            out.append(
                f"- {p.profile.id} ({p.profile.track}, {p.profile.class_label}) × "
                f"{p.posting.company}: {p.posting.title[:60]} ({on}). "
                f"Reference {r['ref'].grade}: {r['ref'].reason[:110]} "
                f"Judge {r['judge'].grade}: {r['judge'].reason[:110]}"
            )
        out.append("")
    out += [
        "## Proposed next steps",
        "",
        "These are suggestions for a person to review. Nothing was changed.",
        "",
    ]
    for kind, rs in sorted(kinds.items(), key=lambda kv: -len(kv[1]))[:3]:
        out.append(
            f"- **{kind}** ({len(rs)} dev pairs): read the examples above and look for "
            "one general rule; add it with a unit test, then re-score dev before test."
        )
    return "\n".join(out) + "\n"
