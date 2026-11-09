"""Human golden set: sample judged pairs into a blind grading sheet, then score agreement.

The sheet hides the judge's grade and the filter's verdict so the grader is not anchored. Both
are recomputed at scoring time: the judge grade from the on-disk cache, the verdict from the
eligibility rules.
"""

from __future__ import annotations

import csv
import io
import json
import random
from collections import defaultdict
from datetime import date
from pathlib import Path

from sjmb.filters.eligibility import violations
from sjmb.types import Posting, Profile

RUBRIC_LINES = [
    "4 Excellent: eligible, right track and level, most core skills match.",
    "3 Good: eligible, right track and level, partial skill overlap or weaker location fit.",
    "2 Fair: eligible, but an adjacent track or thin overlap.",
    "1 Poor: eligible on paper, but a clear track or skill mismatch; the student would not apply.",
    "0 Unusable: cannot or should not apply (senior-only, wrong level, excluded grad year, "
    "work authorization, closed or past deadline).",
]
COLUMNS = ["pair_no", "profile_id", "posting_id", "student", "job", "your_grade", "notes"]

# Disagreement strata between the eligibility filter and the judge.
FILTER_OUT_JUDGE_OK = "filter excludes, judge grades >= 2"
FILTER_OK_JUDGE_ZERO = "filter allows, judge grades 0"


def stratum(profile: Profile, posting: Posting, grade: int, as_of: date) -> str | None:
    ineligible = bool(violations(profile, posting, as_of))
    if ineligible and grade >= 2:
        return FILTER_OUT_JUDGE_OK
    if not ineligible and grade == 0:
        return FILTER_OK_JUDGE_ZERO
    return None


def band(grade: int) -> str:
    return "0-1" if grade <= 1 else "2" if grade == 2 else "3-4"


def _spread(cands: list[dict], k: int, rng: random.Random, taken: dict[tuple, int]) -> list[dict]:
    """Pick k rows, round-robin over (track, class year) cells, least-covered cells first."""
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for c in cands:
        cells[c["cell"]].append(c)
    for group in cells.values():
        rng.shuffle(group)
    out: list[dict] = []
    while len(out) < k and any(cells.values()):
        order = sorted((c for c in cells if cells[c]), key=lambda c: (taken[c], rng.random()))
        for cell in order:
            if len(out) == k:
                break
            out.append(cells[cell].pop())
            taken[cell] += 1
    return out


def sample_pairs(
    judgments: list[dict],
    profiles: dict[str, Profile],
    postings: dict[str, Posting],
    as_of: date,
    n: int = 100,
    n_disagree: int = 30,
    per_band: int = 20,
    seed: int = 7,
) -> list[dict]:
    """Stratified sample of judged pairs: disagreements in both directions, then each judge
    grade band, then fill to n, always spreading across track x class-year cells."""
    rng = random.Random(seed)
    seen: set[tuple[str, str]] = set()
    rows = []
    for j in judgments:
        key = (j["profile"], j["posting"])
        if j.get("grade") is None or key in seen:
            continue
        seen.add(key)
        prof, post = profiles[j["profile"]], postings[j["posting"]]
        rows.append(
            {
                "profile": prof,
                "posting": post,
                "grade": j["grade"],
                "stratum": stratum(prof, post, j["grade"], as_of),
                "band": band(j["grade"]),
                "cell": (prof.track, prof.class_label),
            }
        )
    rows.sort(key=lambda r: (r["profile"].id, r["posting"].id))  # input-order independent

    taken: dict[tuple, int] = defaultdict(int)
    chosen: list[dict] = []

    def remaining(pred) -> list[dict]:
        ids = {id(c) for c in chosen}
        return [r for r in rows if id(r) not in ids and pred(r)]

    half = n_disagree // 2
    out_ok = remaining(lambda r: r["stratum"] == FILTER_OUT_JUDGE_OK)
    ok_zero = remaining(lambda r: r["stratum"] == FILTER_OK_JUDGE_ZERO)
    k1 = min(len(out_ok), max(half, n_disagree - len(ok_zero)))
    chosen += _spread(out_ok, k1, rng, taken)
    chosen += _spread(ok_zero, min(len(ok_zero), n_disagree - k1), rng, taken)
    for b in ("0-1", "2", "3-4"):
        chosen += _spread(
            remaining(lambda r, b=b: r["stratum"] is None and r["band"] == b), per_band, rng, taken
        )
    chosen += _spread(remaining(lambda r: True), max(0, n - len(chosen)), rng, taken)
    rng.shuffle(chosen)  # strata must not show up as blocks in the sheet
    return chosen[:n]


def student_summary(p: Profile) -> str:
    return (
        f"{p.class_label}, class of {p.grad_year}; {p.major}; {p.track} track, seeking "
        f"{p.seeking.replace('_', '-')}; skills: {', '.join(p.skills)}; wants: {p.interests}; "
        f"locations: {', '.join(p.locations)}; work auth: {p.work_auth.replace('_', ' ')}"
    )


def job_summary(p: Posting, excerpt_chars: int = 240) -> str:
    text = " ".join(p.description.split())
    excerpt = text[:excerpt_chars].rsplit(" ", 1)[0] + ("..." if len(text) > excerpt_chars else "")
    grad = ", ".join(map(str, p.grad_years)) or "not stated"
    return (
        f"{p.company} | {p.title} | {p.location or 'location n/a'} | "
        f"level: {p.role_type.replace('_', '-')} | terms: {', '.join(p.terms) or 'n/a'} | "
        f"deadline: {p.deadline or 'not stated'} | active: {'yes' if p.active else 'no'} | "
        f"auto-extracted grad years: {grad}; work auth: {p.sponsorship.replace('_', ' ')} | "
        f"{excerpt} | {p.url}"
    )


def _cell(value: object) -> object:
    """Keep spreadsheet apps from running posting text as a formula (CSV injection)."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def write_sheet(pairs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    buf.write("# Grade each pair 0-4 in your_grade. Rubric (full text in golden/README.md):\n")
    for line in RUBRIC_LINES:
        buf.write(f"# {line}\n")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COLUMNS)
    for i, r in enumerate(pairs, 1):
        w.writerow(
            [
                i,
                r["profile"].id,
                r["posting"].id,
                _cell(student_summary(r["profile"])),
                _cell(job_summary(r["posting"])),
                "",
                "",
            ]
        )
    path.write_text(buf.getvalue())


def read_judgments(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def confusion_matrix(human: list[int], model: list[int], n_labels: int = 5) -> str:
    head = "human \\ judge | " + " | ".join(map(str, range(n_labels))) + " | total"
    lines = [head, "|".join(["---"] * (n_labels + 2))]
    for h in range(n_labels):
        counts = [
            sum(1 for a, b in zip(human, model, strict=True) if a == h and b == m)
            for m in range(n_labels)
        ]
        lines.append(f"{h} | " + " | ".join(map(str, counts)) + f" | {sum(counts)}")
    return "\n".join(lines)


def agreement_report(human: list[int], model: list[int], strata: list[str | None]) -> str:
    from sjmb.judge.agreement import cohen_kappa

    def summary(h: list[int], m: list[int]) -> list[str]:
        exact = sum(a == b for a, b in zip(h, m, strict=True)) / len(h)
        out = [f"pairs: {len(h)}", f"exact agreement: {exact:.3f}"]
        if len(set(h)) > 1 and len(set(m)) > 1:  # kappa is undefined if a rater is constant
            out += [
                f"cohen kappa (unweighted): {cohen_kappa(h, m):.3f}",
                f"cohen kappa (quadratic): {cohen_kappa(h, m, weights='quadratic'):.3f}",
            ]
        return out

    parts = [
        "## All graded pairs",
        *summary(human, model),
        "",
        "Confusion matrix (rows = human, columns = judge):",
        confusion_matrix(human, model),
    ]
    for name, human_sides_with_filter in (
        (FILTER_OUT_JUDGE_OK, lambda g: g <= 1),
        (FILTER_OK_JUDGE_ZERO, lambda g: g >= 1),
    ):
        idx = [i for i, s in enumerate(strata) if s == name]
        parts += ["", f"## Filter-judge disagreement: {name}"]
        if not idx:
            parts.append("no graded pairs in this subset")
            continue
        h = [human[i] for i in idx]
        m = [model[i] for i in idx]
        with_filter = sum(1 for g in h if human_sides_with_filter(g))
        parts += [
            *summary(h, m),
            f"human sides with the filter: {with_filter}/{len(h)}",
            f"human sides with the judge: {len(h) - with_filter}/{len(h)}",
        ]
    rest = [i for i, s in enumerate(strata) if s is None]
    if rest:
        parts += [
            "",
            "## Pairs where filter and judge agree",
            *summary([human[i] for i in rest], [model[i] for i in rest]),
        ]
    return "\n".join(parts)
