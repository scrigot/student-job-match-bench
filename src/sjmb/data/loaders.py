"""Loaders for public job boards and the SimplifyJobs lists.

Raw responses are cached outside git (default `~/.cache/student-job-match-bench/raw`, override
with SJMB_CACHE_DIR). Only `sjmb/data/fixtures/sample_postings.json` is committed.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import httpx

from sjmb.data.extract import enrich, strip_html
from sjmb.types import Posting, RoleType, Sponsorship

FIXTURE = Path(__file__).parent / "fixtures" / "sample_postings.json"

# Public board tokens for finance, consulting and tech firms (checked 2026-11-09).
GREENHOUSE_BOARDS = [
    "point72",
    "janestreet",
    "imc",
    "drweng",
    "wehrtyou",
    "squarepointcapital",
    "akunacapital",
    "jumptrading",
    "aqr",
    "virtu",
    "pdtpartners",
    "generalatlantic",
    "williamblair",
    "clearviewhealthcarepartners",
    "stripe",
    "robinhood",
    "databricks",
    "figma",
    "airbnb",
    "coinbase",
    "brex",
    "affirm",
    "chime",
    "sofi",
    "mongodb",
    "datadog",
]
LEVER_BOARDS = ["palantir", "wealthfront", "spotify", "belvederetrading"]
SIMPLIFY_LISTS = {
    "simplify-intern-2027": "https://raw.githubusercontent.com/SimplifyJobs/Summer2027-Internships/dev/.github/scripts/listings.json",
    "simplify-newgrad": "https://raw.githubusercontent.com/SimplifyJobs/New-Grad-Positions/dev/.github/scripts/listings.json",
}

USER_AGENT = "student-job-match-bench/0.1 (research benchmark)"
MAX_DESCRIPTION_CHARS = 4000


def cache_dir() -> Path:
    root = os.environ.get("SJMB_CACHE_DIR") or Path.home() / ".cache" / "student-job-match-bench"
    path = Path(root) / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_json(client: httpx.Client, url: str, dest: Path, max_age_hours: float) -> object:
    if dest.exists() and (time.time() - dest.stat().st_mtime) < max_age_hours * 3600:
        return json.loads(dest.read_text())
    resp = client.get(url)
    resp.raise_for_status()
    dest.write_text(resp.text)
    return resp.json()


def fetch_all(max_age_hours: float = 24.0) -> dict[str, Path]:
    """Download every source into the cache. Returns {source_key: path}."""
    out: dict[str, Path] = {}
    raw = cache_dir()
    with httpx.Client(timeout=60, headers={"User-Agent": USER_AGENT}, follow_redirects=True) as c:
        for token in GREENHOUSE_BOARDS:
            url = f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"
            dest = raw / f"greenhouse-{token}.json"
            try:
                _get_json(c, url, dest, max_age_hours)
                out[f"greenhouse-{token}"] = dest
            except httpx.HTTPError as e:
                print(f"skip greenhouse/{token}: {e}")
        for token in LEVER_BOARDS:
            url = f"https://api.lever.co/v0/postings/{token}?mode=json"
            dest = raw / f"lever-{token}.json"
            try:
                _get_json(c, url, dest, max_age_hours)
                out[f"lever-{token}"] = dest
            except httpx.HTTPError as e:
                print(f"skip lever/{token}: {e}")
        for key, url in SIMPLIFY_LISTS.items():
            dest = raw / f"{key}.json"
            try:
                _get_json(c, url, dest, max_age_hours)
                out[key] = dest
            except httpx.HTTPError as e:
                print(f"skip {key}: {e}")
    return out


def _iso(ts: str | int | float | None) -> str | None:
    if ts is None or ts == "":
        return None
    if isinstance(ts, int | float):
        secs = ts / 1000 if ts > 1e11 else ts  # Lever uses ms, Simplify uses s
        return datetime.fromtimestamp(secs, tz=UTC).date().isoformat()
    return str(ts)[:10]


def parse_greenhouse(token: str, payload: dict) -> list[Posting]:
    posts = []
    for j in payload.get("jobs", []):
        posts.append(
            Posting(
                id=f"gh-{token}-{j['id']}",
                source="greenhouse",
                company=(j.get("company_name") or token).strip(),
                title=j.get("title", "").strip(),
                location=(j.get("location") or {}).get("name", ""),
                url=j.get("absolute_url", ""),
                description=strip_html(j.get("content", ""))[:MAX_DESCRIPTION_CHARS],
                posted_at=_iso(j.get("first_published") or j.get("updated_at")),
                deadline=_iso(j.get("application_deadline")),
            )
        )
    return posts


def parse_lever(token: str, payload: list) -> list[Posting]:
    posts = []
    for j in payload:
        cats = j.get("categories") or {}
        body = " ".join(
            [j.get("descriptionPlain", "")]
            + [
                f"{lst.get('text', '')}: {strip_html(lst.get('content', ''))}"
                for lst in j.get("lists", [])
            ]
            + [j.get("additionalPlain", "")]
        )
        posts.append(
            Posting(
                id=f"lv-{token}-{j['id']}",
                source="lever",
                company=token,
                title=j.get("text", "").strip(),
                location=cats.get("location", "") or "",
                url=j.get("hostedUrl", ""),
                description=" ".join(body.split())[:MAX_DESCRIPTION_CHARS],
                posted_at=_iso(j.get("createdAt")),
            )
        )
    return posts


_SIMPLIFY_SPONSOR: dict[str, Sponsorship] = {
    "Offers Sponsorship": "offers",
    "Does Not Offer Sponsorship": "no",
    "U.S. Citizenship is Required": "citizens_only",
}


def parse_simplify(key: str, payload: list) -> list[Posting]:
    role: RoleType = "internship" if "intern" in key else "new_grad"
    posts = []
    for j in payload:
        if not j.get("is_visible", True):
            continue
        # Simplify rows carry no description; the title, company and category are the text.
        desc = f"{j.get('category', '')} role at {j.get('company_name', '')}. " + " ".join(
            j.get("terms", []) + j.get("degrees", [])
        )
        posts.append(
            Posting(
                id=f"sp-{j['id']}",
                source="simplify",
                company=j.get("company_name", "").strip(),
                title=j.get("title", "").strip(),
                location="; ".join(j.get("locations", [])),
                url=j.get("url", ""),
                description=desc.strip(),
                posted_at=_iso(j.get("date_posted")),
                active=bool(j.get("active", True)),
                role_type=role,
                sponsorship=_SIMPLIFY_SPONSOR.get(j.get("sponsorship", ""), "unknown"),
                terms=list(j.get("terms", [])),
            )
        )
    return posts


def load_cache(paths: dict[str, Path] | None = None) -> list[Posting]:
    """Parse every cached raw file into enriched postings."""
    if paths is None:
        paths = {p.stem: p for p in cache_dir().glob("*.json")}
    posts: list[Posting] = []
    for key, path in sorted(paths.items()):
        payload = json.loads(path.read_text())
        if key.startswith("greenhouse-"):
            posts += parse_greenhouse(key.removeprefix("greenhouse-"), payload)
        elif key.startswith("lever-"):
            posts += parse_lever(key.removeprefix("lever-"), payload)
        elif key.startswith("simplify-"):
            posts += parse_simplify(key, payload)
    return [enrich(p) for p in posts]


def load_fixture(path: Path = FIXTURE) -> list[Posting]:
    return [Posting.from_dict(d) for d in json.loads(path.read_text())]


def save_postings(posts: Iterable[Posting], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([p.to_dict() for p in posts], indent=1, ensure_ascii=False) + "\n")


# Per-track quotas for the Simplify slice of the sample. Finance tracks are rare in the raw
# lists, so they are over-sampled relative to their share to keep every slice populated.
SAMPLE_QUOTAS = {
    "ib": 10,
    "markets": 20,
    "consulting": 20,
    "quant": 20,
    "product": 12,
    "data": 15,
    "swe": 15,
    "other": 8,
}


def build_sample(
    posts: list[Posting], per_board: int = 4, seed: int = 7, desc_chars: int = MAX_DESCRIPTION_CHARS
) -> list[Posting]:
    """A small, mixed sample for the committed fixture.

    Keeps a few postings per board (student roles first) plus a track-stratified slice of
    active Simplify rows. Descriptions keep the same cap as the full cache, so the judge
    reads the whole posting.
    """
    rng = random.Random(seed)
    by_board: dict[str, list[Posting]] = {}
    by_track: dict[str, list[Posting]] = {}
    for p in posts:
        if p.source == "simplify":
            if p.active:
                by_track.setdefault(p.track, []).append(p)
        else:
            by_board.setdefault(p.id.split("-")[1], []).append(p)
    sample: list[Posting] = []
    student_roles = ("internship", "new_grad")
    for _, group in sorted(by_board.items()):
        student = [p for p in group if p.role_type in student_roles]
        other = [p for p in group if p.role_type not in student_roles]
        rng.shuffle(student)
        rng.shuffle(other)
        sample += (student + other)[:per_board]
    for track, quota in SAMPLE_QUOTAS.items():
        group = by_track.get(track, [])
        rng.shuffle(group)
        sample += group[:quota]
    for p in sample:
        p.description = p.description[:desc_chars]
    return sample
