"""Fetch the full text of a posting from its own page.

SimplifyJobs rows carry only a title and company, so eligibility rules that live in the posting
(graduation windows, citizenship, program restrictions) are invisible to both the extractor and
the judge. This module fetches the employer's page and returns its description text.

Handlers, most specific first:
- Workday:     the public `wday/cxs` JSON endpoint behind every myworkdayjobs.com page
- Greenhouse:  boards-api.greenhouse.io
- Lever:       api.lever.co
- anything else: the schema.org JobPosting JSON-LD that most career pages embed, falling back
  to the page's visible text

Responses are cached on disk next to the other raw data, so each URL is fetched once.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from sjmb.data.extract import strip_html
from sjmb.data.loaders import USER_AGENT, cache_dir

_REMOVED = re.compile(
    r"no longer (available|accepting|open)|position has been filled|job (is )?not found|"
    r"this job (has )?(expired|closed)|posting has (closed|expired)|page (you requested )?"
    r"(could not be|cannot be) found",
    re.I,
)


@dataclass
class Fetched:
    text: str  # plain-text description, "" when nothing usable was found
    removed: bool  # the employer's page says the posting is gone
    method: str  # which handler produced it, for reporting


def _workday(client: httpx.Client, url: str) -> Fetched | None:
    u = urlparse(url)
    if "myworkdayjobs.com" not in u.netloc:
        return None
    tenant = u.netloc.split(".")[0]
    parts = [p for p in u.path.split("/") if p]
    if "job" not in parts:
        return None
    i = parts.index("job")
    site = parts[i - 1]
    rest = "/".join(parts[i + 1 :])
    api = f"https://{u.netloc}/wday/cxs/{tenant}/{site}/job/{rest}"
    r = client.get(api, headers={"Accept": "application/json"})
    if r.status_code in (404, 410):
        return Fetched("", True, "workday")
    r.raise_for_status()
    info = r.json().get("jobPostingInfo", {})
    text = strip_html(info.get("jobDescription", ""))
    return Fetched(text, not info.get("canApply", True) and not text, "workday")


def _greenhouse(client: httpx.Client, url: str) -> Fetched | None:
    m = re.search(r"greenhouse\.io/(?:embed/job_app\?for=)?([\w-]+)/jobs/(\d+)", url)
    if not m:
        return None
    r = client.get(f"https://boards-api.greenhouse.io/v1/boards/{m[1]}/jobs/{m[2]}")
    if r.status_code in (404, 410):
        return Fetched("", True, "greenhouse")
    r.raise_for_status()
    return Fetched(strip_html(r.json().get("content", "")), False, "greenhouse")


def _lever(client: httpx.Client, url: str) -> Fetched | None:
    m = re.search(r"jobs\.lever\.co/([\w-]+)/([\w-]{36})", url)
    if not m:
        return None
    r = client.get(f"https://api.lever.co/v0/postings/{m[1]}/{m[2]}")
    if r.status_code in (404, 410):
        return Fetched("", True, "lever")
    r.raise_for_status()
    j = r.json()
    parts = [j.get("descriptionPlain", "")]
    parts += [
        f"{x.get('text', '')}: {strip_html(x.get('content', ''))}" for x in j.get("lists", [])
    ]
    parts.append(j.get("additionalPlain", ""))
    return Fetched(" ".join(" ".join(parts).split()), False, "lever")


def _jsonld_description(html: str) -> str:
    for block in re.findall(
        r"<script[^>]+application/ld\+json[^>]*>(.*?)</script>", html, re.S | re.I
    ):
        try:
            data = json.loads(block.strip())
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else data.get("@graph", [data])
        for item in items:
            if isinstance(item, dict) and item.get("@type") == "JobPosting":
                return strip_html(item.get("description", ""))
    return ""


def _generic(client: httpx.Client, url: str) -> Fetched:
    r = client.get(url)
    if r.status_code in (404, 410):
        return Fetched("", True, "generic")
    r.raise_for_status()
    html = r.text
    text = _jsonld_description(html)
    method = "jsonld"
    if not text:
        body = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
        text, method = strip_html(body), "html"
    return Fetched(text, bool(_REMOVED.search(text[:3000])) and len(text) < 3000, method)


def fetch_posting_text(url: str, max_chars: int = 4000) -> Fetched:
    """Fetch (or read from cache) the description behind a posting URL. Never raises on HTTP
    or network errors: a failure comes back as empty text so callers keep the old row."""
    path = cache_dir() / "pages" / f"{hashlib.sha256(url.encode()).hexdigest()[:24]}.json"
    if path.exists():
        d = json.loads(path.read_text())
        return Fetched(d["text"], d["removed"], d["method"])
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.Client(
            timeout=30, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        ) as client:
            got = None
            for handler in (_workday, _greenhouse, _lever):
                got = handler(client, url)
                if got is not None:
                    break
            if got is None:
                got = _generic(client, url)
    except (httpx.HTTPError, ValueError) as e:
        return Fetched("", False, f"error: {type(e).__name__}")
    got = Fetched(" ".join(got.text.split())[:max_chars], got.removed, got.method)
    path.write_text(json.dumps({"url": url, **got.__dict__}))
    return got


def fill_from_pages(posts: list, only_source: str = "simplify") -> dict[str, int]:
    """Replace thin descriptions with the employer page's text and re-extract eligibility fields.

    Source-provided fields (Simplify's role type, terms, sponsorship) are kept; fields the
    source does not provide are re-extracted from the fuller text. A page that says the posting
    is gone marks it inactive.
    """
    from sjmb.data.extract import extract_grad_years, extract_sponsorship

    stats = {"fetched": 0, "removed": 0, "unavailable": 0}
    for p in posts:
        if p.source != only_source:
            continue
        got = fetch_posting_text(p.url)
        if got.removed:
            p.active = False
            stats["removed"] += 1
            continue
        if len(got.text) < 300:
            stats["unavailable"] += 1
            continue
        p.description = got.text
        full = f"{p.title}\n{p.description}"
        p.grad_years = extract_grad_years(full)
        if p.sponsorship == "unknown":
            p.sponsorship = extract_sponsorship(full)
        stats["fetched"] += 1
    return stats
