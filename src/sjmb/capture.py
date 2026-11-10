"""Capture a posting page in a real browser: visible text plus a full-page screenshot.

For career sites that only render in a browser, where the plain fetcher (`data/fetch_text.py`)
gets nothing. The browser is ordinary Chrome driven by Playwright (`uv sync --extra capture`):
no stealth plugins, no altered identity, no CAPTCHA solving. A page that shows a bot check or a
sign-in wall is recorded as `blocked` and left alone.

Captures are cached per URL, so a run can be stopped and resumed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

from sjmb.data.fetch_text import UnsafeURL, check_public_url
from sjmb.data.loaders import cache_dir

# Signs that the site is asking for a human or a login. We stop there.
BLOCK_SIGNS = re.compile(
    r"verify (that )?you are (a )?human|are you a robot|captcha|unusual traffic|"
    r"checking your browser|just a moment\.\.\.|access denied|request blocked|"
    r"enable javascript and cookies|sign in to (continue|view)|log in to (continue|view)",
    re.I,
)
MIN_TEXT = 300  # fewer visible characters than this is not a usable posting
HOST_DELAY_S = 5.0  # pause between two pages on the same site
MAX_TEXT = 12000


@dataclass
class Capture:
    url: str
    status: str  # ok | thin | blocked | error | unsafe
    text: str = ""
    final_url: str = ""
    screenshot: str = ""  # path to a PNG, "" when none was taken
    detail: str = ""


def _dir() -> Path:
    d = cache_dir() / "captures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:24]


def load_capture(url: str) -> Capture | None:
    path = _dir() / f"{_key(url)}.json"
    return Capture(**json.loads(path.read_text())) if path.exists() else None


def classify(text: str) -> str:
    """Decide what a captured page is from its visible text."""
    if BLOCK_SIGNS.search(text[:2000]) and len(text) < 3000:
        return "blocked"
    return "ok" if len(text) >= MIN_TEXT else "thin"


def _save(c: Capture) -> Capture:
    (_dir() / f"{_key(c.url)}.json").write_text(json.dumps(asdict(c)))
    return c


def capture_urls(
    urls: list[str], headed: bool = False, progress=lambda _: None, retry_failed: bool = False
) -> list[Capture]:
    """Capture each URL not already cached. Returns one Capture per input URL, in order."""
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise SystemExit("capture needs Playwright: run `uv sync --extra capture`") from e

    done: dict[str, Capture] = {}
    todo = []
    for u in dict.fromkeys(urls):
        cached = load_capture(u)
        if cached and (cached.status in ("ok", "blocked", "unsafe") or not retry_failed):
            done[u] = cached
        else:
            todo.append(u)
    if todo:
        last_hit: dict[str, float] = {}
        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(channel="chrome", headless=not headed)
            except PlaywrightError:
                browser = pw.chromium.launch(headless=not headed)  # Playwright's own Chromium
            context = browser.new_context(
                viewport={"width": 1280, "height": 1600}, service_workers="block"
            )
            context.route("**/*", _public_only_route())  # every request, redirect and subresource
            page = context.new_page()
            for n, url in enumerate(todo, 1):
                host = urlparse(url).hostname or ""
                wait = HOST_DELAY_S - (time.time() - last_hit.get(host, 0))
                if wait > 0:
                    time.sleep(wait)
                done[url] = _save(_capture_one(page, url, PlaywrightError))
                last_hit[host] = time.time()
                progress(f"captured {n}/{len(todo)}: {done[url].status:8} {url[:70]}")
            browser.close()
    return [done[u] for u in urls]


def _public_only_route():
    """A Playwright route handler that aborts any request to a non-public address.

    Posting URLs come from third-party lists, and a page can redirect or load subresources
    anywhere, so the check runs on every request the browser makes. Host verdicts are cached
    for the run. Residual risk: a host that re-resolves to a private address between this
    check and the browser's own lookup (DNS rebinding) is not covered.
    """
    verdict: dict[str, bool] = {}

    def handle(route) -> None:
        req_url = route.request.url
        if req_url.startswith(("data:", "blob:", "about:")):
            route.continue_()
            return
        host = urlparse(req_url).hostname or ""
        if host not in verdict:
            try:
                check_public_url(req_url)
                verdict[host] = True
            except UnsafeURL:
                verdict[host] = False
        if verdict[host]:
            route.continue_()
        else:
            route.abort("blockedbyclient")

    return handle


def view_url(url: str) -> str:
    """The page that shows the job description. Workable's `/apply` path is only the form."""
    u = urlparse(url)
    if (u.hostname or "").endswith("workable.com") and u.path.rstrip("/").endswith("/apply"):
        return url.split("?")[0].rstrip("/").removesuffix("/apply")
    return url


def _capture_one(page, url: str, playwright_error) -> Capture:
    try:
        check_public_url(url)
    except UnsafeURL as e:
        return Capture(url, "unsafe", detail=str(e))
    try:
        page.goto(view_url(url), wait_until="domcontentloaded", timeout=45000)
        with contextlib.suppress(playwright_error):  # busy pages never go idle
            page.wait_for_load_state("networkidle", timeout=10000)
        check_public_url(page.url)  # a redirect must also land on a public host
        text = " ".join(page.inner_text("body").split())[:MAX_TEXT]
        status = classify(text)
        shot = ""
        if status != "blocked":
            shot = str(_dir() / f"{_key(url)}.png")
            page.screenshot(path=shot, full_page=True)
        return Capture(url, status, text if status != "blocked" else "", page.url, shot)
    except UnsafeURL as e:
        return Capture(url, "unsafe", detail=str(e))
    except playwright_error as e:
        return Capture(url, "error", detail=str(e).splitlines()[0][:200])


def best_text(url: str, fallback: str) -> tuple[str, str]:
    """(text, where it came from) for a posting: a good capture wins over thin stored text."""
    c = load_capture(url)
    if c and c.status == "ok" and len(c.text) > len(fallback):
        return c.text, "capture"
    return fallback, "stored"


def apply_captures(posts: list) -> int:
    """Give postings with thin stored text their captured text, and re-read eligibility fields."""
    from sjmb.data.extract import extract_grad_years, extract_sponsorship
    from sjmb.data.loaders import MAX_DESCRIPTION_CHARS

    changed = 0
    for p in posts:
        if len(p.description) >= MIN_TEXT:
            continue
        text, source = best_text(p.url, p.description)
        if source != "capture":
            continue
        p.description = text[:MAX_DESCRIPTION_CHARS]
        full = f"{p.title}\n{p.description}"
        p.grad_years = extract_grad_years(full)
        if p.sponsorship == "unknown":
            p.sponsorship = extract_sponsorship(full)
        changed += 1
    return changed
