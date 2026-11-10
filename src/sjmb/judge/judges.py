"""Relevance judges: a deterministic stub and an Azure AI Foundry LLM judge.

The stub applies the rubric with rules (eligibility, track, skill overlap). It exists so the
whole pipeline runs offline and in CI. Its grades share logic with the eligibility filter, so
stub results are a smoke test of the plumbing, not a measurement of ranking quality.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from sjmb import azure
from sjmb.azure import cache_root
from sjmb.filters.eligibility import violations
from sjmb.judge.rubric import PROMPT, RUBRIC, RUBRIC_VERSION
from sjmb.retrievers.base import tokenize
from sjmb.types import Posting, Profile

# Tracks that count as "adjacent" (grade 2) rather than a mismatch.
ADJACENT = {
    "ib": {"markets", "consulting"},
    "markets": {"ib", "quant"},
    "quant": {"markets", "data", "swe"},
    "consulting": {"ib", "product"},
    "swe": {"data", "quant", "product"},
    "data": {"swe", "quant", "product"},
    "product": {"swe", "data", "consulting"},
}


@dataclass
class Judgment:
    grade: int | None  # None = the judge reply could not be parsed; excluded from scoring
    reason: str = ""
    cached: bool = False


class Judge(Protocol):
    name: str
    calls: int

    def grade(self, profile: Profile, posting: Posting) -> Judgment: ...


class StubJudge:
    name = "stub"

    def __init__(self, as_of: date):
        self.as_of = as_of
        self.calls = 0

    def grade(self, profile: Profile, posting: Posting) -> Judgment:
        self.calls += 1
        v = violations(profile, posting, self.as_of)
        if v:
            return Judgment(0, "ineligible: " + ", ".join(v))
        text = set(tokenize(posting.text()))
        skill_tokens = [set(tokenize(s)) for s in profile.skills]
        skill_hits = sum(1 for toks in skill_tokens if toks and toks <= text)
        overlap = skill_hits / max(1, len(profile.skills))
        if posting.track == profile.track:
            if overlap >= 0.4:
                return Judgment(4, f"on track, {skill_hits} skills match")
            return Judgment(3, f"on track, {skill_hits} skills match")
        if posting.track in ADJACENT[profile.track]:
            return Judgment(2, f"adjacent track {posting.track}")
        return Judgment(1, f"off track {posting.track}")


def hard_check(profile: Profile, posting: Posting) -> str | None:
    """Eligibility facts that code checks better than the LLM. Runs before any judge call.

    Return a short reason (e.g. "grad year 2029 not in posting window [2028]") when the student
    is clearly ineligible, or None to let the LLM judge the pair.

    The posting fields come from the extractor (sjmb/data/extract.py):
      posting.grad_years  -> list[int], e.g. [2027, 2028]; empty list means "not stated"
      posting.role_type   -> "internship" | "new_grad" | "experienced" | "unknown"
      posting.sponsorship -> "offers" | "no" | "us_persons" | "citizens_only" | "unknown"
    The student fields: profile.grad_year (int), profile.seeking ("internship" | "new_grad"),
      profile.work_auth ("citizen" | "permanent_resident" | "needs_sponsorship").

    Rule of thumb from the golden set: a field the posting does not state never disqualifies.
    """
    # Graduation window: only when the posting states one (empty list = not stated).
    if posting.grad_years and profile.grad_year not in posting.grad_years:
        window = ", ".join(str(y) for y in posting.grad_years)
        return f"grad year {profile.grad_year} not in posting window [{window}]"
    return None


# City aliases seen in posting locations. Keys are lowercase city names from the profiles.
_CITY_ALIASES = {
    "new york": ("new york", "nyc", "manhattan", "brooklyn"),
    "san francisco": ("san francisco", "sf bay", "bay area"),
    "chicago": ("chicago",),
    "boston": ("boston", "cambridge, ma"),
    "seattle": ("seattle", "bellevue"),
    "austin": ("austin",),
    "charlotte": ("charlotte",),
}


def location_match(profile: Profile, posting: Posting) -> bool | None:
    """Is the posting in one of the student's preferred places? None when it has no location."""
    loc = posting.location.lower()
    if not loc.strip():
        return None
    for pref in profile.locations:
        city = pref.split(",")[0].strip().lower()
        if city == "remote":
            if "remote" in loc:
                return True
            continue
        if any(alias in loc for alias in _CITY_ALIASES.get(city, (city,))):
            return True
    return False


def apply_fit_caps(profile: Profile, posting: Posting, j: Judgment) -> Judgment:
    """Rubric rules that code enforces on the judge's fit grade:
    - 3 and 4 require the student's target track, so another track caps at 2. The track is
      the extractor's (posting.track); "other" means the extractor found no track, so no cap.
    - a 4 needs one of the student's preferred locations (golden set: every human 4 has one,
      and 8 of 10 human 3s are right-track roles in other cities).
    """
    if j.grade is None or j.grade == 0:
        return j
    if j.grade > 2 and posting.track not in ("other", profile.track):
        return Judgment(
            2, f"[capped: {posting.track} is not the {profile.track} track] {j.reason}", j.cached
        )
    if j.grade == 4 and location_match(profile, posting) is False:
        return Judgment(3, f"[capped: not a preferred location] {j.reason}", j.cached)
    return j


class AzureJudge:
    """Chat completions on an Azure AI Foundry deployment through the OpenAI v1 route, with
    `model` set to the deployment name.

    Every judgment is cached on disk, keyed by a hash of (rubric version, deployment, profile,
    posting), so a re-run never pays twice. `max_calls` is a hard cap on uncached calls.
    """

    OUTPUT_TOKENS = 60  # a short JSON reply; used for cost estimates

    def __init__(self, as_of: date, max_calls: int = 500, cache_path: Path | None = None, llm=None):
        self.as_of = as_of
        self.deployment = os.environ["AZURE_JUDGE_DEPLOYMENT"]
        self.name = f"azure:{self.deployment}"
        self.max_calls = max_calls
        self.calls = 0
        self.cache_path = cache_path or cache_root() / "judgments.jsonl"
        self.cache: dict[str, Judgment] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                row = json.loads(line)
                grade, reason = row["grade"], row.get("reason", "")
                if grade is None and row.get("raw"):
                    # A failed parse keeps the raw reply, so parser fixes apply for free.
                    try:
                        parsed = parse_judgment(row["raw"])
                        grade, reason = parsed.grade, parsed.reason
                    except ValueError:
                        pass
                self.cache[row["key"]] = Judgment(grade, reason, True)
        self.llm = llm or azure.client()

    def _key(self, profile: Profile, posting: Posting) -> str:
        blob = json.dumps(
            [
                RUBRIC_VERSION,
                self.deployment,
                self.as_of.isoformat(),
                profile.to_dict(),
                posting.to_dict(),
            ],
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def is_cached(self, profile: Profile, posting: Posting) -> bool:
        return self._key(profile, posting) in self.cache

    def estimate_cost(self, pairs: list[tuple[Profile, Posting]]) -> float:
        tokens = sum(azure.estimate_tokens(render_prompt(pr, po, self.as_of)) for pr, po in pairs)
        return azure.estimate_cost(self.deployment, tokens, self.OUTPUT_TOKENS * len(pairs))

    def grade(self, profile: Profile, posting: Posting) -> Judgment:
        # Code checks first: they override any cached LLM answer and cost nothing.
        reason = hard_check(profile, posting)
        if reason is not None:
            return Judgment(0, f"[code] {reason}")
        key = self._key(profile, posting)
        if key in self.cache:
            return apply_fit_caps(profile, posting, self.cache[key])
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"judge budget of {self.max_calls} uncached calls reached")
        prompt = render_prompt(profile, posting, self.as_of)
        j = None
        raw = ""
        for max_tokens in (400, 1500):  # reasoning deployments can spend the first budget
            resp = self.llm.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
            )
            self.calls += 1
            raw = resp.choices[0].message.content or ""
            try:
                j = parse_judgment(raw)
                break
            except ValueError:
                continue
        if j is None:
            # Cache the failure too, so a re-run does not pay for the same pair again.
            j = Judgment(None, "unparsable judge reply")
        row = {
            "key": key,
            "grade": j.grade,
            "reason": j.reason,
            "profile": profile.id,
            "posting": posting.id,
        }
        if j.grade is None:
            row["raw"] = raw[:4000]
        with self.cache_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.cache[key] = j
        return apply_fit_caps(profile, posting, j)


class BudgetExceeded(RuntimeError):
    pass


def render_prompt(profile: Profile, posting: Posting, as_of: date) -> str:
    return PROMPT.format(
        rubric=RUBRIC,
        as_of=as_of.isoformat(),
        role_type=posting.role_type,
        grad_years=", ".join(map(str, posting.grad_years)) or "not stated",
        sponsorship={
            "offers": "sponsorship offered",
            "no": "no visa sponsorship",
            "us_persons": "U.S. citizens or permanent residents only",
            "citizens_only": "U.S. citizens only",
            "unknown": "not stated",
        }[posting.sponsorship],
        track=profile.track,
        seeking={
            "internship": "an internship",
            "new_grad": "a new-grad role (full-time, starting after graduation)",
        }[profile.seeking],
        grad_year=profile.grad_year,
        class_label=profile.class_label,
        major=profile.major,
        skills=", ".join(profile.skills),
        interests=profile.interests,
        work_auth=profile.work_auth,
        locations=", ".join(profile.locations),
        company=posting.company,
        title=posting.title,
        location=posting.location,
        terms=", ".join(posting.terms) or "not stated",
        posted_at=posting.posted_at or "unknown",
        deadline=posting.deadline or "not stated",
        active=posting.active,
        description=posting.description[:4000],
    )


DISQUALIFIERS = set("abcdefg")
_GRADE = re.compile(r'"grade"\s*:\s*([0-4])')


def parse_judgment(content: str) -> Judgment:
    try:
        data = json.loads(content.strip().removeprefix("```json").removesuffix("```").strip())
        reason = str(data.get("reason", ""))[:300]
        if "fit" in data:  # v4+: code, not the model, turns a disqualifier into a 0
            fit = int(data["fit"])
            dq = str(data.get("disqualifier") or "none").strip().lower()
            if dq[:1] in DISQUALIFIERS and (len(dq) == 1 or not dq[1].isalpha()):
                return Judgment(0, f"[{dq[:1]}] {reason}")  # any fit, even 0
            if 0 <= fit <= 4:
                # No disqualifier means eligible, so the grade is at least 1.
                return Judgment(max(fit, 1), reason)
        grade = int(data["grade"])
        if 0 <= grade <= 4:
            return Judgment(grade, reason)
    except (ValueError, KeyError, TypeError):
        pass
    m = _GRADE.search(content)
    if m:
        return Judgment(int(m.group(1)), "parsed from non-JSON reply")
    raise ValueError(f"judge reply has no grade: {content[:200]!r}")


def make_judge(kind: str, as_of: date, max_calls: int = 500) -> Judge:
    if kind == "stub":
        return StubJudge(as_of)
    if kind == "azure":
        missing = [
            v
            for v in ("AZURE_AI_ENDPOINT", "AZURE_AI_KEY", "AZURE_JUDGE_DEPLOYMENT")
            if not os.environ.get(v)
        ]
        if missing:
            raise SystemExit(f"--judge azure needs {', '.join(missing)} (see .env.example)")
        return AzureJudge(as_of, max_calls=max_calls)
    raise ValueError(f"unknown judge {kind!r}")
