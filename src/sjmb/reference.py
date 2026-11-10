"""Reference grader: a stronger model that grades a pair from the full posting text.

It produces *reference labels* for large batches. They are AI labels, not human ones: every
report says so, and a human audit sample is what ties them to a person's judgment.

Backends:
- `azure`:  a chat deployment named by AZURE_REFERENCE_DEPLOYMENT (for example gpt-4.1). When the
            posting text is thin and a screenshot exists, the screenshot is attached.
- `claude`: headless Claude Code (`claude -p --model sonnet`), text only, on the user's own
            subscription.

The grader is blind to the judge. It uses the same written rubric and returns the same JSON,
plus one quoted line of evidence. Answers are cached on disk by content hash.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sjmb import azure
from sjmb.capture import MIN_TEXT, best_text, load_capture
from sjmb.judge.judges import _JSON_OBJECT, Judgment, parse_judgment, render_prompt
from sjmb.judge.rubric import RUBRIC_VERSION
from sjmb.types import Posting, Profile

REFERENCE_TEXT_CHARS = 12000
EVIDENCE_LINE = (
    '\nAlso add an "evidence" field: one short phrase quoted from the posting that decided '
    "your answer, or an empty string."
)


CLAUDE_DENIED_TOOLS = "Bash Edit Write Read Glob Grep WebFetch WebSearch Agent NotebookEdit Skill"


class ReferenceUnavailable(RuntimeError):
    pass


@dataclass
class ReferenceGrader:
    backend: str  # azure | claude
    as_of: date
    max_calls: int = 300
    model: str = ""
    calls: int = 0
    llm: object = None  # injected in tests

    def __post_init__(self) -> None:
        if self.backend == "azure":
            self.model = self.model or os.environ.get("AZURE_REFERENCE_DEPLOYMENT", "")
            if not self.model and self.llm is None:
                raise ReferenceUnavailable("set AZURE_REFERENCE_DEPLOYMENT in .env")
            self.llm = self.llm or azure.client()
        elif self.backend == "claude":
            self.model = self.model or "sonnet"
            if self.llm is None and not shutil.which("claude"):
                raise ReferenceUnavailable("the `claude` CLI is not on PATH")
        else:
            raise ValueError(f"unknown reference backend {self.backend!r}")
        self.name = f"{self.backend}:{self.model}"
        self.cache_path = azure.cache_root() / "reference.jsonl"
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row

    def _posting_for_prompt(self, posting: Posting) -> tuple[Posting, str | None]:
        """The posting with the fullest text we have, and a screenshot path if text is thin."""
        text, _ = best_text(posting.url, posting.description)
        full = Posting.from_dict({**posting.to_dict(), "description": text[:REFERENCE_TEXT_CHARS]})
        shot = None
        if len(text) < MIN_TEXT:
            cap = load_capture(posting.url)
            if cap and cap.screenshot and Path(cap.screenshot).exists():
                shot = cap.screenshot
        return full, shot

    def _key(self, profile: Profile, posting: Posting) -> str:
        blob = json.dumps(
            [
                self.name,
                RUBRIC_VERSION,
                self.as_of.isoformat(),
                profile.to_dict(),
                posting.to_dict(),
            ],
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def is_cached(self, profile: Profile, posting: Posting) -> bool:
        return self._key(profile, self._posting_for_prompt(posting)[0]) in self.cache

    def estimate_cost(self, pairs: list[tuple[Profile, Posting]]) -> float:
        if self.backend != "azure":
            return 0.0  # runs on the Claude subscription, not on Azure credit
        tokens = sum(
            azure.estimate_tokens(render_prompt(pr, self._posting_for_prompt(po)[0], self.as_of))
            for pr, po in pairs
        )
        return azure.estimate_cost(self.model, tokens, 120 * len(pairs))

    def grade(self, profile: Profile, posting: Posting) -> Judgment:
        full, shot = self._posting_for_prompt(posting)
        key = self._key(profile, full)
        if key in self.cache:
            row = self.cache[key]
            return Judgment(
                row["grade"], row.get("reason", ""), True, row.get("fit"), row.get("disqualifier")
            )
        if self.calls >= self.max_calls:
            raise ReferenceUnavailable(f"reference budget of {self.max_calls} calls reached")
        prompt = render_prompt(profile, full, self.as_of, max_chars=REFERENCE_TEXT_CHARS)
        prompt += EVIDENCE_LINE
        raw = self._ask(prompt, shot)
        self.calls += 1
        try:
            j = parse_judgment(raw)
        except ValueError:
            j = Judgment(None, "unparsable reference reply")
        evidence = ""
        try:
            m = _JSON_OBJECT.search(raw)
            evidence = str(json.loads(m.group(0)).get("evidence", ""))[:200] if m else ""
        except (ValueError, AttributeError):
            pass
        row = {
            "key": key,
            "grader": self.name,
            "profile": profile.id,
            "posting": posting.id,
            "grade": j.grade,
            "fit": j.fit,
            "disqualifier": j.disqualifier,
            "reason": j.reason,
            "evidence": evidence,
        }
        if j.grade is None:
            row["raw"] = raw[:4000]
        with self.cache_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.cache[key] = row
        return j

    def _ask(self, prompt: str, screenshot: str | None) -> str:
        if self.backend == "azure":
            content: object = prompt
            if screenshot:
                b64 = base64.b64encode(Path(screenshot).read_bytes()).decode()
                content = [
                    {
                        "type": "text",
                        "text": prompt + "\nThe posting page is attached as an image.",
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]
            resp = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_completion_tokens=3000,  # room for reasoning models
            )
            return resp.choices[0].message.content or ""
        if self.llm is not None:  # test double
            return self.llm(prompt)
        # Headless Claude Code. The prompt goes on stdin; tools are disabled so the model can
        # only answer, and posting text (untrusted) cannot make it act.
        out = subprocess.run(
            [
                "claude",
                "-p",
                "--model",
                self.model,
                "--strict-mcp-config",
                "--no-session-persistence",
                "--disallowedTools",
                CLAUDE_DENIED_TOOLS,
                "--system-prompt",
                "You grade job postings. Reply with JSON only, no prose.",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
            cwd=azure.cache_root(),  # not a project folder: no project instructions load
        )
        if out.returncode != 0:
            raise ReferenceUnavailable(f"claude exited {out.returncode}: {out.stderr[:200]}")
        return out.stdout
