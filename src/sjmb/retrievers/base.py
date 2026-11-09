from __future__ import annotations

import re
from typing import Protocol

import numpy as np

from sjmb.types import Posting

_TOKEN = re.compile(r"[a-z0-9+#&]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or our that the their this "
    "to we will with you your who what about all can do into more not other such than they "
    "us was were which while within".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class Retriever(Protocol):
    name: str

    def fit(self, postings: list[Posting]) -> Retriever: ...

    def scores(self, query: str) -> np.ndarray:
        """Exact score for every posting (higher is better). No approximate search."""
        ...


def top_k(scores: np.ndarray, k: int, allowed: np.ndarray | None = None) -> list[int]:
    """Indices of the k best scores, optionally restricted to a boolean mask."""
    s = scores.astype(float).copy()
    if allowed is not None:
        s[~allowed] = -np.inf
    k = min(k, int(np.isfinite(s).sum()))
    if k <= 0:
        return []
    # Full stable sort: ties (common with BM25 zeros) break by index, the same on every numpy.
    order = np.argsort(-s, kind="stable")[:k]
    return [int(i) for i in order]
