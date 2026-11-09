"""Hybrid retrieval with reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009).

RRF ignores raw score scales, which matters here: BM25 scores are unbounded and cosine scores
sit in [-1, 1], so a weighted sum would need per-query calibration. RRF only needs ranks.
"""

from __future__ import annotations

import numpy as np

from sjmb.types import Posting


def rrf(rank_lists: list[np.ndarray], n: int, k: int = 60, depth: int | None = None) -> np.ndarray:
    """Fuse full score vectors by rank. Returns a fused score per document.

    `depth` limits each list to its top-`depth` ranks (documents below get no credit from it).
    """
    fused = np.zeros(n)
    for scores in rank_lists:
        order = np.argsort(-scores, kind="stable")
        # Documents a retriever did not match at all (BM25 score 0) get no credit from it,
        # so their index order cannot leak into the fused ranking.
        order = order[(scores[order] != 0) & np.isfinite(scores[order])]
        if depth is not None:
            order = order[:depth]
        ranks = np.arange(1, len(order) + 1)
        fused[order] += 1.0 / (k + ranks)
    return fused


class Hybrid:
    def __init__(self, retrievers: list, k: int = 60, depth: int | None = 200):
        self.retrievers, self.k, self.depth = retrievers, k, depth
        self.name = "hybrid-rrf(" + "+".join(r.name for r in retrievers) + ")"

    def fit(self, postings: list[Posting]) -> Hybrid:
        """Sub-retrievers must already be fitted on the same postings; they are shared."""
        self.n = len(postings)
        return self

    def scores(self, query: str) -> np.ndarray:
        return rrf([r.scores(query) for r in self.retrievers], self.n, self.k, self.depth)
